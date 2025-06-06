import dataclasses
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora
from fish_speech.tokenizer import SEMANTIC_TOKENS, FishTokenizer


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class BaseModelArgs:
    model_type: str = "base"

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = False

    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 4

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Initialize the model
    initializer_range: float = 0.02

    # Dummy vars
    is_reward_model: bool = False
    scale_codebook_embeddings: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_head

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        match data["model_type"]:
            case "naive":
                cls = NaiveModelArgs
            case "dual_ar":
                cls = DualARModelArgs
            case _:
                raise ValueError(f"Unknown model type: {data['model_type']}")

        return cls(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


@dataclass
class NaiveModelArgs(BaseModelArgs):
    model_type: str = "naive"


@dataclass
class DualARModelArgs(BaseModelArgs):
    model_type: str = "dual_ar"
    n_fast_layer: int = 4
    fast_dim: int | None = None
    fast_n_head: int | None = None
    fast_n_local_heads: int | None = None
    fast_head_dim: int | None = None
    fast_intermediate_size: int | None = None
    fast_attention_qkv_bias: bool | None = None
    fast_attention_qk_norm: bool | None = None
    fast_attention_o_bias: bool | None = None

    def __post_init__(self):
        super().__post_init__()

        self.fast_dim = self.fast_dim or self.dim
        self.fast_n_head = self.fast_n_head or self.n_head
        self.fast_n_local_heads = self.fast_n_local_heads or self.n_local_heads
        self.fast_head_dim = self.fast_head_dim or self.head_dim
        self.fast_intermediate_size = (
            self.fast_intermediate_size or self.intermediate_size
        )
        self.fast_attention_qkv_bias = (
            self.fast_attention_qkv_bias
            if self.fast_attention_qkv_bias is not None
            else self.attention_qkv_bias
        )
        self.fast_attention_qk_norm = (
            self.fast_attention_qk_norm
            if self.fast_attention_qk_norm is not None
            else self.attention_qk_norm
        )
        self.fast_attention_o_bias = (
            self.fast_attention_o_bias
            if self.fast_attention_o_bias is not None
            else self.attention_o_bias
        )


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_len, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    codebook_logits: Tensor


@dataclass
class BaseTransformerForwardResult:
    logits: Tensor
    hidden_states: Tensor


class BaseTransformer(nn.Module):
    def __init__(
        self,
        config: BaseModelArgs,
        tokenizer: FishTokenizer,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.semantic_token_ids = list(tokenizer.semantic_id_to_token_id.values())

        # Slow transformer
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.head_dim,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

        if init_weights:
            self.apply(self._init_weights)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        if self.max_seq_len >= max_seq_len and self.max_batch_size >= max_batch_size:
            return

        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype=dtype,
            )

    def embed(self, inp: Tensor) -> Tensor:
        embeds = []
        semantic_token_ids_tensor = torch.tensor(
            self.semantic_token_ids, device=inp.device, dtype=inp.dtype
        )

        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(
                inp[:, i + 1] + i * self.config.codebook_size
            )
            embeds.append(emb)

        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)
        vq_embeds_sum[~torch.isin(inp[:, 0], semantic_token_ids_tensor)] = 0
        x = self.embeddings(inp[:, 0]) + vq_embeds_sum

        return x

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> BaseTransformerForwardResult:
        seq_len = inp.size(2)

        # Here we want to merge the embeddings of the codebooks
        x = self.embed(inp)

        freqs_cis = self.freqs_cis[:seq_len]

        # Not that the causal mask here follows the definition of scaled_dot_product_attention
        # That is, FALSE means masked out
        # To maintain consistency, key_padding_mask use TRUE to mask out
        mask = None
        if key_padding_mask is not None:
            causal = self.causal_mask[:seq_len, :seq_len]
            causal = rearrange(causal, "q k -> 1 1 q k")

            atten_mask = rearrange(key_padding_mask, "b s -> b 1 1 s")
            atten_mask = atten_mask.logical_not()
            mask = causal & atten_mask

        # return freqs_cis, mask

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask)

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def forward_generate(
        self,
        inp: Tensor,
        input_pos: Optional[Tensor] = None,
        audio_masks: Optional[Tensor] = None,
        audio_parts: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> BaseTransformerForwardResult:
        # This is used for generation, optimized for torch compile
        # assert (
        #     self.max_seq_len != -1 and self.max_batch_size != -1
        # ), "Please call setup_caches before forward_generate"

        embeds = []
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(
                inp[:, i + 1] + i * self.config.codebook_size
            )
            embeds.append(emb)

        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)

        vq_masks = (inp[:, 0] >= self.tokenizer.semantic_begin_id) & (
            inp[:, 0] <= self.tokenizer.semantic_end_id
        )

        vq_embeds_sum[~vq_masks] = 0
        x = self.embeddings(inp[:, 0]) + vq_embeds_sum

        if self.config.scale_codebook_embeddings:
            # Expand vq_masks to match x's shape
            vq_masks_expanded = vq_masks.unsqueeze(-1).expand_as(x)
            x = torch.where(
                vq_masks_expanded, x / math.sqrt(self.config.num_codebooks + 1), x
            )

        # Audio embeddings
        if audio_parts is not None:
            audio_embeds = self.audio_projector(audio_parts)
            if self.config.scale_codebook_embeddings:
                x[audio_masks] = audio_embeds / math.sqrt(2)
            else:
                x[audio_masks] = audio_embeds

        if input_pos is None:
            input_pos = torch.arange(inp.shape[-1], device=x.device)
            max_seq_len = inp.shape[-1]
        else:
            max_seq_len = self.max_seq_len

        mask = self.causal_mask[None, None, input_pos, :max_seq_len]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        # If prefill, we only calculate the logits of last token
        if x.size(1) > 1 and not return_all:
            x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.is_reward_model:
            token_logits = self.score_output(slow_out)
        elif self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=x,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @staticmethod
    def from_pretrained(
        path: str,
        load_weights: bool = False,
        max_length: int | None = None,
        lora_config: LoraConfig | None = None,
        rope_base: int | None = None,
    ) -> "BaseTransformer":
        config = BaseModelArgs.from_pretrained(str(path))
        if max_length is not None:
            config.max_seq_len = max_length
            logger.info(f"Override max_seq_len to {max_length}")

        if rope_base is not None:
            config.rope_base = rope_base
            logger.info(f"Override rope_base to {rope_base}")

        match config.model_type:
            case "naive":
                model_cls = NaiveTransformer
            case "dual_ar":
                model_cls = DualARTransformer
            case _:
                raise ValueError(f"Unknown model type: {config.model_type}")

        tokenizer = FishTokenizer.from_pretrained(path)

        logger.info(f"Loading model from {path}, config: {config}")
        model = model_cls(config, tokenizer=tokenizer)

        if lora_config is not None:
            setup_lora(model, lora_config)
            logger.info(f"LoRA setup: {lora_config}")

        if load_weights is False:
            logger.info("Randomly initialized model")
        else:

            if "int8" in str(Path(path)):
                logger.info("Using int8 weight-only quantization!")
                from tools.llama.quantize import WeightOnlyInt8QuantHandler

                simple_quantizer = WeightOnlyInt8QuantHandler(model)
                model = simple_quantizer.convert_for_runtime()

            if "int4" in str(Path(path)):
                logger.info("Using int4 quantization!")
                path_comps = path.name.split("-")
                assert path_comps[-2].startswith("g")
                groupsize = int(path_comps[-2][1:])
                from tools.llama.quantize import WeightOnlyInt4QuantHandler

                simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
                model = simple_quantizer.convert_for_runtime()

            weights = torch.load(
                Path(path) / "model.pth",
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )

            if "state_dict" in weights:
                logger.warning(
                    "Using a TextToSemantic LightningModule checkpoint, "
                    "please make sure it is a full model, not a LoRA model."
                )
                weights = weights["state_dict"]

            if next(iter(weights.keys())).startswith("model."):
                logger.info(
                    f"Remove prefix 'model.' created by TextToSemantic LightningModule from keys"
                )
                new_weights = OrderedDict()
                for k, v in weights.items():
                    new_weights[k.replace("model.", "")] = v
                weights = new_weights

            # Remove audio related weights
            for k in list(weights.keys()):
                if "audio_" in k:
                    weights.pop(k)

            # Verify the name and shape of parameters since strict=False in load_state_dict.
            for k, v in model.named_parameters():
                if k not in weights:
                    logger.warning(f"No weight for {k}")
                elif v.shape != weights[k].shape:
                    logger.warning(
                        f"Shape mismatch for {k}: {v.shape} vs {weights[k].shape}"
                    )

            err = model.load_state_dict(weights, strict=False, assign=True)
            logger.info(f"Loaded weights with error: {err}")

        return model

    def save_pretrained(self, path: str, drop_lora: bool = False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(path / "config.json")
        state_dict = self.state_dict()

        if drop_lora:
            for key in list(state_dict.keys()):
                if "lora" not in key:
                    continue

                state_dict.pop(key)
                logger.info(f"Drop LoRA parameter: {key}")

        torch.save(state_dict, path / "model.pth")
        self.tokenizer.save_pretrained(path)


class NaiveTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs, tokenizer: FishTokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        self.codebook_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.codebook_output = nn.Linear(
            config.dim,
            config.codebook_size * config.num_codebooks,
            bias=False,
        )

        self.apply(self._init_weights)

    def decode(self, result: BaseTransformerForwardResult) -> TransformerForwardResult:
        token_logits = result.logits
        x = result.hidden_states

        # Codebook
        codebook_logits = self.codebook_output(self.codebook_norm(x))
        codebook_logits = rearrange(
            codebook_logits, "b n (c d) -> b n c d", c=self.config.num_codebooks
        )

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward(
        self,
        inp: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        result = super().forward(
            inp=inp,
            key_padding_mask=key_padding_mask,
        )
        return self.decode(result)

    def forward_generate(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        result = super().forward_generate(x, input_pos)
        return self.decode(result)


class DualARTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs, tokenizer: FishTokenizer) -> None:
        super().__init__(config, init_weights=False, tokenizer=tokenizer)

        # Project to fast dim if needed
        if config.fast_dim is not None and config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()

        # Fast transformer
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)

        # The equivalent bs is so large that sdpa doesn't work
        override_config = dataclasses.replace(
            config,
            dim=config.fast_dim,
            n_head=config.fast_n_head,
            n_local_heads=config.fast_n_local_heads,
            head_dim=config.fast_head_dim,
            intermediate_size=config.fast_intermediate_size,
            attention_qkv_bias=config.fast_attention_qkv_bias,
            attention_qk_norm=config.fast_attention_qk_norm,
            attention_o_bias=config.fast_attention_o_bias,
        )

        self.fast_layers = nn.ModuleList(
            TransformerBlock(override_config, use_sdpa=False)
            for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.fast_dim,
            config.codebook_size,
            bias=False,
        )

        self.register_buffer(
            "fast_freqs_cis",
            precompute_freqs_cis(
                config.num_codebooks,
                config.fast_head_dim,
                config.rope_base,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().setup_caches(max_batch_size, max_seq_len, dtype)

        # Fast transformer
        # The max seq len here is the number of codebooks
        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.fast_n_local_heads,
                self.config.fast_head_dim,
                dtype=dtype,
            )

    def forward(
        self,
        inp: Tensor,
        labels: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        vq_parts: Optional[Tensor] = None,
        vq_masks: Optional[Tensor] = None,
        vq_require_losses: Optional[Tensor] = None,
        mel_parts: Optional[Tensor] = None,
        mel_masks: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        parent_result = super().forward(
            inp=inp,
            key_padding_mask=key_padding_mask,
            vq_parts=vq_parts,
            vq_masks=vq_masks,
            mel_parts=mel_parts,
            mel_masks=mel_masks,
        )
        token_logits = parent_result.logits
        x = parent_result.hidden_states

        # Fast transformer
        fast_seq_len = self.config.num_codebooks
        fast_mask = self.causal_mask[
            None, None, :fast_seq_len, :fast_seq_len
        ]  # (B, N, Q, K)
        fast_freqs_cis = self.fast_freqs_cis[:fast_seq_len]

        # Extract corresponding parts with labels
        codebook_mask = labels == self.semantic_token_id
        # This gives where input token is <|semantic|>
        x = x[codebook_mask]

        if x.shape[0] == 0:
            # Use dummy input when no vq is required
            x = torch.zeros(
                (4, self.config.dim),
                device=x.device,
                dtype=x.dtype,
            )
            codebooks = torch.zeros(
                (x.shape[0], self.config.num_codebooks - 1),
                device=x.device,
                dtype=torch.int,
            )
        else:
            codebooks = vq_parts[..., :-1][vq_require_losses][
                vq_masks[vq_require_losses]
            ]

        x = self.fast_project_in(x)
        codebook_embeddings = self.fast_embeddings(codebooks)
        x = torch.cat([x[:, None], codebook_embeddings], dim=1)

        for layer in self.fast_layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, fast_freqs_cis, fast_mask, use_reentrant=True)
            else:
                x = layer(x, fast_freqs_cis, fast_mask)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)
        codebook_logits = self.fast_output(fast_out)

        assert codebook_logits.shape[1] == self.config.num_codebooks

        return TransformerForwardResult(
            token_logits=token_logits,
            codebook_logits=codebook_logits,
        )

    def forward_generate_fast(
        self, x: Tensor, input_pos: Optional[Tensor] = None
    ) -> Tensor:
        # Fast transformer
        x = x.view(x.shape[0], 1, -1)

        fast_mask = self.causal_mask[
            None, None, input_pos, : self.config.num_codebooks
        ]  # (B, N, Q, K)
        fast_freqs_cis = self.fast_freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis, fast_mask, input_pos=input_pos)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)  # only take the last token
        codebook_logits = self.fast_output(fast_out)

        return codebook_logits

    def forward_generate(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        audio_masks: Optional[Tensor] = None,
        audio_parts: Optional[Tensor] = None,
    ) -> TransformerForwardResult:
        x = super().forward_generate(x, input_pos, audio_masks, audio_parts)
        x.hidden_states = self.fast_project_in(x.hidden_states)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Tensor = None
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(
            config.dim, total_head_dim, bias=config.attention_qkv_bias
        )
        self.wo = nn.Linear(
            config.n_head * config.head_dim, config.dim, bias=config.attention_o_bias
        )
        self.kv_cache = None

        if config.attention_qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim, config.norm_eps)
            self.k_norm = nn.RMSNorm(config.head_dim, config.norm_eps)

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self.attention_qk_norm = config.attention_qk_norm
        self.config = config

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.attention_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.use_sdpa:
            if mask is None:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    y = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        dropout_p=self.dropout if self.training else 0.0,
                        is_causal=True,
                        # No third party attn_mask here to use flash_attention
                    )
            else:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            y = self.eq_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, q_size)

        return self.wo(y)

    def eq_scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
    ) -> torch.Tensor:
        # This is a standard scaled dot product attention
        # It's low efficient, but it doesn't raise cuda error

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(1, 1, L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        return attn_weight @ value


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    """
    Precomputes frequency tensors for complex exponentials (cis)

    Args:
        seq_len: Length of the sequence for which positional embeddings are needed.
        n_elem: Number of elements in the frequency tensor.
        base: Base value for the frequency scaling (default: 10000).

    Returns:
        A tensor containing the precomputed frequencies in real and imaginary parts (bfloat16).
    """
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
