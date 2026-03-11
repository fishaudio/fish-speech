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

from fish_speech.models.text2semantic.lora import LoraConfig, setup_lora


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

    semantic_begin_id: int = 0
    semantic_end_id: int = 0

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Initialize the model
    initializer_range: float = 0.02

    # Dummy vars
    is_reward_model: bool = False
    scale_codebook_embeddings: bool = False
    audio_embed_dim: Optional[int] = None

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
            case "fish_qwen3_omni":
                return BaseModelArgs._from_fish_qwen3_omni(data)
            case _:
                raise ValueError(f"Unknown model type: {data['model_type']}")

        # Filter out unexpected keyword arguments
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in data.items() if k in valid_keys}

        return cls(**data)

    @staticmethod
    def _from_fish_qwen3_omni(data: dict) -> "DualARModelArgs":
        tc = data["text_config"]
        adc = data["audio_decoder_config"]
        flat = dict(
            model_type="dual_ar",
            vocab_size=tc["vocab_size"],
            n_layer=tc["n_layer"],
            n_head=tc["n_head"],
            n_local_heads=tc.get("n_local_heads", -1),
            head_dim=tc.get("head_dim"),
            dim=tc["dim"],
            intermediate_size=tc.get("intermediate_size"),
            rope_base=tc.get("rope_base", 10000),
            norm_eps=tc.get("norm_eps", 1e-5),
            max_seq_len=tc.get("max_seq_len", 2048),
            dropout=tc.get("dropout", 0.0),
            tie_word_embeddings=tc.get("tie_word_embeddings", True),
            attention_qkv_bias=tc.get("attention_qkv_bias", False),
            attention_o_bias=tc.get("attention_o_bias", False),
            attention_qk_norm=tc.get("attention_qk_norm", False),
            use_gradient_checkpointing=tc.get("use_gradient_checkpointing", True),
            initializer_range=tc.get("initializer_range", 0.02),
            semantic_begin_id=data.get("semantic_start_token_id", 0),
            semantic_end_id=data.get("semantic_end_token_id", 0),
            scale_codebook_embeddings=True,
            norm_fastlayer_input=True,
            audio_embed_dim=adc.get("text_dim", tc["dim"]),
            codebook_size=adc["vocab_size"],
            num_codebooks=adc["num_codebooks"],
            n_fast_layer=adc["n_layer"],
            fast_dim=adc.get("dim"),
            fast_n_head=adc.get("n_head"),
            fast_n_local_heads=adc.get("n_local_heads"),
            fast_head_dim=adc.get("head_dim"),
            fast_intermediate_size=adc.get("intermediate_size"),
            fast_attention_qkv_bias=adc.get("attention_qkv_bias"),
            fast_attention_qk_norm=adc.get("attention_qk_norm"),
            fast_attention_o_bias=adc.get("attention_o_bias"),
        )
        valid_keys = {f.name for f in dataclasses.fields(DualARModelArgs)}
        flat = {k: v for k, v in flat.items() if k in valid_keys and v is not None}
        return DualARModelArgs(**flat)

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
    norm_fastlayer_input: bool = False

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


def _remap_fish_qwen3_omni_keys(weights: OrderedDict) -> OrderedDict:
    if not any(k.startswith(("text_model.", "audio_decoder.")) for k in weights):
        return weights
    new_weights = OrderedDict()
    for k, v in weights.items():
        if k.startswith("text_model.model."):
            new_key = k[len("text_model.model.") :]
        elif k.startswith("audio_decoder."):
            suffix = k[len("audio_decoder.") :]
            new_key = (
                suffix
                if suffix.startswith("codebook_embeddings.")
                else "fast_" + suffix
            )
        else:
            new_key = k
        new_weights[new_key] = v
    return new_weights


class BaseTransformer(nn.Module):
    def __init__(
        self,
        config: BaseModelArgs,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config

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

        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(
                inp[:, i + 1] + i * self.config.codebook_size
            )
            embeds.append(emb)

        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)

        is_semantic = (inp[:, 0] >= self.config.semantic_begin_id) & (
            inp[:, 0] <= self.config.semantic_end_id
        )

        vq_embeds_sum[~is_semantic] = 0

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

        mask = None
        if key_padding_mask is not None:
            causal = self.causal_mask[:seq_len, :seq_len]
            causal = rearrange(causal, "q k -> 1 1 q k")

            atten_mask = rearrange(key_padding_mask, "b s -> b 1 1 s")
            atten_mask = atten_mask.logical_not()
            mask = causal & atten_mask

        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
            else:
                x = layer(x, freqs_cis, mask)

        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        hidden_out = (
            slow_out if getattr(self.config, "norm_fastlayer_input", False) else x
        )

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=hidden_out,
        )

    def forward_generate(
        self,
        inp: Tensor,
        input_pos: Optional[Tensor] = None,
        audio_masks: Optional[Tensor] = None,
        audio_parts: Optional[Tensor] = None,
        return_all: bool = False,
    ) -> BaseTransformerForwardResult:

        # Embedding logic replicated from embed() for compilation compatibility
        embeds = []
        for i in range(self.config.num_codebooks):
            emb = self.codebook_embeddings(
                inp[:, i + 1] + i * self.config.codebook_size
            )
            embeds.append(emb)

        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)

        vq_masks = (inp[:, 0] >= self.config.semantic_begin_id) & (
            inp[:, 0] <= self.config.semantic_end_id
        )

        vq_embeds_sum[~vq_masks] = 0
        x = self.embeddings(inp[:, 0]) + vq_embeds_sum

        if self.config.scale_codebook_embeddings:
            vq_masks_expanded = vq_masks.unsqueeze(-1).expand_as(x)
            x = torch.where(
                vq_masks_expanded, x / math.sqrt(self.config.num_codebooks + 1), x
            )

        # Audio embeddings
        if audio_parts is not None:
            # Note: This assumes self.audio_projector exists if audio_parts is used
            # It seems missing in init, but we keep existing logic
            if hasattr(self, "audio_projector"):
                audio_embeds = self.audio_projector(audio_parts)
                if self.config.scale_codebook_embeddings:
                    x[audio_masks] = audio_embeds / math.sqrt(2)
                else:
                    x[audio_masks] = audio_embeds
            else:
                logger.warning("audio_parts provided but model has no audio_projector")

        if input_pos is None:
            input_pos = torch.arange(inp.shape[-1], device=x.device)
            max_seq_len = inp.shape[-1]
        else:
            max_seq_len = self.max_seq_len

        mask = self.causal_mask[None, None, input_pos, :max_seq_len]  # (B, N, Q, K)
        freqs_cis = self.freqs_cis[input_pos]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)

        if x.size(1) > 1 and not return_all:
            x = x[:, -1:]

        slow_out = self.norm(x)

        if self.config.is_reward_model:
            token_logits = self.score_output(slow_out)
        elif self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        hidden_out = (
            slow_out if getattr(self.config, "norm_fastlayer_input", False) else x
        )

        return BaseTransformerForwardResult(
            logits=token_logits,
            hidden_states=hidden_out,
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
        from fish_speech.utils.model_type import get_fish_model_type

        fish_model_type = get_fish_model_type()
        config = BaseModelArgs.from_pretrained(str(path))
        if max_length is not None:
            config.max_seq_len = max_length
            logger.info(f"Override max_seq_len to {max_length}")

        if rope_base is not None:
            config.rope_base = rope_base
            logger.info(f"Override rope_base to {rope_base}")

        # Validate checkpoint type before tokenizer loading to surface a clear error.
        path_obj = Path(path)
        config_json_path = path_obj / "config.json" if path_obj.is_dir() else path_obj
        if config_json_path.exists():
            with open(config_json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            raw_model_type = raw_data.get("model_type", "unknown")
            if fish_model_type == "s1" and raw_model_type == "fish_qwen3_omni":
                raise ValueError(
                    "FISH_MODEL_TYPE=s1 but checkpoint has model_type=fish_qwen3_omni. "
                    "This is an S2 checkpoint. Set FISH_MODEL_TYPE=s2 or use an S1 checkpoint."
                )
            if fish_model_type == "s2" and raw_model_type == "dual_ar":
                raise ValueError(
                    "FISH_MODEL_TYPE=s2 but checkpoint has model_type=dual_ar. "
                    "This is an S1 checkpoint. Set FISH_MODEL_TYPE=s1 or use an S2 checkpoint."
                )

        if fish_model_type == "s1":
            from fish_speech.tokenizer_s1 import FishTokenizerS1

            tokenizer = FishTokenizerS1.from_pretrained(str(path))
            logger.info("Loaded S1 tiktoken tokenizer")
        else:
            from fish_speech.tokenizer import FishTokenizer

            tokenizer = FishTokenizer.from_pretrained(str(path))
            logger.info("Loaded S2 HuggingFace tokenizer")

        config.semantic_begin_id = tokenizer.semantic_begin_id
        config.semantic_end_id = tokenizer.semantic_end_id
        logger.info(
            f"Injected Semantic IDs into Config: {config.semantic_begin_id}-{config.semantic_end_id}"
        )

        match config.model_type:
            case "naive":
                model_cls = NaiveTransformer
            case "dual_ar":
                model_cls = DualARTransformer
            case _:
                raise ValueError(f"Unknown model type: {config.model_type}")

        logger.info(f"Loading model from {path}, config: {config}")
        # Initialize model without passing tokenizer explicitly to __init__
        model = model_cls(config)
        # Attach tokenizer to model instance for inference convenience (optional, but good for user scripts)
        model.tokenizer = tokenizer

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

            path_obj = Path(path)
            index_json = path_obj / "model.safetensors.index.json"
            single_st = path_obj / "model.safetensors"
            pth_file = path_obj / "model.pth"

            if index_json.exists():
                logger.info("Loading sharded safetensors weights")
                from safetensors.torch import load_file as st_load_file

                with open(index_json) as f:
                    st_index = json.load(f)
                shard_files = sorted(set(st_index["weight_map"].values()))
                weights = OrderedDict()
                for shard in shard_files:
                    weights.update(st_load_file(str(path_obj / shard), device="cpu"))
                weights = _remap_fish_qwen3_omni_keys(weights)
            elif single_st.exists():
                logger.info("Loading single safetensors weights")
                from safetensors.torch import load_file as st_load_file

                weights = OrderedDict(st_load_file(str(single_st), device="cpu"))
                weights = _remap_fish_qwen3_omni_keys(weights)
            elif pth_file.exists():
                weights = torch.load(
                    pth_file,
                    map_location="cpu",
                    mmap=True,
                    weights_only=True,
                )
                if "state_dict" in weights:
                    weights = weights["state_dict"]
                if weights and next(iter(weights.keys())).startswith("model."):
                    weights = OrderedDict(
                        (k.replace("model.", ""), v) for k, v in weights.items()
                    )
                for k in list(weights.keys()):
                    if "audio_" in k:
                        weights.pop(k)
            else:
                raise FileNotFoundError(f"No model weights found in {path_obj}")

            err = model.load_state_dict(weights, strict=False, assign=True)
            logger.info(f"Model weights loaded - Status: {err}")

        if lora_config is not None:
            setup_lora(model, lora_config)
            logger.info(f"LoRA setup: {lora_config}")

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

        torch.save(state_dict, path / "model.pth")
        if hasattr(self, "tokenizer"):
            self.tokenizer.save_pretrained(path)


class NaiveTransformer(BaseTransformer):
    def __init__(self, config: NaiveModelArgs) -> None:
        super().__init__(config, init_weights=False)

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
    def __init__(self, config: NaiveModelArgs) -> None:
        super().__init__(config, init_weights=False)

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
        token_labels = labels[:, 0]

        # [MODIFIED] Use config instead of tokenizer
        codebook_mask = (token_labels >= self.config.semantic_begin_id) & (
            token_labels <= self.config.semantic_end_id
        )

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
            all_codebooks = labels[:, 1:, :]
            all_codebooks_permuted = all_codebooks.permute(0, 2, 1)
            semantic_codebooks = all_codebooks_permuted[codebook_mask]
            codebooks = semantic_codebooks[:, :-1]

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
