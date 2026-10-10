import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.types import Device

from fish_speech.models.text2semantic.fish_omni.config import (
    AudioDecoderConfig,
    FishOmniConfig,
)
from fish_speech.models.text2semantic.fish_omni.layers import (
    ZeroCenteredRMSNorm,
    grouped_attention,
)

# Repetition-aware sampling of the main token, fixed by the training recipe: when
# the first draw repeats one of the last RAS_WINDOW_FRAMES main tokens, a second
# draw with these parameters is used instead.
RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95
MIN_TEMPERATURE = 1e-5
TOWER_CHECKPOINT_PREFIXES = {
    "audio_decoder.codebook_embeddings.": "codebook_embeddings.",
    "audio_decoder_mtp_shared.layers.": "shared_layers.",
    "audio_decoder_mtp.layers.": "residual_layers.",
    "audio_decoder_mtp.heads.": "codebook_heads.",
}


def sample_top_k_top_p(
    logits: Tensor,
    temperature: float | Tensor,
    top_k: int | Tensor,
    top_p: float | Tensor,
    *,
    check_finite: bool = True,
) -> Tensor:
    """One id per row; drops the token crossing top_p, except the highest-ranked one.

    Sampling parameters may be 0-dim device tensors (CUDA graphs); check_finite
    syncs with the host.
    """
    if isinstance(temperature, Tensor):
        temperature = temperature.clamp(min=MIN_TEMPERATURE)
    else:
        temperature = max(temperature, MIN_TEMPERATURE)
    if check_finite and not torch.isfinite(logits).all():
        raise FloatingPointError("non-finite logits; refusing to sample invalid audio")
    values, ids = torch.sort(logits.float() / temperature, descending=True, dim=-1)
    ranks = torch.arange(values.shape[-1], device=values.device)
    dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
    dropped[..., 0] = False
    probabilities = values.masked_fill(dropped, -torch.inf).softmax(-1)
    noise = torch.empty_like(probabilities).exponential_()
    choice = (probabilities / noise).argmax(-1, keepdim=True)
    return ids.gather(-1, choice).squeeze(-1)


class TowerAttention(nn.Module):
    """Gated attention over time; checkpoint wqkv rows are [Q | Gate | K | V]."""

    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.config = config
        q_width = config.n_head * config.head_dim
        kv_width = config.n_local_heads * config.head_dim
        linear = dict(bias=False, dtype=dtype, device=device)
        norm = dict(dtype=dtype, device=device)
        self.wqkv = nn.Linear(config.dim, 2 * q_width + 2 * kv_width, **linear)
        self.wo = nn.Linear(q_width, config.dim, **linear)
        # Always on, whatever audio_decoder_config.attention_qk_norm says: released S3
        # configs set it to false, but their q_norm/k_norm weights are all far from the
        # zero init, so the decoder was trained with q/k norm.
        self.q_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps, **norm)
        self.k_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps, **norm)
        self.k_cache: Tensor | None = None
        self.v_cache: Tensor | None = None
        # CUDA-graph decode attends this many cache rows; the mask hides the rest.
        self.static_kv_len: int | None = None

    def setup_cache(self, max_seq_len: int) -> None:
        shape = (1, self.config.n_local_heads, max_seq_len, self.config.head_dim)
        self.k_cache = self.wqkv.weight.new_zeros(shape)
        self.v_cache = self.wqkv.weight.new_zeros(shape)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Interleaved (non-neox) partial RoPE, computed in fp32."""
        width = self.config.rotary_dim
        exponents = torch.arange(0, width, 2, device=x.device, dtype=torch.float32)
        angles = positions.float()[:, None] * self.config.rope_base ** (
            -exponents / width
        )
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        even, odd = x[..., :width:2].float(), x[..., 1:width:2].float()
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), -1)
        return torch.cat((rotated.flatten(-2).to(x.dtype), x[..., width:]), dim=-1)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        cfg = self.config
        length = len(positions)
        q_width, kv_width = cfg.n_head * cfg.head_dim, cfg.n_local_heads * cfg.head_dim
        q, gate, k, v = self.wqkv(x).split((q_width, q_width, kv_width, kv_width), -1)
        q = self.q_norm(q.reshape(length, cfg.n_head, cfg.head_dim))
        k = self.k_norm(k.reshape(length, cfg.n_local_heads, cfg.head_dim))
        q = self.rotate(q, positions).transpose(0, 1).unsqueeze(0)
        k = self.rotate(k, positions).transpose(0, 1).unsqueeze(0)
        v = v.reshape(length, cfg.n_local_heads, cfg.head_dim)
        v = v.transpose(0, 1).unsqueeze(0)
        if self.k_cache is not None:
            if self.static_kv_len is not None:
                end = self.static_kv_len
            else:
                end = int(positions[-1]) + 1
                assert end <= self.k_cache.shape[2], "tower cache capacity exceeded"
            self.k_cache.index_copy_(2, positions, k)
            self.v_cache.index_copy_(2, positions, v)
            k, v = self.k_cache[:, :, :end], self.v_cache[:, :, :end]
            key_positions = torch.arange(end, device=x.device)
        else:
            key_positions = positions
        mask = key_positions[None, :] <= positions[:, None]
        out = grouped_attention(q, k, v, mask)
        out = out.squeeze(0).transpose(0, 1).reshape(length, q_width)
        return self.wo(out * gate.sigmoid())


class FP32OutputLinear(nn.Linear):
    """FP16 GEMM with FP32 output, keeping values beyond the FP16 range."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != torch.float16:
            return super().forward(x)
        if x.is_cuda:
            out = torch.mm(
                x.reshape(-1, self.in_features), self.weight.T, out_dtype=torch.float32
            )
            out = out.reshape(*x.shape[:-1], self.out_features)
        else:
            out = F.linear(x.float(), self.weight.float())
        return out if self.bias is None else out + self.bias.float()


class TowerMLP(nn.Module):
    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        linear = dict(bias=False, dtype=dtype, device=device)
        self.w1 = nn.Linear(config.dim, config.intermediate_size, **linear)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, **linear)
        # The checkpoint's down projections exceed 65504 under FP16.
        self.w2 = FP32OutputLinear(config.intermediate_size, config.dim, **linear)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TowerTransformerLayer(nn.Module):
    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        args = dict(dtype=dtype, device=device)
        self.attention = TowerAttention(config, **args)
        self.attention_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.ffn_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.feed_forward = TowerMLP(config, **args)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        dtype = self.attention.wqkv.weight.dtype
        residual = x.float() if dtype == torch.float16 else x
        residual = residual + self.attention(
            self.attention_norm(residual).to(dtype), positions
        )
        return residual + self.feed_forward(self.ffn_norm(residual).to(dtype))


class TowerLayer(nn.Module):
    """Mix a code embedding into the hidden state, then one pre-norm block."""

    def __init__(
        self,
        config: AudioDecoderConfig,
        *,
        shared: bool,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.shared = shared
        args = dict(dtype=dtype, device=device)
        # Checkpoint name of the embedding differs between shared / residual layers.
        name = "semantic_embedding" if shared else "codebook_embedding"
        self.add_module(name, nn.Embedding(config.vocab_size, config.dim, **args))
        self.enorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.hnorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False, **args)
        self.transformer_layer = TowerTransformerLayer(config, **args)
        self.final_layernorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps, **args)

    def forward(
        self,
        hidden: Tensor,
        codes: Tensor,
        semantic: Tensor,
        positions: Tensor,
        shared_input: Tensor | None = None,
    ) -> Tensor:
        embedding = self.semantic_embedding if self.shared else self.codebook_embedding
        codes = codes.clamp(0, embedding.num_embeddings - 1)
        e = torch.where(semantic[:, None], embedding(codes), 0.0)
        if shared_input is not None:
            e = e + shared_input
        mixed = self.eh_proj(torch.cat((self.enorm(e), self.hnorm(hidden)), dim=-1))
        out = self.final_layernorm(self.transformer_layer(mixed, positions))
        return out.to(mixed.dtype)


class FishOmniAudioDecoder(nn.Module):
    """Codebook tower: predicts codes 1..N-1 of each frame, attending over time.

    A frame row is [main token id, code 0, ..., code N-1]; code 0 is the main
    token's offset in the semantic band.
    """

    def __init__(
        self,
        config: FishOmniConfig,
        *,
        dtype: torch.dtype = torch.float16,
        device: Device = None,
    ) -> None:
        super().__init__()
        cfg = self.config = config.audio_decoder
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id
        args = dict(dtype=dtype, device=device)
        self.codebook_embeddings = nn.Embedding(
            cfg.vocab_size * cfg.num_codebooks, cfg.dim, **args
        )
        self.shared_layers = nn.ModuleList(
            TowerLayer(cfg, shared=True, **args)
            for _ in range(cfg.shared_enrichment_layers)
        )
        self.residual_layers = nn.ModuleList(
            TowerLayer(cfg, shared=False, **args) for _ in range(cfg.num_codebooks - 1)
        )
        self.codebook_heads = nn.ModuleList(
            nn.Linear(cfg.dim, cfg.vocab_size, bias=False, **args)
            for _ in range(cfg.num_codebooks - 1)
        )
        # Ids the main token is drawn from (eos + the semantic band), built lazily
        # on the tower's device: creating it inside CUDA-graph capture is illegal.
        self.main_candidates: Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.codebook_embeddings.weight.device

    def setup_caches(self, max_seq_len: int) -> None:
        for layer in (*self.shared_layers, *self.residual_layers):
            layer.transformer_layer.attention.setup_cache(max_seq_len)

    def is_semantic(self, tokens: Tensor) -> Tensor:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: Tensor, first: int, end: int) -> Tensor:
        """Summed embeddings of codes first..end-1 per frame."""
        if first == end:
            return self.codebook_embeddings.weight.new_zeros(
                (len(frames), self.config.dim)
            )
        offsets = torch.arange(first, end, device=self.device) * self.config.vocab_size
        rows = frames[:, 1 + first : 1 + end] + offsets
        return F.embedding_bag(rows, self.codebook_embeddings.weight, mode="sum")

    def embed_backbone(self, frames: Tensor, text_embeddings: Tensor) -> Tensor:
        """Backbone input: semantic frames add codes 1..K to the token embedding."""
        frames = frames.to(self.device)
        count = self.config.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count).to(text_embeddings.device)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0]).to(text_embeddings.device)
        return torch.where(semantic[:, None], mixed, text_embeddings)

    def run_shared(
        self,
        hidden: Tensor,
        c0: Tensor,
        semantic: Tensor,
        positions: Tensor,
        frames: Tensor,
    ) -> Tensor:
        # The first shared layer also sees the codes the backbone did not.
        unseen = self.embedding_sum(
            frames, 1 + self.config.vq_backbone_codebooks, self.config.num_codebooks
        )
        unseen = torch.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(self.shared_layers):
            hidden = layer(
                hidden, c0, semantic, positions, unseen if index == 0 else None
            )
        return hidden

    def prefill(
        self, hidden: Tensor, positions: Tensor, frames: Tensor, next_frames: Tensor
    ) -> None:
        """Teacher-force the tower cache; row p is labelled with frame p + 1."""
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, positions, frames)
        for index, layer in enumerate(self.residual_layers):
            hidden = layer(hidden, next_frames[:, index + 1], semantic, positions)

    def decode_step(
        self,
        logits: Tensor,
        hidden: Tensor,
        positions: Tensor,
        frames: Tensor,
        history: Tensor,
        *,
        temperature: float | Tensor,
        top_k: int | Tensor,
        top_p: float | Tensor,
        check_finite: bool = True,
    ) -> Tensor:
        """Sample the frame after each row; the main token is drawn over eos + semantics."""
        if self.main_candidates is None or self.main_candidates.device != self.device:
            eos = torch.tensor([self.eos_id], device=self.device)
            band = torch.arange(
                self.semantic_begin, self.semantic_end + 1, device=self.device
            )
            self.main_candidates = torch.cat((eos, band)).sort().values
        candidates = self.main_candidates
        candidate_logits = logits[:, candidates]
        first = candidates[
            sample_top_k_top_p(
                candidate_logits, temperature, top_k, top_p, check_finite=check_finite
            )
        ]
        retry = candidates[
            sample_top_k_top_p(
                candidate_logits,
                RAS_TEMPERATURE,
                RAS_TOP_K,
                RAS_TOP_P,
                check_finite=False,
            )
        ]
        # Only audio tokens are resampled, as in S2: the window starts with prompt rows,
        # which hold the user turn's eos, and a stop must not be redrawn.
        repeated = (history[-RAS_WINDOW_FRAMES:] == first[:, None]).any(-1)
        repeated &= self.is_semantic(first)
        main = torch.where(repeated, retry, first)
        c0 = (main - self.semantic_begin).clamp(0, self.config.vocab_size - 1)
        semantic = self.is_semantic(main)
        hidden = self.run_shared(hidden, c0, semantic, positions, frames)
        codes = [c0]
        for layer, head in zip(self.residual_layers, self.codebook_heads):
            hidden = layer(hidden, codes[-1], semantic, positions)
            codes.append(
                sample_top_k_top_p(
                    head(hidden), temperature, top_k, top_p, check_finite=check_finite
                )
            )
        return torch.stack((main, *codes), dim=-1)


@torch.no_grad()
def load_audio_decoder_weights(
    model: FishOmniAudioDecoder, weights: Iterable[tuple[str, Tensor]]
) -> None:
    """Copy audio_decoder* checkpoint tensors into model, each parameter exactly once."""
    targets = dict(model.named_parameters())
    loaded = set()
    for key, tensor in weights:
        name = next(
            (
                new + key[len(old) :]
                for old, new in TOWER_CHECKPOINT_PREFIXES.items()
                if key.startswith(old)
            ),
            None,
        )
        if name not in targets:
            raise ValueError(f"unknown tower weight {key}")
        if name in loaded:
            raise ValueError(f"tower weight filled twice: {key}")
        target = targets[name]
        if target.shape != tensor.shape:
            raise ValueError(f"{key}: shape {tensor.shape} != {target.shape}")
        value = tensor.to(device=target.device, dtype=target.dtype)
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{key} cannot be represented in {target.dtype}")
        target.copy_(value)
        loaded.add(name)
    if missing := targets.keys() - loaded:
        raise ValueError(f"tower weights not loaded: {sorted(missing)[:8]}")
    logger.info(f"FishOmni tower: loaded {len(loaded)} tensors on {model.device}")
