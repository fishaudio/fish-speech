import math
import typing as tp
from dataclasses import dataclass
from typing import List, Optional, Union

import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from dac.model.base import CodecMixin
from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


@dataclass
class VQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    semantic_distill_z: torch.Tensor | None = None


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    block_size: int = 2048
    n_layer: int = 8
    n_head: int = 8
    dim: int = 512
    intermediate_size: int = 1536
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True  # to be compatible with conv1d input/output
    pos_embed_type: str = "rope"  # can be "rope" or "conformer"
    max_relative_position: int = 128  # for conformer-style relative position embedding

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        assert self.pos_embed_type in [
            "rope",
            "conformer",
        ], "pos_embed_type must be either 'rope' or 'conformer'"


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return (
            k_out[:, :, : input_pos.max() + 1, :],
            v_out[:, :, : input_pos.max() + 1, :],
        )

    def clear_cache(self, prompt_len):
        self.k_cache[:, :, prompt_len:, :].fill_(0)
        self.v_cache[:, :, prompt_len:, :].fill_(0)


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Only compute RoPE frequencies if using RoPE
        if config.pos_embed_type == "rope":
            freqs_cis = precompute_freqs_cis(
                self.config.block_size, self.config.head_dim, self.config.rope_base
            )
            self.register_buffer("freqs_cis", freqs_cis)
        else:
            self.register_buffer("freqs_cis", None)

        causal_mask = torch.tril(
            torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def setup_caches(self, max_batch_size, max_seq_length):
        """
        This method will only be called during inference when using KV cache.
        """
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.weight.dtype
        device = self.norm.weight.device

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                head_dim,
                dtype,
            ).to(device)

        self.use_kv_cache = True

    def forward(
        self,
        x: Tensor,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if self.config.pos_embed_type == "rope":
            assert (
                self.freqs_cis is not None
            ), "RoPE frequencies must be initialized for RoPE positional embedding"
            freqs_cis = self.freqs_cis[input_pos]
        else:
            freqs_cis = None

        if mask is None:  # in case of non-causal model
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., : input_pos.max() + 1]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = LayerScale(config.dim, inplace=True)

    def forward(
        self,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
    ) -> Tensor:
        h = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        )
        out = h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate
        self.pos_embed_type = config.pos_embed_type

        # Add relative position embedding for conformer-style
        if self.pos_embed_type == "conformer":
            self.max_relative_position = config.max_relative_position
            num_pos_embeddings = 2 * config.max_relative_position + 1
            self.rel_pos_embeddings = nn.Parameter(
                torch.zeros(num_pos_embeddings, self.head_dim)
            )
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)

    def _compute_conformer_pos_scores(self, q: Tensor, seqlen: int) -> Tensor:
        # q: [B, H, S, D]
        # Returns: [B, H, S, S]
        positions = torch.arange(seqlen, device=q.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # [S, S]
        relative_positions = torch.clamp(
            relative_positions + self.max_relative_position,
            0,
            2 * self.max_relative_position,
        )
        rel_embeddings = self.rel_pos_embeddings[relative_positions]  # [S, S, D]

        # Compute attention scores with relative position embeddings
        q = q.transpose(1, 2)  # [B, S, H, D]
        rel_logits = torch.matmul(q, rel_embeddings.transpose(-2, -1))  # [B, S, H, S]
        rel_logits = rel_logits.transpose(1, 2)  # [B, H, S, S]
        return rel_logits

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        if self.pos_embed_type == "rope":
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.pos_embed_type == "conformer":
            # Compute attention scores
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Add relative position embeddings for conformer-style
            rel_scores = self._compute_conformer_pos_scores(q, seqlen)
            scores = scores + rel_scores

            # Apply attention
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))

            attn = F.softmax(scores, dim=-1)
            if self.attn_dropout_rate > 0 and self.training:
                attn = F.dropout(attn, p=self.attn_dropout_rate)

            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout_rate if self.training else 0.0,
                attn_mask=mask,
            )
            # is_causal=True)
        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.head_dim * self.n_head)
        )
        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


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


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-2,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class WindowLimitedTransformer(Transformer):
    """
    Transformer with window limited attention, causal.
    """

    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 512,
        window_size: Optional[int] = None,
        causal: bool = True,
        look_ahead_conv: nn.Module = None,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.look_ahead_conv = (
            look_ahead_conv if look_ahead_conv is not None else nn.Identity()
        )
        self.input_proj = (
            nn.Linear(input_dim, config.dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim)
            if input_dim != config.dim
            else nn.Identity()
        )

    def make_window_limited_mask(
        self,
        max_length: int,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Make mask to form window limited attention.
        """
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
            row_indices = torch.arange(max_length).view(-1, 1)
            window_size = self.window_size or max_length
            valid_range = (row_indices - window_size + 1).clamp(min=0)
            column_indices = torch.arange(max_length)
            mask = (column_indices >= valid_range) & mask.bool()
        else:
            raise NotImplementedError
        mask = mask.bool()[None, None]
        return mask

    def make_mask(
        self,
        max_length: int,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Make ordinary mask if window size is not specified.
        """
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
        else:
            mask = torch.ones(max_length, max_length)
            mask = mask.bool()[None, None]
            for i, x_len in enumerate(x_lens):
                mask[:x_len, i] = 0
        mask = mask.bool()[None, None]
        return mask

    def forward(
        self,
        x: Tensor,
        x_lens: Optional[Tensor] = None,
    ) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.input_proj(x)  # (B, T, D)
        x = self.look_ahead_conv(x)
        input_pos = torch.arange(x.shape[1], device=x.device)
        # construct mask to form window limited attention
        max_length = x.shape[1]
        if self.window_size is not None:
            mask = self.make_window_limited_mask(max_length, x_lens)
        else:
            mask = self.make_mask(max_length, x_lens)
        mask = mask.to(x.device)
        x = super().forward(x, input_pos, mask)
        x = self.output_proj(x)  # (B, T, D)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


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


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right
    before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class CausalConvNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
        padding=None,
    ):
        super(CausalConvNet, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        pad = self.padding
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


class CausalTransConvNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=None
    ):
        super(CausalTransConvNet, self).__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


def CausalWNConv1d(*args, **kwargs):
    return CausalConvNet(*args, **kwargs).weight_norm()


def CausalWNConvTranspose1d(*args, **kwargs):
    return CausalTransConvNet(*args, **kwargs).weight_norm()


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=1),
        )
        self.causal = causal

    def forward(self, x):
        y = self.block(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            if self.causal:
                x = x[..., :-pad]
            else:
                x = x[..., pad // 2 : -pad // 2]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        transformer_module = (
            nn.Identity()
            if n_t_layer == 0
            else (
                WindowLimitedTransformer(
                    causal=causal,
                    input_dim=dim,
                    window_size=512,
                    config=transformer_general_config(
                        n_layer=n_t_layer,
                        n_head=dim // 64,
                        dim=dim,
                        intermediate_size=dim * 3,
                    ),
                )
            )
        )
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv_class(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            transformer_module,
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        n_transformer_layers: list = [0, 0, 4, 4],
        transformer_general_config: ModelArgs = None,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        # Create first convolution
        self.block = [conv_class(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride, n_t_layer in zip(strides, n_transformer_layers):
            d_model *= 2
            self.block += [
                EncoderBlock(
                    d_model,
                    stride=stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            conv_class(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_trans_class = CausalWNConvTranspose1d if causal else WNConvTranspose1d
        transformer_module = (
            nn.Identity()
            if n_t_layer == 0
            else (
                WindowLimitedTransformer(
                    causal=causal,
                    input_dim=input_dim,
                    window_size=None,
                    config=transformer_general_config(
                        n_layer=n_t_layer,
                        n_head=input_dim // 64,
                        dim=input_dim,
                        intermediate_size=input_dim * 3,
                    ),
                )
            )
        )
        self.block = nn.Sequential(
            # transformer_module,
            Snake1d(input_dim),
            conv_trans_class(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: list = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        # Add first conv layer
        layers = [conv_class(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (stride, n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            conv_class(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        quantizer: torch.nn.Module = None,
        sample_rate: int = 44100,
        causal: bool = True,
        encoder_transformer_layers: List[int] = [0, 0, 0, 0],
        decoder_transformer_layers: List[int] = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            latent_dim,
            causal=causal,
            n_transformer_layers=encoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )

        self.quantizer = quantizer

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            causal=causal,
            n_transformer_layers=decoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

        self.frame_length = self.hop_length * 4

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        audio_lengths: torch.Tensor = None,
        n_quantizers: int = None,
        **kwargs,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        # pad to multiple of self.frame_length
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        # print(audio_data.shape)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.frame_length) * self.frame_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        if audio_lengths is None:
            audio_lengths = torch.LongTensor([length + right_pad]).to(audio_data.device)

        z = self.encoder(audio_data)
        vq_results = self.quantizer(z, n_quantizers, **kwargs)
        indices = vq_results.codes
        indices_lens = torch.ceil(audio_lengths / self.frame_length).long()
        return indices, indices_lens

    def decode(self, indices: torch.Tensor, feature_lengths):
        if indices.ndim == 2:
            indices = indices[None]

        z = self.quantizer.decode(indices)
        audio_lengths = feature_lengths * self.frame_length
        return self.decoder(z), audio_lengths

    def forward(
        self,
        audio_data: torch.Tensor,
        template: torch.Tensor = None,
        mask: torch.Tensor = None,
        sample_rate: int = None,
        n_quantizers: int = None,
        **kwargs,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        vq_results = self.encode(audio_data, n_quantizers, **kwargs)
        z = vq_results[0] if isinstance(vq_results, tuple) else vq_results.z
        x = self.decode(z)
        return x[..., :length], vq_results
