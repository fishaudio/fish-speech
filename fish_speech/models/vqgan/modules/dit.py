import math
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
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


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = FeedForward(
            frequency_embedding_size, hidden_size, out_dim=hidden_size
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> torch.Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
    ):
        super().__init__()
        assert dim % n_head == 0

        self.dim = dim
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, q, freqs_cis, kv=None, mask=None):
        bsz, seqlen, _ = q.shape

        if kv is None:
            kv = q

        kv_seqlen = kv.shape[1]

        q = self.wq(q).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.wk(kv).view(bsz, kv_seqlen, self.n_head, self.head_dim)
        v = self.wv(kv).view(bsz, kv_seqlen, self.n_head, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis[:seqlen])
        k = apply_rotary_emb(k, freqs_cis[:kv_seqlen])

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, in_dim, intermediate_size, out_dim=None):
        super().__init__()
        self.w1 = nn.Linear(in_dim, intermediate_size)
        self.w3 = nn.Linear(in_dim, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, out_dim or in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_self_attention=True,
        use_cross_attention=False,
    ):
        super().__init__()

        self.use_self_attention = use_self_attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if use_self_attention:
            self.mix = Attention(hidden_size, num_heads)
        else:
            self.mix = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=7,
                padding=3,
                bias=True,
                groups=hidden_size,
            )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = Attention(hidden_size, num_heads)
            self.adaLN_modulation_cross = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )
            self.adaLN_modulation_cross_condition = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(
        self,
        x,
        condition,
        freqs_cis,
        self_mask=None,
        cross_condition=None,
        cross_mask=None,
    ):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(condition).chunk(6, dim=-1)

        # Self-attention
        inp = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.use_self_attention:
            inp = self.mix(inp, freqs_cis=freqs_cis, mask=self_mask)
        else:
            inp = self.mix(inp.mT).mT
        x = x + gate_msa * inp

        # Cross-attention
        if self.use_cross_attention:
            (
                shift_cross,
                scale_cross,
                gate_cross,
            ) = self.adaLN_modulation_cross(
                condition
            ).chunk(3, dim=-1)

            (
                shift_cross_condition,
                scale_cross_condition,
            ) = self.adaLN_modulation_cross_condition(cross_condition).chunk(2, dim=-1)

            inp = modulate(self.norm3(x), shift_cross, scale_cross)
            inp = self.cross_attn(
                inp,
                freqs_cis=freqs_cis,
                kv=modulate(
                    self.norm4(cross_condition),
                    shift_cross_condition,
                    scale_cross_condition,
                ),
                mask=cross_mask,
            )
            x = x + gate_cross * inp

        # MLP
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        diffusion_num_layers,
        channels=160,
        mlp_ratio=4.0,
        max_seq_len=16384,
        condition_dim=512,
        style_dim=None,
        cross_condition_dim=None,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.time_embedder = TimestepEmbedder(hidden_size)
        self.condition_embedder = FeedForward(
            condition_dim, int(hidden_size * mlp_ratio), out_dim=hidden_size
        )

        if cross_condition_dim is not None:
            self.cross_condition_embedder = FeedForward(
                cross_condition_dim, int(hidden_size * mlp_ratio), out_dim=hidden_size
            )

        self.use_style = style_dim is not None
        if self.use_style:
            self.style_embedder = FeedForward(
                style_dim, int(hidden_size * mlp_ratio), out_dim=hidden_size
            )

        self.diffusion_blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    use_self_attention=i % 4 == 0,
                    use_cross_attention=cross_condition_dim is not None,
                )
                for i in range(diffusion_num_layers)
            ]
        )

        # Downsample & upsample blocks
        self.input_embedder = FeedForward(
            channels, int(hidden_size * mlp_ratio), out_dim=hidden_size
        )
        self.final_layer = FinalLayer(hidden_size, channels)

        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(max_seq_len, hidden_size // num_heads)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize input embedding:
        self.input_embedder.apply(self.init_weight)
        self.time_embedder.mlp.apply(self.init_weight)
        self.condition_embedder.apply(self.init_weight)

        if self.use_style:
            self.style_embedder.apply(self.init_weight)

        if hasattr(self, "cross_condition_embedder"):
            self.cross_condition_embedder.apply(self.init_weight)

        for block in self.diffusion_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            block.mix.apply(self.init_weight)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        self.final_layer.linear.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x,
        time,
        condition,
        style=None,
        self_mask=None,
        cross_condition=None,
        cross_mask=None,
    ):
        # Embed inputs
        x = self.input_embedder(x)
        t = self.time_embedder(time)

        condition = self.condition_embedder(condition)

        if self.use_style:
            style = self.style_embedder(style)

        if cross_condition is not None:
            cross_condition = self.cross_condition_embedder(cross_condition)
            cross_condition = t[:, None, :] + cross_condition

        # Merge t, condition, and style
        condition = t[:, None, :] + condition
        if self.use_style:
            condition = condition + style[:, None, :]

        if self_mask is not None:
            self_mask = self_mask[:, None, None, :]

        if cross_mask is not None:
            cross_mask = cross_mask[:, None, None, :]

        # DiT
        for block in self.diffusion_blocks:
            x = block(
                x,
                condition,
                self.freqs_cis,
                self_mask=self_mask,
                cross_condition=cross_condition,
                cross_mask=cross_mask,
            )

        x = self.final_layer(x, condition)

        return x


if __name__ == "__main__":
    model = DiT(
        hidden_size=384,
        num_heads=6,
        diffusion_num_layers=12,
        channels=160,
        condition_dim=512,
        style_dim=256,
    )
    bs, seq_len = 8, 1024
    x = torch.randn(bs, seq_len, 160)
    condition = torch.randn(bs, seq_len, 512)
    style = torch.randn(bs, 256)
    mask = torch.ones(bs, seq_len, dtype=torch.bool)
    mask[0, 5:] = False
    time = torch.arange(bs)
    print(time)
    out = model(x, time, condition, style, self_mask=mask)
    print(out.shape)  # torch.Size([2, 100, 160])

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.1f}M")
