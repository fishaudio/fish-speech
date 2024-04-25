from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .wavenet import WaveNet


class ReferenceEncoder(WaveNet):
    def __init__(
        self,
        input_channels: Optional[int] = None,
        output_channels: Optional[int] = None,
        residual_channels: int = 512,
        residual_layers: int = 20,
        dilation_cycle: Optional[int] = 4,
        num_heads: int = 8,
        latent_len: int = 4,
    ):
        super().__init__(
            input_channels=input_channels,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle=dilation_cycle,
        )

        self.head_dim = residual_channels // num_heads
        self.num_heads = num_heads

        self.latent_len = latent_len
        self.latent = nn.Parameter(torch.zeros(1, self.latent_len, residual_channels))

        self.q = nn.Linear(residual_channels, residual_channels, bias=True)
        self.kv = nn.Linear(residual_channels, residual_channels * 2, bias=True)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(residual_channels, residual_channels)
        self.proj_drop = nn.Dropout(0.1)

        self.norm = nn.LayerNorm(residual_channels)
        self.mlp = nn.Sequential(
            nn.Linear(residual_channels, residual_channels * 4),
            nn.SiLU(),
            nn.Linear(residual_channels * 4, residual_channels),
        )
        self.output_projection_attn = nn.Linear(residual_channels, output_channels)

        torch.nn.init.trunc_normal_(self.latent, std=0.02)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, attn_mask=None):
        x = super().forward(x).mT
        B, N, C = x.shape

        # Calculate mask
        if attn_mask is not None:
            assert attn_mask.shape == (B, N) and attn_mask.dtype == torch.bool

            attn_mask = attn_mask[:, None, None, :].expand(
                B, self.num_heads, self.latent_len, N
            )

        q_latent = self.latent.expand(B, -1, -1)
        q = (
            self.q(q_latent)
            .reshape(B, self.latent_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))
        x = self.output_projection_attn(x)
        x = x.mean(1)

        return x


if __name__ == "__main__":
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        model = ReferenceEncoder(
            input_channels=128,
            output_channels=64,
            residual_channels=384,
            residual_layers=20,
            dilation_cycle=4,
            num_heads=8,
        )
        x = torch.randn(4, 128, 64)
        mask = torch.ones(4, 64, dtype=torch.bool)
        y = model(x, mask)
        print(y.shape)
        loss = F.mse_loss(y, torch.randn(4, 64))
        loss.backward()
