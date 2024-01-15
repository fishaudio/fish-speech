from math import log2
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vector_quantize_pytorch import LFQ, GroupedResidualVQ, VectorQuantize


class VQEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        vq_channels: int = 1024,
        codebook_size: int = 2048,
        downsample: int = 1,
        codebook_groups: int = 1,
        codebook_layers: int = 1,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()

        if codebook_groups > 1 or codebook_layers > 1:
            self.vq = GroupedResidualVQ(
                dim=vq_channels,
                codebook_size=codebook_size,
                threshold_ema_dead_code=threshold_ema_dead_code,
                kmeans_init=True,
                groups=codebook_groups,
                num_quantizers=codebook_layers,
            )
        else:
            self.vq = VectorQuantize(
                dim=vq_channels,
                codebook_size=codebook_size,
                threshold_ema_dead_code=threshold_ema_dead_code,
                kmeans_init=True,
            )

        self.codebook_groups = codebook_groups
        self.codebook_layers = codebook_layers
        self.downsample = downsample
        self.conv_in = nn.Conv1d(
            in_channels, vq_channels, kernel_size=downsample, stride=downsample
        )
        self.conv_out = nn.Sequential(
            nn.Upsample(scale_factor=downsample, mode="nearest")
            if downsample > 1
            else nn.Identity(),
            nn.Conv1d(vq_channels, in_channels, kernel_size=1, stride=1),
        )

    @property
    def mode(self):
        if self.codebook_groups > 1 and self.codebook_layers > 1:
            return "grouped-residual"
        elif self.codebook_groups > 1:
            return "grouped"
        elif self.codebook_layers > 1:
            return "residual"
        else:
            return "single"

    def forward(self, x, x_mask):
        # x: [B, C, T], x_mask: [B, 1, T]
        x_len = x.shape[2]

        if x_len % self.downsample != 0:
            x = F.pad(x, (0, self.downsample - x_len % self.downsample))
            x_mask = F.pad(x_mask, (0, self.downsample - x_len % self.downsample))

        x = self.conv_in(x)
        q, indices, loss = self.vq(x.mT)
        q = q.mT

        if self.codebook_groups > 1:
            loss = loss.mean()

        x = self.conv_out(q) * x_mask
        x = x[:, :, :x_len]

        # Post process indices
        if self.mode == "grouped-residual":
            indices = rearrange(indices, "g b t r -> b (g r) t")
        elif self.mode == "grouped":
            indices = rearrange(indices, "g b t 1 -> b g t")
        elif self.mode == "residual":
            indices = rearrange(indices, "1 b t r -> b r t")
        else:
            indices = rearrange(indices, "b t -> b 1 t")

        return x, indices, loss

    def decode(self, indices):
        # Undo rearrange
        if self.mode == "grouped-residual":
            indices = rearrange(indices, "b (g r) t -> g b t r", g=self.codebook_groups)
        elif self.mode == "grouped":
            indices = rearrange(indices, "b g t -> g b t 1")
        elif self.mode == "residual":
            indices = rearrange(indices, "b r t -> 1 b t r")
        else:
            indices = rearrange(indices, "b 1 t -> b t")

        q = self.vq.get_output_from_indices(indices)

        # Edge case for single vq
        if self.mode == "single":
            q = rearrange(q, "b (t c) -> b t c", t=indices.shape[-1])

        x = self.conv_out(q.mT)

        return x


if __name__ == "__main__":
    # Test VQEncoder
    for group, layer in [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (4, 1),
        (4, 2),
    ]:
        encoder = VQEncoder(
            in_channels=1024,
            vq_channels=1024,
            codebook_size=2048,
            downsample=1,
            codebook_groups=group,
            codebook_layers=layer,
            threshold_ema_dead_code=2,
        )
        x = torch.randn(2, 1024, 100)
        x_mask = torch.ones(2, 1, 100)
        x, indices, loss = encoder(x, x_mask)
        x = encoder.decode(indices)
        assert x.shape == (2, 1024, 100)
