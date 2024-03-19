from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from vector_quantize_pytorch import LFQ, ResidualVQ


class DownsampleResidualVectorQuantizer(nn.Module):
    """
    Downsampled version of ResidualVectorQuantize
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        min_quantizers: int = 4,
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
    ):
        super().__init__()
        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        # self.vq = ResidualVQ(
        #     dim=all_dims[-1],
        #     num_quantizers=n_codebooks,
        #     codebook_dim=codebook_dim,
        #     threshold_ema_dead_code=2,
        #     codebook_size=codebook_size,
        #     kmeans_init=False,
        # )

        self.vq = LFQ(
            dim=all_dims[-1],
            codebook_size=2**14,
            entropy_loss_weight=0.1,
            diversity_gamma=1.0,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Conv1d(
                    all_dims[idx],
                    all_dims[idx + 1],
                    kernel_size=factor,
                    stride=factor,
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.ConvTranspose1d(
                    all_dims[idx + 1],
                    all_dims[idx],
                    kernel_size=factor,
                    stride=factor,
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

    def forward(self, z):
        original_shape = z.shape
        z = self.downsample(z)
        z, indices, loss = self.vq(z.mT)
        z = self.upsample(z.mT)
        loss = loss.mean()

        # Pad or crop z to match original shape
        diff = original_shape[-1] - z.shape[-1]
        left = diff // 2
        right = diff - left

        if diff > 0:
            z = F.pad(z, (left, right))
        elif diff < 0:
            z = z[..., left:-right]

        return z, indices, loss

    # def from_codes(self, codes: torch.Tensor):
    #     z_q, z_p, codes = super().from_codes(codes)
    #     z_q = self.upsample(z_q)
    #     return z_q, z_p, codes

    # def from_latents(self, latents: torch.Tensor):
    #     z_q, z_p, codes = super().from_latents(latents)
    #     z_q = self.upsample(z_q)
    #     return z_q, z_p, codes


if __name__ == "__main__":
    rvq = DownsampleResidualVectorQuantizer(
        quantizer_dropout=1.0,
        min_quantizers=1,
        codebook_size=256,
        downsample_factor=(2, 2),
    )
    x = torch.randn(16, 512, 80)

    result = rvq(x)
    print(result.latents.shape, result.codes.shape, result.z.shape)

    y = rvq.from_codes(result.codes)
    print(y[0].shape)

    y = rvq.from_latents(result.latents)
    print(y[0].shape)
