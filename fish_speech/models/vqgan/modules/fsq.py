from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from vector_quantize_pytorch import GroupedResidualFSQ

from .convnext import ConvNeXtBlock


@dataclass
class FSQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor


class DownsampleFiniteScalarQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        n_groups: int = 1,
        levels: tuple[int] = (8, 5, 5, 5),  # Approximate 2**10
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.residual_fsq = GroupedResidualFSQ(
            dim=all_dims[-1],
            levels=levels,
            num_quantizers=n_codebooks,
            groups=n_groups,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose1d(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )

    def forward(self, z) -> FSQResult:
        original_shape = z.shape
        z = self.downsample(z)
        quantized, indices = self.residual_fsq(z.mT)
        result = FSQResult(
            z=quantized.mT,
            codes=indices.mT,
            latents=z,
        )
        result.z = self.upsample(result.z)

        # Pad or crop z to match original shape
        diff = original_shape[-1] - result.z.shape[-1]
        left = diff // 2
        right = diff - left

        if diff > 0:
            result.z = F.pad(result.z, (left, right))
        elif diff < 0:
            result.z = result.z[..., left:-right]

        return result

    # def from_codes(self, codes: torch.Tensor):
    #     z_q, z_p, codes = self.residual_fsq.get_output_from_indices(codes)
    #     z_q = self.upsample(z_q)
    #     return z_q, z_p, codes

    # def from_latents(self, latents: torch.Tensor):
    #     z_q, z_p, codes = super().from_latents(latents)
    #     z_q = self.upsample(z_q)
    #     return z_q, z_p, codes


if __name__ == "__main__":
    rvq = DownsampleFiniteScalarQuantize(
        n_codebooks=1,
        downsample_factor=(2, 2),
    )
    x = torch.randn(16, 512, 80)

    result = rvq(x)
    print(rvq)
    print(result.latents.shape, result.codes.shape, result.z.shape)

    # y = rvq.from_codes(result.codes)
    # print(y[0].shape)

    # y = rvq.from_latents(result.latents)
    # print(y[0].shape)
