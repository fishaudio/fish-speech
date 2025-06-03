import math
import typing as tp
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from dac.nn.quantize import ResidualVectorQuantize
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


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


# ConvNeXt Block copied from https://github.com/fishaudio/fish-diffusion/blob/main/fish_diffusion/modules/convnext.py
class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        convnet_type = CausalConvNet
        self.dwconv = convnet_type(
            dim,
            dim,
            kernel_size=kernel_size,
            # padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
            dilation=dilation,
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)

        if apply_residual:
            x = input + x

        return x


@dataclass
class VQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    semantic_distill_z: torch.Tensor | None = None


class DownsampleResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 4096,
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
        pre_module: nn.Module | None = None,
        post_module: nn.Module | None = None,
        semantic_predictor_module: nn.Module | None = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.0,
        )

        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        convnet_type = CausalConvNet
        transconvnet_type = CausalTransConvNet

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    convnet_type(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    transconvnet_type(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )
        self.apply(self._init_weights)
        self.pre_module = (
            pre_module if pre_module is not None else nn.Identity()
        )  # leave for transformer, LSTM or Mamba or something else
        self.post_module = post_module if post_module is not None else nn.Identity()
        self.semantic_predictor_module = (
            semantic_predictor_module
            if semantic_predictor_module is not None
            else nn.Identity()
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, z, n_quantizers: int = None, semantic_len: torch.Tensor = None, **kwargs
    ):
        # z: (B, D, T)
        original_shape = z.shape
        if semantic_len is None:
            semantic_len = torch.LongTensor([z.shape[-1]])
        z = self.downsample(z)
        z = self.pre_module(z)  # B, T, D
        (
            semantic_z,
            semantic_codes,
            semantic_latents,
            semantic_commitment_loss,
            semantic_codebook_loss,
        ) = self.semantic_quantizer(z)
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            residual_z, n_quantizers=n_quantizers
        )
        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = torch.cat([semantic_codes, codes], dim=1)
        latents = torch.cat([semantic_latents, latents], dim=1)
        z = self.post_module(z)
        z = self.upsample(z)
        # z: (B, D, T)

        # semantic distillation (disabled here since only used in training)
        # semantic_distill_z = self.semantic_predictor_module(semantic_z, semantic_len).mT  # wav2vec target is B, T, D

        # Pad or crop z to match original shape
        diff = original_shape[-1] - z.shape[-1]
        right = 0
        left = abs(diff) - right

        if diff > 0:
            z = F.pad(z, (left, right))
        elif diff < 0:
            z = z[..., left:]

        results = VQResult(
            z=z,
            codes=codes,
            latents=latents,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

        return results

    # def encode(self, z):
    #     z = self.downsample(z)
    #     z = self.pre_module(z)
    #     _, indices, _, _, _ = self.quantizer(z.mT)
    #     indices = rearrange(indices, "g b l r -> b (g r) l")
    #     return indices
    #
    def decode(self, indices: torch.Tensor):
        # indices = rearrange(indices, "b (g r) l -> g b l r", g=self.residual_fsq.groups)

        # print(f"indices: {indices.shape}, semantic_quantizer.codebook_size: {self.semantic_quantizer.codebook_size}, quantizer.codebook_size: {self.quantizer.codebook_size}, semantic min: {indices[:, 0].min()}, max: {indices[:, 0].max()}, quantizer min: {indices[:, 1:].min()}, max: {indices[:, 1:].max()}")

        new_indices = torch.zeros_like(indices)
        new_indices[:, 0] = torch.clamp(
            indices[:, 0], max=self.semantic_quantizer.codebook_size - 1
        )
        new_indices[:, 1:] = torch.clamp(
            indices[:, 1:], max=self.quantizer.codebook_size - 1
        )

        z_q_semantic = self.semantic_quantizer.from_codes(new_indices[:, :1])[0]
        z_q_residual = self.quantizer.from_codes(new_indices[:, 1:])[0]
        z_q = z_q_semantic + z_q_residual
        z_q = self.post_module(z_q)
        z_q = self.upsample(z_q)
        return z_q

    # def from_latents(self, latents: torch.Tensor):
    #     z_q, z_p, codes = super().from_latents(latents)
    #     z_q = self.upsample(z_q)
    #     return z_q, z_p, codes


if __name__ == "__main__":
    rvq = DownsampleResidualVectorQuantize(
        input_dim=512,
        n_codebooks=8,
        codebook_dim=8,
        codebook_size=1024,
        quantizer_dropout=0.5,
        downsample_factor=[2, 2],
    )
    rvq.eval()
    x = torch.randn(2, 512, 442)

    result = rvq(x)
    print(rvq)
    print(result.latents.shape, result.codes.shape, result.z.shape)

    # y = rvq.from_codes(result.codes)
    # print(y[0].shape)

    # y = rvq.from_latents(

    result1 = rvq(x[:, :, :40])
    print(result1.latents.shape, result1.codes.shape, result1.z.shape)

    assert torch.allclose(result.z[:, :, :40], result1.z, atol=1e-8)
    print("Success")
