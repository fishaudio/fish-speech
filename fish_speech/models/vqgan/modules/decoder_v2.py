from functools import partial
from math import prod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_parametrizations(self):
        for conv in self.convs1:
            remove_parametrizations(conv)
        for conv in self.convs2:
            remove_parametrizations(conv)


class ParralelBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int] = (3, 7, 11),
        dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        assert len(kernel_sizes) == len(dilation_sizes)

        self.blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_sizes):
            self.blocks.append(ResBlock(channels, k, d))

    def forward(self, x):
        xs = [block(x) for block in self.blocks]

        return torch.stack(xs, dim=0).mean(dim=0)


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        hop_length: int = 512,
        upsample_rates: tuple[int] = (8, 8, 2, 2, 2),
        upsample_kernel_sizes: tuple[int] = (16, 16, 8, 2, 2),
        resblock_kernel_sizes: tuple[int] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
        use_template: bool = True,
        pre_conv_kernel_size: int = 7,
        post_conv_kernel_size: int = 7,
        post_activation: Callable = partial(nn.SiLU, inplace=True),
        checkpointing: bool = False,
    ):
        super().__init__()

        assert (
            prod(upsample_rates) == hop_length
        ), f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = weight_norm(
            nn.Conv1d(
                num_mels,
                upsample_initial_channel,
                pre_conv_kernel_size,
                1,
                padding=get_padding(pre_conv_kernel_size),
            )
        )

        self.hop_length = hop_length
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.noise_convs = nn.ModuleList()
        self.use_template = use_template
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

            if not use_template:
                continue

            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.resblocks.append(
                ParralelBlock(ch, resblock_kernel_sizes, resblock_dilation_sizes)
            )

        self.activation_post = post_activation()
        self.conv_post = weight_norm(
            nn.Conv1d(
                ch,
                1,
                post_conv_kernel_size,
                1,
                padding=get_padding(post_conv_kernel_size),
            )
        )
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # Gradient checkpointing
        self.checkpointing = checkpointing

    def forward(self, x, template=None):
        if self.use_template and template is None:
            length = x.shape[-1] * self.hop_length
            template = (
                torch.randn(x.shape[0], 1, length, device=x.device, dtype=x.dtype)
                * 0.003
            )

        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            if self.use_template:
                x = x + self.noise_convs[i](template)

            if self.training and self.checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    self.resblocks[i],
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.resblocks[i](x)

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_parametrizations(self):
        for up in self.ups:
            remove_parametrizations(up)
        for block in self.resblocks:
            block.remove_parametrizations()
        remove_parametrizations(self.conv_pre)
        remove_parametrizations(self.conv_post)
