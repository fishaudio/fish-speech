# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from fish_speech.models.vq_diffusion.bigvgan.activations import Snake, SnakeBeta
from fish_speech.models.vq_diffusion.bigvgan.alias_free_torch import Activation1d
from fish_speech.models.vq_diffusion.bigvgan.utils import get_padding, init_weights
from fish_speech.models.vqgan.spectrogram import LogMelSpectrogram

LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h

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

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList(
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
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # total number of conv layers

        if (
            activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif (
            activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale)
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGANModule(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, h):
        super(BigVGANModule, self).__init__()
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == "1" else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # post conv
        if (
            h.activation == "snake"
        ):  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif (
            h.activation == "snakebeta"
        ):  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class BigVGAN(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/bigvgan-24k-100band/g_05000000",
        config_file: Optional[str] = None,
    ):
        super().__init__()

        if config_file is None:
            config_file = Path(checkpoint_path).parent / "config.json"

        with open(config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.model = BigVGANModule(self.h)

        state_dict = torch.load(checkpoint_path, map_location="cpu")["generator"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.remove_weight_norm()

        self.mel_transform = LogMelSpectrogram(
            sample_rate=self.h.sampling_rate,
            n_fft=self.h.n_fft,
            win_length=self.h.win_size,
            hop_length=self.h.hop_size,
            f_min=self.h.fmin,
            f_max=self.h.fmax,
            n_mels=self.h.num_mels,
        )

    @torch.no_grad()
    def decode(self, mel):
        mel = F.pad(mel, (0, 10), "reflect")
        y = self.model(mel)
        y = y[:, :, : -self.h.hop_size * 10]
        return y

    @torch.no_grad()
    def encode(self, x):
        return self.mel_transform(x)


if __name__ == "__main__":
    import librosa
    import soundfile as sf

    x = "data/StarRail/Chinese/罗刹/archive_luocha_2.wav"
    model = BigVGAN()

    wav, sr = librosa.load(x, sr=24000, mono=True)
    wav = torch.from_numpy(wav).float()[None]
    mel = model.encode(wav)

    wav = model.decode(mel)[0].mT
    sf.write("test.wav", wav.cpu().numpy(), 24000)
