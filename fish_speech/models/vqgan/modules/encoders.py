from math import log2
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import LFQ, GroupedResidualVQ, VectorQuantize

from fish_speech.models.vqgan.modules.modules import WN
from fish_speech.models.vqgan.modules.transformer import (
    MultiHeadAttention,
    RelativePositionTransformer,
)
from fish_speech.models.vqgan.utils import sequence_mask


# * Ready and Tested
class ConvDownSampler(nn.Module):
    def __init__(
        self,
        dims: list,
        kernel_sizes: list,
        strides: list,
    ):
        super().__init__()

        self.dims = dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.total_strides = np.prod(self.strides)

        self.convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            in_channels=self.dims[i],
                            out_channels=self.dims[i + 1],
                            kernel_size=self.kernel_sizes[i],
                            stride=self.strides[i],
                            padding=(self.kernel_sizes[i] - 1) // 2,
                        ),
                        nn.LayerNorm(self.dims[i + 1], elementwise_affine=True),
                        nn.GELU(),
                    ]
                )
                for i in range(len(self.dims) - 1)
            ]
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for conv, norm, act in self.convs:
            x = conv(x)
            x = norm(x.mT).mT
            x = act(x)

        return x


# * Ready and Tested
class TextEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
        gin_channels=0,
        speaker_cond_layer=0,
        use_vae=True,
        use_embedding=False,
    ):
        """Text Encoder for VITS model.

        Args:
            in_channels (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            n_heads (int): Number of attention heads for the Transformer layers.
            n_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout (float): Dropout rate for the Transformer layers.
            gin_channels (int, optional): Number of channels for speaker embedding. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.use_embedding = use_embedding

        if use_embedding:
            self.proj_in = nn.Embedding(in_channels, hidden_channels)
        else:
            self.proj_in = nn.Conv1d(in_channels, hidden_channels, 1)

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            window_size=4,
            gin_channels=gin_channels,
            speaker_cond_layer=speaker_cond_layer,
        )
        self.proj_out = nn.Conv1d(
            hidden_channels, out_channels * 2 if use_vae else out_channels, 1
        )
        self.use_vae = use_vae

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor = None,
        noise_scale: float = 1,
    ):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """

        if self.use_embedding:
            x = self.proj_in(x.long()).mT * x_mask
        else:
            x = self.proj_in(x) * x_mask

        x = self.encoder(x, x_mask, g=g)
        x = self.proj_out(x) * x_mask

        if self.use_vae is False:
            return x

        m, logs = torch.split(x, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs) * x_mask * noise_scale
        return z, m, logs, x, x_mask


# * Ready and Tested
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        noise_scale: float = 1,
    ):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs) * x_mask * noise_scale
        return z, m, logs, x_mask


# TODO: Ready for testing
class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 192,
        out_channels: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.Mish(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            nn.Mish(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            nn.Mish(),
        )
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.apply(self._init_weights)

        self.encoder = WN(
            hidden_channels,
            kernel_size=3,
            dilation_rate=1,
            n_layers=num_layers,
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, mels, mel_masks: torch.Tensor):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
        """

        x = self.in_proj(mels) * mel_masks
        x = self.encoder(x, mel_masks)

        # Avg Pooling
        x = x * mel_masks
        x = self.out_proj(x)
        x = torch.sum(x, dim=-1) / torch.sum(mel_masks, dim=-1)
        x = x[..., None]

        return x


class VQEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        vq_channels: int = 1024,
        codebook_size: int = 2048,
        downsample: int = 1,
        codebook_groups: int = 1,
    ):
        super().__init__()

        if codebook_groups > 1:
            self.vq = GroupedResidualVQ(
                dim=vq_channels,
                codebook_size=codebook_size,
                threshold_ema_dead_code=2,
                kmeans_init=False,
                groups=codebook_groups,
                num_quantizers=1,
            )
        else:
            self.vq = VectorQuantize(
                dim=vq_channels,
                codebook_size=codebook_size,
                threshold_ema_dead_code=2,
                kmeans_init=False,
            )

        self.codebook_groups = codebook_groups
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

        return x, indices, loss

    def decode(self, indices):
        q = self.vq.get_output_from_indices(indices)

        if q.shape[1] != indices.shape[1] and indices.ndim != 4:
            q = q.view(q.shape[0], indices.shape[1], -1)
        q = q.mT

        x = self.conv_out(q)

        return x
