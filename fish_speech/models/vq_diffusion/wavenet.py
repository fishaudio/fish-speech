import math

import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DiffusionEmbedding(nn.Module):
    """Diffusion Step Embedding"""

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearNorm(nn.Module):
    """LinearNorm Projection"""

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvNorm(nn.Module):
    """1D Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, d_encoder, residual_channels, use_linear_bias=False, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv_layer = ConvNorm(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = LinearNorm(
            residual_channels, residual_channels, use_linear_bias
        )
        self.conditioner_projection = ConvNorm(
            d_encoder, 2 * residual_channels, kernel_size=1
        )
        self.output_projection = ConvNorm(
            residual_channels, 2 * residual_channels, kernel_size=1
        )

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step

        y = self.conv_layer(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(nn.Module):
    """
    WaveNet
    https://www.deepmind.com/blog/wavenet-a-generative-model-for-raw-audio
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=128,
        d_encoder=256,
        residual_channels=512,
        residual_layers=20,
        use_linear_bias=False,
        dilation_cycle=None,
    ):
        super(WaveNet, self).__init__()

        self.input_projection = ConvNorm(in_channels, residual_channels, kernel_size=1)
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4, use_linear_bias),
            Mish(),
            LinearNorm(residual_channels * 4, residual_channels, use_linear_bias),
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_encoder,
                    residual_channels,
                    use_linear_bias=use_linear_bias,
                    dilation=2 ** (i % dilation_cycle) if dilation_cycle else 1,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = ConvNorm(
            residual_channels, residual_channels, kernel_size=1
        )
        self.output_projection = ConvNorm(
            residual_channels, out_channels, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.conv.weight)

    def forward(self, x, diffusion_step, x_masks, condition):
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x * x_masks, condition * x_masks, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 128, T]

        x = x * x_masks

        return x
