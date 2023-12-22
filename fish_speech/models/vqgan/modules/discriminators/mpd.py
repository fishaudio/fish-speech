import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        *,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        channels: tuple[int] = (1, 64, 128, 256, 512, 1024),
    ) -> None:
        super(DiscriminatorP, self).__init__()

        self.period = period
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    )
                )
                for in_channels, out_channels in zip(channels[:-1], channels[1:])
            ]
        )

        self.conv_post = weight_norm(
            nn.Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "constant")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.silu(x, inplace=True)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int] = (2, 3, 5, 7, 11)) -> None:
        super().__init__()

        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period=period) for period in periods]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        scores, feature_map = [], []

        for disc in self.discriminators:
            res, fmap = disc(x)

            scores.append(res)
            feature_map.append(fmap)

        return scores, feature_map
