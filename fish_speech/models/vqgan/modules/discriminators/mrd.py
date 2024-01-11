import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class DiscriminatorR(torch.nn.Module):
    def __init__(
        self,
        *,
        n_fft: int = 1024,
        hop_length: int = 120,
        win_length: int = 600,
    ):
        super(DiscriminatorR, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
            ]
        )

        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)

        for conv in self.convs:
            x = conv(x)
            x = F.silu(x, inplace=True)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        x = F.pad(
            x,
            (
                (self.n_fft - self.hop_length) // 2,
                (self.n_fft - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        )
        x = x.squeeze(1)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            return_complex=True,
        )
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(torch.nn.Module):
    def __init__(self, resolutions: list[tuple[int]]):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                )
                for n_fft, hop_length, win_length in resolutions
            ]
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
