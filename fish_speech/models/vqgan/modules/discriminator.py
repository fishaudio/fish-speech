import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        blocks = []
        convs = [
            (1, 64, (3, 9), 1, (1, 4)),
            (64, 128, (3, 9), (1, 2), (1, 4)),
            (128, 256, (3, 9), (1, 2), (1, 4)),
            (256, 512, (3, 9), (1, 2), (1, 4)),
            (512, 1024, (3, 3), 1, (1, 1)),
            (1024, 1, (3, 3), 1, (1, 1)),
        ]

        for idx, (in_channels, out_channels, kernel_size, stride, padding) in enumerate(
            convs
        ):
            blocks.append(
                weight_norm(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                )
            )

            if idx != len(convs) - 1:
                blocks.append(nn.SiLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x[:, None])[:, 0]


if __name__ == "__main__":
    model = Discriminator()
    print(sum(p.numel() for p in model.parameters()) / 1_000_000)
    x = torch.randn(1, 128, 1024)
    y = model(x)
    print(y.shape)
    print(y)
