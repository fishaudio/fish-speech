import torch
from torch import nn

from .convnext import ConvNeXtEncoder
from .hifigan import HiFiGANGenerator


class FireflyBase(nn.Module):
    def __init__(self, ckpt_path: str = None):
        super().__init__()

        self.backbone = ConvNeXtEncoder(
            input_channels=160,
            depths=[3, 3, 9, 3],
            dims=[128, 256, 384, 512],
            drop_path_rate=0.2,
            kernel_sizes=[7],
        )

        self.head = HiFiGANGenerator(
            hop_length=512,
            upsample_rates=[8, 8, 2, 2, 2],
            upsample_kernel_sizes=[16, 16, 4, 4, 4],
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            num_mels=512,
            upsample_initial_channel=512,
            use_template=True,
            pre_conv_kernel_size=13,
            post_conv_kernel_size=13,
        )

        if ckpt_path is None:
            return

        state_dict = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any("generator." in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        self.load_state_dict(state_dict, strict=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        if x.ndim == 2:
            x = x[:, None, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x
