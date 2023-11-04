import torch
import torch.nn as nn


class MultiCondLayer(nn.Module):
    def __init__(
        self,
        gin_channels: int,
        out_channels: int,
        n_cond: int,
    ):
        """MultiCondLayer of VITS model.

        Args:
            gin_channels (int): Number of conditioning tensor channels.
            out_channels (int): Number of output tensor channels.
            n_cond (int): Number of conditions.
        """
        super().__init__()
        self.n_cond = n_cond

        self.cond_layers = nn.ModuleList()
        for _ in range(n_cond):
            self.cond_layers.append(nn.Linear(gin_channels, out_channels))

    def forward(self, cond: torch.Tensor, x_mask: torch.Tensor):
        """
        Shapes:
            - cond: :math:`[B, C, N]`
            - x_mask: :math`[B, 1, T]`
        """

        cond_out = torch.zeros_like(cond)
        for i in range(self.n_cond):
            cond_in = self.cond_layers[i](cond.mT).mT
            cond_out = cond_out + cond_in
        return cond_out * x_mask
