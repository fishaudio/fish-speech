import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from fish_speech.models.vqgan.utils import fused_add_tanh_sigmoid_multiply

LRELU_SLOPE = 0.1


# ! PosteriorEncoder
# ! ResidualCouplingLayer
class WaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        out_channels=None,
        in_channels=None,
    ):
        super(WaveNet, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.n_layers = n_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        self.in_channels = in_channels
        if in_channels is not None:
            self.proj_in = nn.Conv1d(in_channels, hidden_channels, 1)

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            res_skip_channels = (
                2 * hidden_channels if i < n_layers - 1 else hidden_channels
            )
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

        self.out_channels = out_channels
        if out_channels is not None:
            self.out_layer = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_mask=None):
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if self.in_channels is not None:
            x = self.proj_in(x)

        output = torch.zeros_like(x)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            acts = fused_add_tanh_sigmoid_multiply(x_in, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = x + res_acts
                if x_mask is not None:
                    x = x * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

        if x_mask is not None:
            x = output * x_mask

        if self.out_channels is not None:
            x = self.out_layer(x)

        return x

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_parametrizations(self.cond_layer)
        for l in self.in_layers:
            remove_parametrizations(l)
        for l in self.res_skip_layers:
            remove_parametrizations(l)
