import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from fish_speech.models.vqgan.utils import fused_add_tanh_sigmoid_multiply

LRELU_SLOPE = 0.1


# ! PosteriorEncoder
# ! ResidualCouplingLayer
class WN(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
        out_channels=None,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = weight_norm(cond_layer, name="weight")

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

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts

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


# ! StochasticDurationPredictor
# ! ResidualCouplingBlock
# TODO convert to class method
class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x
