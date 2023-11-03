import math
from dataclasses import dataclass

import torch
from encodec.quantization.core_vq import VectorQuantization
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from fish_speech.models.vqgan.utils import (
    convert_pad_shape,
    fused_add_tanh_sigmoid_multiply,
    get_padding,
    init_weights,
)

LRELU_SLOPE = 0.1


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 4,
        n_flows: int = 4,
        gin_channels: int = 512,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ResidualCouplingLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


@dataclass
class SemanticEncoderOutput:
    loss: torch.Tensor
    mean: torch.Tensor
    logs: torch.Tensor
    z: torch.Tensor


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        hidden_channels: int = 384,
        out_channels: int = 192,
        num_heads: int = 2,
        num_layers: int = 8,
        input_downsample: bool = True,
        code_book_size: int = 2048,
        freeze_vq: bool = False,
        gin_channels: int = 512,
    ):
        super().__init__()

        # Feature Encoder
        down_sample = 2 if input_downsample else 1

        self.in_proj = nn.Conv1d(
            in_channels, in_channels, kernel_size=down_sample, stride=down_sample
        )
        self.vq = VectorQuantization(
            dim=in_channels,
            codebook_size=code_book_size,
            threshold_ema_dead_code=2,
            kmeans_init=False,
            kmeans_iters=50,
        )

        # Init weights of in_proj to mimic the effect of avg pooling
        nn.init.normal_(
            self.in_proj.weight, mean=1 / (down_sample * in_channels), std=0.01
        )
        self.in_proj.bias.data.zero_()

        self.feature_in = nn.Linear(in_channels, hidden_channels)
        self.g_in = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_channels,
                    num_heads,
                    window_size=4,
                    window_heads_share=True,
                    proximal_init=True,
                    proximal_bias=False,
                    use_relative_attn=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_proj = nn.Linear(hidden_channels, out_channels * 2)

        self.input_downsample = input_downsample

        if freeze_vq:
            for p in self.vq.parameters():
                p.requires_grad = False

            for p in self.vq_in.parameters():
                p.requires_grad = False

    def forward(self, x, key_padding_mask=None, g=None) -> SemanticEncoderOutput:
        # x: (batch, seq_len, channels)

        assert key_padding_mask.size(1) == x.size(
            1
        ), f"key_padding_mask shape {key_padding_mask.size()} does not match features shape {x.size()}"

        # Encode Features
        features = self.in_proj(x.transpose(1, 2))
        features, _, loss = self.vq(features)
        features = features.transpose(1, 2)

        if self.input_downsample:
            features = F.interpolate(
                features.transpose(1, 2), scale_factor=2, mode="nearest"
            ).transpose(1, 2)

        # Shape may change due to downsampling, let's cut it to the same size
        if features.shape[1] != key_padding_mask.shape[1]:
            assert abs(features.shape[1] - key_padding_mask.shape[1]) <= 1
            min_len = min(features.shape[1], key_padding_mask.shape[1])
            features = features[:, :min_len]
            key_padding_mask = key_padding_mask[:, :min_len]

        features = self.feature_in(features)
        g = self.g_in(g).transpose(1, 2)
        features = features + g

        for block in self.blocks:
            features = block(features, key_padding_mask=key_padding_mask)

        stats = self.out_proj(features).transpose(1, 2)
        stats = torch.masked_fill(stats, key_padding_mask.unsqueeze(1), 0)
        mean, logs = torch.chunk(stats, 2, dim=1)

        return SemanticEncoderOutput(
            loss=loss,
            mean=mean,
            logs=logs,
            z=mean + torch.randn_like(mean) * torch.exp(logs) * 0.5,
        )


class WaveNet(nn.Module):
    def __init__(
        self,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
        p_dropout=0,
    ):
        super(WaveNet, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = weight_norm(
                nn.Conv1d(
                    hidden_channels,
                    2 * hidden_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = weight_norm(
                nn.Conv1d(hidden_channels, res_skip_channels, 1), name="weight"
            )
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None):
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

        return output * x_mask

    def remove_parametrizations(self):
        if self.gin_channels != 0:
            nn.utils.remove_parametrizations(self.cond_layer)
        for l in self.in_layers:
            nn.utils.remove_parametrizations(l)
        for l in self.res_skip_layers:
            nn.utils.remove_parametrizations(l)


@dataclass
class PosteriorEncoderOutput:
    z: torch.Tensor
    mean: torch.Tensor
    logs: torch.Tensor


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 192,
        hidden_channels: int = 192,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 16,
        gin_channels: int = 512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask, g=None):
        g = g.detach()
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return PosteriorEncoderOutput(
            z=z,
            mean=m,
            logs=logs,
        )


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 192,
        out_channels: int = 512,
        num_heads: int = 2,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, hidden_channels))
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_channels,
                    num_heads,
                    use_relative_attn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, mels, mels_key_padding_mask=None):
        x = self.in_proj(mels)
        x = torch.cat([self.query.expand(x.shape[0], -1, -1), x], dim=1)

        mels_key_padding_mask = torch.cat(
            [
                torch.ones(x.shape[0], 1, dtype=torch.bool, device=x.device),
                mels_key_padding_mask,
            ],
            dim=1,
        )
        for block in self.blocks:
            x = block(x, key_padding_mask=mels_key_padding_mask)

        x = self.out_proj(x[:, 0])

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels,
        n_heads,
        mlp_ratio=4 * 2 / 3,
        p_dropout=0.0,
        window_size=4,
        window_heads_share=True,
        proximal_init=True,
        proximal_bias=False,
        use_relative_attn=True,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(channels)

        if use_relative_attn:
            self.attn = RelativeAttention(
                channels,
                n_heads,
                p_dropout,
                window_size,
                window_heads_share,
                proximal_init,
                proximal_bias,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=n_heads,
                dropout=p_dropout,
                batch_first=True,
            )

        self.mlp_norm = RMSNorm(channels)
        self.mlp = SwiGLU(channels, int(channels * mlp_ratio), channels, drop=p_dropout)

    def forward(self, x, key_padding_mask=None):
        norm_x = self.attn_norm(x)

        if isinstance(self.attn, RelativeAttention):
            attn = self.attn(norm_x, key_padding_mask=key_padding_mask)
        else:
            attn, _ = self.attn(
                norm_x, norm_x, norm_x, key_padding_mask=key_padding_mask
            )

        x = x + attn
        x = x + self.mlp(self.mlp_norm(x))

        return x


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(drop)
        self.norm = RMSNorm(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)

        x = x1 * self.act(x2)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class RelativeAttention(nn.Module):
    def __init__(
        self,
        channels,
        n_heads,
        p_dropout=0.0,
        window_size=4,
        window_heads_share=True,
        proximal_init=True,
        proximal_bias=False,
    ):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = window_heads_share
        self.proximal_init = proximal_init
        self.proximal_bias = proximal_bias

        self.k_channels = channels // n_heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if window_heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.qkv.weight)

        if proximal_init:
            with torch.no_grad():
                # Sync qk weights
                self.qkv.weight.data[: self.channels] = self.qkv.weight.data[
                    self.channels : self.channels * 2
                ]
                self.qkv.bias.data[: self.channels] = self.qkv.bias.data[
                    self.channels : self.channels * 2
                ]

    def forward(self, x, key_padding_mask=None):
        # x: (batch, seq_len, channels)
        batch_size, seq_len, _ = x.size()
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.n_heads, self.k_channels)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = torch.unbind(qkv, dim=0)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))

        if self.window_size is not None:
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, seq_len
            )
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels), key_relative_embeddings
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local

        if self.proximal_bias:
            scores = scores + self._attention_bias_proximal(seq_len).to(
                device=scores.device, dtype=scores.dtype
            )

        # key_padding_mask: (batch, seq_len)
        if key_padding_mask is not None:
            assert key_padding_mask.size() == (
                batch_size,
                seq_len,
            ), f"key_padding_mask shape {key_padding_mask.size()} does not match x shape {x.size()}"
            assert (
                key_padding_mask.dtype == torch.bool
            ), f"key_padding_mask dtype {key_padding_mask.dtype} is not bool"

            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(
                -1, self.n_heads, -1, -1
            )
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, seq_len
            )
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        return output.reshape(batch_size, seq_len, self.n_heads * self.k_channels)

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
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

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_parametrizations(self):
        for l in self.convs1:
            remove_parametrizations(l)
        for l in self.convs2:
            remove_parametrizations(l)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
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

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_parametrizations(self):
        for l in self.convs:
            remove_parametrizations(l)


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            g = self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_parametrizations(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_parametrizations(l)
        for l in self.resblocks:
            l.remove_parametrizations()


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class EnsembleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(EnsembleDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
