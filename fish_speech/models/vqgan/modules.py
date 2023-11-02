import math
from dataclasses import dataclass

import torch
from encodec.quantization.core_vq import VectorQuantization
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from fish_speech.models.vqgan.utils import convert_pad_shape, get_padding, init_weights

LRELU_SLOPE = 0.1


@dataclass
class VQEncoderOutput:
    loss: torch.Tensor
    features: torch.Tensor


class VQEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        channels: int = 192,
        num_mels: int = 128,
        num_heads: int = 2,
        num_feature_layers: int = 2,
        num_speaker_layers: int = 4,
        num_mixin_layers: int = 4,
        input_downsample: bool = True,
        code_book_size: int = 2048,
        freeze_vq: bool = False,
    ):
        super().__init__()

        # Feature Encoder
        down_sample = 2 if input_downsample else 1

        self.vq_in = nn.Linear(in_channels * down_sample, in_channels)
        self.vq = VectorQuantization(
            dim=in_channels,
            codebook_size=code_book_size,
            threshold_ema_dead_code=2,
            kmeans_init=True,
            kmeans_iters=50,
        )

        self.feature_in = nn.Linear(in_channels, channels)
        self.feature_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    channels,
                    num_heads,
                    window_size=4,
                    window_heads_share=True,
                    proximal_init=True,
                    proximal_bias=False,
                    use_relative_attn=True,
                )
                for _ in range(num_feature_layers)
            ]
        )

        # Speaker Encoder
        self.speaker_query = nn.Parameter(torch.randn(1, 1, channels))
        self.speaker_in = nn.Linear(num_mels, channels)
        self.speaker_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    channels,
                    num_heads,
                    use_relative_attn=False,
                )
                for _ in range(num_speaker_layers)
            ]
        )

        # Final Mixer
        self.mixer_in = nn.ModuleList(
            [
                TransformerBlock(
                    channels,
                    num_heads,
                    window_size=4,
                    window_heads_share=True,
                    proximal_init=True,
                    proximal_bias=False,
                    use_relative_attn=True,
                )
                for _ in range(num_mixin_layers)
            ]
        )

        self.input_downsample = input_downsample

        if freeze_vq:
            for p in self.vq.parameters():
                p.requires_grad = False

            for p in self.vq_in.parameters():
                p.requires_grad = False

    def forward(self, x, mels, key_padding_mask=None):
        # x: (batch, seq_len, channels)
        # x: (batch, seq_len, 128)

        if self.input_downsample and key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, ::2]

        # Merge Channels
        if self.input_downsample:
            feature_0, feature_1 = x[:, ::2], x[:, 1::2]
            min_len = min(feature_0.size(1), feature_1.size(1))
            x = torch.cat([feature_0[:, :min_len], feature_1[:, :min_len]], dim=2)

        # Encode Features
        features = self.vq_in(x)
        assert key_padding_mask.size(1) == features.size(
            1
        ), f"key_padding_mask shape {key_padding_mask.size()} is not (batch_size, seq_len)"

        features, _, loss = self.vq(features, mask=~key_padding_mask)

        if self.input_downsample:
            features = F.interpolate(
                features.transpose(1, 2), scale_factor=2
            ).transpose(1, 2)

        features = self.feature_in(features)
        for block in self.feature_blocks:
            features = block(features, key_padding_mask=key_padding_mask)

        # Encode Speaker
        speaker = self.speaker_in(x)
        speaker = torch.cat(
            [self.speaker_query.expand(speaker.shape[0], -1, -1), speaker], dim=1
        )
        for block in self.speaker_blocks:
            speaker = block(mels, key_padding_mask=key_padding_mask)

        # Mix
        x = features + speaker[:, :1]
        for block in self.mixer_in:
            x = block(x, key_padding_mask=key_padding_mask)

        return VQEncoderOutput(
            loss=loss,
            features=x.transpose(1, 2),
        )


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
            ), f"key_padding_mask shape {key_padding_mask.size()} is not (batch_size, seq_len)"
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


class ResBlock1(torch.nn.Module):
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

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
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

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


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

    def forward(self, x):
        x = self.conv_pre(x)

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

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


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
