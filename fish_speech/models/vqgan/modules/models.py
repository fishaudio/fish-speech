import copy
import math

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from fish_speech.models.vqgan.modules import attentions, commons, modules
from fish_speech.models.vqgan.modules.commons import get_padding, init_weights
from fish_speech.models.vqgan.modules.rvq import DownsampleResidualVectorQuantizer


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        spec_channels,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        codebook_size=1024,
        num_codebooks=2,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.spec_proj = nn.Conv1d(spec_channels, hidden_channels, 1)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.vq = DownsampleResidualVectorQuantizer(
            input_dim=hidden_channels,
            n_codebooks=num_codebooks,
            codebook_size=codebook_size,
            min_quantizers=num_codebooks,
            downsample_factor=(2,),
        )

        self.decoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
            isflow=True,
            gin_channels=gin_channels,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, ge):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )

        y = self.spec_proj(y * y_mask) * y_mask
        y = self.encoder(y * y_mask, y_mask)
        quantized = self.vq(y)
        y = self.decoder(quantized.z * y_mask, y_mask, g=ge)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
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
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
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
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class WNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
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
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm = modules.LayerNorm(out_channels)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        out = self.proj(x) * x_mask
        out = self.norm(out)
        return out


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

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

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
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


class DiscriminatorP(torch.nn.Module):
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
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
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
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class EnsembledDiscriminator(torch.nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), use_spectral_norm=False):
        super(EnsembledDiscriminator, self).__init__()
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


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        *,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        freeze_quantizer=False,
        codebook_size=1024,
        num_codebooks=2,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels

        self.enc_p = FeatureEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            codebook_size=codebook_size,
            num_codebooks=num_codebooks,
            gin_channels=gin_channels,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        self.ref_enc = modules.MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )

        if freeze_quantizer:
            self.enc_p.spec_proj.requires_grad_(False)
            self.enc_p.encoder.requires_grad_(False)
            self.enc_p.vq.requires_grad_(False)

    def forward(self, y, y_lengths):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        ge = self.ref_enc(y * y_mask, y_mask)

        x, m_p, logs_p, y_mask, quantized = self.enc_p(y, y_lengths, ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=ge)

        return (
            o,
            ids_slice,
            y_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, y, y_lengths, noise_scale=0.5):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        ge = self.ref_enc(y * y_mask, y_mask)
        x, m_p, logs_p, y_mask, quantized = self.enc_p(y, y_lengths, ge)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_mask, (z, z_p, m_p, logs_p)

    def infer_posterior(self, y, y_lengths):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        ge = self.ref_enc(y * y_mask, y_mask)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        o = self.dec(z * y_mask, g=ge)
        return o, y_mask, (z, m_q, logs_q)

    # @torch.no_grad()
    # def decode(self, codes, text, refer, noise_scale=0.5):
    #     refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
    #     refer_mask = torch.unsqueeze(
    #         commons.sequence_mask(refer_lengths, refer.size(2)), 1
    #     ).to(refer.dtype)
    #     ge = self.ref_enc(refer * refer_mask, refer_mask)

    #     y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
    #     text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

    #     quantized = self.quantizer.decode(codes)
    #     if self.semantic_frame_rate == "25hz":
    #         quantized = F.interpolate(
    #             quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
    #         )

    #     x, m_p, logs_p, y_mask = self.enc_p(
    #         quantized, y_lengths, text, text_lengths, ge
    #     )
    #     z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

    #     z = self.flow(z_p, y_mask, g=ge, reverse=True)

    #     o = self.dec((z * y_mask)[:, :, :], g=ge)
    #     return o

    # def extract_latent(self, x):
    #     ssl = self.ssl_proj(x)
    #     quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
    #     return codes.transpose(0, 1)


if __name__ == "__main__":
    model = SynthesizerTrn(
        spec_channels=1025,
        segment_size=20480,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        gin_channels=512,
        freeze_quantizer=True,
    )

    state_dict_g = torch.load("checkpoints/gpt_sovits_g_488k.pth", map_location="cpu")
    # state_dict_d = torch.load("checkpoints/gpt_sovits_d_488k.pth", map_location="cpu")
    # keys = set(model.state_dict().keys())
    # state_dict_g = {k.replace("encoder2.", "decoder."): v for k, v in state_dict_g.items() if k in keys}

    # new_state = {}
    # for k, v in state_dict_g.items():
    #     new_state["generator." + k] = v

    # for k, v in state_dict_d.items():
    #     new_state["discriminator." + k] = v

    # torch.save(new_state, "checkpoints/gpt_sovits_488k.pth")

    # print(EnsembledDiscriminator().load_state_dict(state_dict_d, strict=False))
    print(model.load_state_dict(state_dict_g, strict=False))

    # y = torch.randn(3, 1025, 20480)
    # y_lengths = torch.tensor([20480, 19000, 18000])

    import librosa
    import soundfile as sf

    from fish_speech.models.vqgan.spectrogram import LinearSpectrogram

    spec = LinearSpectrogram(
        n_fft=2048, win_length=2048, hop_length=640, mode="pow2_sqrt"
    )

    audio, _ = librosa.load(
        "/***REMOVED***/workspace/llm-multimodal-test/data/Rail_ZH/星/dbc16cc114ca1700.wav",
        sr=32000,
    )

    y = spec(torch.tensor(audio).unsqueeze(0))
    y_lengths = torch.tensor([y.size(2)])

    o, ids_slice, y_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), quantized = model(
        y, y_lengths
    )
    print(o.shape)

    o, y_mask, (z, z_p, m_p, logs_p) = model.infer(y, y_lengths)
    print(o.shape)

    o, y_mask, (z, m_q, logs_q) = model.infer_posterior(y, y_lengths)
    print(o.shape)

    o = o.squeeze(0).T.detach().cpu().numpy()
    sf.write("test.wav", o, 32000)
