import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from fish_speech.models.vits_decoder.modules import attentions, commons, modules

from .commons import get_padding, init_weights
from .mrte import MRTE
from .vq_encoder import VQEncoder


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        codebook_size=264,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.encoder_text = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.text_embedding = nn.Embedding(codebook_size, hidden_channels)

        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, text, text_lengths, ge):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )

        y = self.ssl_proj(y * y_mask) * y_mask

        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.unsqueeze(
            commons.sequence_mask(text_lengths, text.size(1)), 1
        ).to(y.dtype)
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)

        y = self.mrte(y, y_mask, text, text_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask


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
        if g != None:
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
        super().__init__()
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
        codebook_size=264,
        vq_mask_ratio=0.0,
        ref_mask_ratio=0.0,
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
        self.vq_mask_ratio = vq_mask_ratio
        self.ref_mask_ratio = ref_mask_ratio

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            codebook_size=codebook_size,
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

        self.vq = VQEncoder()
        for param in self.vq.parameters():
            param.requires_grad = False

    def forward(
        self, audio, audio_lengths, gt_specs, gt_spec_lengths, text, text_lengths
    ):
        y_mask = torch.unsqueeze(
            commons.sequence_mask(gt_spec_lengths, gt_specs.size(2)), 1
        ).to(gt_specs.dtype)
        ge = self.ref_enc(gt_specs * y_mask, y_mask)

        if self.training and self.ref_mask_ratio > 0:
            bs = audio.size(0)
            mask_speaker_len = int(bs * self.ref_mask_ratio)
            mask_indices = torch.randperm(bs)[:mask_speaker_len]
            audio[mask_indices] = 0

        quantized = self.vq(audio, audio_lengths)

        # Block masking, block_size = 4
        block_size = 4
        if self.training and self.vq_mask_ratio > 0:
            reduced_length = quantized.size(-1) // block_size
            mask_length = int(reduced_length * self.vq_mask_ratio)
            mask_indices = torch.randperm(reduced_length)[:mask_length]
            short_mask = torch.zeros(
                quantized.size(0),
                quantized.size(1),
                reduced_length,
                device=quantized.device,
                dtype=torch.float,
            )
            short_mask[:, :, mask_indices] = 1.0
            long_mask = short_mask.repeat_interleave(block_size, dim=-1)
            long_mask = F.interpolate(
                long_mask, size=quantized.size(-1), mode="nearest"
            )
            quantized = quantized.masked_fill(long_mask > 0.5, 0)

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, gt_spec_lengths, text, text_lengths, ge
        )
        z, m_q, logs_q, y_mask = self.enc_q(gt_specs, gt_spec_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, gt_spec_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=ge)

        return (
            o,
            ids_slice,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    @torch.no_grad()
    def infer(
        self,
        audio,
        audio_lengths,
        gt_specs,
        gt_spec_lengths,
        text,
        text_lengths,
        noise_scale=0.5,
    ):
        y_mask = torch.unsqueeze(
            commons.sequence_mask(gt_spec_lengths, gt_specs.size(2)), 1
        ).to(gt_specs.dtype)
        ge = self.ref_enc(gt_specs * y_mask, y_mask)
        quantized = self.vq(audio, audio_lengths)

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, audio_lengths, text, text_lengths, ge
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec(z * y_mask, g=ge)
        return o

    @torch.no_grad()
    def infer_posterior(
        self,
        gt_specs,
        gt_spec_lengths,
    ):
        y_mask = torch.unsqueeze(
            commons.sequence_mask(gt_spec_lengths, gt_specs.size(2)), 1
        ).to(gt_specs.dtype)
        ge = self.ref_enc(gt_specs * y_mask, y_mask)
        z, m_q, logs_q, y_mask = self.enc_q(gt_specs, gt_spec_lengths, g=ge)
        o = self.dec(z * y_mask, g=ge)

        return o

    @torch.no_grad()
    def decode(self, codes, text, refer, noise_scale=0.5):
        # TODO: not tested yet

        ge = None
        if refer is not None:
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(
                commons.sequence_mask(refer_lengths, refer.size(2)), 1
            ).to(refer.dtype)
            ge = self.ref_enc(refer * refer_mask, refer_mask)

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        quantized = F.interpolate(
            quantized, size=int(quantized.shape[-1] * 2), mode="nearest"
        )

        x, m_p, logs_p, y_mask = self.enc_p(
            quantized, y_lengths, text, text_lengths, ge
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o


if __name__ == "__main__":
    import librosa
    from transformers import AutoTokenizer

    from fish_speech.utils.spectrogram import LinearSpectrogram

    model = SynthesizerTrn(
        spec_channels=1025,
        segment_size=20480 // 640,
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
        upsample_rates=[8, 8, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16, 8, 2, 2],
        gin_channels=512,
    )

    ckpt = "checkpoints/Bert-VITS2/G_0.pth"
    # Try to load the model
    print(f"Loading model from {ckpt}")
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    # d_checkpoint = torch.load(
    #     "checkpoints/Bert-VITS2/D_0.pth", map_location="cpu", weights_only=True
    # )["model"]
    # print(checkpoint.keys())

    checkpoint.pop("dec.cond.weight")
    checkpoint.pop("enc_q.enc.cond_layer.weight_v")

    # new_checkpoint = {}
    # for k, v in checkpoint.items():
    #     new_checkpoint["generator." + k] = v

    # for k, v in d_checkpoint.items():
    #     new_checkpoint["discriminator." + k] = v

    # torch.save(new_checkpoint, "checkpoints/Bert-VITS2/ensemble.pth")
    # exit()

    print(model.load_state_dict(checkpoint, strict=False))

    # Test

    ref_audio = librosa.load("data/source/云天河/云天河-旁白/《薄太太》第0025集-yth_24.wav", sr=32000)[
        0
    ]
    input_audio = librosa.load(
        "data/source/云天河/云天河-旁白/《薄太太》第0025集-yth_24.wav", sr=32000
    )[0]
    ref_audio = input_audio
    text = "博兴只知道身边的小女人没睡着，他又凑到她耳边压低了声线。阮苏眉睁眼，不觉得你老公像英雄吗？阮苏还是没反应，这男人是不是有病？刚才那冰冷又强势的样子，和现在这幼稚无赖的样子，根本就判若二人。"
    encoded_text = AutoTokenizer.from_pretrained("fishaudio/fish-speech-1")
    spec = LinearSpectrogram(n_fft=2048, hop_length=640, win_length=2048)

    ref_audio = torch.tensor(ref_audio).unsqueeze(0).unsqueeze(0)
    ref_spec = spec(ref_audio)

    input_audio = torch.tensor(input_audio).unsqueeze(0).unsqueeze(0)
    text = encoded_text(text, return_tensors="pt")["input_ids"]
    print(ref_audio.size(), ref_spec.size(), input_audio.size(), text.size())

    o, y_mask, (z, z_p, m_p, logs_p) = model.infer(
        input_audio,
        torch.LongTensor([input_audio.size(2)]),
        ref_spec,
        torch.LongTensor([ref_spec.size(2)]),
        text,
        torch.LongTensor([text.size(1)]),
    )
    print(o.size(), y_mask.size(), z.size(), z_p.size(), m_p.size(), logs_p.size())

    # Save output
    # import soundfile as sf

    # sf.write("output.wav", o.squeeze().detach().numpy(), 32000)
