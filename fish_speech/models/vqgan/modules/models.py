import torch
from torch import nn

from fish_speech.models.vqgan.modules.decoder import Generator
from fish_speech.models.vqgan.modules.encoders import (
    PosteriorEncoder,
    SpeakerEncoder,
    TextEncoder,
    VQEncoder,
)
from fish_speech.models.vqgan.modules.flow import ResidualCouplingBlock
from fish_speech.models.vqgan.utils import rand_slice_segments, sequence_mask


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        *,
        in_channels,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        n_flows,
        n_layers_q,
        n_layers_spk,
        n_layers_flow,
        kernel_size,
        p_dropout,
        speaker_cond_layer,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        codebook_size,
        kmeans_ckpt=None,
    ):
        super().__init__()

        self.segment_size = segment_size

        self.vq = VQEncoder(
            in_channels=in_channels,
            vq_channels=in_channels,
            codebook_size=codebook_size,
            kmeans_ckpt=kmeans_ckpt,
        )
        self.enc_p = TextEncoder(
            in_channels,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
            speaker_cond_layer=speaker_cond_layer,
        )
        self.enc_spk = SpeakerEncoder(
            in_channels=spec_channels,
            hidden_channels=inter_channels,
            out_channels=gin_channels,
            num_heads=n_heads,
            num_layers=n_layers_spk,
            p_dropout=p_dropout,
        )
        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=n_layers_flow,
            n_flows=n_flows,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            n_layers_q,
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

    def forward(self, x, x_lengths, specs):
        x = x.mT

        min_length = min(x.shape[2], specs.shape[2])
        if min_length % 2 != 0:
            min_length -= 1

        x = x[:, :, :min_length]
        specs = specs[:, :, :min_length]
        x_lengths = torch.clamp(x_lengths, max=min_length)

        spec_masks = torch.unsqueeze(sequence_mask(x_lengths, specs.shape[2]), 1).to(
            specs.dtype
        )
        x_masks = torch.unsqueeze(sequence_mask(x_lengths, x.shape[2]), 1).to(x.dtype)

        g = self.enc_spk(specs, spec_masks)
        x, vq_loss = self.vq(x, x_masks)

        _, m_p, logs_p, _, _ = self.enc_p(x, x_masks, g=g)
        z_q, m_q, logs_q, _ = self.enc_q(specs, spec_masks, g=g)
        z_p = self.flow(z_q, spec_masks, g=g, reverse=False)

        z_slice, ids_slice = rand_slice_segments(z_q, x_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return (
            o,
            ids_slice,
            x_masks,
            spec_masks,
            (z_q, z_p),
            (m_p, logs_p),
            (m_q, logs_q),
            vq_loss,
        )

    def infer(self, x, x_lengths, specs, max_len=None, noise_scale=0.35):
        x = x.mT
        spec_masks = torch.unsqueeze(sequence_mask(x_lengths, specs.shape[2]), 1).to(
            specs.dtype
        )
        x_masks = torch.unsqueeze(sequence_mask(x_lengths, x.shape[2]), 1).to(x.dtype)
        g = self.enc_spk(specs, spec_masks)
        x, vq_loss = self.vq(x, x_masks)
        z_p, m_p, logs_p, h_text, _ = self.enc_p(
            x, x_masks, g=g, noise_scale=noise_scale
        )
        z_p = self.flow(z_p, x_masks, g=g, reverse=True)

        o = self.dec((z_p * x_masks)[:, :, :max_len], g=g)
        return o

    def reconstruct(self, specs, spec_lengths, max_len=None, noise_scale=0.35):
        spec_masks = torch.unsqueeze(sequence_mask(spec_lengths, specs.shape[2]), 1).to(
            specs.dtype
        )
        g = self.enc_spk(specs, spec_masks)
        z_q, m_q, logs_q, _ = self.enc_q(
            specs, spec_masks, g=g, noise_scale=noise_scale
        )
        o = self.dec((z_q * spec_masks)[:, :, :max_len], g=g)

        return o
