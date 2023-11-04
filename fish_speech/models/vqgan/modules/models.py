import torch
from torch import nn

from fish_speech.models.vqgan.modules.decoder import Generator
from fish_speech.models.vqgan.modules.encoders import (
    PosteriorEncoder,
    SpeakerEncoder,
    TextEncoder,
)
from fish_speech.models.vqgan.utils import rand_slice_segments


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        in_channels,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        n_layers_q,
        n_layers_spk,
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
    ):
        super().__init__()

        self.segment_size = segment_size

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

    def forward(self, x, x_lengths, y):
        g = self.enc_spk(y, x_lengths)
        z_p, m_p, logs_p, _, x_mask = self.enc_p(x, x_lengths, g=g)
        z_q, m_q, logs_q, y_mask = self.enc_q(y, x_lengths, g=g)
        z_slice, ids_slice = rand_slice_segments(z_q, x_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return (
            o,
            ids_slice,
            x_mask,
            y_mask,
            (z_q, z_p),
            (m_p, logs_p),
            (m_q, logs_q),
        )

    def infer(self, x, x_lengths, y, max_len=None):
        g = self.enc_spk(y, x_lengths)
        z_p, m_p, logs_p, h_text, x_mask = self.enc_p(x, x_lengths, g=g)
        # z_p_audio, m_p_audio, logs_p_audio = self.flow(z_p_text, m_p_text, logs_p_text, x_mask, g=g, reverse=True)

        o = self.dec((z_p * x_mask)[:, :, :max_len], g=g)
        return o

    def reconstruct(self, x, x_lengths, max_len=None):
        g = self.enc_spk(x, x_lengths)
        z_q, m_q, logs_q, x_mask = self.enc_q(x, x_lengths, g=g)
        o = self.dec((z_q * x_mask)[:, :, :max_len], g=g)

        return o
