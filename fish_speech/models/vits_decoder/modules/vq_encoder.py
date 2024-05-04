import torch
from torch import nn

from fish_speech.models.vqgan.modules.fsq import DownsampleFiniteScalarQuantize
from fish_speech.models.vqgan.modules.wavenet import WaveNet
from fish_speech.utils.spectrogram import LogMelSpectrogram


class VQEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.encoder = WaveNet(
            input_channels=128,
            residual_channels=768,
            residual_layers=20,
            dilation_cycle=4,
        )

        self.quantizer = DownsampleFiniteScalarQuantize(
            input_dim=768, n_codebooks=1, n_groups=2, levels=[8, 5, 5, 5]
        )

        self.spec = LogMelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_mels=128,
            f_min=0.0,
            f_max=8000.0,
        )

        self.eval()
        e = self.load_state_dict(
            torch.load("checkpoints/vq-gan-group-fsq-2x1024.pth", map_location="cpu"),
            strict=False,
        )

        assert len(e.missing_keys) == 0, e.missing_keys
        assert all(
            k.startswith("decoder.")
            or k.startswith("quality_projection.")
            or k.startswith("discriminator.")
            for k in e.unexpected_keys
        ), e.unexpected_keys

    @torch.no_grad()
    def forward(self, audios, audio_lengths, sr=None):
        mel_spec = self.spec(audios, sample_rate=sr)

        if sr is not None:
            audio_lengths = audio_lengths * 44100 // sr

        mel_lengths = audio_lengths // self.spec.hop_length
        mel_masks = (
            torch.arange(mel_spec.shape[2], device=mel_spec.device)
            < mel_lengths[:, None]
        )
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mel_spec * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(mels) * mel_masks_float_conv
        encoded_features = self.quantizer(encoded_features).z * mel_masks_float_conv

        return encoded_features
