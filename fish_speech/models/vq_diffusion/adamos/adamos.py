import librosa
import torch
from torch import nn

from fish_speech.models.vqgan.spectrogram import LogMelSpectrogram

from .encoder import ConvNeXtEncoder
from .hifigan import HiFiGANGenerator


class ADaMoSHiFiGANV1(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/adamos-generator-1640000.pth",
    ):
        super().__init__()

        self.backbone = ConvNeXtEncoder(
            input_channels=128,
            depths=[3, 3, 9, 3],
            dims=[128, 256, 384, 512],
            drop_path_rate=0,
            kernel_sizes=(7,),
        )

        self.head = HiFiGANGenerator(
            hop_length=512,
            upsample_rates=(4, 4, 2, 2, 2, 2, 2),
            upsample_kernel_sizes=(8, 8, 4, 4, 4, 4, 4),
            resblock_kernel_sizes=(3, 7, 11, 13),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5)),
            num_mels=512,
            upsample_initial_channel=1024,
            use_template=False,
            pre_conv_kernel_size=13,
            post_conv_kernel_size=13,
        )
        self.sampling_rate = 44100

        ckpt_state = torch.load(checkpoint_path, map_location="cpu")

        if "state_dict" in ckpt_state:
            ckpt_state = ckpt_state["state_dict"]

        if any(k.startswith("generator.") for k in ckpt_state):
            ckpt_state = {
                k.replace("generator.", ""): v
                for k, v in ckpt_state.items()
                if k.startswith("generator.")
            }

        self.load_state_dict(ckpt_state)
        self.eval()

        self.mel_transform = LogMelSpectrogram(
            sample_rate=44100,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            f_min=40,
            f_max=16000,
            n_mels=128,
        )

    @torch.no_grad()
    def decode(self, mel):
        y = self.backbone(mel)
        y = self.head(y)

        return y

    @torch.no_grad()
    def encode(self, x):
        return self.mel_transform(x)


if __name__ == "__main__":
    import soundfile as sf

    x = "data/StarRail/Chinese/罗刹/archive_luocha_2.wav"
    model = ADaMoSHiFiGANV1()

    wav, sr = librosa.load(x, sr=44100, mono=True)
    wav = torch.from_numpy(wav).float()[None]
    mel = model.encode(wav)

    wav = model.decode(mel)[0].mT
    sf.write("test.wav", wav.cpu().numpy(), 44100)
