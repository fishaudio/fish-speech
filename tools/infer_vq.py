import librosa
import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import OmegaConf

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=True)
def main():
    with initialize(version_base="1.3", config_path="../fish_speech/configs"):
        cfg = compose(config_name="vq_naive_40hz")

    model: LightningModule = instantiate(cfg.model)
    state_dict = torch.load(
        "results/vq_naive_40hz/checkpoints/step_000675000.ckpt",
        # "results/vq_naive_25hz/checkpoints/step_000100000.ckpt",
        map_location=model.device,
    )["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()
    logger.info("Restored model from checkpoint")

    # Load audio
    audio = librosa.load("record1.wav", sr=model.sampling_rate, mono=True)[0]
    audios = torch.from_numpy(audio).to(model.device)[None, None, :]
    logger.info(
        f"Loaded audio with {audios.shape[2] / model.sampling_rate:.2f} seconds"
    )

    # VQ Encoder
    audio_lengths = torch.tensor(
        [audios.shape[2]], device=model.device, dtype=torch.long
    )
    mel_masks, gt_mels, text_features, indices, loss_vq = model.vq_encode(
        audios, audio_lengths
    )
    logger.info(
        f"VQ Encoded, indices: {indices.shape} equavilent to "
        + f"{1/(audios.shape[2] / model.sampling_rate / indices.shape[1]):.2f} Hz"
    )

    # VQ Decoder
    audioa = librosa.load(
        "data/AiShell/wav/train/S0121/BAC009S0121W0125.wav",
        sr=model.sampling_rate,
        mono=True,
    )[0]
    audioa = torch.from_numpy(audioa).to(model.device)[None, None, :]
    mel = model.mel_transform(audioa)
    mel1_masks = torch.ones([mel.shape[0], 1, mel.shape[2]], device=model.device)

    speaker_features = model.speaker_encoder(mel, mel1_masks)

    speaker_features = model.speaker_encoder(gt_mels, mel_masks)
    speaker_features = torch.zeros_like(speaker_features)
    decoded_mels = model.vq_decode(text_features, speaker_features, gt_mels, mel_masks)
    fake_audios = model.vocoder(decoded_mels)

    # Save audio
    fake_audio = fake_audios[0, 0].cpu().numpy().astype(np.float32)
    sf.write("fake.wav", fake_audio, model.sampling_rate)


if __name__ == "__main__":
    main()
