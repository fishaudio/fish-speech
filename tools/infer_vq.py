import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import OmegaConf

from fish_speech.models.vqgan.utils import sequence_mask

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=True)
def main():
    with initialize(version_base="1.3", config_path="../fish_speech/configs"):
        cfg = compose(config_name="vqgan")

    model: LightningModule = instantiate(cfg.model)
    state_dict = torch.load(
        "results/vqgan/checkpoints/step_000110000.ckpt",
        map_location=model.device,
    )["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()
    logger.info("Restored model from checkpoint")

    # Load audio
    audio = librosa.load("0.wav", sr=model.sampling_rate, mono=True)[0]
    audios = torch.from_numpy(audio).to(model.device)[None, None, :]
    logger.info(
        f"Loaded audio with {audios.shape[2] / model.sampling_rate:.2f} seconds"
    )

    # VQ Encoder
    audio_lengths = torch.tensor(
        [audios.shape[2]], device=model.device, dtype=torch.long
    )

    features = gt_mels = model.mel_transform(audios, sample_rate=model.sampling_rate)

    if model.downsample is not None:
        features = model.downsample(features)

    mel_lengths = audio_lengths // model.hop_length
    feature_lengths = (
        audio_lengths
        / model.hop_length
        / (model.downsample.total_strides if model.downsample is not None else 1)
    ).long()

    feature_masks = torch.unsqueeze(
        sequence_mask(feature_lengths, features.shape[2]), 1
    ).to(gt_mels.dtype)
    mel_masks = torch.unsqueeze(sequence_mask(mel_lengths, gt_mels.shape[2]), 1).to(
        gt_mels.dtype
    )

    # vq_features is 50 hz, need to convert to true mel size
    text_features = model.mel_encoder(features, feature_masks)
    text_features, indices, _ = model.vq_encoder(text_features, feature_masks)

    logger.info(
        f"VQ Encoded, indices: {indices.shape} equavilent to "
        + f"{1/(audios.shape[2] / model.sampling_rate / indices.shape[2]):.2f} Hz"
    )

    text_features = F.interpolate(text_features, size=gt_mels.shape[2], mode="nearest")

    # Sample mels
    decoded_mels = model.decoder(text_features, mel_masks)
    fake_audios = model.generator(decoded_mels)

    # Save audio
    fake_audio = fake_audios[0, 0].cpu().numpy().astype(np.float32)
    sf.write("fake.wav", fake_audio, model.sampling_rate)


if __name__ == "__main__":
    main()
