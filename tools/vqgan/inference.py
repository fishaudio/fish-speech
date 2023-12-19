from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import OmegaConf

from fish_speech.models.vqgan.utils import sequence_mask
from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


@torch.no_grad()
@torch.autocast(device_type="cuda", enabled=True)
@click.command()
@click.option(
    "--input-path",
    "-i",
    default="data/Genshin/Chinese/派蒙/vo_WYLQ103_10_paimon_04.wav",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-path", "-o", default="fake.wav", type=click.Path(path_type=Path)
)
@click.option("--config-name", "-cfg", default="vqgan_pretrain")
@click.option(
    "--checkpoint-path", "-ckpt", default="checkpoints/vqgan/step_000380000_wo.ckpt"
)
def main(input_path, output_path, config_name, checkpoint_path):
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    model: LightningModule = instantiate(cfg.model)
    state_dict = torch.load(
        checkpoint_path,
        map_location=model.device,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()
    logger.info("Restored model from checkpoint")

    if input_path.suffix in AUDIO_EXTENSIONS:
        logger.info(f"Processing in-place reconstruction of {input_path}")
        # Load audio
        audio, _ = librosa.load(
            input_path,
            sr=model.sampling_rate,
            mono=True,
        )
        audios = torch.from_numpy(audio).to(model.device)[None, None, :]
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=model.device, dtype=torch.long
        )

        features = gt_mels = model.mel_transform(
            audios, sample_rate=model.sampling_rate
        )

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
        _, indices, _ = model.vq_encoder(text_features, feature_masks)

        if indices.ndim == 4 and indices.shape[1] == 1 and indices.shape[3] == 1:
            indices = indices[:, 0, :, 0]
        else:
            logger.error(f"Unknown indices shape: {indices.shape}")
            return

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())
    elif input_path.suffix == ".npy":
        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(model.device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
    else:
        raise ValueError(f"Unknown input type: {input_path}")

    # Restore
    indices = indices.unsqueeze(1).unsqueeze(-1)
    mel_lengths = indices.shape[2] * (
        model.downsample.total_strides if model.downsample is not None else 1
    )
    mel_lengths = torch.tensor([mel_lengths], device=model.device, dtype=torch.long)
    mel_masks = torch.ones(
        (1, 1, mel_lengths), device=model.device, dtype=torch.float32
    )

    text_features = model.vq_encoder.decode(indices)

    logger.info(
        f"VQ Encoded, indices: {indices.shape} equivalent to "
        + f"{1/(mel_lengths[0] * model.hop_length / model.sampling_rate / indices.shape[2]):.2f} Hz"
    )

    text_features = F.interpolate(text_features, size=mel_lengths[0], mode="nearest")

    # Sample mels
    decoded_mels = model.decoder(text_features, mel_masks)
    fake_audios = model.generator(decoded_mels)
    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {fake_audios.shape[-1] / model.sampling_rate:.2f} seconds"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].cpu().numpy().astype(np.float32)
    sf.write("fake.wav", fake_audio, model.sampling_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
