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
@click.option("--checkpoint-path", "-ckpt", default="checkpoints/vqgan-v1.pth")
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
        encoded = model.encode(audios, audio_lengths)
        indices = encoded.indices[0]

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
    feature_lengths = torch.tensor([indices.shape[1]], device=model.device)
    decoded = model.decode(indices=indices[None], feature_lengths=feature_lengths)
    fake_audios = decoded.audios
    audio_time = fake_audios.shape[-1] / model.sampling_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].cpu().numpy().astype(np.float32)
    sf.write("fake.wav", fake_audio, model.sampling_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
