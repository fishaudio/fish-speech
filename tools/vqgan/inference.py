from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import OmegaConf

from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model(config_name, checkpoint_path, device="cuda"):
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    model: LightningModule = instantiate(cfg.model)
    state_dict = torch.load(
        checkpoint_path,
        map_location=model.device,
    )

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    logger.info("Restored model from checkpoint")

    return model


@torch.no_grad()
@click.command()
@click.option(
    "--input-path",
    "-i",
    default="test.wav",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-path", "-o", default="fake.wav", type=click.Path(path_type=Path)
)
@click.option("--config-name", "-cfg", default="vqgan_pretrain")
@click.option(
    "--checkpoint-path",
    "-ckpt",
    default="checkpoints/vq-gan-group-fsq-2x1024.pth",
)
@click.option(
    "--device",
    "-d",
    default="cuda",
)
def main(input_path, output_path, config_name, checkpoint_path, device):
    model = load_model(config_name, checkpoint_path, device=device)

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
        indices = model.encode(audios, audio_lengths)[0][0]

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
    fake_audios = model.decode(
        indices=indices[None], feature_lengths=feature_lengths, return_audios=True
    )
    audio_time = fake_audios.shape[-1] / model.sampling_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.sampling_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
