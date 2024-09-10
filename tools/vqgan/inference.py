from pathlib import Path

import click
import hydra
import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from tools.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model(config_name, checkpoint_path, device="cuda"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../fish_speech/configs"):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model: {result}")
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
@click.option("--config-name", default="firefly_gan_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
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
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(
            audio, sr, model.spec_transform.sample_rate
        )

        audios = audio[None].to(device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.spec_transform.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        indices = model.encode(audios, audio_lengths)[0][0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())
    elif input_path.suffix == ".npy":
        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
    else:
        raise ValueError(f"Unknown input type: {input_path}")

    # Restore
    feature_lengths = torch.tensor([indices.shape[1]], device=device)
    fake_audios, _ = model.decode(
        indices=indices[None], feature_lengths=feature_lengths
    )
    audio_time = fake_audios.shape[-1] / model.spec_transform.sample_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.spec_transform.sample_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
