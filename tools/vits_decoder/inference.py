from pathlib import Path

import click
import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from fish_speech.utils.file import AUDIO_EXTENSIONS

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)


def load_model(config_name, checkpoint_path, device="cuda"):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
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
    default="test.npy",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--reference-path",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option(
    "--text",
    type=str,
    default="-",
)
@click.option(
    "--tokenizer",
    type=str,
    default="fishaudio/fish-speech-1",
)
@click.option(
    "--output-path", "-o", default="fake.wav", type=click.Path(path_type=Path)
)
@click.option("--config-name", "-cfg", default="vits_decoder")
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
def main(
    input_path,
    reference_path,
    text,
    tokenizer,
    output_path,
    config_name,
    checkpoint_path,
    device,
):
    model = load_model(config_name, checkpoint_path, device=device)

    assert input_path.suffix == ".npy", f"Expected .npy file, got {input_path.suffix}"

    logger.info(f"Processing precomputed indices from {input_path}")
    indices = np.load(input_path)
    indices = torch.from_numpy(indices).to(model.device).long()
    assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"

    # Extract reference audio
    if reference_path is not None:
        assert (
            reference_path.suffix in AUDIO_EXTENSIONS
        ), f"Expected audio file, got {reference_path.suffix}"
        reference_audio, sr = librosa.load(reference_path, sr=model.sampling_rate)
        reference_audio = torch.from_numpy(reference_audio).to(model.device).float()
        reference_spec = model.spec_transform(reference_audio[None])
        reference_embedding = model.generator.encode_ref(
            reference_spec,
            torch.tensor([reference_spec.shape[-1]], device=model.device),
        )
        logger.info(
            f"Loaded reference audio from {reference_path}, shape: {reference_audio.shape}"
        )
    else:
        reference_embedding = torch.zeros(
            1, model.generator.gin_channels, 1, device=model.device
        )
        logger.info("No reference audio provided, use zero embedding")

    # Extract text
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    encoded_text = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    logger.info(f"Encoded text: {encoded_text.shape}")

    # Restore
    feature_lengths = torch.tensor([indices.shape[1]], device=model.device)
    quantized = model.generator.vq.indicies_to_vq_features(
        indices=indices[None], feature_lengths=feature_lengths
    )
    logger.info(f"Restored VQ features: {quantized.shape}")

    # Decode
    fake_audios = model.generator.decode(
        quantized,
        torch.tensor([quantized.shape[-1]], device=model.device),
        encoded_text,
        torch.tensor([encoded_text.shape[-1]], device=model.device),
        ge=reference_embedding,
    )
    logger.info(
        f"Generated audio: {fake_audios.shape}, equivalent to {fake_audios.shape[-1] / model.sampling_rate:.2f} seconds"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.sampling_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    main()
