import os
import subprocess as sp
import sys
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from random import Random

import click
import numpy as np
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files, load_filelist

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)
# This file is used to convert the audio files to text files using the Whisper model.
# It's mainly used to generate the training data for the VQ model.

backends = torchaudio.list_audio_backends()

if "ffmpeg" in backends:
    backend = "ffmpeg"
else:
    backend = "soundfile"

RANK = int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))

logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{extra[rank]} - <level>{message}</level>"
)
logger.configure(extra={"rank": f"RANK: {RANK} / {WORLD_SIZE}"})
logger.remove()
logger.add(sys.stderr, format=logger_format)


@lru_cache(maxsize=1)
def get_model(
    config_name: str = "modded_dac_vq",
    checkpoint_path: str = "checkpoints/openaudio-s1-mini/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    device: str | torch.device = "cuda",
):
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

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model")
    return model


@torch.inference_mode()
def process_batch(files: list[Path], model) -> float:
    wavs = []
    audio_lengths = []
    new_files = []
    max_length = total_time = 0

    for file in files:
        try:
            wav, sr = torchaudio.load(
                str(file), backend=backend
            )  # Need to install libsox-dev
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = torchaudio.functional.resample(
            wav.cuda(), sr, model.spec_transform.sample_rate
        )[0]
        total_time += len(wav) / model.spec_transform.sample_rate
        max_length = max(max_length, len(wav))

        wavs.append(wav)
        audio_lengths.append(len(wav))
        new_files.append(file)

    files = new_files

    # Pad to max length
    for i, wav in enumerate(wavs):
        wavs[i] = torch.nn.functional.pad(wav, (0, max_length - len(wav)), "constant")

    audios = torch.stack(wavs, dim=0)[:, None]
    audio_lengths = torch.tensor(audio_lengths, device=model.device, dtype=torch.long)

    # Calculate lengths
    indices, feature_lengths = model.encode(audios, audio_lengths)

    # Save to disk
    outputs = indices.cpu().numpy()

    for file, length, feature, audio_length in zip(
        files, feature_lengths, outputs, audio_lengths
    ):
        feature = feature[:, :length]

        # (T,)
        with open(file.with_suffix(".npy"), "wb") as f:
            np.save(f, feature)

    return total_time


@click.command()
@click.argument("folder")
@click.option("--num-workers", default=1)
@click.option("--config-name", default="modded_dac_vq")
@click.option(
    "--checkpoint-path",
    default="checkpoints/openaudio-s1-mini/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
)
@click.option("--batch-size", default=64)
@click.option("--filelist", default=None, type=Path)
def main(
    folder: str,
    num_workers: int,
    config_name: str,
    checkpoint_path: str,
    batch_size: int,
    filelist: Path,
):
    if num_workers > 1 and WORLD_SIZE != num_workers:
        assert WORLD_SIZE == 1, "You should either use SLURM or this launcher, not both"

        logger.info(f"Spawning {num_workers} workers")

        if torch.cuda.is_available():
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is None:
                visible_devices = list(range(torch.cuda.device_count()))
            else:
                visible_devices = visible_devices.split(",")
        else:
            # Set to empty string to avoid using GPU
            visible_devices = [""]

        processes = []
        for i in range(num_workers):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(visible_devices[i % len(visible_devices)])
            env["SLURM_PROCID"] = str(i)
            env["SLURM_NTASKS"] = str(num_workers)

            processes.append(
                sp.Popen(
                    [sys.executable] + sys.argv.copy(),
                    env=env,
                )
            )

        for p in processes:
            p.wait()

        logger.info(f"All workers finished")
        return

    # This is a worker
    logger.info(f"Starting worker")
    if filelist:
        files = [i[0] for i in load_filelist(filelist)]
    else:
        files = list_files(folder, AUDIO_EXTENSIONS, recursive=True, sort=False)

    print(f"Found {len(files)} files")
    files = [Path(f) for f in files if not Path(f).with_suffix(".npy").exists()]

    total_files = len(files)
    files = files[RANK::WORLD_SIZE]
    logger.info(f"Processing {len(files)}/{total_files} files")

    # Batch processing
    total_time = 0
    begin_time = time.time()
    processed_files = 0
    model = get_model(config_name, checkpoint_path)

    for n_batch, idx in enumerate(range(0, len(files), batch_size)):
        batch = files[idx : idx + batch_size]
        batch_time = process_batch(batch, model)

        total_time += batch_time
        processed_files += len(batch)

        if (n_batch + 1) % 10 == 0:
            eta = (
                (time.time() - begin_time)
                / processed_files
                * (len(files) - processed_files)
            )
            logger.info(
                f"Processed {processed_files} files, {total_time / 3600:.2f} hours of audio, "
                + f"ETA: {timedelta(seconds=round(eta))}s"
            )

    logger.info(
        f"Finished processing {len(files)} files, {total_time / 3600:.2f} hours of audio"
    )


if __name__ == "__main__":
    main()
