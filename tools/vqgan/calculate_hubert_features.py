# This file is used to convert the audio files to text files using the Whisper model.
# It's mainly used to generate the training data for the VQ model.

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
from loguru import logger
from transformers import HubertModel

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files

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
def get_hubert_model():
    model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-large")
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.half()
    model.eval()

    logger.info(f"Loaded model")
    return model


def process_batch(files: list[Path]):
    model = get_hubert_model()

    wavs = []
    max_length = total_time = 0

    for file in files:
        wav, sr = torchaudio.load(file)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = torchaudio.functional.resample(wav.cuda(), sr, 16000)[0]

        if len(wav) > sr * 60:
            wav = wav[: sr * 60]

        wavs.append(wav)
        total_time += len(wav) / sr
        max_length = max(max_length, len(wav))

    # Pad to max length
    attention_mask = torch.ones(len(wavs), max_length, dtype=torch.float)
    feature_lengths = []

    if max_length % 320 != 0:
        max_length += 320 - max_length % 320

    for i, wav in enumerate(wavs):
        attention_mask[i, len(wav) :] = 0
        feature_lengths.append(int(len(wav) / 320))
        wavs[i] = torch.nn.functional.pad(wav, (0, max_length - len(wav)), "constant")

    wavs = torch.stack(wavs, dim=0).half()
    attention_mask = attention_mask.cuda()

    # Calculate lengths
    with torch.no_grad():
        outputs = model(wavs, attention_mask=attention_mask)

    # Save to disk
    outputs = outputs.last_hidden_state.cpu().numpy()

    for file, length, feature, wav in zip(files, feature_lengths, outputs, wavs):
        feature = feature[:length]

        # (T, 1024)
        with open(file.with_suffix(".npy"), "wb") as f:
            np.save(f, feature)

    return total_time


@click.command()
@click.argument("folder")
@click.option("--num-workers", default=1)
def main(folder: str, num_workers: int):
    if num_workers > 1 and WORLD_SIZE != num_workers:
        assert WORLD_SIZE == 1, "You should either use SLURM or this launcher, not both"

        logger.info(f"Spawning {num_workers} workers")

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is None:
            visible_devices = list(range(torch.cuda.device_count()))
        else:
            visible_devices = visible_devices.split(",")

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
    files = list_files(folder, AUDIO_EXTENSIONS, recursive=True, sort=True)
    Random(42).shuffle(files)

    total_files = len(files)
    files = files[RANK::WORLD_SIZE]
    logger.info(f"Processing {len(files)}/{total_files} files")

    # Batch size 64
    total_time = 0
    begin_time = time.time()
    processed_files = 0

    for n_batch, idx in enumerate(range(0, len(files), 32)):
        batch = files[idx : idx + 32]
        batch_time = process_batch(batch)
        total_time += batch_time
        processed_files += len(batch)

        if (n_batch + 1) % 10 == 0:
            eta = (
                (time.time() - begin_time)
                / processed_files
                * (len(files) - processed_files)
            )
            logger.info(
                f"Processed {processed_files} files, {total_time / 3600:.2f} hours of audio, ETA: {timedelta(seconds=round(eta))}s"
            )

        # Stop after 1000 hours
        if total_time * WORLD_SIZE > 3600 * 1000:
            break

    logger.info(
        f"Finished processing {len(files)} files, {total_time / 3600:.2f} hours of audio"
    )


if __name__ == "__main__":
    main()
