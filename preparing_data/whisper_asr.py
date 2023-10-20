# This file is used to convert the audio files to text files using the Whisper model.
# It's mainly used to generate the training data for the VQ model.

import os
import subprocess as sp
import time
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from random import Random

import click
import numpy as np
import torch
from loguru import logger
from transformers import WhisperProcessor
from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim

from speech_lm.models.flash_whisper import FlashWhisperForConditionalGeneration

RANK_STR = ""


@lru_cache(maxsize=1)
def get_whisper_model():
    model = FlashWhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium"
    ).cuda()
    model.eval()
    logger.info(f"{RANK_STR}Loaded model")

    return model


@lru_cache(maxsize=1)
def get_whisper_processor():
    return WhisperProcessor.from_pretrained("openai/whisper-medium")


def transcribe_batch(files: list[str]):
    wavs = [load_audio(file, 16000) for file in files]
    total_time = sum([len(wav) for wav in wavs]) / 16000
    wavs = [pad_or_trim(wav) for wav in wavs]

    wavs = torch.from_numpy(np.stack(wavs)).float().cuda()
    mels = log_mel_spectrogram(wavs).cuda()
    model = get_whisper_model()

    with torch.no_grad():
        outputs = model.generate(
            input_features=mels,
            max_length=448,
            do_sample=False,
        )

    processor = get_whisper_processor()
    transcriptions = processor.batch_decode(outputs, skip_special_tokens=False)
    tokens = [",".join(map(str, line.cpu().tolist())) for line in outputs]
    transcriptions = [
        f"{token}\t{transcription}"
        for token, transcription in zip(tokens, transcriptions)
    ]

    return transcriptions, total_time


@click.command()
@click.argument("folder")
@click.option("--rank", default=0)
@click.option("--world-size", default=1)
@click.option("--num-workers", default=1)
def main(folder: str, rank: int, world_size: int, num_workers: int):
    global RANK_STR

    if num_workers > 1 and world_size != num_workers:
        RANK_STR = "[Master] "
        logger.info(f"{RANK_STR}Spawning {num_workers} workers")

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is None:
            visible_devices = list(range(torch.cuda.device_count()))
        else:
            visible_devices = visible_devices.split(",")

        processes = []
        for i in range(num_workers):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(visible_devices[i % len(visible_devices)])
            args = [
                "python",
                __file__,
                "--rank",
                str(i),
                "--world-size",
                str(num_workers),
                folder,
            ]
            processes.append(
                sp.Popen(
                    args,
                    env=env,
                )
            )

        for p in processes:
            p.wait()

        logger.info(f"{RANK_STR}All workers finished")
        return

    # This is a worker
    RANK_STR = f"[Rank: {rank}] "
    logger.info(f"{RANK_STR}Starting worker")

    files = [
        str(file)
        for file in Path(folder).rglob("*")
        if file.suffix in [".wav", ".flac"]
    ]

    logger.info(f"{RANK_STR}Found {len(files)} files")

    files = sorted(files)
    Random(42).shuffle(files)
    files = files[rank::world_size]
    logger.info(f"{RANK_STR}Processing {len(files)} files")

    # Batch size 64
    total_time = 0
    begin_time = time.time()
    processed_files = 0

    for n_batch, idx in enumerate(range(0, len(files), 64)):
        batch = files[idx : idx + 64]
        trascriptions, batch_time = transcribe_batch(batch)
        total_time += batch_time
        processed_files += len(batch)

        if (n_batch + 1) % 10 == 0:
            eta = (
                (time.time() - begin_time)
                / processed_files
                * (len(files) - processed_files)
            )
            logger.info(
                f"{RANK_STR}Processed {processed_files} files, {total_time / 3600:.2f} hours of audio, ETA: {timedelta(seconds=round(eta))}s"
            )

        # Write to file
        for file, transcription in zip(batch, trascriptions):
            Path(file).with_suffix(".whisper.txt").write_text(
                transcription, encoding="utf-8"
            )

        # Stop if total time is more than 1000 / world_size hours
        if total_time > 1000 / world_size * 3600:
            break

    logger.info(
        f"{RANK_STR}Finished processing {len(files)} files, {total_time / 3600:.2f} hours of audio"
    )


if __name__ == "__main__":
    main()
