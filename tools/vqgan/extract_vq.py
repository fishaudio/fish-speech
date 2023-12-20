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
from einops import rearrange
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning import LightningModule
from loguru import logger
from omegaconf import OmegaConf

from fish_speech.models.vqgan.utils import sequence_mask
from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files

# register eval resolver
OmegaConf.register_new_resolver("eval", eval)
# This file is used to convert the audio files to text files using the Whisper model.
# It's mainly used to generate the training data for the VQ model.


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
    config_name: str = "vqgan",
    checkpoint_path: str = "checkpoints/vqgan/step_000380000.ckpt",
):
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

    logger.info(f"Loaded model")
    return model


def process_batch(files: list[Path], model) -> float:
    wavs = []
    audio_lengths = []
    max_length = total_time = 0

    for file in files:
        wav, sr = torchaudio.load(file)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = torchaudio.functional.resample(wav.cuda(), sr, model.sampling_rate)[0]
        wavs.append(wav)
        total_time += len(wav) / model.sampling_rate
        max_length = max(max_length, len(wav))
        audio_lengths.append(len(wav))

    # Pad to max length
    for i, wav in enumerate(wavs):
        wavs[i] = torch.nn.functional.pad(wav, (0, max_length - len(wav)), "constant")

    audios = torch.stack(wavs, dim=0)[:, None]
    audio_lengths = torch.tensor(audio_lengths, device=model.device, dtype=torch.long)

    # Calculate lengths
    with torch.no_grad():
        # VQ Encoder
        features = gt_mels = model.mel_transform(
            audios, sample_rate=model.sampling_rate
        )

        if model.downsample is not None:
            features = model.downsample(features)

        feature_lengths = (
            audio_lengths
            / model.hop_length
            / (model.downsample.total_strides if model.downsample is not None else 1)
        ).long()

        feature_masks = torch.unsqueeze(
            sequence_mask(feature_lengths, features.shape[2]), 1
        ).to(gt_mels.dtype)

        text_features = model.mel_encoder(features, feature_masks)
        _, indices, _ = model.vq_encoder(text_features, feature_masks)

        if indices.ndim == 4:
            # Grouped vq
            assert indices.shape[-1] == 1, f"Residual vq is not supported"
            indices = indices.squeeze(-1)
        elif indices.ndim == 2:
            # Single vq
            indices = indices.unsqueeze(0)
        else:
            raise ValueError(f"Invalid indices shape {indices.shape}")

        indices = rearrange(indices, "c b t -> b c t")

    # Save to disk
    outputs = indices.cpu().numpy()

    for file, length, feature, audio in zip(files, feature_lengths, outputs, audios):
        feature = feature[:, :length]

        # (T,)
        with open(file.with_suffix(".npy"), "wb") as f:
            np.save(f, feature)

    return total_time


@click.command()
@click.argument("folder")
@click.option("--num-workers", default=1)
@click.option("--config-name", default="vqgan_pretrain")
@click.option(
    "--checkpoint-path",
    default="checkpoints/vqgan-v1.pth",
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
    if filelist:
        with open(filelist, "r", encoding="utf-8") as f:
            #files = [Path(line..strip().split("|")[0]) for line in f]
            files = set()
            countSame = 0
            countNotFound = 0
            for line in f.readlines():
                file = Path(line.strip().split("|")[0])
                if file in files:
                    print(f"重复音频文本：{line}")
                    countSame += 1
                    continue
                if not os.path.isfile(file):
                # 过滤数据集错误：不存在对应音频
                    print(f"没有找到对应的音频：{file}")
                    countNotFound += 1
                    continue
                files.add(file)
        files = list(files)
        print(f"总重复音频数：{countSame}，总未找到的音频数:{countNotFound}")
    else:
        files = list_files(folder, AUDIO_EXTENSIONS, recursive=True, sort=True)
    Random(42).shuffle(files)

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
