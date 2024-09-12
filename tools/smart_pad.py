import random
from multiprocessing import Pool
from pathlib import Path

import click
import librosa
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from tools.file import AUDIO_EXTENSIONS, list_files

threshold = 10 ** (-50 / 20.0)


def process(file):
    waveform, sample_rate = torchaudio.load(str(file), backend="sox")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    loudness = librosa.feature.rms(
        y=waveform.numpy().squeeze(), frame_length=2048, hop_length=512, center=True
    )[0]

    for i in range(len(loudness) - 1, 0, -1):
        if loudness[i] > threshold:
            break

    end_silent_time = (len(loudness) - i) * 512 / sample_rate

    if end_silent_time <= 0.3:
        random_time = random.uniform(0.3, 0.7) - end_silent_time
        waveform = F.pad(
            waveform, (0, int(random_time * sample_rate)), mode="constant", value=0
        )

    for i in range(len(loudness)):
        if loudness[i] > threshold:
            break

    start_silent_time = i * 512 / sample_rate

    if start_silent_time > 0.02:
        waveform = waveform[:, int((start_silent_time - 0.02) * sample_rate) :]

    torchaudio.save(uri=str(file), src=waveform, sample_rate=sample_rate)


@click.command()
@click.argument("source", type=Path)
@click.option("--num-workers", type=int, default=12)
def main(source, num_workers):
    files = list(list_files(source, AUDIO_EXTENSIONS, recursive=True))

    with Pool(num_workers) as p:
        list(tqdm(p.imap_unordered(process, files), total=len(files)))


if __name__ == "__main__":
    main()
