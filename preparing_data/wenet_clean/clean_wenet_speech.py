import json
from pathlib import Path
import subprocess

import librosa
import soundfile as sf
import torch
import torchaudio
from fish_audio_preprocess.utils.separate_audio import (
    separate_audio,
    merge_tracks,
    init_model,
)
from tqdm import tqdm
import time
import os
import tempfile

rank = int(os.environ.get("SLURM_PROCID", 0))
world_size = int(os.environ.get("SLURM_NTASKS", 1))
device = torch.device("cuda:0")
print(f"Rank {rank}/{world_size} on {device}")


def main():
    meta_path = Path("dataset/tts/WenetSpeech/WenetSpeech.json")
    dataset_path = Path("dataset/tts/WenetSpeech")
    cleaned_path = Path("dataset/tts/WenetSpeech/cleaned")
    if not cleaned_path.exists():
        cleaned_path.mkdir(parents=True)

    demucs = init_model("htdemucs", device)
    print("Model loaded")

    with open(meta_path) as f:
        dataset = json.load(f)["audios"]

    print(f"Dataset loaded, {len(dataset)} samples")
    dataset = dataset[rank::world_size]
    print(f"Dataset split, {len(dataset)} samples")

    for data_idx, data in enumerate(dataset):
        done_path = cleaned_path / data["aid"] / "done"
        done_path.parent.mkdir(parents=True, exist_ok=True)

        if done_path.exists():
            continue

        print(f"Processing {data_idx}/{len(dataset)} at rank {rank}")

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                subprocess.check_call(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(dataset_path / data["path"]),
                        "-c:a",
                        "pcm_s16le",
                        "-threads",
                        "0",
                        "-ar",
                        "24000",
                        str(f.name),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                raw_audio, sr = librosa.load(f.name, sr=None, mono=True)

            raw_audio = torch.from_numpy(raw_audio[None]).to(device)
            audio = torchaudio.functional.resample(
                raw_audio, orig_freq=sr, new_freq=demucs.samplerate
            )
            # Make it 2 channels
            audio = torch.cat([audio, audio], dim=0)
            tracks = separate_audio(demucs, audio, shifts=1, num_workers=0, progress=False)
            audio = merge_tracks(tracks, filter=["vocals"])[0]
            vocals, sr = (
                torchaudio.functional.resample(
                    audio, orig_freq=demucs.samplerate, new_freq=24000
                ),
                24000,
            )
            vocals = vocals.cpu().numpy()

            for idx, segment in enumerate(data["segments"]):
                if segment["confidence"] <= 0.95:
                    continue

                # Load audio
                begin = int(segment["begin_time"] * sr)
                end = int(segment["end_time"] * sr)
                segment_audio = vocals[begin:end]

                # Write audio
                temp_path = cleaned_path / data["aid"] / f"S{idx:05d}.wav"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(temp_path, segment_audio, samplerate=sr)

                # Write text
                temp_path = temp_path.with_suffix(".txt")
                temp_path.write_text(segment["text"])

            # Write done file
            done_path.write_text("")
        except Exception as e:
            print(f"Error {e} on {data_idx}/{len(dataset)} at rank {rank}")
            time.sleep(10)
            continue

    print("Done")


if __name__ == "__main__":
    main()
