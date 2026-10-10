"""Decode local reference recordings, including macOS Voice Memos M4A files."""

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def decode_reference_audio(
    source: bytes | str | os.PathLike, sample_rate: int
) -> np.ndarray:
    """Return finite, mono float32 samples at the codec's sample rate."""
    if isinstance(source, bytes):
        if not source:
            raise ValueError(
                "The reference recording is empty. Please upload an audio file."
            )
        soundfile_source = io.BytesIO(source)
    else:
        source = Path(source)
        if not source.is_file():
            raise ValueError(
                "The reference recording could not be found. Please upload it again."
            )
        soundfile_source = source

    try:
        samples, original_rate = sf.read(
            soundfile_source, dtype="float32", always_2d=True
        )
    except sf.LibsndfileError:
        # libsndfile cannot read AAC/M4A. Use a seekable input file because some
        # Voice Memos recordings store their MP4 metadata after the audio data.
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            ffmpeg = next(
                (
                    str(p)
                    for p in (
                        Path("/opt/homebrew/bin/ffmpeg"),
                        Path("/usr/local/bin/ffmpeg"),
                    )
                    if p.is_file()
                ),
                None,
            )
        if ffmpeg is None:
            raise ValueError(
                "FFmpeg is needed for this recording format. Please upload WAV audio."
            )
        with tempfile.TemporaryDirectory(prefix="fish-reference-") as temp:
            if isinstance(source, bytes):
                input_path = Path(temp) / "reference.audio"
                input_path.write_bytes(source)
            else:
                input_path = source
            result = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-c:a",
                    "pcm_f32le",
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=120,
            )
        if result.returncode != 0:
            raise ValueError(
                "The reference recording could not be decoded. Please upload a valid audio file."
            )
        audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    else:
        audio = samples.mean(axis=1, dtype=np.float32)
        if original_rate != sample_rate:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), original_rate, sample_rate
            ).numpy()

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(
            "The reference recording contains no usable audio. Please upload another recording."
        )
    return np.ascontiguousarray(audio, dtype=np.float32)


def install_reference_audio_support():
    """Apply the decoder to the pinned runtime without changing its source."""
    from fish_speech.inference_engine.reference_loader import ReferenceLoader

    def load_audio(self, reference_audio, sr):
        return decode_reference_audio(reference_audio, sr)

    ReferenceLoader.load_audio = load_audio
