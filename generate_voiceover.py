"""Generate a WAV through the running local S1 Mini web interface."""

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from gradio_client import Client, handle_file

SERVER_URL = "http://127.0.0.1:7860"


def generate(text, output, reference_audio=None, reference_text="", seed=42):
    if not text.strip():
        raise ValueError("Provide non-empty English text.")
    destination = Path(output).expanduser().resolve()
    if destination.suffix.lower() != ".wav":
        raise ValueError("Output must have a .wav extension.")
    if destination.exists():
        raise FileExistsError(
            f"Output already exists; choose a new filename: {destination}"
        )
    reference = None
    if reference_audio:
        reference = Path(reference_audio).expanduser().resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"Reference audio not found: {reference}")
        if not reference_text.strip():
            raise ValueError("Provide the exact reference-audio transcript.")
    elif reference_text.strip():
        raise ValueError("Reference text requires reference audio.")

    with tempfile.TemporaryDirectory(prefix="fish-voiceover-") as download_dir:
        # Keep stdout as one machine-readable JSON response for the MCP adapter.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                client = Client(SERVER_URL, verbose=False, download_files=download_dir)
            except Exception as exc:
                raise RuntimeError(
                    "Cannot reach the voice server. Start Fish Speech.command, "
                    "wait for http://127.0.0.1:7860, then retry."
                ) from exc
            result = client.predict(
                text,
                "",
                handle_file(str(reference)) if reference else None,
                reference_text,
                1024,
                200,
                0.8,
                1.1,
                0.8,
                seed,
                "on",
                api_name="/partial",
            )
        audio_file, error = result
        if error or not audio_file:
            raise RuntimeError(
                f"Voice generation failed: {error or 'No audio returned'}"
            )
        audio, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
        if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
            raise RuntimeError("The server returned empty, silent, or invalid audio.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids overwriting an existing file, even after inference.
        with destination.open("xb") as target, open(audio_file, "rb") as source:
            shutil.copyfileobj(source, target)
        return {
            "path": str(destination),
            "duration_seconds": round(len(audio) / sample_rate, 3),
            "sample_rate": sample_rate,
            "channels": audio.shape[1],
            "model": "OpenAudio S1 Mini",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text")
    text_group.add_argument("--text-file", type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-audio")
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        text = (
            args.text
            if args.text is not None
            else args.text_file.read_text(encoding="utf-8")
        )
        result = generate(
            text, args.output, args.reference_audio, args.reference_text, args.seed
        )
    except Exception as exc:
        print(f"Voiceover error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
