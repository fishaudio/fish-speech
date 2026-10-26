#!/usr/bin/env python3
"""Benchmark a running Fish Speech HTTP API without importing Torch.

Examples:
    python tools/benchmark_tts.py --api-url http://127.0.0.1:8080 --runs 3
    python tools/benchmark_tts.py --streaming --save-dir /tmp/fish-bench
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
import urllib.error
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TEXTS = {
    "short": "Hello world.",
    "medium": (
        "The quick brown fox jumps over the lazy dog. This is a second sentence "
        "to measure generation speed."
    ),
    "long": (
        "Good morning. I wanted to let you know that the furniture will arrive on "
        "Friday between nine and eleven. Please make sure the front gate is open, "
        "and call me if the delivery team needs help finding the building. Thank "
        "you very much for your patience."
    ),
}


def post_json(url: str, payload: dict, timeout: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def wav_info(audio: bytes, *, streaming: bool = False) -> dict:
    try:
        with wave.open(io.BytesIO(audio)) as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            frames = wav.getnframes()
            if streaming:
                # A streamed WAV header cannot carry a final data size (it is
                # often 0xFFFFFFFF), so measure the bytes actually received.
                position = 12
                while position + 8 <= len(audio):
                    chunk = audio[position : position + 4]
                    size = int.from_bytes(audio[position + 4 : position + 8], "little")
                    if chunk == b"data":
                        frames = max(0, len(audio) - position - 8) // (channels * width)
                        break
                    position += 8 + size + (size & 1)
            return {
                "sample_rate": rate,
                "channels": channels,
                "duration_s": frames / rate if rate else 0.0,
            }
    except (wave.Error, EOFError):
        return {}


def run_once(url: str, text: str, args) -> dict:
    payload = {
        "text": text,
        "format": "wav",
        "streaming": args.streaming,
        "seed": args.seed,
        "normalize": True,
    }
    if args.voice:
        payload["reference_id"] = args.voice
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.chunk_length is not None:
        payload["chunk_length"] = args.chunk_length
    if args.single_chunk:
        payload["single_chunk"] = True

    started = time.perf_counter()
    first_byte = None
    audio = bytearray()
    with post_json(url, payload, args.timeout) as response:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            if first_byte is None:
                first_byte = time.perf_counter() - started
            audio.extend(chunk)
    elapsed = time.perf_counter() - started
    info = wav_info(bytes(audio), streaming=args.streaming)
    duration = info.get("duration_s", 0.0)
    result = {
        "wall_s": round(elapsed, 3),
        "first_byte_s": round(first_byte, 3) if first_byte is not None else None,
        "bytes": len(audio),
        "rtf": round(elapsed / duration, 3) if duration else None,
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()},
    }
    if args.save_dir:
        directory = Path(args.save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{args.label}-{args.case_name}-{args.run_index}.wav"
        path.write_bytes(audio)
        result["path"] = str(path.resolve())
    return result


def summarize(samples: list[dict]) -> dict:
    summary = {}
    for key in ("wall_s", "first_byte_s", "rtf", "duration_s"):
        values = [s[key] for s in samples if s.get(key) is not None]
        if values:
            summary[key] = {
                "median": round(statistics.median(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--text", action="append", help="Text to benchmark (repeatable)"
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voice")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--chunk-length", type=int)
    parser.add_argument("--single-chunk", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--save-dir")
    parser.add_argument("--label", default="fish")
    parser.add_argument("--output", default="benchmark-results.json")
    args = parser.parse_args()

    url = args.api_url.rstrip("/") + "/v1/tts"
    defaults_url = args.api_url.rstrip("/") + "/v1/tts/defaults"
    try:
        with urllib.request.urlopen(defaults_url, timeout=args.timeout) as response:
            defaults = json.load(response)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"API is not reachable at {defaults_url}: {exc}")

    texts = {f"custom-{i + 1}": text for i, text in enumerate(args.text or [])}
    if not texts:
        texts = DEFAULT_TEXTS

    for _ in range(args.warmup):
        with post_json(
            url, {"text": "Warm up.", "format": "wav", "seed": args.seed}, args.timeout
        ) as response:
            response.read()

    cases = []
    for case_name, text in texts.items():
        samples = []
        for index in range(args.runs):
            args.case_name = case_name
            args.run_index = index
            samples.append(run_once(url, text, args))
        cases.append(
            {
                "case": case_name,
                "chars": len(text),
                "samples": samples,
                "summary": summarize(samples),
            }
        )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "streaming": args.streaming,
        "defaults": defaults,
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")

    print("| case | chars | audio s | wall s (median) | RTF (median) | first byte s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        summary = case["summary"]
        print(
            f"| {case['case']} | {case['chars']} | "
            f"{summary.get('duration_s', {}).get('median', '-')} | "
            f"{summary.get('wall_s', {}).get('median', '-')} | "
            f"{summary.get('rtf', {}).get('median', '-')} | "
            f"{summary.get('first_byte_s', {}).get('median', '-')} |"
        )
    print(f"\nWrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
