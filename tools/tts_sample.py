"""Send text passages to a Fish Speech server with a reference voice and save the audio.

    python tools/tts_sample.py -r test.wav -t "第一句" -t "第二句" -o outputs/asmr

The reference transcript comes from --ref-text or a .lab / .txt file next to the audio.
"""

import argparse
import sys
from pathlib import Path

import ormsgpack
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", "-u", default="http://127.0.0.1:8080/v1/tts")
    parser.add_argument("--ref-audio", "-r", default="test.wav", help="reference audio")
    parser.add_argument("--ref-text", default=None, help="transcript of the reference")
    parser.add_argument(
        "--text",
        "-t",
        action="extend",
        nargs="+",
        required=True,
        help="one passage per argument (repeat -t for several)",
    )
    parser.add_argument("--out-dir", "-o", default="outputs/tts_sample")
    parser.add_argument("--prefix", default="sample", help="output file name prefix")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--format", default="wav")
    parser.add_argument("--timeout", type=float, default=600)
    return parser.parse_args()


def reference_text(audio: Path, given: str | None) -> str:
    if given:
        return given
    for suffix in (".lab", ".txt"):
        sidecar = audio.with_suffix(suffix)
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8").strip()
    sys.exit(
        f"no transcript for {audio}: pass --ref-text or write "
        f"{audio.with_suffix('.lab')}"
    )


def main() -> None:
    args = parse_args()
    ref_audio = Path(args.ref_audio)
    if not ref_audio.is_file():
        sys.exit(f"reference audio not found: {ref_audio}")
    ref_text = reference_text(ref_audio, args.ref_text)
    print(f"reference: {ref_audio}  text: {ref_text!r}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling = {
        key: value
        for key, value in (
            ("temperature", args.temperature),
            ("top_p", args.top_p),
            ("max_new_tokens", args.max_new_tokens),
        )
        if value is not None
    }
    headers = {"Content-Type": "application/msgpack"}
    reference = {"audio": ref_audio.read_bytes(), "text": ref_text}

    for index, text in enumerate(args.text):
        payload = {
            "text": text,
            "references": [reference],
            "format": args.format,
            **sampling,
        }
        response = requests.post(
            args.url,
            data=ormsgpack.packb(payload),
            headers=headers,
            timeout=args.timeout,
        )
        if response.status_code != 200:
            print(f"{index}: HTTP {response.status_code}: {response.text[:300]}")
            continue
        out_path = out_dir / f"{args.prefix}_{index:02d}.{args.format}"
        out_path.write_bytes(response.content)
        print(f"{out_path}: {len(text)} chars -> {len(response.content)} bytes")


if __name__ == "__main__":
    main()
