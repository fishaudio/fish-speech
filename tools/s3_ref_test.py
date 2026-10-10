import argparse
import io
import sys
import time
from pathlib import Path

import ormsgpack
import requests
import soundfile as sf

# Mixed-language sample texts for voice cloning checks.
DEFAULT_TEXTS = [
    "你好，欢迎使用语音合成服务，今天的天气非常不错。",
    "这是一段用参考音频做声音克隆的测试，看看音色像不像。",
    "Get the trust fund to the bank early, before the storm arrives tonight.",
    "他停顿了一下，低声说道：其实，我早就知道这件事了，只是一直没说出口。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send reference-voice TTS requests to a running server and "
        "report latency and real-time factor per request. The reference transcript "
        "comes from --ref-text or a .lab / .txt file next to the audio."
    )
    parser.add_argument("--url", "-u", default="http://127.0.0.1:8080/v1/tts")
    parser.add_argument("--ref-audio", "-r", default="test.wav", help="reference audio")
    parser.add_argument("--ref-text", help="exact transcript of the reference audio")
    parser.add_argument("--text", "-t", action="append", help="text to synthesize")
    parser.add_argument("--repeat", type=int, default=1, help="requests per text")
    parser.add_argument("--out-dir", "-o", default="outputs/s3_ref_test")
    parser.add_argument(
        "--format", default="wav", choices=["wav", "mp3", "opus", "pcm"]
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-new-tokens", type=int)
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
    info = sf.info(str(ref_audio))
    print(f"reference: {ref_audio} ({info.duration:.2f}s)  text: {ref_text!r}")

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
    reference = {"audio": ref_audio.read_bytes(), "text": ref_text}

    health_url = args.url.rsplit("/v1/", 1)[0] + "/v1/health"
    try:
        requests.get(health_url, timeout=5).raise_for_status()
    except requests.RequestException as exc:
        sys.exit(f"server not reachable at {args.url}: {exc}")

    total_audio = total_latency = 0.0
    succeeded = sent = 0
    for index, text in enumerate(args.text or DEFAULT_TEXTS):
        for run in range(args.repeat):
            payload = {
                "text": text,
                "references": [reference],
                "format": args.format,
                **sampling,
            }
            started = time.perf_counter()
            response = requests.post(
                args.url,
                data=ormsgpack.packb(payload),
                headers={"Content-Type": "application/msgpack"},
                timeout=args.timeout,
            )
            latency = time.perf_counter() - started
            name = f"{index:02d}_{run}.{args.format}"
            sent += 1
            if response.status_code != 200:
                print(f"{name}: HTTP {response.status_code}: {response.text[:300]}")
                continue
            (out_dir / name).write_bytes(response.content)
            if args.format == "pcm":
                print(f"{name}: latency {latency:6.2f}s  {text[:30]}")
                continue
            data, sample_rate = sf.read(io.BytesIO(response.content))
            duration = len(data) / sample_rate
            print(
                f"{name}: latency {latency:6.2f}s  audio {duration:6.2f}s  "
                f"RTF {latency / duration:.3f}  {text[:30]}"
            )
            succeeded += 1
            total_audio += duration
            total_latency += latency

    print("-" * 72)
    print(f"{succeeded}/{sent} decoded, outputs in {out_dir}")
    if total_audio:
        print(
            f"total audio {total_audio:.2f}s, total latency {total_latency:.2f}s, "
            f"mean RTF {total_latency / total_audio:.3f}"
        )


if __name__ == "__main__":
    main()
