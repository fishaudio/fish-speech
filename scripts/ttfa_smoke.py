#!/usr/bin/env python3
"""One TTS request (streaming or oneshot): measure TTFA and save WAV. Prints ttfa_s, ttfa_audio_s, total_s."""
import argparse
import sys
import time

WAV_HEADER_SIZE = 44


def main() -> None:
    p = argparse.ArgumentParser(description="TTS request with TTFA/timing metrics")
    p.add_argument("--url", required=True, help="Base URL (e.g. http://127.0.0.1:8080)")
    p.add_argument("--output", required=True, help="Output WAV path")
    p.add_argument("--reference-id", default=None, help="Optional reference_id")
    p.add_argument("--text", default=None, help="Text to synthesize (default depends on --oneshot)")
    p.add_argument("--oneshot", action="store_true", help="Non-streaming request (streaming: false)")
    args = p.parse_args()
    if args.text is None:
        args.text = "Short test." if args.oneshot else "Hello, this is an e2e smoke test."

    try:
        import urllib.request
    except ImportError:
        print("ERROR: urllib.request required", file=sys.stderr)
        sys.exit(1)

    url = f"{args.url.rstrip('/')}/v1/tts"
    body = {"text": args.text, "streaming": not args.oneshot}
    if args.reference_id:
        body["reference_id"] = args.reference_id
    import json
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    t1: float | None = None  # first byte
    t2: float | None = None  # first audio byte (after 44-byte header)
    out_bytes: list[bytes] = []
    bytes_read = 0

    with urllib.request.urlopen(req, timeout=60) as resp:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            if t1 is None:
                t1 = time.perf_counter()
            out_bytes.append(chunk)
            bytes_read += len(chunk)
            if t2 is None and bytes_read > WAV_HEADER_SIZE:
                t2 = time.perf_counter()
    t3 = time.perf_counter()

    with open(args.output, "wb") as f:
        f.write(b"".join(out_bytes))

    ttfa_s = (t1 - t0) if t1 is not None else 0.0
    ttfa_audio_s = (t2 - t0) if t2 is not None else ttfa_s
    total_s = t3 - t0
    print(f"  ttfa_s={ttfa_s:.3f}  ttfa_audio_s={ttfa_audio_s:.3f}  total_s={total_s:.3f}  bytes={bytes_read}")


if __name__ == "__main__":
    main()
