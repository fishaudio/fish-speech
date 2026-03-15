#!/usr/bin/env python3
"""
Poll /v1/debug/memory at a fixed interval; append one JSON line only when metrics change.
Usage:
  python tools/memory_metrics.py
  python tools/memory_metrics.py --interval 0.25 --out metrics.jsonl
  python tools/memory_metrics.py --all   # write every sample (no change filter)
Stop with Ctrl+C.
"""

import argparse
import json
import sys
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(
        description="Log GPU memory metrics to a JSONL file (on change)"
    )
    p.add_argument(
        "--url",
        default="http://127.0.0.1:18081/v1/debug/memory",
        help="Debug memory endpoint URL",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Sample interval in seconds (default: 0.25)",
    )
    p.add_argument(
        "--out",
        default="memory_metrics.jsonl",
        help="Output JSONL file (default: memory_metrics.jsonl)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Write every sample; default is only when allocated/reserved/max change",
    )
    args = p.parse_args()

    req = urllib.request.Request(args.url)
    line_count = 0
    last_key = None
    try:
        with open(args.out, "a") as f:
            while True:
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        data = json.loads(r.read().decode())
                except Exception as e:
                    print(f"{time.time():.0f} error: {e}", file=sys.stderr)
                    time.sleep(args.interval)
                    continue
                ts = time.time()
                row = {
                    "ts": ts,
                    "allocated_gb": data.get("allocated_gb"),
                    "reserved_gb": data.get("reserved_gb"),
                    "max_allocated_gb": data.get("max_allocated_gb"),
                }
                if "models" in data:
                    row["models"] = data["models"]
                key = (row["allocated_gb"], row["reserved_gb"], row["max_allocated_gb"])
                if args.all or key != last_key:
                    last_key = key
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    line_count += 1
                    print(
                        f"{ts:.0f} alloc={row['allocated_gb']} reserved={row['reserved_gb']} max={row['max_allocated_gb']} GB",
                        file=sys.stderr,
                    )
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nStopped. Wrote {line_count} lines to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
