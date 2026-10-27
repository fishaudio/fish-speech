#!/bin/sh
set -eu

# Resolve checkpoints and uv's environment relative to this script.
cd -- "$(dirname -- "$0")"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/fish-speech/inductor}"
mkdir -p -- "$TORCHINDUCTOR_CACHE_DIR"

exec uv run --project environments/dgx-spark --locked --extra cu130 python tools/api_server.py --device cuda --listen 0.0.0.0:8080 "$@"
