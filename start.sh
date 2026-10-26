#!/bin/sh
# One-command DGX Spark launcher: install/refresh the locked environment, fetch
# the selected model if needed, build the WebUI once, and start the API.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd -- "$ROOT"

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != aarch64 ]; then
    echo "start.sh targets Linux ARM64 (DGX Spark); got $(uname -s) $(uname -m)." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# Optional subcommands. API flags can still be passed directly to ./start.sh.
case "${1:-}" in
    mcp)
        shift
        exec "$ROOT/start-mcp.sh" "$@"
        ;;
    api)
        shift
        ;;
esac

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/fish-speech/inductor}"
mkdir -p -- "$TORCHINDUCTOR_CACHE_DIR"

PY="$ROOT/environments/dgx-spark/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Creating the DGX Spark CUDA 13 environment..."
    uv sync --project environments/dgx-spark --locked --python 3.12 --extra cu130
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "WARNING: nvidia-smi cannot see the GPU; the API will fail to start." >&2
        echo "         Check the driver, then rerun ./start.sh." >&2
    fi
fi

MODEL="${FISH_MODEL:-fishaudio/s2-pro}"
IS_S2=0
case "$MODEL" in
    s2-pro|S2-Pro|fishaudio/s2-pro)
        HF_MODEL="fishaudio/s2-pro"
        LOCAL_CHECKPOINT="$ROOT/checkpoints/s2-pro"
        IS_S2=1
        ;;
    drama-3|Drama-3|fishaudio/drama-3)
        HF_MODEL="fishaudio/drama-3"
        LOCAL_CHECKPOINT="$ROOT/checkpoints/drama-3"
        ;;
    *)
        HF_MODEL="$MODEL"
        LOCAL_CHECKPOINT="$MODEL"
        ;;
esac

CHECKPOINT=""
if [ -f "$LOCAL_CHECKPOINT/model.safetensors.index.json" ] || \
   [ -f "$LOCAL_CHECKPOINT/model.safetensors" ]; then
    CHECKPOINT="$LOCAL_CHECKPOINT"
    echo "Using local checkpoint: $CHECKPOINT"
else
    echo "Using Hugging Face model: $HF_MODEL"
    if [ "${FISH_OFFLINE:-0}" = "1" ]; then
        echo "Offline mode: expecting the model to be fully cached."
    elif [ "${FISH_SKIP_DOWNLOAD:-0}" = "1" ]; then
        echo "Skipping model download (FISH_SKIP_DOWNLOAD=1)."
    else
        "$ROOT/environments/dgx-spark/.venv/bin/hf" download "$HF_MODEL" --max-workers 8
    fi
fi

if [ ! -f awesome_webui/dist/index.html ] && [ "${FISH_SKIP_WEBUI:-0}" != "1" ]; then
    if command -v npm >/dev/null 2>&1; then
        echo "Building the WebUI (first run only)..."
        if ! (cd awesome_webui && npm ci && npm run build); then
            echo "WARNING: WebUI build failed; the API will still start." >&2
        fi
    else
        echo "WARNING: npm not found; /ui will be unavailable until the WebUI is built." >&2
    fi
fi

LISTEN="${FISH_LISTEN:-0.0.0.0:8080}"
DEVICE_ARGS=""
if [ -n "${FISH_DEVICE:-}" ]; then
    DEVICE_ARGS="--device $FISH_DEVICE"
fi
OFFLINE_ARGS=""
if [ "${FISH_OFFLINE:-0}" = "1" ]; then
    OFFLINE_ARGS="--offline"
fi
COMPILE_ARGS=""
if [ "$IS_S2" = "1" ] && [ "${FISH_COMPILE:-1}" != "0" ]; then
    # S2-Pro steady-state decode is ~34% faster with Inductor; the one-time
    # compile adds roughly 50 s to startup. Drama 3 rejects --compile.
    COMPILE_ARGS="--compile"
fi

echo "Starting Fish Speech API on http://$LISTEN (WebUI: /ui)"
if [ -n "$CHECKPOINT" ]; then
    CODEC="$CHECKPOINT/codec.pth"
    if [ ! -f "$CODEC" ] && [ -f "$ROOT/checkpoints/s2-pro/codec.pth" ]; then
        CODEC="$ROOT/checkpoints/s2-pro/codec.pth"
    fi
    # shellcheck disable=SC2086
    exec "$ROOT/start-api.sh" \
        --llama-checkpoint-path "$CHECKPOINT" \
        --decoder-checkpoint-path "$CODEC" \
        --listen "$LISTEN" $DEVICE_ARGS $OFFLINE_ARGS $COMPILE_ARGS "$@"
else
    # shellcheck disable=SC2086
    exec "$ROOT/start-api.sh" \
        --model "$HF_MODEL" \
        --listen "$LISTEN" $DEVICE_ARGS $OFFLINE_ARGS $COMPILE_ARGS "$@"
fi
