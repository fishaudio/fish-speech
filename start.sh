#!/usr/bin/env bash
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

# Resolve the model before setup so CLI selection controls compilation too.
MODEL=""
CHECKPOINT=""
MODEL_SET=0
CHECKPOINT_SET=0
EXPLICIT_COMPILE=0
API_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model|--llama-checkpoint-path)
            if [ "$#" -lt 2 ] || [[ "$2" == --* ]] || [ -z "$2" ]; then
                echo "$1 requires a value." >&2
                exit 1
            fi
            if [ "$1" = --model ]; then
                MODEL="$2"; MODEL_SET=1
            else
                CHECKPOINT="$2"; CHECKPOINT_SET=1
            fi
            shift 2
            ;;
        --model=*) MODEL="${1#*=}"; MODEL_SET=1; shift ;;
        --llama-checkpoint-path=*) CHECKPOINT="${1#*=}"; CHECKPOINT_SET=1; shift ;;
        --compile) EXPLICIT_COMPILE=1; API_ARGS+=("$1"); shift ;;
        *) API_ARGS+=("$1"); shift ;;
    esac
done
if [ "$MODEL_SET" = 1 ] && [ "$CHECKPOINT_SET" = 1 ]; then
    echo "Choose either --model or --llama-checkpoint-path." >&2
    exit 1
fi
if { [ "$MODEL_SET" = 1 ] && [ -z "$MODEL" ]; } || \
   { [ "$CHECKPOINT_SET" = 1 ] && [ -z "$CHECKPOINT" ]; }; then
    echo "Model and checkpoint values must not be empty." >&2
    exit 1
fi
if [ "$MODEL_SET" = 0 ] && [ "$CHECKPOINT_SET" = 0 ]; then
    MODEL="${FISH_MODEL:-fishaudio/s2-pro}"
    # FISH_MODEL can explicitly select a local directory; never auto-detect
    # repository checkpoints for a Hub model ID.
    case "$MODEL" in
        /*|./*|../*) CHECKPOINT="$MODEL"; MODEL="" ;;
        *) if [ -d "$MODEL" ]; then CHECKPOINT="$MODEL"; MODEL=""; fi ;;
    esac
fi
case "$MODEL" in
    s2-pro|S2-Pro) MODEL="fishaudio/s2-pro" ;;
    drama-3|Drama-3) MODEL="fishaudio/drama-3" ;;
esac
if [ "$MODEL" = fishaudio/drama-3 ] && [ "$EXPLICIT_COMPILE" = 1 ]; then
    echo "Drama 3 does not support --compile; omit it." >&2
    exit 1
fi
LAUNCH_ARGS=(--listen "${FISH_LISTEN:-0.0.0.0:8080}")
if [ -n "$CHECKPOINT" ]; then
    LAUNCH_ARGS+=(--llama-checkpoint-path "$CHECKPOINT")
else
    LAUNCH_ARGS+=(--model "$MODEL")
fi
if [ -n "${FISH_DEVICE:-}" ]; then LAUNCH_ARGS+=(--device "$FISH_DEVICE"); fi
if [ "${FISH_OFFLINE:-0}" = 1 ]; then LAUNCH_ARGS+=(--offline); fi
if [ "$MODEL" = fishaudio/s2-pro ] && [ "${FISH_COMPILE:-1}" != 0 ] && [ "$EXPLICIT_COMPILE" = 0 ]; then
    LAUNCH_ARGS+=(--compile)
fi
# The API resolver handles revision, offline mode, and codec fallback together.
# Avoid a separate download that could ignore those settings.

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

exec "$ROOT/start-api.sh" "${LAUNCH_ARGS[@]}" "${API_ARGS[@]}"
