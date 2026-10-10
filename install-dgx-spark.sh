#!/bin/sh
# Run as your normal user; sudo is used only for Ubuntu system packages.
set -eu

cd -- "$(dirname -- "$0")"

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != aarch64 ]; then
    echo "This installer requires Linux ARM64 (aarch64), as used by DGX Spark." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this script as your normal user, without sudo, to own the .venv." >&2
    exit 1
fi

if ! command -v apt-get >/dev/null 2>&1 || ! command -v sudo >/dev/null 2>&1; then
    echo "This installer requires Ubuntu's apt-get and sudo." >&2
    exit 1
fi

echo "Installing system dependencies (sudo may ask for your password)..."
sudo apt-get update
sudo apt-get install -y build-essential python3-dev portaudio19-dev libsox-dev libsndfile1 ffmpeg

echo "Installing the Python 3.12 / CUDA 13 environment with uv..."
uv sync --python 3.12

cat <<'EOF'
Installation complete. Download the model weights if you do not have them yet:
  uv run --locked hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro

Start the WebUI:
  uv run --locked python tools/run_webui.py --device cuda

See docs/en/dgx-spark.md for GPU and audio diagnostics.
EOF
