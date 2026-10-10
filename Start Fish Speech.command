#!/bin/zsh
set -eu
cd "${0:A:h}"
mkdir -p .cache/matplotlib .cache/gradio .local-voice logs
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export GRADIO_TEMP_DIR="$PWD/.cache/gradio"
export HF_HOME="$PWD/.cache/huggingface"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export GRADIO_ANALYTICS_ENABLED=False
if /usr/bin/curl -fsS --max-time 2 http://127.0.0.1:7860/config 2>/dev/null | /usr/bin/grep -q 'S1 Mini'; then
    /usr/bin/open http://127.0.0.1:7860
    print 'Fish Speech is already running at http://127.0.0.1:7860'
    exit 0
fi
print 'Starting S1 Mini. Keep this Terminal window open; press Control-C to stop.'
exec ./venv/bin/python -u local_voice.py "$@"
