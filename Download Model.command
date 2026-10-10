#!/bin/zsh
set -eu
cd "${0:A:h}"
export HF_HOME="$PWD/.cache/huggingface"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
./venv/bin/python -u - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'cocktailpeanut/oa',
    revision='be35f06bb5c7604fc2197f631bc3cd4e3286af30',
    local_dir='checkpoints/openaudio-s1-mini',
    max_workers=3,
)
print('S1 Mini download complete. You can now run Start Fish Speech.command.')
PY
