#!/usr/bin/env bash
# Pre-encode reference WAV+txt to references/. Optional: upload to server.
# Env: PREENCODE_INPUT_DIR, PREENCODE_OUTPUT_DIR, PREENCODE_CHECKPOINT, PREENCODE_REF_ID, UPLOAD, SERVER_URL
# Usage: ./scripts/preencode.sh   or   UPLOAD=1 ./scripts/preencode.sh
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

INPUT_DIR="${PREENCODE_INPUT_DIR:-data/voice_references}"
OUTPUT_DIR="${PREENCODE_OUTPUT_DIR:-references}"
CHECKPOINT="${PREENCODE_CHECKPOINT:-checkpoints/s2-pro/codec.pth}"
REF_ID="${PREENCODE_REF_ID:-}"
UPLOAD="${UPLOAD:-0}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080}"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: PREENCODE_INPUT_DIR=$INPUT_DIR not found. Create folder with <stem>.wav + <stem>.txt"
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: PREENCODE_CHECKPOINT=$CHECKPOINT not found. E.g. PREENCODE_CHECKPOINT=../../checkpoints/s2-pro/codec.pth"
  exit 1
fi

set -- --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --checkpoint-path "$CHECKPOINT"
[[ -n "$REF_ID" ]] && set -- "$@" --ref-id "$REF_ID"
[[ "$UPLOAD" =~ ^(1|true|yes|YES)$ ]] && set -- "$@" --upload --server-url "$SERVER_URL"

exec uv run python tools/preencode_references.py "$@"
