#!/usr/bin/env bash
# Upload existing references/ to server (no encoding). Server skips if hash matches.
# Env: PREENCODE_OUTPUT_DIR, SERVER_URL (or BASE_URL)
# Usage: ./scripts/upload_references.sh   or   make upload-references
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${PREENCODE_OUTPUT_DIR:-references}"
SERVER_URL="${SERVER_URL:-${BASE_URL:-http://127.0.0.1:8080}}"

exec uv run python tools/preencode_references.py \
  --output-dir "$OUTPUT_DIR" \
  --upload-only --server-url "$SERVER_URL"
