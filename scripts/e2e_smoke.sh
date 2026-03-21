#!/usr/bin/env bash
# E2E smoke: server must be running (e.g. ./scripts/run_server_32gb.sh).
# Waits for health, runs TTS (streaming + oneshot), checks /v1/debug/memory.
# Usage: PORT=8080 ./scripts/e2e_smoke.sh   or: make e2e
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
CONTAINER="${CONTAINER:-fish-speech}"
HEALTH_TIMEOUT="${E2E_HEALTH_TIMEOUT:-120}"
OUT_DIR="${E2E_OUT_DIR:-$REPO_ROOT/results/e2e}"
mkdir -p "$OUT_DIR"

echo "=== Fish Speech E2E smoke ==="
echo "  BASE_URL=$BASE_URL  CONTAINER=$CONTAINER  OUT_DIR=$OUT_DIR"
echo ""

echo "Waiting for server (timeout ${HEALTH_TIMEOUT}s)..."
WAIT=0
until curl -sf --connect-timeout 2 "$BASE_URL/v1/health" >/dev/null 2>&1; do
  sleep 2
  WAIT=$((WAIT + 2))
  if [[ $WAIT -ge $HEALTH_TIMEOUT ]]; then
    echo "ERROR: server not ready in ${HEALTH_TIMEOUT}s"
    exit 1
  fi
done
echo "Server ready after ${WAIT}s."
echo ""

# Optional: use first reference from server (e.g. after preencode + upload)
REF_ID=""
if command -v jq >/dev/null 2>&1; then
  REF_ID=$(curl -sf --compressed "$BASE_URL/v1/references/list" 2>/dev/null | jq -r '.reference_ids[0] // empty' 2>/dev/null) || true
fi
if [[ -n "$REF_ID" ]]; then
  echo "Using reference_id=$REF_ID for TTS"
fi
echo ""

# Streaming TTS (with TTFA metrics when python3 + ttfa_smoke.py available)
echo "--- TTS streaming ---"
STREAM_OUT="$OUT_DIR/e2e_stream.wav"
TTFA_OK=0
if command -v python3 >/dev/null 2>&1 && [[ -f "$REPO_ROOT/scripts/ttfa_smoke.py" ]]; then
  TTFA_ARGS="--url $BASE_URL --output $STREAM_OUT"
  [[ -n "$REF_ID" ]] && TTFA_ARGS="$TTFA_ARGS --reference-id $REF_ID"
  if python3 "$REPO_ROOT/scripts/ttfa_smoke.py" $TTFA_ARGS && [[ -s "$STREAM_OUT" ]]; then
    TTFA_OK=1
  else
    echo "  (ttfa_smoke failed, falling back to curl)"
  fi
fi
if [[ "$TTFA_OK" -ne 1 ]]; then
  if [[ -n "$REF_ID" ]]; then
    TTS_JSON="{\"text\":\"Hello, this is an e2e smoke test.\",\"streaming\":true,\"reference_id\":\"$REF_ID\"}"
  else
    TTS_JSON='{"text":"Hello, this is an e2e smoke test.","streaming":true}'
  fi
  curl -sf -X POST "$BASE_URL/v1/tts" \
    -H "Content-Type: application/json" \
    -d "$TTS_JSON" \
    --output "$STREAM_OUT" --max-time 60
fi
if [[ ! -s "$STREAM_OUT" ]]; then
  echo "ERROR: streaming TTS produced empty or missing file: $STREAM_OUT"
  exit 1
fi
echo "  OK: $STREAM_OUT ($(wc -c < "$STREAM_OUT") bytes)"
echo ""

# Oneshot TTS (with timing metrics when python3 + ttfa_smoke.py available)
echo "--- TTS oneshot ---"
ONESHOT_OUT="$OUT_DIR/e2e_oneshot.wav"
ONESHOT_TTFA_OK=0
if command -v python3 >/dev/null 2>&1 && [[ -f "$REPO_ROOT/scripts/ttfa_smoke.py" ]]; then
  ONESHOT_ARGS="--url $BASE_URL --output $ONESHOT_OUT --oneshot"
  [[ -n "$REF_ID" ]] && ONESHOT_ARGS="$ONESHOT_ARGS --reference-id $REF_ID"
  if python3 "$REPO_ROOT/scripts/ttfa_smoke.py" $ONESHOT_ARGS && [[ -s "$ONESHOT_OUT" ]]; then
    ONESHOT_TTFA_OK=1
  else
    echo "  (ttfa_smoke oneshot failed, falling back to curl)"
  fi
fi
if [[ "$ONESHOT_TTFA_OK" -ne 1 ]]; then
  if [[ -n "$REF_ID" ]]; then
    ONESHOT_JSON="{\"text\":\"Short test.\",\"streaming\":false,\"reference_id\":\"$REF_ID\"}"
  else
    ONESHOT_JSON='{"text":"Short test.","streaming":false}'
  fi
  curl -sf -X POST "$BASE_URL/v1/tts" \
    -H "Content-Type: application/json" \
    -d "$ONESHOT_JSON" \
    --output "$ONESHOT_OUT" --max-time 60
fi
if [[ ! -s "$ONESHOT_OUT" ]]; then
  echo "ERROR: oneshot TTS produced empty or missing file: $ONESHOT_OUT"
  exit 1
fi
echo "  OK: $ONESHOT_OUT ($(wc -c < "$ONESHOT_OUT") bytes)"
echo ""

# Memory
echo "--- Memory ---"
curl -sf --compressed "$BASE_URL/v1/debug/memory" | tee "$OUT_DIR/memory_after.json" | (command -v jq >/dev/null 2>&1 && jq -c '{allocated_gb, reserved_gb, max_allocated_gb}' 2>/dev/null || cat) || true
echo ""

echo "E2E smoke passed. Outputs in $OUT_DIR"
