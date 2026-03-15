#!/usr/bin/env bash
# E2E memory check: two TTS requests to verify VRAM is freed between requests (no OOM on 32GB).
# Server must be running. Usage: PORT=8080 ./scripts/e2e_memory.sh   or: make e2e-memory
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PORT="${PORT:-8080}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
HEALTH_TIMEOUT="${E2E_HEALTH_TIMEOUT:-120}"
OUT_DIR="${E2E_OUT_DIR:-$REPO_ROOT/results/e2e}"
mkdir -p "$OUT_DIR"

echo "=== Fish Speech E2E memory (2 requests) ==="
echo "  BASE_URL=$BASE_URL"
echo ""

echo "Waiting for server (timeout ${HEALTH_TIMEOUT}s)..."
WAIT=0
until curl -sf --connect-timeout 2 "$BASE_URL/v1/health" >/dev/null 2>&1; do
  sleep 2
  WAIT=$((WAIT + 2))
  if [[ $WAIT -ge $HEALTH_TIMEOUT ]]; then
    echo "ERROR: server not ready"
    exit 1
  fi
done
echo "Server ready."
echo ""

MEM1="$OUT_DIR/memory_before.json"
MEM2="$OUT_DIR/memory_after_req1.json"
MEM3="$OUT_DIR/memory_after_req2.json"

echo "Memory before:"
curl -sf "$BASE_URL/v1/debug/memory" | tee "$MEM1" | (command -v jq >/dev/null 2>&1 && jq -c '{allocated_gb, reserved_gb}' || cat)
echo ""

echo "Request 1 (streaming)..."
curl -sf -X POST "$BASE_URL/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, this is the first request.","streaming":true}' \
  --output "$OUT_DIR/e2e_mem_req1.wav" --max-time 60
echo "Memory after request 1:"
curl -sf "$BASE_URL/v1/debug/memory" | tee "$MEM2" | (command -v jq >/dev/null 2>&1 && jq -c '{allocated_gb, reserved_gb, max_allocated_gb}' || cat)
echo ""

sleep 2
echo "Request 2 (streaming, OOM check)..."
if curl -sf -X POST "$BASE_URL/v1/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"Second request to confirm memory is freed.","streaming":true}' \
  --output "$OUT_DIR/e2e_mem_req2.wav" --max-time 60; then
  echo "Request 2: OK"
else
  echo "Request 2: FAIL (possible OOM)"
  exit 1
fi
echo "Memory after request 2:"
curl -sf "$BASE_URL/v1/debug/memory" | tee "$MEM3" | (command -v jq >/dev/null 2>&1 && jq -c '{allocated_gb, reserved_gb, max_allocated_gb}' || cat)
echo ""

echo "E2E memory passed (two requests succeeded). Outputs in $OUT_DIR"
