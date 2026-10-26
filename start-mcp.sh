#!/bin/sh
# Start the Fish Speech MCP server. The TTS API must already be running.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd -- "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if [ ! -x environments/mcp/.venv/bin/python ]; then
    echo "Creating the MCP client environment..."
    uv sync --project environments/mcp --locked
fi

exec uv run --project environments/mcp --locked python tools/mcp_server.py "$@"
