#!/bin/zsh
set -eu
cd "${0:A:h}"
./venv/bin/python -m venv .venv-mcp
./.venv-mcp/bin/python -m pip install -r local_mcp_requirements.txt
print 'Voice MCP adapter installed. See VOICEOVER_SETUP.md to connect your apps.'
