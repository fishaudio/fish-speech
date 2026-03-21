# Fish Speech — 32GB VRAM Docker server and E2E tests
# Use from repo root. See docs/docker-32gb-rtx5090.md

PORT ?= 8080
CONTAINER ?= fish-speech
BASE_URL ?= http://127.0.0.1:$(PORT)

.PHONY: run-server stop-server e2e e2e-memory preencode preencode-upload upload-references test help

help:
	@echo "Fish Speech 32GB server and E2E"
	@echo "  make run-server        - Start API server in Docker (COMPILE=0)"
	@echo "  make run-server COMPILE=1 - Start with torch.compile (warmup 2-10 min)"
	@echo "  make stop-server       - Stop container"
	@echo "  make e2e               - E2E smoke (server must be up)"
	@echo "  make e2e-memory        - E2E memory (2 TTS requests)"
	@echo "  make preencode         - Pre-encode WAV+txt to references/"
	@echo "  make preencode-upload  - Pre-encode and upload to server"
	@echo "  make upload-references - Upload existing references/ only"
	@echo "  make test              - Run pytest"
	@echo "Override: PORT=8080 PREENCODE_CHECKPOINT=../../checkpoints/s2-pro/codec.pth"

run-server:
	WORKSPACE_DIR="$(WORKSPACE_DIR)" CHECKPOINTS_DIR="$(CHECKPOINTS_DIR)" PORT="$(PORT)" COMPILE="$(COMPILE)" ./scripts/run_server_32gb.sh

stop-server:
	docker rm -f $(CONTAINER) 2>/dev/null || true

e2e:
	PORT="$(PORT)" BASE_URL="$(BASE_URL)" ./scripts/e2e_smoke.sh

e2e-memory:
	PORT="$(PORT)" BASE_URL="$(BASE_URL)" ./scripts/e2e_memory.sh

preencode:
	PREENCODE_INPUT_DIR="$(PREENCODE_INPUT_DIR)" PREENCODE_OUTPUT_DIR="$(PREENCODE_OUTPUT_DIR)" \
	PREENCODE_CHECKPOINT="$(PREENCODE_CHECKPOINT)" PREENCODE_REF_ID="$(PREENCODE_REF_ID)" \
	./scripts/preencode.sh

preencode-upload:
	PREENCODE_INPUT_DIR="$(PREENCODE_INPUT_DIR)" PREENCODE_OUTPUT_DIR="$(PREENCODE_OUTPUT_DIR)" \
	PREENCODE_CHECKPOINT="$(PREENCODE_CHECKPOINT)" PREENCODE_REF_ID="$(PREENCODE_REF_ID)" \
	UPLOAD=1 SERVER_URL="$(BASE_URL)" ./scripts/preencode.sh

upload-references:
	PREENCODE_OUTPUT_DIR="$(PREENCODE_OUTPUT_DIR)" BASE_URL="$(BASE_URL)" ./scripts/upload_references.sh

test:
	uv run pytest tests/ -v --tb=short
