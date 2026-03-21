# Running the API server on 32GB VRAM (e.g. RTX 5090)

Minimal Docker-based setup to run the TTS API server on a single GPU with ~32GB VRAM. The same image and env work on RTX 3090/4090/5090.

## What we did (32GB tuning summary)

- **KV cache cap**: `FISH_CACHE_MAX_SEQ_LEN=384` and `FISH_MAX_NEW_TOKENS_CAP=80` to keep VRAM under ~32GB.
- **Memory between requests**: LLM `clear_caches()` after each request (KV cache released) plus `gc` / `empty_cache` so the second request does not OOM.
- **Back-pressure**: LLM and DAC do not run in parallel; codes are passed on CPU to avoid holding extra tensors on GPU.
- **Debug**: `/v1/debug/memory` returns allocated/reserved/max VRAM; optional `?dump=1` for PyTorch snapshot.
- **Non-streaming TTS**: Views fixed to consume `engine.inference()` and concatenate segment/final chunks into one WAV.
- **Repro**: This doc + `scripts/run_server_32gb.sh` + optional `WORKSPACE_DIR` when fish-speech lives inside another repo (e.g. checkpoints in parent).

## Quick start

From the **fish-speech repo root**:

```bash
# Build image, download checkpoints, start server (no torch.compile; fast startup)
./scripts/run_server_32gb.sh
```

- First run: builds the image and downloads `fishaudio/s2-pro` into `checkpoints/s2-pro`.
- Server listens on **http://127.0.0.1:8080** (override with `PORT=18081 ./scripts/run_server_32gb.sh`).

**Using the Makefile** (from repo root):

```bash
make run-server          # start server (PORT=8080)
make stop-server         # stop container
make e2e                 # smoke: health + TTS streaming/oneshot + memory (server must be up)
make e2e-memory          # two TTS requests + memory (OOM check)
make help                # list targets
```

If fish-speech is inside a parent repo and checkpoints are in the parent:

```bash
make run-server WORKSPACE_DIR=/path/to/parent CHECKPOINTS_DIR=checkpoints/s2-pro
```

## With torch.compile (faster inference after warmup)

Warmup runs at startup so the server only reports healthy after compilation. First start can take **several minutes** (2–10+), then requests are fast.

```bash
COMPILE=1 ./scripts/run_server_32gb.sh
```

Wait until health returns 200:

```bash
curl -s http://127.0.0.1:8080/v1/health
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Host port for the API |
| `CONTAINER` | fish-speech | Docker container name |
| `COMPILE` | 0 | Set to 1 to enable torch.compile (warmup at startup) |
| `IMAGE` | fish-speech-webui:cu129 | Docker image to run |
| `CHECKPOINTS_DIR` | checkpoints/s2-pro | Path to s2-pro checkpoints (relative to repo or `WORKSPACE_DIR`) |
| `WORKSPACE_DIR` | (unset) | If set, mount this dir as `/workspace` so both repo and checkpoints are visible (e.g. parent repo path). |

## 32GB tuning (already set by the script)

- `FISH_CACHE_MAX_SEQ_LEN=384` — KV cache size (smaller = less VRAM).
- `FISH_MAX_NEW_TOKENS_CAP=80` — cap on generated tokens per request to avoid OOM.
- `PYTHONPATH=/workspace` — so the container uses the mounted repo (e.g. your branch).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces fragmentation.

## E2E tests

With the server already running:

```bash
make e2e           # health + streaming TTS + oneshot TTS + /v1/debug/memory; writes to results/e2e/
make e2e-memory   # two streaming TTS requests + memory before/after (checks VRAM is freed, no OOM)
```

Or run scripts directly (same env vars as Makefile):

```bash
PORT=8080 ./scripts/e2e_smoke.sh      # smoke
PORT=8080 ./scripts/e2e_memory.sh     # two requests + memory
```

Outputs (WAV + `memory_*.json`) go to `results/e2e/` by default; override with `E2E_OUT_DIR=/path`.

## Manual smoke (curl)

After the server is up:

```bash
# Health
curl -s http://127.0.0.1:8080/v1/health

# Streaming TTS (no reference; or use reference_id if you added one)
curl -X POST http://127.0.0.1:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","streaming":true}' \
  --output test.wav

# GPU memory stats
curl -s http://127.0.0.1:8080/v1/debug/memory | jq
```

### Reference voice (optional)

Without a reference, the server uses the model’s default voice. To clone a specific voice you need a **reference**: a short clean speech recording (about 10–30 seconds) and its **exact transcript** (text that was spoken).

**There is no bundled sample.** Use your own recording (e.g. 10–30 s of yourself reading a paragraph), or any clean speech WAV + transcript (e.g. a sample from [LibriSpeech](https://www.openslr.org/12/) or similar). The WAV should be mono, 44.1 kHz is typical; the `.txt` or `.lab` must match what is said in the audio.

**Two ways to add a reference:**

**1) Pre-encode (offline, recommended for 32GB)** — encodes once so the server doesn’t load full WAV at runtime:

- Put in a folder one or more pairs: `<stem>.wav` + `<stem>.txt` (or `<stem>.lab`). Example: `data/voice_references/en.wav` and `data/voice_references/en.txt`.
- Run (from repo root, with venv/conda that has fish-speech deps):

  ```bash
  python tools/preencode_references.py --input-dir ./data/voice_references --output-dir references
  ```

  This creates `references/en/en.codes.pt` and `references/en/en.lab`. With `--ref-id NAME` all files go under `references/NAME/`.
- Ensure the server can see `references/` (e.g. same mount as repo, or copy into the container). Then call TTS with `"reference_id": "en"`.

**2) API (upload at runtime)** — server encodes on first use (uses more VRAM for that request):

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
  -F "id=my_voice" \
  -F "audio=@/path/to/recording.wav" \
  -F "text=Exact transcript of what is said in the audio."
```

- List IDs: `curl -s http://127.0.0.1:8080/v1/references/list`
- TTS with this voice: include `"reference_id": "my_voice"` in the JSON body of `POST /v1/tts`.

## Logs

```bash
docker logs -f fish-speech
```

## Stopping

```bash
make stop-server
# or
docker rm -f fish-speech
```
