# Server

This page covers server-side inference for Fish Audio runtime, including S2-Pro and OpenAudio S1-Mini pipelines.

## API Server Inference

Fish Speech provides an HTTP API server entrypoint at `tools/api_server.py`.

### Start the server locally

```bash
# S2-Pro (default)
python tools/api_server.py \
  --listen 0.0.0.0:8080
```

For S1-Mini, set `FISH_MODEL_TYPE=s1` before startup:

```bash
FISH_MODEL_TYPE=s1 python tools/api_server.py --listen 0.0.0.0:8080
```

Model defaults and overrides:

- `FISH_MODEL_TYPE=s2` (default) uses `checkpoints/s2-pro` and `checkpoints/s2-pro/codec.pth`
- `FISH_MODEL_TYPE=s1` uses `checkpoints/openaudio-s1-mini` and `checkpoints/openaudio-s1-mini/codec.pth`
- optional overrides: `LLAMA_CHECKPOINT_PATH`, `DECODER_CHECKPOINT_PATH`

Common options:

- `--compile`: enable `torch.compile` optimization
- `--half`: use fp16 mode
- `--api-key`: require bearer token authentication
- `--workers`: set worker process count

### Health check

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

Expected response:

```json
{"status":"ok"}
```

### Main API endpoint

- `POST /v1/tts` for text-to-speech generation
- `POST /v1/vqgan/encode` for VQ encode
- `POST /v1/vqgan/decode` for VQ decode

## WebUI Inference

For WebUI usage, see:

- [WebUI Inference](https://speech.fish.audio/inference/#webui-inference)

## Docker

For Docker-based server or WebUI deployment, see:

- [Docker Setup](https://speech.fish.audio/install/#docker-setup)

You can also start the server profile directly with Docker Compose:

```bash
docker compose --profile server up
```
