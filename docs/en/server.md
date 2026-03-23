# Server

This page covers server-side inference for Fish Audio S2, plus quick links for WebUI inference and Docker deployment.

## API Server Inference

Fish Speech provides an HTTP API server entrypoint at `tools/api_server.py`.

### Start the server locally

```bash
python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 0.0.0.0:8080
```

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

### Python client example

The base TTS model is selected when the server starts. In the example above, the server is started with the `checkpoints/s2-pro` weights, so every request sent to `http://127.0.0.1:8080/v1/tts` will use **S2-Pro** automatically. There is no separate per-request `model` field in `tools/api_client.py` for local server calls.

```bash
python tools/api_client.py \
  --url http://127.0.0.1:8080/v1/tts \
  --text "Hello from Fish Speech" \
  --output s2-pro-demo
```

If you want to select a saved reference voice, use `--reference_id`. This chooses the **voice reference**, not the base TTS model:

```bash
python tools/api_client.py \
  --url http://127.0.0.1:8080/v1/tts \
  --text "Hello from Fish Speech" \
  --reference_id my-speaker \
  --output s2-pro-demo
```

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
