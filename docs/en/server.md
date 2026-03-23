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
- `GET /v1/references/compatibility` for the runtime prompt-bundle contract

### Reusing a VQ-encoded prompt bundle

For advanced integrations, a client can encode a reference audio once with `/v1/vqgan/encode`, fetch the runtime compatibility contract from `/v1/references/compatibility`, then send the resulting prompt bundle directly to `/v1/tts` as `reference_payload`.

This is useful when your own backend wants to cache and replay prompt tokens without resending raw reference audio on every TTS request.

Example flow:

1. Call `POST /v1/vqgan/encode`
2. Call `GET /v1/references/compatibility`
3. Store a prompt bundle on your backend:

```json
{
  "reference_id": "debug-sky",
  "reference_text": "Hello there.",
  "prompt_tokens": [[1, 2], [3, 4], [5, 6]],
  "reference_fingerprint": "sha256:...",
  "compatibility": {
    "artifact_schema_version": 1,
    "codec_checkpoint_sha256": "sha256:...",
    "decoder_config_name": "modded_dac_vq",
    "text2semantic_checkpoint_sha256": "sha256:...",
    "tokenizer_sha256": "sha256:...",
    "num_codebooks": 3,
    "semantic_begin_id": 1024,
    "sample_rate_hz": 24000
  }
}
```

4. Send that bundle to `POST /v1/tts`:

```json
{
  "text": "Say hello",
  "reference_payload": {
    "reference_id": "debug-sky",
    "reference_text": "Hello there.",
    "prompt_tokens": [[1, 2], [3, 4], [5, 6]],
    "reference_fingerprint": "sha256:...",
    "compatibility": {
      "artifact_schema_version": 1,
      "codec_checkpoint_sha256": "sha256:...",
      "decoder_config_name": "modded_dac_vq",
      "text2semantic_checkpoint_sha256": "sha256:...",
      "tokenizer_sha256": "sha256:...",
      "num_codebooks": 3,
      "semantic_begin_id": 1024,
      "sample_rate_hz": 24000
    }
  }
}
```

Request precedence is:

- `reference_payload`
- `references`
- `reference_id`

So if `reference_payload` is present, the server ignores inline `references` and `reference_id`.

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
