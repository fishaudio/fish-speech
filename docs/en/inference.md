# Inference

This fork defaults to the shared Hugging Face cache. Use `--model fishaudio/s2-pro`
or `--model fishaudio/drama-3` with the API server or text-to-semantic CLI.
Existing explicit checkpoint-path commands below remain supported. See
[model cache and migration](model-cache.md) for revisions and offline mode.

Fish Audio ships two models: **S2-Pro** and **Drama 3**. They share the same command line and API server, and the model type is read from the checkpoint's `config.json`, so every section below is organized as "task → command for each model".

Finish [Installation](install.md) and download the [model weights](install.md#download-weights) first.

!!! warning "Drama 3 runtime limits"
    Drama 3 runs on NVIDIA GPUs only: CPU, Apple MPS and AMD ROCm are not supported, and neither is `--compile`. See [Requirements](install.md#requirements) for GPU memory.

!!! note
    V100 and other GPUs older than SM80 have no bfloat16, so Drama 3 switches to float16 there (keeping the residual stream in fp32). This path has not been tested on real hardware yet; if you get silence, noise or a "non-finite logits" error, please open an issue.

## Command Line Inference

### S2-Pro

S2-Pro uses a three-step pipeline: "VQ tokens → Semantic tokens → audio".

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip step 1.

**1. Get VQ tokens from reference audio**

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

You should get a `fake.npy` and a `fake.wav`.

**2. Generate Semantic tokens from text**

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    # --compile
```

This command will create a `codes_N` file in the working directory, where N is an integer starting from 0.

!!! note
    You may want to use `--compile` to fuse CUDA kernels for faster inference. However, we recommend using our sglang inference acceleration optimization.
    Correspondingly, if you do not plan to use acceleration, you can comment out the `--compile` parameter.

!!! info
    For GPUs that do not support bf16, you may need to use the `--half` parameter.

**3. Generate vocals from semantic tokens**

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

After that, you will get a `fake.wav` file.

### Drama 3

Drama 3 does it in one command; `--output` writes the decoded audio directly:

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/drama-3 \
    --text "The text you want to convert" \
    --prompt-audio "reference.wav" \
    --prompt-text "Your reference text" \
    --output "output.wav" \
    --device cuda:0
```

On older GPUs, list several devices, for example `--device cuda:0,cuda:1,cuda:2,cuda:3`; the last one holds the audio decoder and the codec. Plain `--device cuda` uses up to the first four visible GPUs.

## API Server

### Start S2-Pro

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq
```

### Start Drama 3

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/drama-3 \
    --device cuda:0
```

!!! note
    Drama 3 does not support `--compile`. On older GPUs, list several devices with `--device`; the last one holds the audio decoder and the codec.

Both models share these startup options:

- `--compile`: enable `torch.compile` optimization (S2-Pro only)
- `--half`: use fp16 mode
- `--api-key`: require bearer token authentication
- `--workers`: set the worker process count
- `--max-text-length`: cap the text length of a single request; 0 means no limit

Health check:

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

A response of `{"status":"ok"}` means the server is up.

### API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/tts` | Text-to-speech; the request body is msgpack |
| `GET` | `/v1/tts/defaults` | Sampling defaults of the loaded model |
| `GET` | `/v1/health` | Health check |
| `POST` | `/v1/vqgan/encode` | Reference audio → VQ tokens |
| `POST` | `/v1/vqgan/decode` | VQ tokens → audio |
| `POST` | `/v1/references/add` | Save a reference voice; form fields `id`, `audio` (file), `text` |
| `GET` | `/v1/references/list` | List saved voices |
| `POST` | `/v1/references/update` | Rename; form fields `old_reference_id`, `new_reference_id` |
| `DELETE` | `/v1/references/delete` | Delete; form field `reference_id` |
| `GET` | `/ui` | WebUI (built into the Docker image; build it first when running from source) |

Once a voice is saved, use it with `--reference_id` (the audio field needs its MIME type, otherwise the server cannot read it):

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
    -F id=my-speaker \
    -F "audio=@reference.wav;type=audio/wav" \
    -F text="Transcript of the reference audio"

python tools/api_client.py --url http://127.0.0.1:8080/v1/tts \
    --text "Hello" --reference_id my-speaker --output demo
```

### Send a request

Which model is used depends on the weights the server loaded at startup; there is no per-request model field.

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "Hello from Fish Speech" \
    --output demo
```

Common arguments:

| Argument | Description |
| --- | --- |
| `--reference_audio` / `--reference_text` | Reference audio and its transcript; both need the same number of values |
| `--reference_speaker` | Speaker ID (0–15) of each reference audio, written as a <code>&lt;&#124;speaker:N&#124;&gt;</code> prefix (the main Drama 3 usage) |
| `--reference_id` | Use a saved reference voice, e.g. `my-speaker` |
| `--format` | Output format: `wav`, `pcm`, `mp3` or `opus`; defaults to `wav` |
| `--streaming` | Stream the audio back |
| `--single_chunk` | Generate the whole text in one pass, without chunking (Drama 3 only) |
| `--chunk_length` | Chunk length, 200 by default (S2-Pro only) |
| `--max_new_tokens` | Maximum frames per chunk; unset uses the server default of 1024, 0 uses the model's own cap |
| `--temperature` / `--top_p` | Sampling; unset uses the model defaults (0.7 / 0.7) |
| `--seed` | Fix the random seed (it does not fix the timbre) |
| `--api_key` | Required when the server runs with `--api-key` |
| `--no-play` | Do not play the audio afterwards (it plays by default; needed on machines without an audio device) |

!!! note
    `repetition_penalty` is accepted for compatibility only; S2-Pro and Drama 3 both ignore it and use built-in repetition-aware sampling.

!!! tip
    To start the server with Docker, see [Docker Setup](install.md#docker-setup) (the server image already includes the WebUI: open `http://localhost:8080/ui` once it is running).

## Model differences

| Item | S2-Pro | Drama 3 |
| --- | --- | --- |
| Hardware | GPU / CPU | NVIDIA GPU only, no `--compile` |
| Long text | Chunked by `chunk_length` | Split automatically into chunks of about 300 UTF-8 bytes with at most three speakers, cut at sentence ends where possible; the first chunk and the latest one are fed back as context |
| Length limit per pass | 1024 frames per batch by default; `max_new_tokens=0` runs up to the context length | 1024 frames (about 47 seconds) per chunk by default, which with `single_chunk` covers the whole text; `max_new_tokens=0` means 600 frames per chunk, or 2048 (about 95 seconds) with `single_chunk`; always capped at 10,240 frames minus the prompt length |
| Speakers | The order of the references decides the role | <code>&lt;&#124;speaker:0&#124;&gt;</code>–<code>&lt;&#124;speaker:15&#124;&gt;</code> tags decide the role; reference order does not matter |
| `chunk_length` | Takes effect | No effect |
| Sampling defaults | temperature 0.7, top-p 0.7, top-k 30 | Same |
| `repetition_penalty` | Ignored | Ignored |

Drama 3 also has two environment variables: `FISH_DRAMA3_FP8_EXPERTS=0` or `1` forces dequantized or FP8 experts, and `FISH_DRAMA3_CUDA_GRAPH=0` turns off CUDA-graph decoding.

## WebUI Inference

### Awesome WebUI

Awesome WebUI is a modernized Web interface built with TypeScript, offering richer features and a better user experience.

The server Docker images already include the built WebUI: open `http://localhost:8080/ui` once the container is running. The steps below are for running from source.

**Build WebUI:**

You need to have Node.js and npm installed on your local machine or server.

1. Enter the `awesome_webui` directory:
   ```bash
   cd awesome_webui
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build the WebUI:
   ```bash
   npm run build
   ```

**Start Backend Server:**

After building the WebUI, return to the project root and start the API server (`--compile` is S2-Pro only; drop it when loading Drama 3):

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**Access:**

Once the server is running, you can access it via your browser:
`http://localhost:8888/ui`
