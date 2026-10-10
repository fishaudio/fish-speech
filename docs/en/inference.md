# Inference

The Fish Audio S2 model requires a large amount of VRAM. We recommend using a GPU with at least 24GB for inference.

## Download Weights

First, you need to download the model weights:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## Command Line Inference

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

### 1. Get VQ tokens from reference audio

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

You should get a `fake.npy` and a `fake.wav`.

### 2. Generate Semantic tokens from text:

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

### 3. Generate vocals from semantic tokens:

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

After that, you will get a `fake.wav` file.

## S3-drama

S3-drama uses the same command line and API server as S2-Pro; the model type is read from the checkpoint's `config.json`.

!!! warning
    S3-drama runs on NVIDIA GPUs only. CPU, Apple MPS and AMD ROCm are not supported, and neither is `--compile`.

| GPU | Mode | GPU memory |
| --- | --- | --- |
| SM89 or newer (L40, H100, H20) | MoE experts stay in FP8, CUDA-graph decoding, one GPU | about 41 GB |
| Older GPUs (V100, A100) | MoE experts dequantized and spread over several GPUs | about 70 GB in total |

!!! note
    V100 and other GPUs older than SM80 have no bfloat16, so S3 switches to float16 there (keeping the residual stream in fp32). This path has not been tested on real hardware yet; if you get silence, noise or a "non-finite logits" error, please open an issue.

### Download weights

```bash
hf download fishaudio/s3-drama-it-fp8 --local-dir checkpoints/s3-drama-fp8
```

If the checkpoint directory has no `codec.pth`, the S2-Pro codec at `checkpoints/s2-pro/codec.pth` is used.

### Command line inference

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/s3-drama-fp8 \
    --text "The text you want to convert" \
    --prompt-audio "reference.wav" \
    --prompt-text "Your reference text" \
    --output "output.wav" \
    --device cuda:0
```

`--output` writes the decoded audio directly. On older GPUs, list several devices, for example `--device cuda:0,cuda:1,cuda:2,cuda:3`; the last one holds the audio decoder and the codec. Plain `--device cuda` uses up to the first four visible GPUs.

### API server

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s3-drama-fp8 \
    --device cuda:0
```

With Docker, run the server image against the S3 checkpoint. On SM89 or newer cards, expose a single GPU as below; on older cards, use `--gpus all`:

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s3-drama-fp8/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### Differences from S2-Pro

- Long text is split into chunks of about 300 UTF-8 bytes with at most three speakers each, cut at sentence ends where possible, and the chunks are generated one after another. The first chunk and the latest one are fed back as context to keep the voices consistent. By default each chunk produces at most 600 frames (about 28 seconds).
- Set `single_chunk` (`--single-chunk` on the command line) to generate the whole text in one pass instead, by default at most 2048 frames (about 95 seconds). In either mode, a larger `max_new_tokens` is capped at 10,240 frames minus the prompt length.
- Speaker tags `<|speaker:0|>` to `<|speaker:15|>` are accepted, and untagged text is speaker 0. To give speakers their own voices, set each reference's `speaker` field (`--reference_speaker` in `tools/api_client.py`), which tags a transcript that does not start with a speaker tag, or start the transcript with that speaker's tag yourself (for example `<|speaker:1|>`); the order of the references does not decide which speaker they belong to.
- `chunk_length` and `repetition_penalty` have no effect; S3 has built-in repetition-aware sampling. Its default sampling is temperature 1.0, top-p 0.9, top-k 30.
- With `streaming` enabled, audio is returned chunk by chunk.
- `FISH_S3_FP8_EXPERTS=0` or `1` forces dequantized or FP8 experts, and `FISH_S3_CUDA_GRAPH=0` turns off CUDA-graph decoding.

## WebUI Inference

### Awesome WebUI

Awesome WebUI is a modernized Web interface built with TypeScript, offering richer features and a better user experience.

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

After building the WebUI, return to the project root and start the API server:

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**Access:**

Once the server is running, you can access it via your browser:
`http://localhost:8888/ui`
