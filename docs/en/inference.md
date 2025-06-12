# Inference

As the vocoder model has been changed, you need more VRAM than before, 12GB is recommended for fluently inference.

We support command line, HTTP API and WebUI for inference, you can choose any method you like.

## Download Weights

First you need to download the model weights:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## Command Line Inference

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

### 1. Get VQ tokens from reference audio

```bash
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

You should get a `fake.npy` and a `fake.wav`.

### 2. Generate semantic tokens from text:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    --compile
```

This command will create a `codes_N` file in the working directory, where N is an integer starting from 0.

!!! note
    You may want to use `--compile` to fuse CUDA kernels for faster inference (~15 tokens/second -> ~150 tokens/second, on RTX 4090 GPU).
    Correspondingly, if you do not plan to use acceleration, you can comment out the `--compile` parameter.

!!! info
    For GPUs that do not support bf16, you may need to use the `--half` parameter.

### 3. Generate vocals from semantic tokens:

!!! warning "Future Warning"
    We have kept the interface accessible from the original path (tools/vqgan/inference.py), but this interface may be removed in subsequent releases, so please change your code as soon as possible.

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

## HTTP API Inference

We provide a HTTP API for inference. You can use the following command to start the server:

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> If you want to speed up inference, you can add the `--compile` parameter.

After that, you can view and test the API at http://127.0.0.1:8080/.

## GUI Inference 
[Download client](https://github.com/AnyaCoder/fish-speech-gui/releases)

## WebUI Inference

You can start the WebUI using the following command:

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

Or simply

```bash
python -m tools.run_webui
```
> If you want to speed up inference, you can add the `--compile` parameter.


!!! note
    You can save the label file and reference audio file in advance to the `references` folder in the main directory (which you need to create yourself), so that you can directly call them in the WebUI.

!!! note
    You can use Gradio environment variables, such as `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` to configure WebUI.

Enjoy!
