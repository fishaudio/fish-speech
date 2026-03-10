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

## WebUI Inference

Coming soon.
