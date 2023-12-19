# Inference

In the plan, inference is expected to support both command line and webui methods, but currently, only the command-line reasoning function has been completed.  

!!! note
    Overall, reasoning consists of several parts:

    1. Encode a given 5-10 seconds of voice using VQGAN.
    2. Input the encoded semantic tokens and the corresponding text into the language model as an example.
    3. Given a new piece of text, let the model generate the corresponding semantic tokens.
    4. Input the generated semantic tokens into VQGAN to decode and generate the corresponding voice.

## Command Line Inference

Download the required `vqgan` and `text2semantic` models from our Hugging Face repository.
    
```bash
huggingface-cli download fishaudio/speech-lm-v1 vqgan-v1.pth --local-dir checkpoints
huggingface-cli download fishaudio/speech-lm-v1 text2semantic-400m-v0.2-4k.pth --local-dir checkpoints
```

### 1. Generate prompt from voice:

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

```bash
python tools/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/vqgan-v1.pth"
```
You should get a `fake.npy` file.

### 2. Generate semantic tokens from text:
```bash
python tools/llama/generate.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/text2semantic-400m-v0.2-4k.pth" \
    --num-samples 2 \
    --compile
```

This command will create a `codes_N` file in the working directory, where N is an integer starting from 0.

!!! note
    You may want to use `--compile` to fuse CUDA kernels for faster inference (~30 tokens/second -> ~500 tokens/second).
    Correspondingly, if you do not plan to use acceleration, you can comment out the `--compile` parameter.

!!! info
    For GPUs that do not support bf16, you may need to use the `--half` parameter.

### 3. Generate vocals from semantic tokens:
```bash
python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/vqgan-v1.pth"
```
