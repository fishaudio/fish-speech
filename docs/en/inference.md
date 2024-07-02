# Inference

Inference support command line, HTTP API and web UI.

!!! note
    Overall, reasoning consists of several parts:

    1. Encode a given ~10 seconds of voice using VQGAN.
    2. Input the encoded semantic tokens and the corresponding text into the language model as an example.
    3. Given a new piece of text, let the model generate the corresponding semantic tokens.
    4. Input the generated semantic tokens into VITS / VQGAN to decode and generate the corresponding voice.

In version 1.1, we recommend using VITS for decoding, as it performs better than VQGAN in both timbre and pronunciation.

## Command Line Inference

Download the required `vqgan` and `text2semantic` models from our Hugging Face repository.
    
```bash
huggingface-cli download fishaudio/fish-speech-1 vq-gan-group-fsq-2x1024.pth --local-dir checkpoints
huggingface-cli download fishaudio/fish-speech-1 text2semantic-sft-medium-v1.1-4k.pth --local-dir checkpoints
huggingface-cli download fishaudio/fish-speech-1 vits_decoder_v1.1.ckpt --local-dir checkpoints
huggingface-cli download fishaudio/fish-speech-1 firefly-gan-base-generator.ckpt --local-dir checkpoints
```

### 1. Generate prompt from voice:

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

```bash
python tools/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
```
You should get a `fake.npy` file.

### 2. Generate semantic tokens from text:
```bash
python tools/llama/generate.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    --config-name dual_ar_2_codebook_medium \
    --checkpoint-path "checkpoints/text2semantic-sft-medium-v1.1-4k.pth" \
    --num-samples 2 \
    --compile
```

This command will create a `codes_N` file in the working directory, where N is an integer starting from 0.

!!! note
    You may want to use `--compile` to fuse CUDA kernels for faster inference (~30 tokens/second -> ~500 tokens/second).
    Correspondingly, if you do not plan to use acceleration, you can comment out the `--compile` parameter.

!!! info
    For GPUs that do not support bf16, you may need to use the `--half` parameter.

!!! warning
    If you are using your own fine-tuned model, please be sure to carry the `--speaker` parameter to ensure the stability of pronunciation.

### 3. Generate vocals from semantic tokens:

#### VITS Decoder
```bash
python tools/vits_decoder/inference.py \
    --checkpoint-path checkpoints/vits_decoder_v1.1.ckpt \
    -i codes_0.npy -r ref.wav \
    --text "The text you want to generate"
```

#### VQGAN Decoder (not recommended)
```bash
python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
```

## HTTP API Inference

We provide a HTTP API for inference. You can use the following command to start the server:

```bash
python -m tools.api \
    --listen 0.0.0.0:8000 \
    --llama-checkpoint-path "checkpoints/text2semantic-sft-medium-v1.1-4k.pth" \
    --llama-config-name dual_ar_2_codebook_medium \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

After that, you can view and test the API at http://127.0.0.1:8000/.  

!!! info
    You should use following parameters to start VITS decoder:

    ```bash
    --decoder-config-name vits_decoder_finetune \
    --decoder-checkpoint-path "checkpoints/vits_decoder_v1.1.ckpt" # or your own model
    ```

## WebUI Inference

You can start the WebUI using the following command:

```bash
python -m tools.webui \
    --llama-checkpoint-path "checkpoints/text2semantic-sft-medium-v1.1-4k.pth" \
    --llama-config-name dual_ar_2_codebook_medium \
    --vqgan-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --vits-checkpoint-path "checkpoints/vits_decoder_v1.1.ckpt"
```

!!! info
    You should use following parameters to start VITS decoder:

    ```bash
    --decoder-config-name vits_decoder_finetune \
    --decoder-checkpoint-path "checkpoints/vits_decoder_v1.1.ckpt" # or your own model
    ```

!!! note
    You can use Gradio environment variables, such as `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` to configure WebUI.

Enjoy!
