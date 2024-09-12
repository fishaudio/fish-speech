# Inference

Inference support command line, HTTP API and web UI.

!!! note
    Overall, reasoning consists of several parts:

    1. Encode a given ~10 seconds of voice using VQGAN.
    2. Input the encoded semantic tokens and the corresponding text into the language model as an example.
    3. Given a new piece of text, let the model generate the corresponding semantic tokens.
    4. Input the generated semantic tokens into VITS / VQGAN to decode and generate the corresponding voice.

## Command Line Inference

Download the required `vqgan` and `llama` models from our Hugging Face repository.

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

### 1. Generate prompt from voice:

!!! note
    If you plan to let the model randomly choose a voice timbre, you can skip this step.

```bash
python tools/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

You should get a `fake.npy` file.

### 2. Generate semantic tokens from text:

```bash
python tools/llama/generate.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.4" \
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

#### VQGAN Decoder

```bash
python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

## HTTP API Inference

We provide a HTTP API for inference. You can use the following command to start the server:

```bash
python -m tools.api \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/fish-speech-1.4" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

If you want to speed up inference, you can add the --compile parameter.

After that, you can view and test the API at http://127.0.0.1:8080/.

Below is an example of sending a request using `tools/post_api.py`.

```bash
python -m tools.post_api \
    --text "Text to be input" \
    --reference_audio "Path to reference audio" \
    --reference_text "Text content of the reference audio" \
    --streaming True
```

The above command indicates synthesizing the desired audio according to the reference audio information and returning it in a streaming manner.

The following example demonstrates that you can use **multiple** reference audio paths and reference audio texts at once. Separate them with spaces in the command.

```bash
python -m tools.post_api \
    --text "Text to input" \
    --reference_audio "reference audio path1" "reference audio path2" \
    --reference_text "reference audio text1" "reference audio text2"\
    --streaming False \
    --output "generated" \
    --format "mp3"
```

The above command synthesizes the desired `MP3` format audio based on the information from multiple reference audios and saves it as `generated.mp3` in the current directory.

## GUI Inference 
[Download client](https://github.com/AnyaCoder/fish-speech-gui/releases/tag/v0.1.0)

## WebUI Inference

You can start the WebUI using the following command:

```bash
python -m tools.webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.4" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

!!! note
    You can use Gradio environment variables, such as `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` to configure WebUI.

Enjoy!
