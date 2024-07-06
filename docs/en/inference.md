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
huggingface-cli download fishaudio/fish-speech-1.2 --local-dir checkpoints/fish-speech-1.2
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
    --checkpoint-path "checkpoints/fish-speech-1.2" \
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
    --llama-checkpoint-path "checkpoints/fish-speech-1.2" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

If you want to speed up inference, you can add the --compile parameter.

After that, you can view and test the API at http://127.0.0.1:8000/.

Below is an example of sending a request using `tools/post_api.py`.

```bash
python -m tools.post_api \
    --text "Text to be input" \
    --reference_audio "Path to reference audio" \
    --reference_text "Text content of the reference audio"
    --streaming True
```

The above command indicates synthesizing the desired audio according to the reference audio information and returning it in a streaming manner.

If you need to randomly select reference audio based on `{SPEAKER}` and `{EMOTION}`, configure it according to the following steps:

### 1. Create a `ref_data` folder in the root directory of the project.

### 2. Create a directory structure similar to the following within the `ref_data` folder.

```
.
├── SPEAKER1
│    ├──EMOTION1
│    │    ├── 21.15-26.44.lab
│    │    ├── 21.15-26.44.wav
│    │    ├── 27.51-29.98.lab
│    │    ├── 27.51-29.98.wav
│    │    ├── 30.1-32.71.lab
│    │    └── 30.1-32.71.flac
│    └──EMOTION2
│         ├── 30.1-32.71.lab
│         └── 30.1-32.71.mp3
└── SPEAKER2
    └─── EMOTION3
          ├── 30.1-32.71.lab
          └── 30.1-32.71.mp3
```

That is, first place `{SPEAKER}` folders in `ref_data`, then place `{EMOTION}` folders under each speaker, and place any number of `audio-text pairs` under each emotion folder.

### 3. Enter the following command in the virtual environment

```bash
python tools/gen_ref.py

```

### 4. Call the API.

```bash
python -m tools.post_api \
    --text "Text to be input" \
    --speaker "${SPEAKER1}" \
    --emotion "${EMOTION1}"
    --streaming True
```

The above example is for testing purposes only.

## WebUI Inference

You can start the WebUI using the following command:

```bash
python -m tools.webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.2" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

!!! note
    You can use Gradio environment variables, such as `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` to configure WebUI.

Enjoy!
