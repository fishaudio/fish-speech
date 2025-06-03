# Introduction

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>

!!! warning
    We assume no responsibility for any illegal use of the codebase. Please refer to the local laws regarding DMCA (Digital Millennium Copyright Act) and other relevant laws in your area. <br/>
    This codebase and all models are released under the CC-BY-NC-SA-4.0 license.

<p align="center">
   <img src="../assets/figs/diagram.png" width="75%">
</p>

## Requirements

- GPU Memory: 4GB (for inference), 8GB (for fine-tuning)
- System: Linux, Windows

## Windows Setup

!!! info "Attention"

    We strongly suggest non-professional windows users use our official GUI to run the project. [GUI is here](https://github.com/AnyaCoder/fish-speech-gui).

Professional Windows users may consider using WSL2 or Docker to run the codebase.

```bash
# Create a python 3.10 virtual environment, you can also use virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# Install pytorch
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install fish-speech
pip3 install -e .

# (Enable acceleration) Install triton-windows
pip install https://github.com/AnyaCoder/fish-speech/releases/download/v0.1.0/triton_windows-0.1.0-py3-none-any.whl
```

## Linux Setup

See [pyproject.toml](../../pyproject.toml) for details.
```bash
# Create a python 3.10 virtual environment, you can also use virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# Install pytorch
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# (Ubuntu / Debian User) Install sox + ffmpeg
apt install libsox-dev ffmpeg 

# (Ubuntu / Debian User) Install pyaudio 
apt install build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0
    
# Install fish-speech
pip3 install -e .[stable]
```

## MacOS setup

If you want to perform inference on MPS, please add the `--device mps` flag.
Please refer to [this PR](https://github.com/fishaudio/fish-speech/pull/461#issuecomment-2284277772) for a comparison of inference speeds.

!!! warning
    The `compile` option is not officially supported on Apple Silicon devices, so there is no guarantee that inference speed will improve.

```bash
# install dependencies
brew install portaudio
# create a python 3.10 virtual environment, you can also use virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech
# install pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
# install fish-speech
pip install -e .[stable]
```

## Docker Setup

1. Install NVIDIA Container Toolkit:

    To use GPU for model training and inference in Docker, you need to install NVIDIA Container Toolkit:

    For Ubuntu users:

    ```bash
    # Add repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    # Install nvidia-container-toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    # Restart Docker service
    sudo systemctl restart docker
    ```

    For users of other Linux distributions, please refer to: [NVIDIA Container Toolkit Install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

2. Pull and run the fish-speech image

    ```shell
    # Pull the image
    docker pull fishaudio/fish-speech:latest-dev
    # Run the image
    docker run -it \
        --name fish-speech \
        --gpus all \
        -p 7860:7860 \
        fishaudio/fish-speech:latest-dev \
        zsh
    # If you need to use a different port, please modify the -p parameter to YourPort:7860
    ```

3. Download model dependencies

    Make sure you are in the terminal inside the docker container, then download the required `vqgan` and `llama` models from our huggingface repository.

    ```bash
    huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
    ```

4. Configure environment variables and access WebUI

    In the terminal inside the docker container, enter `export GRADIO_SERVER_NAME="0.0.0.0"` to allow external access to the gradio service inside docker.
    Then in the terminal inside the docker container, enter `python tools/run_webui.py` to start the WebUI service.

    If you're using WSL or MacOS, visit [http://localhost:7860](http://localhost:7860) to open the WebUI interface.

    If it's deployed on a server, replace localhost with your server's IP.

## Changelog

- 2024/12/03: Updated Fish-Speech to 1.5 version, supports more languages, and reaches SOTA in the Open-Source field.
- 2024/09/10: Updated Fish-Speech to 1.4 version, with an increase in dataset size and a change in the quantizer's n_groups from 4 to 8.
- 2024/07/02: Updated Fish-Speech to 1.2 version, remove VITS Decoder, and greatly enhanced zero-shot ability.
- 2024/05/10: Updated Fish-Speech to 1.1 version, implement VITS decoder to reduce WER and improve timbre similarity.
- 2024/04/22: Finished Fish-Speech 1.0 version, significantly modified VQGAN and LLAMA models.
- 2023/12/28: Added `lora` fine-tuning support.
- 2023/12/27: Add `gradient checkpointing`, `causual sampling`, and `flash-attn` support.
- 2023/12/19: Updated webui and HTTP API.
- 2023/12/18: Updated fine-tuning documentation and related examples.
- 2023/12/17: Updated `text2semantic` model, supporting phoneme-free mode.
- 2023/12/13: Beta version released, includes VQGAN model and a language model based on LLAMA (phoneme support only).

## Acknowledgements

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
