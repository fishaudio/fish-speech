## Requirements

- GPU Memory: 12GB (Inference)
- System: Linux, WSL

## Setup

First you need install pyaudio and sox, which is used for audio processing.

``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

pip install -e .
```

### UV

```bash

uv sync --python 3.12
```
### Intel Arc XPU support

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

conda install libstdcxx -c conda-forge

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -e .
```

!!! warning
    The `compile` option is not supported on windows and macOS, if you want to run with compile, you need to install trition by yourself.
