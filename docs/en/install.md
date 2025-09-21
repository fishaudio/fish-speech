## Requirements

- GPU Memory: 12GB (Inference)
- System: Linux, WSL

## System Setup

First you need install pyaudio and sox, which is used for audio processing.

``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Select the correct cuda version for your system from [cu126, cu128, cu129]
pip install -e .[cu129]
# Or for cpu only
pip install -e .[cpu]
# You can also omit the extra if you want to use the default torch index
pip install -e .
```

### UV

```bash
# Select the correct cuda version for your system from [cu126, cu128, cu129]
uv sync --python 3.12 --extra cu129
# Or for cpu only
uv sync --python 3.12 --extra cpu
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


## Docker Setup

See [inference](./inference.md) to use docker for the webui or the API server.
