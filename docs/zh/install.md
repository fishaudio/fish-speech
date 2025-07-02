## 系统要求

- GPU 内存：12GB（推理）
- 系统：Linux、WSL

## 安装

首先需要安装 pyaudio 和 sox，用于音频处理。

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

### Intel Arc XPU 支持

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

conda install libstdcxx -c conda-forge

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -e .
```

!!! warning
    `compile` 选项在 Windows 和 macOS 上不受支持，如果您想使用 compile 运行，需要自己安装 triton。
