## Requisitos

- Memória GPU: 12GB (Inferência)
- Sistema: Linux, WSL

## Configuração

Primeiro você precisa instalar pyaudio e sox, que são usados para processamento de áudio.

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

### Suporte para Intel Arc XPU

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

conda install libstdcxx -c conda-forge

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -e .
```

!!! warning
    A opção `compile` não é suportada no Windows e macOS, se você quiser executar com compile, precisa instalar o triton por conta própria.
