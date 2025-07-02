## Requisitos

- Memoria GPU: 12GB (para inferencia)
- Sistema: Linux, WSL

## Configuración

Primero necesitas instalar pyaudio y sox, que se utilizan para el procesamiento de audio.

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
### Soporte para Intel Arc XPU

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

conda install libstdcxx -c conda-forge

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -e .
```

!!! advertencia
    La opción `compile` no es compatible con Windows ni macOS. Si deseas ejecutar con compilación, debes instalar Triton manualmente.