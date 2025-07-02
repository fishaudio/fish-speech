## 시스템 요구사항

- GPU 메모리: 12GB (추론)
- 시스템: Linux, WSL

## 설정

먼저 오디오 처리에 사용되는 pyaudio와 sox를 설치해야 합니다.

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

### Intel Arc XPU 지원

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

conda install libstdcxx -c conda-forge

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -e .
```

!!! warning
    `compile` 옵션은 Windows와 macOS에서 지원되지 않습니다. compile로 실행하려면 triton을 직접 설치해야 합니다.
