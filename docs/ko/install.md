## 요구 사양

| 모델 | GPU 메모리 | GPU |
| --- | --- | --- |
| Drama 3 | 32 GB 이상 | NVIDIA GPU 전용(CPU, Apple MPS, AMD ROCm 미지원, `--compile` 미지원) |
| S2-Pro | 24 GB 이상 | NVIDIA GPU 또는 CPU |

- 시스템: Linux, WSL

!!! note
    Drama 3의 H20 단일 카드 실측 피크 메모리는 약 41 GB이므로 여유를 두세요. V100 등 SM80 이전 GPU에는 bfloat16이 없어 Drama 3는 float16으로 전환됩니다(잔차 스트림은 fp32 유지).

## 시스템 설정

Fish Audio Drama 3와 S2-Pro는 동일한 설치 방법을 공유합니다. 개발 환경에 가장 잘 맞는 방법을 선택하세요.

**사전 요구 사항**: 오디오 처리에 필요한 시스템 의존성을 먼저 설치합니다:
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU 버전 설치 (CUDA 버전 선택: cu126, cu128, cu129)
# cu126 또는 cu128을 사용하는 경우 아래의 "cu129" 두 곳을 모두 바꾸세요.
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[cu129]"

# CPU 버전만 설치
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu]"

# 기본 설치 (PyTorch 기본 인덱스 사용)
pip install -e .

# pyaudio 설치 중 오류가 발생하면 다음 명령을 사용해 보세요:
# conda install pyaudio
# 그런 다음 pip install -e . 를 다시 실행하세요
```

### UV

UV는 의존성 해결과 설치가 더 빠릅니다:

```bash
# GPU 버전 설치 (CUDA 버전 선택: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# CPU 버전만 설치
uv sync --python 3.12 --extra cpu
```

### Intel Arc XPU 지원

Intel Arc GPU를 사용한다면 XPU 지원과 함께 설치하세요(S2-Pro 전용, Drama 3는 NVIDIA GPU가 필요합니다):

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# 필요한 C++ 표준 라이브러리 설치
conda install libstdcxx -c conda-forge

# Intel XPU를 지원하는 PyTorch 설치
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Fish Speech 설치
pip install -e .
```

!!! warning
    `compile` 옵션은 Windows와 macOS에서 지원되지 않습니다. compile을 사용하려면 Triton을 수동으로 설치해야 합니다.

## 가중치 다운로드

두 모델은 동일한 명령줄과 API 서버를 공유하며, 모델 종류는 가중치 디렉터리의 `config.json`에서 자동으로 판별됩니다. 같은 `checkpoints` 디렉터리에 두세요.

다운로드에는 Hugging Face CLI가 필요합니다:

```bash
pip install -U "huggingface_hub[cli]"
```

### Drama 3

```bash
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

가중치 디렉터리에 `codec.pth`가 없으면 `checkpoints/s2-pro/codec.pth`의 S2-Pro codec을 사용하므로 S2-Pro도 함께 다운로드하세요.

### S2-Pro

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

가중치를 내려받았다면 [추론](inference.md)을 참고하세요.

## Docker 설정

Drama 3와 S2-Pro는 여러 Docker 배포 방식을 제공합니다. Docker Hub의 사전 빌드 이미지를 쓰거나, Docker Compose로 로컬 빌드하거나, 직접 커스텀 이미지를 만들 수 있습니다.

GPU(기본 CUDA126) 및 CPU용 API 서버 이미지를 제공합니다. 이미지에는 WebUI가 포함되어 있어 컨테너를 띄운 뒤 `http://localhost:8080/ui`를 열면 바로 사용할 수 있습니다. GPU 이미지는 두 모델을 모두 실행하고, CPU 이미지는 S2-Pro만 지원합니다.

### 사전 요구사항

- Docker 및 Docker Compose 설치됨
- (GPU 사용 시) NVIDIA Docker runtime 설치됨(Drama 3는 필수)
- GPU 메모리: Drama 3는 32 GB 이상, S2-Pro는 24 GB 이상 권장

### 사전 빌드 이미지 실행

Drama 3: SM89 이상에서는 아래처럼 카드를 한 장만 노출하세요. 구형 카드는 `--gpus all`을 사용합니다:

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

S2-Pro:

```bash
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s2-pro \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### Docker Compose 사용

개발이나 커스터마이징이 필요하다면 Docker Compose로 로컬 빌드하여 실행하는 것을 권장합니다:

```bash
# 먼저 리포지토리를 클론합니다
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# API 서버 시작(기본값은 checkpoints/s2-pro)
docker compose --profile server up

# Drama 3를 불러오고 DEVICE로 한 장의 카드에 고정(미설정 시 최대 4장)
LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 DEVICE=cuda:0 docker compose --profile server up

# 컴파일 최적화로 API 서버 시작(S2-Pro 전용)
COMPILE=1 docker compose --profile server up

# CPU 전용 배포(S2-Pro 전용)
BACKEND=cpu docker compose --profile server up
```

#### Docker Compose 환경 변수

환경 변수로 배포를 커스터마이징할 수 있습니다:

```bash
# .env 파일 예시
BACKEND=cuda              # 또는 cpu
COMPILE=1                 # 컴파일 최적화 활성화
API_PORT=8080             # API 서버 포트
UV_VERSION=0.8.15         # UV 패키지 매니저 버전
```

명령을 실행하면 이미지가 빌드되고 컨테이너가 시작됩니다. API 서버는 `http://localhost:8080`, WebUI는 `http://localhost:8080/ui`에서 접속할 수 있습니다.

### 수동 Docker 빌드

빌드를 세밀하게 제어하고 싶다면:

```bash
# CUDA를 지원하는 API 서버 이미지 빌드
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# CPU 전용 이미지 빌드 (멀티 플랫폼 지원)
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target server \
    -t fish-speech-server:cpu .

# 개발용 이미지 빌드
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### 빌드 인자

- `BACKEND`: `cuda` 또는 `cpu`(기본값: `cuda`)
- `CUDA_VER`: CUDA 버전(기본값: `12.6.0`)
- `UV_EXTRA`: CUDA용 UV 확장(기본값: `cu126`)
- `UBUNTU_VER`: Ubuntu 버전(기본값: `24.04`)
- `PY_VER`: Python 버전(기본값: `3.12`)

### 볼륨 마운트

두 방법 모두 아래 디렉터리 마운트가 필요합니다:

- `./checkpoints:/app/checkpoints` - 모델 가중치 디렉터리
- `./references:/app/references` - 참조 오디오 디렉터리

### 환경 변수

- `COMPILE=1` - `torch.compile` 활성화(약 10배 속도 향상)
- `LLAMA_CHECKPOINT_PATH` - 메인 모델 가중치 경로(예: `checkpoints/drama-3`, `checkpoints/s2-pro`)
- `DECODER_CHECKPOINT_PATH` - codec 가중치 경로. 비워 두면 서버가 가중치 디렉터리에서 찾습니다
- `DEVICE` - 추론 디바이스(예: `cuda:0`). Drama 3는 지정한 한 장만 사용하고, 미설정 시 최대 4장
- `FISH_DRAMA3_FP8_EXPERTS=0` 또는 `1` - Drama 3의 전문가를 역양자화 / FP8로 강제(기본값은 GPU에 따라 자동 선택)
- `FISH_DRAMA3_CUDA_GRAPH=0` - Drama 3의 CUDA graph 디코딩 비활성화
- `API_SERVER_NAME=0.0.0.0` - API 서버 호스트
- `API_SERVER_PORT=8080` - API 서버 포트

!!! note
    Docker 컨테이너는 `/app/checkpoints`에서 모델 가중치를 읽습니다. 컨테이너를 시작하기 전에 필요한 가중치를 내려받으세요.

!!! warning
    GPU 지원에는 NVIDIA Docker runtime이 필요합니다. CPU 전용이라면 `--gpus all`을 제거하고 CPU 이미지를 사용하세요.
