## 요구 사양

- GPU 메모리: 12GB (추론 시)
- 시스템: Linux, WSL

## 시스템 설정

OpenAudio는 다양한 설치 방법을 지원합니다. 자신의 개발 환경에 가장 적합한 방법을 선택하세요.

**사전 요구사항**: 오디오 처리를 위한 시스템 의존성을 설치합니다:
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU 버전 설치 (CUDA 버전 선택: cu126, cu128, cu129)
pip install -e .[cu129]

# CPU 버전만 설치
pip install -e .[cpu]

# 기본 설치 (PyTorch 기본 인덱스 사용)
pip install -e .
```

### UV

UV는 더 빠른 의존성 해결 및 설치를 제공합니다:

```bash
# GPU 버전 설치 (CUDA 버전 선택: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# CPU 버전만 설치
uv sync --python 3.12 --extra cpu
```
### Intel Arc XPU 지원

Intel Arc GPU 사용자는 다음을 통해 XPU 지원을 설치하세요:

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
    `compile` 옵션은 Windows와 macOS에서 지원되지 않습니다. 컴파일을 활성화하여 실행하려면 Triton을 직접 설치해야 합니다.


## Docker 설정

OpenAudio S1 시리즈 모델은 다양한 요구에 부응하기 위해 여러 Docker 배포 옵션을 제공합니다. Docker Hub의 사전 빌드된 이미지를 사용하거나, Docker Compose로 로컬에서 빌드하거나, 수동으로 사용자 정의 이미지를 빌드할 수 있습니다.

WebUI와 API 서버 모두에 대해 GPU(기본값 CUDA 12.6) 및 CPU 버전의 Docker 이미지를 제공합니다. Docker Hub의 사전 빌드된 이미지를 사용하거나, Docker Compose로 로컬에서 빌드하거나, 수동으로 사용자 정의 이미지를 빌드할 수 있습니다. 로컬에서 빌드하려면 아래 지침을 따르세요. 사전 빌드된 이미지를 사용하려면 [추론 가이드](inference.md)를 직접 참조하세요.

### 사전 요구사항

- Docker 및 Docker Compose 설치
- NVIDIA Docker 런타임 설치 (GPU 지원용)
- CUDA 추론을 위한 최소 12GB의 GPU 메모리

### Docker Compose 사용

개발 또는 사용자 정의를 위해 Docker Compose를 사용하여 로컬에서 빌드하고 실행할 수 있습니다:

```bash
# 먼저 리포지토리를 클론합니다
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# CUDA로 WebUI 시작
docker compose --profile webui up

# 컴파일 최적화로 WebUI 시작
COMPILE=1 docker compose --profile webui up

# API 서버 시작
docker compose --profile server up

# 컴파일 최적화로 API 서버 시작
COMPILE=1 docker compose --profile server up

# CPU 전용 배포
BACKEND=cpu docker compose --profile webui up
```

#### Docker Compose 환경 변수

환경 변수를 사용하여 배포를 사용자 정의할 수 있습니다:

```bash
# .env 파일 예시
BACKEND=cuda              # 또는 cpu
COMPILE=1                 # 컴파일 최적화 활성화
GRADIO_PORT=7860         # WebUI 포트
API_PORT=8080            # API 서버 포트
UV_VERSION=0.8.15        # UV 패키지 관리자 버전
```

이 명령은 이미지를 빌드하고 컨테이너를 실행합니다. WebUI는 `http://localhost:7860`에서, API 서버는 `http://localhost:8080`에서 접근할 수 있습니다.

### 수동 Docker 빌드

빌드 프로세스를 사용자 정의하려는 고급 사용자를 위해:

```bash
# CUDA를 지원하는 WebUI 이미지 빌드
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

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
    --target webui \
    -t fish-speech-webui:cpu .

# 개발용 이미지 빌드
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### 빌드 인자

- `BACKEND`: `cuda` 또는 `cpu` (기본값: `cuda`)
- `CUDA_VER`: CUDA 버전 (기본값: `12.6.0`)
- `UV_EXTRA`: CUDA용 UV 추가 패키지 (기본값: `cu126`)
- `UBUNTU_VER`: Ubuntu 버전 (기본값: `24.04`)
- `PY_VER`: Python 버전 (기본값: `3.12`)

### 볼륨 마운트

두 방법 모두 다음 디렉토리를 마운트해야 합니다:

- `./checkpoints:/app/checkpoints` - 모델 가중치 디렉토리
- `./references:/app/references` - 참조 오디오 파일 디렉토리

### 환경 변수

- `COMPILE=1` - `torch.compile`을 활성화하여 추론 속도 향상 (약 10배)
- `GRADIO_SERVER_NAME=0.0.0.0` - WebUI 서버 호스트
- `GRADIO_SERVER_PORT=7860` - WebUI 서버 포트
- `API_SERVER_NAME=0.0.0.0` - API 서버 호스트
- `API_SERVER_PORT=8080` - API 서버 포트

!!! note
    Docker 컨테이너는 모델 가중치가 `/app/checkpoints`에 마운트될 것으로 예상합니다. 컨테이너를 시작하기 전에 필요한 모델 가중치를 다운로드했는지 확인하세요.

!!! warning
    GPU 지원에는 NVIDIA Docker 런타임이 필요합니다. CPU 전용 배포의 경우 `--gpus all` 플래그를 제거하고 CPU 이미지를 사용하세요.
