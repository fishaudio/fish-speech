## Requirements

- GPU Memory: 12GB (Inference)
- System: Linux, WSL

## System Setup

OpenAudio supports multiple installation methods. Choose the one that best fits your development environment.

**Prerequisites**: Install system dependencies for audio processing:
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU installation (choose your CUDA version: cu126, cu128, cu129)
pip install -e .[cu129]

# CPU-only installation
pip install -e .[cpu]

# Default installation (uses PyTorch default index)
pip install -e .
```

### UV

UV provides faster dependency resolution and installation:

```bash
# GPU installation (choose your CUDA version: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# CPU-only installation
uv sync --python 3.12 --extra cpu
```
### Intel Arc XPU support

For Intel Arc GPU users, install with XPU support:

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Install required C++ standard library
conda install libstdcxx -c conda-forge

# Install PyTorch with Intel XPU support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Install Fish Speech
pip install -e .
```

!!! warning
    The `compile` option is not supported on windows and macOS, if you want to run with compile, you need to install trition by yourself.


## Docker Setup

OpenAudio S1 series model provides multiple Docker deployment options to suit different needs. You can use pre-built images from Docker Hub, build locally with Docker Compose, or manually build custom images.

We provided Docker images for both WebUI and API server on both GPU(CUDA126 for default) and CPU. You can use the pre-built images from Docker Hub, or build locally with Docker Compose, or manually build custom images. If you want to build locally, follow the instructions below. If you just want to use the pre-built images, follow [inference guide](en/inference.md) to use directly.

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- At least 12GB GPU memory for CUDA inference

# Use docker compose

For development or customization, you can use Docker Compose to build and run locally:

```bash
# Clone the repository first
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Start WebUI with CUDA
docker compose --profile webui up

# Start WebUI with compile optimization
COMPILE=1 docker compose --profile webui up

# Start API server
docker compose --profile server up

# Start API server with compile optimization  
COMPILE=1 docker compose --profile server up

# For CPU-only deployment
BACKEND=cpu docker compose --profile webui up
```

#### Environment Variables for Docker Compose

You can customize the deployment using environment variables:

```bash
# .env file example
BACKEND=cuda              # or cpu
COMPILE=1                 # Enable compile optimization
GRADIO_PORT=7860         # WebUI port
API_PORT=8080            # API server port
UV_VERSION=0.8.15        # UV package manager version
```

The comand will build the image and run the container. You can access the WebUI at `http://localhost:7860` and the API server at `http://localhost:8080`.

### Manual Docker Build

For advanced users who want to customize the build process:

```bash
# Build WebUI image with CUDA support
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

# Build API server image with CUDA support
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# Build CPU-only images (supports multi-platform)
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target webui \
    -t fish-speech-webui:cpu .

# Build development image
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### Build Arguments

- `BACKEND`: `cuda` or `cpu` (default: `cuda`)
- `CUDA_VER`: CUDA version (default: `12.6.0`)
- `UV_EXTRA`: UV extra for CUDA (default: `cu126`)
- `UBUNTU_VER`: Ubuntu version (default: `24.04`)
- `PY_VER`: Python version (default: `3.12`)

### Volume Mounts

Both methods require mounting these directories:

- `./checkpoints:/app/checkpoints` - Model weights directory
- `./references:/app/references` - Reference audio files directory

### Environment Variables

- `COMPILE=1` - Enable torch.compile for faster inference (~10x speedup)
- `GRADIO_SERVER_NAME=0.0.0.0` - WebUI server host
- `GRADIO_SERVER_PORT=7860` - WebUI server port
- `API_SERVER_NAME=0.0.0.0` - API server host  
- `API_SERVER_PORT=8080` - API server port

!!! note
    The Docker containers expect model weights to be mounted at `/app/checkpoints`. Make sure to download the required model weights before starting the containers.

!!! warning
    GPU support requires NVIDIA Docker runtime. For CPU-only deployment, remove the `--gpus all` flag and use CPU images.
