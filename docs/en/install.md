## Requirements

- GPU Memory: 24GB (Inference)
- System: Linux, WSL

## System Setup

Fish Audio S2 supports multiple installation methods. Choose the one that best fits your development environment.

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

# If you encounter an error during installation due to pyaudio, consider using the following command:
# conda install pyaudio
# Then run pip install -e . again
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
    The `compile` option is not supported on Windows and macOS. If you want to run with compile, you need to install Triton manually.


## Docker Setup

Fish Audio S2 series model provides multiple Docker deployment options to suit different needs. You can use pre-built images from Docker Hub, build locally with Docker Compose, or manually build custom images.

We provide Docker images for both WebUI and API server on both GPU (CUDA126 by default) and CPU. You can use the pre-built images from Docker Hub, build locally with Docker Compose, or manually build custom images. If you want to build locally, follow the instructions below. If you only want to use pre-built images, follow the [inference guide](inference.md).

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- At least 24GB GPU memory for CUDA inference

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
CUDA_VER=12.9.0          # CUDA base image version (e.g. 12.6.0 for older drivers)
UV_EXTRA=cu129           # PyTorch CUDA variant (cu126, cu128, cu129) — must match CUDA_VER
```

The command will build the image and run the container. You can access the WebUI at `http://localhost:7860` and the API server at `http://localhost:8080`.

### Manual Docker Build

For advanced users who want to customize the build process:

```bash
# Build WebUI image with CUDA support
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.9.0 \
    --build-arg UV_EXTRA=cu129 \
    --target webui \
    -t fish-speech-webui:cuda .

# Build API server image with CUDA support
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.9.0 \
    --build-arg UV_EXTRA=cu129 \
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

### AMD ROCm support

Fish Speech runs on AMD GPUs via ROCm. The ROCm image is based on the official `rocm/pytorch` image, which already ships a gfx-tuned PyTorch, so no separate torch install is needed. Verified on RDNA4 (Radeon AI PRO R9700 / gfx1201) with ROCm 7.2.3; RDNA3 (gfx1100/gfx1101) should also work.

**Prerequisites:**

- AMD GPU with ROCm support (RDNA3 / RDNA4)
- ROCm drivers installed on the host
- Docker with GPU passthrough (`/dev/kfd` and `/dev/dri`)

**Using Docker Compose:**

```bash
# WebUI
docker compose -f compose.rocm.yml --profile webui up --build

# API server
docker compose -f compose.rocm.yml --profile server up --build
```

**Manual build and run:**

```bash
docker build -f docker/Dockerfile.rocm --target webui -t fish-speech-webui:rocm .

docker run \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    -e ROCBLAS_USE_HIPBLASLT=0 \
    -v ./checkpoints:/app/checkpoints \
    -p 7860:7860 \
    fish-speech-webui:rocm
```

!!! note
    `ROCBLAS_USE_HIPBLASLT=0` is set by default for RDNA4 (gfx1201) stability; RDNA3 users may not need it. Fish Speech uses `scaled_dot_product_attention`, which dispatches to ROCm's AOTriton flash-attention backend automatically — no custom kernel build is required. The first run is slower while MIOpen auto-tunes kernels. `torch.compile` is enabled by default (`COMPILE=1`); set `COMPILE=0` to disable.
