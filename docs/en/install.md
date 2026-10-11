## Requirements

| Model | GPU memory | GPU |
| --- | --- | --- |
| Drama 3 | at least 32 GB | NVIDIA GPU only: no CPU, Apple MPS or AMD ROCm, and no `--compile` |
| S2-Pro | at least 24 GB | NVIDIA GPU or CPU |

- System: Linux, WSL

!!! note
    Peak Drama 3 memory measured on a single H20 is about 41 GB, so leave some headroom. V100 and other GPUs older than SM80 have no bfloat16, so Drama 3 switches to float16 there (keeping the residual stream in fp32).

## System Setup

Fish Audio Drama 3 and S2-Pro share the same installation. Choose the one that best fits your development environment.

**Prerequisites**: Install system dependencies for audio processing:
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU installation (choose your CUDA version: cu126, cu128, cu129)
# Replace both "cu129" values when using cu126 or cu128.
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[cu129]"

# CPU-only installation
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu]"

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

For Intel Arc GPU users, install with XPU support (S2-Pro only; Drama 3 needs an NVIDIA GPU):

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

## Download Weights

Both models share the same command line and API server; the model type is read from the checkpoint's `config.json`. Keep them in the same `checkpoints` directory.

Downloading needs the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

### Drama 3

```bash
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

If the checkpoint directory has no `codec.pth`, the S2-Pro codec at `checkpoints/s2-pro/codec.pth` is used, so download S2-Pro as well.

### S2-Pro

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

Once the weights are downloaded, see [Inference](inference.md).

## Docker Setup

Drama 3 and S2-Pro provide multiple Docker deployment options to suit different needs. You can use pre-built images from Docker Hub, build locally with Docker Compose, or manually build custom images.

We provide Docker images for the API server on both GPU (CUDA126 by default) and CPU. The images already include the WebUI: open `http://localhost:8080/ui` once the container is running. The GPU image runs both models; the CPU image only supports S2-Pro.

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support; required by Drama 3)
- GPU memory: at least 32 GB for Drama 3, 24 GB recommended for S2-Pro

### Run the pre-built image

Drama 3: on SM89 or newer cards expose a single GPU as below; on older cards use `--gpus all`:

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

### Use docker compose

For development or customization, you can use Docker Compose to build and run locally:

```bash
# Clone the repository first
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Start API server (loads checkpoints/s2-pro by default)
docker compose --profile server up

# Load Drama 3 and pin it to one GPU with DEVICE (up to four are used when unset)
LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 DEVICE=cuda:0 docker compose --profile server up

# Start API server with compile optimization (S2-Pro only)
COMPILE=1 docker compose --profile server up

# For CPU-only deployment (S2-Pro only)
BACKEND=cpu docker compose --profile server up
```

#### Environment Variables for Docker Compose

You can customize the deployment using environment variables:

```bash
# .env file example
BACKEND=cuda              # or cpu
COMPILE=1                 # Enable compile optimization
API_PORT=8080            # API server port
UV_VERSION=0.8.15        # UV package manager version
CUDA_VER=12.9.0          # CUDA base image version (e.g. 12.6.0 for older drivers)
UV_EXTRA=cu129           # PyTorch CUDA variant (cu126, cu128, cu129) — must match CUDA_VER
```

The command will build the image and run the container. You can access the API server at `http://localhost:8080` and the WebUI at `http://localhost:8080/ui`.

### Manual Docker Build

For advanced users who want to customize the build process:

```bash
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
    --target server \
    -t fish-speech-server:cpu .

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
- `LLAMA_CHECKPOINT_PATH` - Main model weights path, e.g. `checkpoints/drama-3` or `checkpoints/s2-pro`
- `DECODER_CHECKPOINT_PATH` - Codec weights path; leave empty and the server resolves it from the checkpoint directory
- `DEVICE` - Inference device, e.g. `cuda:0`; Drama 3 uses just that card when set, up to four when unset
- `FISH_DRAMA3_FP8_EXPERTS=0` or `1` - Force dequantized or FP8 experts for Drama 3 (chosen from the GPU by default)
- `FISH_DRAMA3_CUDA_GRAPH=0` - Disable Drama 3 CUDA-graph decoding
- `API_SERVER_NAME=0.0.0.0` - API server host  
- `API_SERVER_PORT=8080` - API server port

!!! note
    The Docker containers expect model weights to be mounted at `/app/checkpoints`. Make sure to download the required model weights before starting the containers.

!!! warning
    GPU support requires NVIDIA Docker runtime. For CPU-only deployment, remove the `--gpus all` flag and use CPU images.

### AMD ROCm support

Fish Speech runs on AMD GPUs via ROCm (S2-Pro only; Drama 3 checkpoints need CUDA). The ROCm image is based on the official `rocm/pytorch` image, which already ships a gfx-tuned PyTorch, so no separate torch install is needed. Verified on RDNA4 (Radeon AI PRO R9700 / gfx1201) with ROCm 7.2.3; RDNA3 (gfx1100/gfx1101) should also work.

**Prerequisites:**

- AMD GPU with ROCm support (RDNA3 / RDNA4)
- ROCm drivers installed on the host
- Docker with GPU passthrough (`/dev/kfd` and `/dev/dri`)

**Using Docker Compose:**

```bash
# API server
docker compose -f compose.rocm.yml --profile server up --build
```

**Manual build and run:**

```bash
docker build -f docker/Dockerfile.rocm --target server -t fish-speech-server:rocm .

docker run \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    -e ROCBLAS_USE_HIPBLASLT=0 \
    -v ./checkpoints:/app/checkpoints \
    -p 8080:8080 \
    fish-speech-server:rocm
```

!!! note
    `ROCBLAS_USE_HIPBLASLT=0` is set by default for RDNA4 (gfx1201) stability; RDNA3 users may not need it. Fish Speech uses `scaled_dot_product_attention`, which dispatches to ROCm's AOTriton flash-attention backend automatically — no custom kernel build is required. The first run is slower while MIOpen auto-tunes kernels. `torch.compile` is enabled by default (`COMPILE=1`); set `COMPILE=0` to disable.
