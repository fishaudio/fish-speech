## 系统要求

- GPU 显存：24GB（用于推理）
- 系统：Linux、WSL

## 系统设置

Fish Audio S2 支持多种安装方式。请选择最适合你当前开发环境的方案。

**前置依赖**：先安装音频处理所需的系统依赖：
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU 安装（选择 CUDA 版本：cu126、cu128、cu129）
pip install -e .[cu129]

# 仅 CPU 安装
pip install -e .[cpu]

# 默认安装（使用 PyTorch 默认索引）
pip install -e .

# 如果因 pyaudio 导致安装报错，可以先执行：
# conda install pyaudio
# 然后重新执行 pip install -e .
```

### UV

UV 可以更快地完成依赖解析与安装：

```bash
# GPU 安装（选择 CUDA 版本：cu126、cu128、cu129）
uv sync --python 3.12 --extra cu129

# 仅 CPU 安装
uv sync --python 3.12 --extra cpu
```

### Intel Arc XPU 支持

如果你使用 Intel Arc GPU，可按以下方式安装 XPU 支持：

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# 安装必需的 C++ 标准库
conda install libstdcxx -c conda-forge

# 安装支持 Intel XPU 的 PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# 安装 Fish Speech
pip install -e .
```

!!! warning
    `compile` 选项暂不支持 Windows 和 macOS。若你希望启用 compile，请手动安装 Triton。

## Docker 设置

Fish Audio S2 系列模型提供多种 Docker 部署方式，适配不同场景。你可以直接使用 Docker Hub 预构建镜像，也可以用 Docker Compose 本地构建，或手动构建自定义镜像。

我们提供 WebUI 与 API Server 的 GPU（默认 CUDA126）和 CPU 镜像。你可以直接用 Docker Hub 镜像，也可以在本地构建。如果你只想使用预构建镜像，请参考[inference guide](inference.md)。

### 前置条件

- 已安装 Docker 和 Docker Compose
- （GPU 场景）已安装 NVIDIA Docker runtime
- CUDA 推理建议至少 24GB 显存

# 使用 Docker Compose

如果你需要开发或自定义，推荐使用 Docker Compose 在本地构建并运行：

```bash
# 先克隆仓库
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# 使用 CUDA 启动 WebUI
docker compose --profile webui up

# 启用 compile 优化启动 WebUI
COMPILE=1 docker compose --profile webui up

# 启动 API Server
docker compose --profile server up

# 启用 compile 优化启动 API Server
COMPILE=1 docker compose --profile server up

# 仅 CPU 部署
BACKEND=cpu docker compose --profile webui up
```

#### Docker Compose 环境变量

你可以通过环境变量定制部署参数：

```bash
# .env 文件示例
BACKEND=cuda              # 或 cpu
COMPILE=1                 # 启用 compile 优化
GRADIO_PORT=7860          # WebUI 端口
API_PORT=8080             # API Server 端口
UV_VERSION=0.8.15         # UV 包管理器版本
```

命令执行后会自动构建镜像并启动容器。你可以通过 `http://localhost:7860` 访问 WebUI，通过 `http://localhost:8080` 访问 API Server。

### 手动 Docker 构建

如果你需要更细粒度的构建控制，可以手动构建：

```bash
# 构建支持 CUDA 的 WebUI 镜像
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

# 构建支持 CUDA 的 API Server 镜像
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# 构建仅 CPU 镜像（支持多平台）
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target webui \
    -t fish-speech-webui:cpu .

# 构建开发镜像
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### 构建参数

- `BACKEND`：`cuda` 或 `cpu`（默认：`cuda`）
- `CUDA_VER`：CUDA 版本（默认：`12.6.0`）
- `UV_EXTRA`：UV 的 CUDA 扩展（默认：`cu126`）
- `UBUNTU_VER`：Ubuntu 版本（默认：`24.04`）
- `PY_VER`：Python 版本（默认：`3.12`）

### 卷挂载

两种方法都需要挂载以下目录：

- `./checkpoints:/app/checkpoints` - 模型权重目录
- `./references:/app/references` - 参考音频目录

### 环境变量

- `COMPILE=1` - 启用 `torch.compile`，可提升推理速度（约 10 倍）
- `GRADIO_SERVER_NAME=0.0.0.0` - WebUI 服务地址
- `GRADIO_SERVER_PORT=7860` - WebUI 服务端口
- `API_SERVER_NAME=0.0.0.0` - API 服务地址
- `API_SERVER_PORT=8080` - API 服务端口

!!! note
    Docker 容器默认从 `/app/checkpoints` 读取模型权重。启动容器前请先下载好所需权重。

!!! warning
    GPU 支持需要 NVIDIA Docker runtime。若仅使用 CPU，请移除 `--gpus all` 并使用 CPU 镜像。
