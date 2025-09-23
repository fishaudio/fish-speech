## 系统要求

- GPU 内存：12GB（推理）
- 系统：Linux、WSL

## 系统设置

OpenAudio 支持多种安装方式，请选择最适合您开发环境的方法。

**先决条件**：安装用于音频处理的系统依赖项：
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU 安装 (选择您的 CUDA 版本: cu126, cu128, cu129)
pip install -e .[cu129]

# 仅 CPU 安装
pip install -e .[cpu]

# 默认安装 (使用 PyTorch 官方源)
pip install -e .
```

### UV

UV 提供了更快的依赖解析和安装速度：

```bash
# GPU 安装 (选择您的 CUDA 版本: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# 仅 CPU 安装
uv sync --python 3.12 --extra cpu
```
### Intel Arc XPU 支持

对于 Intel Arc GPU 用户，请按以下方式安装以获得 XPU 支持：

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# 安装所需的 C++ 标准库
conda install libstdcxx -c conda-forge

# 安装支持 Intel XPU 的 PyTorch
pip install --pre torch torchvision toraudio --index-url https://download.pytorch.org/whl/nightly/xpu

# 安装 Fish Speech
pip install -e .
```

!!! warning
    `compile` 选项在 Windows 和 macOS 上不受支持。如果希望通过编译运行，您需要自行安装 Triton。


## Docker 设置

OpenAudio S1 系列模型提供了多种 Docker 部署选项以满足不同需求。您可以使用 Docker Hub 上的预构建镜像，通过 Docker Compose 在本地构建，或手动构建自定义镜像。

我们为 WebUI 和 API 服务器提供了 GPU (默认为 CUDA 12.6) 和 CPU 两种版本的 Docker 镜像。您可以直接使用 Docker Hub 上的预构建镜像，或通过 Docker Compose 在本地构建，也可以手动构建自定义镜像。如果希望在本地构建，请遵循以下说明。如果只想使用预构建镜像，请直接查阅 [推理指南](inference.md) 中的说明。

### 先决条件

- 已安装 Docker 和 Docker Compose
- 已安装 NVIDIA Docker 运行时 (用于 GPU 支持)
- 至少 12GB 的 GPU 显存用于 CUDA 推理

### 使用 Docker Compose

对于开发或自定义需求，您可以使用 Docker Compose 在本地构建和运行：

```bash
# 首先克隆本仓库
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# 使用 CUDA 启动 WebUI
docker compose --profile webui up

# 启动带编译优化的 WebUI
COMPILE=1 docker compose --profile webui up

# 启动 API 服务器
docker compose --profile server up

# 启动带编译优化的 API 服务器
COMPILE=1 docker compose --profile server up

# 仅 CPU 部署
BACKEND=cpu docker compose --profile webui up
```

#### Docker Compose 环境变量

您可以使用环境变量自定义部署：

```bash
# .env 文件示例
BACKEND=cuda              # 或 cpu
COMPILE=1                 # 启用编译优化
GRADIO_PORT=7860         # WebUI 端口
API_PORT=8080            # API 服务器端口
UV_VERSION=0.8.15        # UV 包管理器版本
```

该命令将构建镜像并运行容器。您可以在 `http://localhost:7860` 访问 WebUI，在 `http://localhost:8080` 访问 API 服务器。

### 手动 Docker 构建

对于需要自定义构建流程的高级用户：

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

# 构建支持 CUDA 的 API 服务器镜像
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# 构建仅 CPU 镜像 (支持多平台)
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

- `BACKEND`: `cuda` 或 `cpu` (默认: `cuda`)
- `CUDA_VER`: CUDA 版本 (默认: `12.6.0`)
- `UV_EXTRA`: 用于 CUDA 的 UV 附加包 (默认: `cu126`)
- `UBUNTU_VER`: Ubuntu 版本 (默认: `24.04`)
- `PY_VER`: Python 版本 (默认: `3.12`)

### 卷挂载

两种方法都需要挂载以下目录：

- `./checkpoints:/app/checkpoints` - 模型权重目录
- `./references:/app/references` - 参考音频文件目录

### 环境变量

- `COMPILE=1` - 启用 `torch.compile` 以加速推理 (约提速10倍)
- `GRADIO_SERVER_NAME=0.0.0.0` - WebUI 服务器主机
- `GRADIO_SERVER_PORT=7860` - WebUI 服务器端口
- `API_SERVER_NAME=0.0.0.0` - API 服务器主机
- `API_SERVER_PORT=8080` - API 服务器端口

!!! note
    Docker 容器期望模型权重挂载在 `/app/checkpoints` 路径。在启动容器前，请确保已下载所需的模型权重。

!!! warning
    GPU 支持需要 NVIDIA Docker 运行时。对于仅 CPU 部署，请移除 `--gpus all` 标志并使用 CPU 镜像。
