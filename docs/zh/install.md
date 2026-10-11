## 系统要求

| 模型 | GPU 显存 | 显卡 |
| --- | --- | --- |
| Drama 3 | 至少 32 GB | 仅 NVIDIA GPU，不支持 CPU、Apple MPS、AMD ROCm，也不支持 `--compile` |
| S2-Pro | 至少 24 GB | NVIDIA GPU 或 CPU |

- 系统：Linux、WSL

## 系统设置

Fish Audio Drama 3 与 S2-Pro 共用同一套安装方式。请选择最适合你当前开发环境的方案。

**前置依赖**：先安装音频处理所需的系统依赖：
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU 安装（选择 CUDA 版本：cu126、cu128、cu129）
# 使用 cu126 或 cu128 时，请同时替换下面两个命令中的 "cu129"。
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[cu129]"

# 仅 CPU 安装
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu]"

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

如果你使用 Intel Arc GPU，可按以下方式安装 XPU 支持（仅 S2-Pro，Drama 3 需要 NVIDIA GPU）：

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

## 下载权重

两个模型共用同一套命令行和 API Server，模型类型由权重目录里的 `config.json` 自动识别。请把它们放在同一个 `checkpoints` 目录下。

下载需要 Hugging Face 命令行工具：

```bash
pip install -U "huggingface_hub[cli]"
```

### Drama 3

```bash
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

如果权重目录中没有 `codec.pth`，会改用 `checkpoints/s2-pro/codec.pth` 中的 S2-Pro codec，所以 S2-Pro 建议一并下载。

### S2-Pro

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

下载好权重后，用法见[推理](inference.md)。

## Docker 设置

Drama 3 与 S2-Pro 提供多种 Docker 部署方式，适配不同场景。你可以直接使用 Docker Hub 预构建镜像，也可以用 Docker Compose 本地构建，或手动构建自定义镜像。

我们提供 API Server 的 GPU（默认 CUDA126）和 CPU 镜像。镜像已内置 WebUI，容器起来后打开 `http://localhost:8080/ui` 即可访问界面。GPU 镜像可运行两个模型，CPU 镜像仅支持 S2-Pro。

### 前置条件

- 已安装 Docker 和 Docker Compose
- （GPU 场景）已安装 NVIDIA Docker runtime（Drama 3 必须）
- 显存：Drama 3 至少 32 GB，S2-Pro 建议至少 24 GB

### 运行预构建镜像

Drama 3：在 SM89 及以上的显卡上只暴露一张卡，较早的显卡请使用 `--gpus all`：

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

S2-Pro：

```bash
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s2-pro \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### 使用 Docker Compose

如果你需要开发或自定义，推荐使用 Docker Compose 在本地构建并运行：

```bash
# 先克隆仓库
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# 启动 API Server（默认加载 checkpoints/s2-pro）
docker compose --profile server up

# 加载 Drama 3，并用 DEVICE 把它固定在单张卡上（不设置则最多使用四张）
LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 DEVICE=cuda:0 docker compose --profile server up

# 启用 compile 优化启动 API Server（仅 S2-Pro）
COMPILE=1 docker compose --profile server up

# 仅 CPU 部署（仅 S2-Pro）
BACKEND=cpu docker compose --profile server up
```

#### Docker Compose 环境变量

你可以通过环境变量定制部署参数：

```bash
# .env 文件示例
BACKEND=cuda              # 或 cpu
COMPILE=1                 # 启用 compile 优化
API_PORT=8080             # API Server 端口
UV_VERSION=0.8.15         # UV 包管理器版本
```

命令执行后会自动构建镜像并启动容器。你可以通过 `http://localhost:8080` 访问 API Server，通过 `http://localhost:8080/ui` 访问 WebUI。

### 手动 Docker 构建

如果你需要更细粒度的构建控制，可以手动构建：

```bash
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
    --target server \
    -t fish-speech-server:cpu .

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
- `LLAMA_CHECKPOINT_PATH` - 主模型权重路径，例如 `checkpoints/drama-3` 或 `checkpoints/s2-pro`
- `DECODER_CHECKPOINT_PATH` - codec 权重路径，留空则由服务端按权重目录自行查找
- `DEVICE` - 推理设备，例如 `cuda:0`；Drama 3 设置为单张卡时只用它，留空最多使用四张卡
- `FISH_DRAMA3_FP8_EXPERTS=0` 或 `1` - 强制 Drama 3 使用反量化或 FP8 专家（默认按显卡自动选择）
- `FISH_DRAMA3_CUDA_GRAPH=0` - 关闭 Drama 3 的 CUDA graph 解码
- `API_SERVER_NAME=0.0.0.0` - API 服务地址
- `API_SERVER_PORT=8080` - API 服务端口

!!! note
    Docker 容器默认从 `/app/checkpoints` 读取模型权重。启动容器前请先下载好所需权重。

!!! warning
    GPU 支持需要 NVIDIA Docker runtime。若仅使用 CPU，请移除 `--gpus all` 并使用 CPU 镜像。
