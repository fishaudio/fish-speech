## 系统要求

- GPU 内存：24GB（推理）
- 系统：Linux、WSL

## 系统设置

FishAudio S2支持多种安装方式，请选择最适合您开发环境的方法。

**先决条件**：安装用于音频处理的系统依赖项：
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech
pip install -e .
# 如果你没有安装上文的前两个依赖，这里会因为pyaudio无法安装而报错，可以考虑使用下面这一行指令。
# conda install pyaudio 
# 随后再次运行pip install -e .即可
```

### UV

UV 提供了更快的依赖解析和安装速度：

```bash
# GPU 安装 (选择您的 CUDA 版本: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# 仅 CPU 安装
uv sync --python 3.12 --extra cpu
```

## Docker 设置

Fish Audio系列模型提供了多种 Docker 部署选项以满足不同需求。您可以使用 Docker Hub 上的预构建镜像，通过 Docker Compose 在本地构建，或手动构建自定义镜像。

未完待续。
