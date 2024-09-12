# 介绍

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/lengyue233/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/lengyue233/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>

!!! warning
    我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规. <br/>
    此代码库与所有模型根据 CC-BY-NC-SA-4.0 许可证发布.

<p align="center">
   <img src="../assets/figs/diagram.png" width="75%">
</p>

## 要求

- GPU 内存: 4GB (用于推理), 8GB (用于微调)
- 系统: Linux, Windows

## Windows 配置

Windows 专业用户可以考虑 WSL2 或 docker 来运行代码库。

```bash
# 创建一个 python 3.10 虚拟环境, 你也可以用 virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# 安装 pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 fish-speech
pip3 install -e .

# (开启编译加速) 安装 triton-windows
pip install https://github.com/AnyaCoder/fish-speech/releases/download/v0.1.0/triton_windows-0.1.0-py3-none-any.whl
```

Windows 非专业用户可考虑以下为免 Linux 环境的基础运行方法（附带模型编译功能，即 `torch.compile`）：

1. 解压项目压缩包。
2. 点击 `install_env.bat` 安装环境。
3. 若需要开启编译加速则执行这一步:
    1. 使用如下链接下载 LLVM 编译器。
        - [LLVM-17.0.6（原站站点下载）](https://huggingface.co/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true)
        - [LLVM-17.0.6（镜像站点下载）](https://hf-mirror.com/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true)
        - 下载完 `LLVM-17.0.6-win64.exe` 后，双击进行安装，选择合适的安装位置，最重要的是勾选 `Add Path to Current User` 添加环境变量。
        - 确认安装完成。
    2. 下载安装 Microsoft Visual C++ 可再发行程序包，解决潜在 .dll 丢失问题。
        - [MSVC++ 14.40.33810.0 下载](https://aka.ms/vs/17/release/vc_redist.x64.exe)
    3. 下载安装 Visual Studio 社区版以获取 MSVC++ 编译工具, 解决 LLVM 的头文件依赖问题。
        - [Visual Studio 下载](https://visualstudio.microsoft.com/zh-hans/downloads/)
        - 安装好 Visual Studio Installer 之后，下载 Visual Studio Community 2022
        - 如下图点击`修改`按钮，找到`使用C++的桌面开发`项，勾选下载
    4. 下载安装 [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64)
4. 双击 `start.bat` 打开训练推理 WebUI 管理界面. 如有需要，可照下列提示修改`API_FLAGS`.

!!! info "可选"

    想启动 推理 WebUI 界面？编辑项目根目录下的 `API_FLAGS.txt`, 前三行修改成如下格式:
    ```
    --infer
    # --api
    # --listen ...
    ...
    ```

!!! info "可选"

    想启动 API 服务器？编辑项目根目录下的 `API_FLAGS.txt`, 前三行修改成如下格式:
    ```
    # --infer
    --api
    --listen ...
    ...
    ```

!!! info "可选"

    双击 `run_cmd.bat` 进入本项目的 conda/python 命令行环境

## Linux 配置

```bash
# 创建一个 python 3.10 虚拟环境, 你也可以用 virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# 安装 pytorch
pip3 install torch torchvision torchaudio

# 安装 fish-speech
pip3 install -e .[stable]

# (Ubuntu / Debian 用户) 安装 sox
apt install libsox-dev
```

## Docker 配置

1. 安装 NVIDIA Container Toolkit：

    Docker 如果想使用 GPU 进行模型训练和推理，需要安装 NVIDIA Container Toolkit ：

    对于 Ubuntu 用户：

    ```bash
    # 添加远程仓库
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    # 安装 nvidia-container-toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    # 重启 Docker 服务
    sudo systemctl restart docker
    ```

    对于使用其他 Linux 发行版的用户，安装指南请参考：[NVIDIA Container Toolkit Install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

    注：对于中国大陆的用户，您可能需要使用代理来完成相关工具的安装。

2. 拉取并运行 fish-speech 镜像

    ```shell
    # 拉取镜像
    docker pull lengyue233/fish-speech
    # 运行镜像
    docker run -it \
        --name fish-speech \
        --gpus all \
        -p 7860:7860 \
        lengyue233/fish-speech \
        zsh
    # 如果需要使用其他端口，请修改 -p 参数为 YourPort:7860
    ```

3. 下载模型依赖

    确保您在 docker 容器内的终端，然后再从我们的 huggingface 仓库下载所需的 `vqgan` 和 `llama` 模型。

    ```bash
    huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
    ```

    对于中国大陆用户，可以通过镜像站下载。

    ```bash
    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
    ```

4. 配置环境变量，访问 WebUI

    在 docker 容器内的终端，输入 `export GRADIO_SERVER_NAME="0.0.0.0"` ，从而让外部可以访问 docker 内的 gradio 服务。
    接着在 docker 容器内的终端，输入 `python tools/webui.py` 即可开启 WebUI 服务。

    如果是 WSL 或者是 MacOS ，访问 [http://localhost:7860](http://localhost:7860) 即可打开 WebUI 界面。

    如果是部署在服务器上，更换 localhost 为您的服务器 ip 即可。

## 更新日志

- 2024/09/10: 更新了 Fish-Speech 到 1.4, 增加了数据集大小， quantizer n_groups 4 -> 8.
- 2024/07/02: 更新了 Fish-Speech 到 1.2 版本，移除 VITS Decoder，同时极大幅度提升 zero-shot 能力.
- 2024/05/10: 更新了 Fish-Speech 到 1.1 版本，引入了 VITS Decoder 来降低口胡和提高音色相似度.
- 2024/04/22: 完成了 Fish-Speech 1.0 版本, 大幅修改了 VQGAN 和 LLAMA 模型.
- 2023/12/28: 添加了 `lora` 微调支持.
- 2023/12/27: 添加了 `gradient checkpointing`, `causual sampling` 和 `flash-attn` 支持.
- 2023/12/19: 更新了 Webui 和 HTTP API.
- 2023/12/18: 更新了微调文档和相关例子.
- 2023/12/17: 更新了 `text2semantic` 模型, 支持无音素模式.
- 2023/12/13: 测试版发布, 包含 VQGAN 模型和一个基于 LLAMA 的语言模型 (只支持音素).

## 致谢

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
