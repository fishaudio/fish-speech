??? info "测试环境"
    系统版本： Ubuntu 22.04

    镜像源：[清华镜像源](https://pypi.tuna.tsinghua.edu.cn/)

    GPU：Nvidia A100

    CUDA版本: cuda12.2

## 安装cmake
```
sudo apt update
sudo apt install cmake
```
## 创建虚拟环境
创建一个 python 3.10 虚拟环境, 你也可以用 virtualenv
```
conda create -n fish-speech python=3.10
conda activate fish-speech
conda install -c anaconda cmake
```

## 安装 pytorch nightly 版本
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

## 安装 flash-attn (可选)
flash-attn是一个加速注意力机制计算的库，如果你在安装该部分时出现了问题，你也可以跳过此部分的安装，但是可能会导致推理速度变慢。
```
pip3 install ninja packaging
# 可以根据内存的大小增加或者减少MAX_JOBS的值
MAX_JOBS=4
pip3 install flash-attn --no-build-isolation
```
!!! info
    安装速度较慢，请耐心等待，如果你想要查看安装进度，可以从github下载软件包后手动安装。

# 安装 fish-speech
```
pip3 install -e .
```
??? bug "无法使用pip3 install -e"
    如果你无法使用pip3 install -e，请先执行
    ```
    pip install --upgrade pip
    ```