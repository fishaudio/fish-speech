想要使用wsl，请先按照[微软官方的教程](https://learn.microsoft.com/en-us/windows/wsl/install)进行wsl2安装，我们推荐使用官方提供的Ubuntu22.04镜像
!!! note "提示"
    如果可能的话，请不要把wsl装在C盘

??? info "测试环境"
    系统版本： Ubuntu 22.04 on WSL2

    镜像源：[清华镜像源](https://pypi.tuna.tsinghua.edu.cn/)

    GPU：Nvidia RTX 4060 Laptop

    CUDA版本: cuda12.2

安装完成后在命令行窗口执行`wsl`进入linux子系统并导航到你指定的安装目录clone fish-speech的github库
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
flash-attn是一个加速注意力机制计算的库，我们不建议在wsl上安装flsh-attn，因为可能会出现无法预知的依赖问题。
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
