我们推荐在Ubuntu系统进行安装，因为在Ubuntu系统下可以获得更好的nvidia驱动支持。

```
    # 创建一个 python 3.10 虚拟环境, 你也可以用 virtualenv
    conda create -n fish-speech python=3.10
    conda activate fish-speech

    # 安装 pytorch nightly 版本
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

    # 安装 flash-attn (适用于linux)
    pip3 install ninja 
    # 可以根据内存的大小增加或者减少
    MAX_JOBS=4 
    # 如果flash-attn加速模块出现问题，你也可以跳过此部分
    pip3 install flash-attn --no-build-isolation

    # 安装 fish-speech
    pip3 install -e .
```
