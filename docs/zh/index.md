# 介绍

!!! warning
    我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规.

此代码库根据 `BSD-3-Clause` 许可证发布, 所有模型根据 CC-BY-NC-SA-4.0 许可证发布.

<p align="center">
<img src="/assets/figs/diagram.png" width="75%">
</p>

## 要求
- GPU内存: 2GB (用于推理), 16GB (用于微调)
- 系统: Linux (全部功能), Windows (仅推理, 不支持 `flash-attn`, 不支持 `torch.compile`)

因此, 我们强烈建议 Windows 用户使用 WSL2 或 docker 来运行代码库.

## 设置
```bash
# 创建一个 python 3.10 虚拟环境, 你也可以用 virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# 安装 pytorch nightly 版本
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# 安装 flash-attn (适用于linux)
pip3 install ninja && MAX_JOBS=4 pip3 install flash-attn --no-build-isolation

# 安装 fish-speech
pip3 install -e .
```

## 更新日志

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
