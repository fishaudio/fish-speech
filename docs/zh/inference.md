# 推理

Fish Audio S2 模型需要较大的显存，我们推荐您使用至少24GB的GPU进行推理。

## 下载权重

首先您需要下载模型权重：

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## 命令行推理

!!! note
    如果您计划让模型随机选择音色，可以跳过此步骤。

### 1. 从参考音频获取 VQ tokens

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

您应该会得到一个 `fake.npy` 和一个 `fake.wav`。

### 2. 从文本生成 Semantic tokens：

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "您想要转换的文本" \
    --prompt-text "您的参考文本" \
    --prompt-tokens "fake.npy" \
    # --compile
```

此命令将在工作目录中创建一个 `codes_N` 文件，其中 N 是从 0 开始的整数。

!!! note
    您可能希望使用 `--compile` 来融合 CUDA 内核以实现更快的推理，但是我们更推荐您使用我们sglang的推理加速优化。
    相应地，如果您不计划使用加速，可以注释掉 `--compile` 参数。

!!! info
    对于不支持 bf16 的 GPU，您可能需要使用 `--half` 参数。

### 3. 从语义令牌生成声音：

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

之后你会得到一个fake.wav文件。

## WebUI 推理

未完待续。
