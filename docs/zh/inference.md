# 推理

由于声码器模型已更改，您需要比以前更多的 VRAM，建议使用 12GB 进行流畅推理。

我们支持命令行、HTTP API 和 WebUI 进行推理，您可以选择任何您喜欢的方法。

## 下载权重

首先您需要下载模型权重：

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## 命令行推理

!!! note
    如果您计划让模型随机选择音色，可以跳过此步骤。

### 1. 从参考音频获取 VQ 令牌

```bash
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

您应该会得到一个 `fake.npy` 和一个 `fake.wav`。

### 2. 从文本生成语义令牌：

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "您想要转换的文本" \
    --prompt-text "您的参考文本" \
    --prompt-tokens "fake.npy" \
    --compile
```

此命令将在工作目录中创建一个 `codes_N` 文件，其中 N 是从 0 开始的整数。

!!! note
    您可能希望使用 `--compile` 来融合 CUDA 内核以实现更快的推理（~15 token/秒 -> ~150 token/秒，在RTX 4090 GPU上）。
    相应地，如果您不计划使用加速，可以注释掉 `--compile` 参数。

!!! info
    对于不支持 bf16 的 GPU，您可能需要使用 `--half` 参数。

### 3. 从语义令牌生成声音：

!!! warning "未来警告"
    我们保留了从原始路径（tools/vqgan/inference.py）访问接口的能力，但此接口可能在后续版本中被删除，因此请尽快更改您的代码。

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

## HTTP API 推理

我们提供HTTP API进行推理。您可以使用以下命令启动服务器：

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> 如果您想要加速推理，可以添加 `--compile` 参数。

之后，您可以在 http://127.0.0.1:8080/ 查看和测试API。

## GUI 推理 
[下载客户端](https://github.com/AnyaCoder/fish-speech-gui/releases)

## WebUI 推理

您可以使用以下命令启动WebUI：

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

或者简单地

```bash
python -m tools.run_webui
```
> 如果您想要加速推理，可以添加 `--compile` 参数。

!!! note
    您可以提前将标签文件和参考音频文件保存到主目录的 `references` 文件夹中（需要自己创建），这样就可以在WebUI中直接调用它们。

!!! note
    您可以使用Gradio环境变量，如 `GRADIO_SHARE`、`GRADIO_SERVER_PORT`、`GRADIO_SERVER_NAME` 来配置WebUI。

尽情享受吧！
