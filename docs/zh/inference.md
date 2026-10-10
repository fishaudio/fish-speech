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

## S3-drama

S3-drama 与 S2-Pro 共用同一套命令行和 API Server，模型类型会根据权重目录中的 `config.json` 自动识别。

!!! warning
    S3-drama 只能在 NVIDIA GPU 上运行，不支持 CPU、Apple MPS 和 AMD ROCm，也不支持 `--compile`。

| GPU | 运行方式 | 显存 |
| --- | --- | --- |
| SM89 及以上（L40、H100、H20） | MoE 专家保持 FP8，使用 CUDA graph 解码，单卡运行 | 约 41 GB |
| 较早的 GPU（V100、A100） | MoE 专家反量化后分布到多张卡上 | 合计约 70 GB |

!!! note
    V100 等 SM80 之前的显卡不支持 bfloat16，S3 会自动改用 float16（残差流保持 fp32）。这条路径尚未在真实硬件上测试过；如果出现静音、噪声或 "non-finite logits" 报错，请提交 issue。

### 下载权重

```bash
hf download fishaudio/s3-drama-it-fp8 --local-dir checkpoints/s3-drama-fp8
```

如果权重目录中没有 `codec.pth`，会改用 `checkpoints/s2-pro/codec.pth` 中的 S2-Pro codec。

### 命令行推理

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/s3-drama-fp8 \
    --text "您想要转换的文本" \
    --prompt-audio "reference.wav" \
    --prompt-text "您的参考文本" \
    --output "output.wav" \
    --device cuda:0
```

`--output` 会直接写出解码后的音频。在较早的 GPU 上，可以指定多张卡，例如 `--device cuda:0,cuda:1,cuda:2,cuda:3`，最后一张卡负责音频解码器和 codec。只写 `--device cuda` 时，最多使用前四张可见的 GPU。

### API Server

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s3-drama-fp8 \
    --device cuda:0
```

使用 Docker 时，用 server 镜像加载 S3 权重。在 SM89 及以上的显卡上，按下面的方式只暴露一张卡；较早的显卡请使用 `--gpus all`：

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

使用 Docker Compose 时设置同样的环境变量；`DEVICE=cuda:0` 让 S3 只用一张卡，不设置则最多使用四张卡：

```bash
LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 DEVICE=cuda:0 docker compose --profile server up
```

### 与 S2-Pro 的区别

- 长文本会切成约 300 字节（UTF-8）、每块最多 3 个说话人的块，尽量在句末断开，再逐块依次生成；第一块和最近一块会作为上下文喂回模型，保持音色一致。默认每块最多生成 1024 帧（约 47 秒）。
- 设置 `single_chunk`（命令行用 `--single-chunk`）可以整段一次生成，此时默认的 1024 帧上限作用于整段文本，长文本请调大 `max_new_tokens`，或设为 0 以使用 2048 帧（约 95 秒）。两种模式下，更大的 `max_new_tokens` 都会被限制在 10,240 帧减去 prompt 长度以内。
- 支持 `<|speaker:0|>` 到 `<|speaker:15|>` 的说话人标签，没写标签的文本算作 speaker 0。要让不同说话人用各自的声音，请给每段参考音频设置 `speaker` 字段（`tools/api_client.py` 中为 `--reference_speaker`），它会给开头没有说话人标签的转写文本补上标签；也可以直接在转写文本开头写上对应的标签（例如 `<|speaker:1|>`）。参考音频的先后顺序并不决定它属于哪个说话人。
- `chunk_length` 和 `repetition_penalty` 不起作用，S3 内置了重复感知采样。默认采样参数与 S2-Pro 相同，沿用官方 Fish Audio SDK：temperature 0.7、top-p 0.7、top-k 30。
- 开启 `streaming` 时，音频按块依次返回。
- `FISH_S3_FP8_EXPERTS=0` 或 `1` 可强制使用反量化或 FP8 专家，`FISH_S3_CUDA_GRAPH=0` 可关闭 CUDA graph 解码。

## WebUI 推理

### Awesome WebUI

Awesome WebUI 是一个基于 TypeScript 开发的现代化 Web 界面，提供更丰富的功能和更好的交互体验。

**构建 WebUI：**

您需要先在本地或者服务器上安装 Node.js 和 npm。

1. 进入 `awesome_webui` 目录：
   ```bash
   cd awesome_webui
   ```
2. 安装依赖：
   ```bash
   npm install
   ```
3. 构建 WebUI：
   ```bash
   npm run build
   ```

**启动后端服务器：**

WebUI 构建完成后，返回项目根目录，启动 API 服务器：

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**访问：**

在服务器启动后，您可以通过浏览器访问以下地址体验：
`http://localhost:8888/ui`
