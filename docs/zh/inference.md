# 推理

Fish Audio 提供两个模型：**S2-Pro** 和 **Drama 3**。两者共用同一套命令行和 API Server，模型类型由权重目录里的 `config.json` 自动识别，因此下面每一节都按「任务 → 两种模型的命令」组织。

开始前请先完成[安装](install.md)，并下载好[模型权重](install.md#下载权重)。

!!! warning "Drama 3 的运行限制"
    Drama 3 只能在 NVIDIA GPU 上运行，不支持 CPU、Apple MPS 和 AMD ROCm，也不支持 `--compile`。显存需求见[系统要求](install.md#系统要求)。

!!! note
    V100 等 SM80 之前的显卡不支持 bfloat16，Drama 3 会自动改用 float16（残差流保持 fp32）。这条路径尚未在真实硬件上测试过；如果出现静音、噪声或 "non-finite logits" 报错，请提交 issue。

## 命令行推理

### S2-Pro

S2-Pro 走「VQ tokens → Semantic tokens → 音频」三步流水线。

!!! note
    如果您计划让模型随机选择音色，可以跳过第 1 步。

**1. 从参考音频获取 VQ tokens**

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

您应该会得到一个 `fake.npy` 和一个 `fake.wav`。

**2. 从文本生成 Semantic tokens**

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "您想要转换的文本" \
    --prompt-text "您的参考文本" \
    --prompt-tokens "fake.npy" \
    # --compile
```

此命令将在工作目录中创建一个 `codes_N` 文件，其中 N 是从 0 开始的整数。

!!! note
    您可能希望使用 `--compile` 来融合 CUDA 内核以实现更快的推理，但是我们更推荐您使用我们 sglang 的推理加速优化。
    相应地，如果您不计划使用加速，可以注释掉 `--compile` 参数。

!!! info
    对于不支持 bf16 的 GPU，您可能需要使用 `--half` 参数。

**3. 从语义令牌生成声音**

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

之后你会得到一个 `fake.wav` 文件。

### Drama 3

Drama 3 一步到位，`--output` 直接写出解码后的音频：

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/drama-3 \
    --text "您想要转换的文本" \
    --prompt-audio "reference.wav" \
    --prompt-text "您的参考文本" \
    --output "output.wav" \
    --device cuda:0
```

在较早的 GPU 上可以指定多张卡，例如 `--device cuda:0,cuda:1,cuda:2,cuda:3`，最后一张卡负责音频解码器和 codec。只写 `--device cuda` 时，最多使用前四张可见的 GPU。

## API Server

### 启动 S2-Pro

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq
```

### 启动 Drama 3

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/drama-3 \
    --device cuda:0
```

!!! note
    Drama 3 不支持 `--compile`；在较早的 GPU 上用 `--device` 列出多张卡，最后一张卡负责音频解码器和 codec。

两个模型共用这些启动参数：

- `--compile`：启用 `torch.compile` 优化（仅 S2-Pro）
- `--half`：使用 fp16 模式
- `--api-key`：要求 bearer token 鉴权
- `--workers`：设置工作进程数
- `--max-text-length`：限制单次请求的文本长度，0 表示不限制

健康检查：

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

返回 `{"status":"ok"}` 即表示服务正常。

### API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/tts` | 文本转语音，请求体为 msgpack |
| `GET` | `/v1/tts/defaults` | 当前加载模型的采样默认值 |
| `GET` | `/v1/health` | 健康检查 |
| `POST` | `/v1/vqgan/encode` | 参考音频 → VQ tokens |
| `POST` | `/v1/vqgan/decode` | VQ tokens → 音频 |
| `POST` | `/v1/references/add` | 保存参考音色，表单字段 `id`、`audio`（文件）、`text` |
| `GET` | `/v1/references/list` | 列出已保存的音色 |
| `POST` | `/v1/references/update` | 重命名，表单字段 `old_reference_id`、`new_reference_id` |
| `DELETE` | `/v1/references/delete` | 删除，表单字段 `reference_id` |
| `GET` | `/ui` | WebUI（Docker 镜像已内置，源码运行需先构建） |

保存参考音色后，请求时用 `--reference_id` 引用即可（音频字段要带 MIME 类型，否则服务端无法识别）：

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
    -F id=my-speaker \
    -F "audio=@reference.wav;type=audio/wav" \
    -F text="参考音频的转写文本"

python tools/api_client.py --url http://127.0.0.1:8080/v1/tts \
    --text "Hello" --reference_id my-speaker --output demo
```

### 发送请求

使用哪个模型取决于服务端启动时加载的权重，请求里没有单独的模型字段。

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "Hello from Fish Speech" \
    --output demo
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--reference_audio` / `--reference_text` | 参考音频与对应转写文本，两者数量必须一致 |
| `--reference_speaker` | 每段参考音频的说话人 ID（0–15），会写成 <code>&lt;&#124;speaker:N&#124;&gt;</code> 前缀（Drama 3 的主要用法） |
| `--reference_id` | 使用已保存的参考音色，例如 `my-speaker` |
| `--format` | 输出格式：`wav`、`pcm`、`mp3`、`opus`，默认 `wav` |
| `--streaming` | 流式返回音频 |
| `--single_chunk` | 整段一次生成，不分块（仅 Drama 3） |
| `--chunk_length` | 分块长度，默认 200（仅 S2-Pro） |
| `--max_new_tokens` | 每块最多生成的帧数，留空用服务端默认 1024；设为 0 使用模型自身上限 |
| `--temperature` / `--top_p` | 采样参数，留空使用模型默认（0.7 / 0.7） |
| `--seed` | 固定随机种子（不能用来固定音色） |
| `--api_key` | 服务端开启 `--api-key` 时必填 |
| `--no-play` | 合成后不播放音频（默认会播放；无音频设备的机器上请加上） |

!!! note
    `repetition_penalty` 仅为兼容而保留，S2-Pro 和 Drama 3 都不使用它，两者都采用内置的重复感知采样。

!!! tip
    用 Docker 启动服务见 [Docker 设置](install.md#docker-设置)（server 镜像已内置 WebUI，容器起来后打开 `http://localhost:8080/ui`）。

## 模型差异

| 项目 | S2-Pro | Drama 3 |
| --- | --- | --- |
| 硬件 | GPU / CPU | 仅 NVIDIA GPU，不支持 `--compile` |
| 长文本 | 按 `chunk_length` 分块 | 自动按约 300 字节（UTF-8）、每块最多 3 个说话人切分，尽量在句末断开；第一块和最近一块回喂作为上下文 |
| 单次长度上限 | 默认每批 1024 帧；`max_new_tokens=0` 时一直生成到上下文用完 | 默认每块 1024 帧（约 47 秒），`single_chunk` 时整段文本也是 1024 帧；`max_new_tokens=0` 时每块 600 帧，`single_chunk` 时 2048 帧（约 95 秒）；始终不超过 10,240 帧减去 prompt 长度 |
| 说话人 | 参考音频的顺序决定角色 | <code>&lt;&#124;speaker:0&#124;&gt;</code>–<code>&lt;&#124;speaker:15&#124;&gt;</code> 标签决定角色，参考音频顺序无关 |
| `chunk_length` | 生效 | 不起作用 |
| 采样默认值 | temperature 0.7、top-p 0.7、top-k 30 | 同左 |
| `repetition_penalty` | 忽略 | 忽略 |

Drama 3 还有两个环境变量：`FISH_DRAMA3_FP8_EXPERTS=0` 或 `1` 可强制使用反量化或 FP8 专家，`FISH_DRAMA3_CUDA_GRAPH=0` 可关闭 CUDA graph 解码。

## WebUI 推理

### Awesome WebUI

Awesome WebUI 是一个基于 TypeScript 开发的现代化 Web 界面，提供更丰富的功能和更好的交互体验。

server Docker 镜像已内置构建好的 WebUI，容器启动后直接打开 `http://localhost:8080/ui` 即可。以下步骤适用于从源码运行。

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

WebUI 构建完成后，返回项目根目录，启动 API 服务器（`--compile` 仅 S2-Pro，加载 Drama 3 时去掉）：

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**访问：**

在服务器启动后，您可以通过浏览器访问以下地址体验：
`http://localhost:8888/ui`
