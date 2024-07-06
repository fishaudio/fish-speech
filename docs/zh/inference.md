# 推理

推理支持命令行, http api, 以及 webui 三种方式.

!!! note
    总的来说, 推理分为几个部分:

    1. 给定一段 ~10 秒的语音, 将它用 VQGAN 编码.
    2. 将编码后的语义 token 和对应文本输入语言模型作为例子.
    3. 给定一段新文本, 让模型生成对应的语义 token.
    4. 将生成的语义 token 输入 VQGAN 解码, 生成对应的语音.

## 命令行推理

从我们的 huggingface 仓库下载所需的 `vqgan` 和 `llama` 模型。

```bash
huggingface-cli download fishaudio/fish-speech-1.2 --local-dir checkpoints/fish-speech-1.2
```

对于中国大陆用户，可使用 mirror 下载。

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/fish-speech-1.2 --local-dir checkpoints/fish-speech-1.2
```

### 1. 从语音生成 prompt:

!!! note
    如果你打算让模型随机选择音色, 你可以跳过这一步.

```bash
python tools/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
```

你应该能得到一个 `fake.npy` 文件.

### 2. 从文本生成语义 token:

```bash
python tools/llama/generate.py \
    --text "要转换的文本" \
    --prompt-text "你的参考文本" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.2" \
    --num-samples 2 \
    --compile
```

该命令会在工作目录下创建 `codes_N` 文件, 其中 N 是从 0 开始的整数.

!!! note
    您可能希望使用 `--compile` 来融合 cuda 内核以实现更快的推理 (~30 个 token/秒 -> ~500 个 token/秒).  
    对应的, 如果你不打算使用加速, 你可以注释掉 `--compile` 参数.

!!! info
    对于不支持 bf16 的 GPU, 你可能需要使用 `--half` 参数.

### 3. 从语义 token 生成人声:

#### VQGAN 解码

```bash
python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
```

## HTTP API 推理

运行以下命令来启动 HTTP 服务:

```bash
python -m tools.api \
    --listen 0.0.0.0:8000 \
    --llama-checkpoint-path "checkpoints/fish-speech-1.2" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --decoder-config-name firefly_gan_vq

如果你想要加速推理，可以加上--compile参数。

# 推荐中国大陆用户运行以下命令来启动 HTTP 服务:
HF_ENDPOINT=https://hf-mirror.com python -m ...
```

随后, 你可以在 `http://127.0.0.1:8000/` 中查看并测试 API.

下面是使用`tools/post_api.py`发送请求的示例。

```bash
python -m tools.post_api \
    --text "要输入的文本" \
    --reference_audio "参考音频路径" \
    --reference_text "参考音频的文本内容"
    --streaming True
```

上面的命令表示按照参考音频的信息，合成所需的音频并流式返回.

如果需要通过`{说话人}`和`{情绪}`随机选择参考音频，那么就根据下列步骤配置：

### 1. 在项目根目录创建`ref_data`文件夹.

### 2. 在`ref_data`文件夹内创建类似如下结构的目录.

```
.
├── SPEAKER1
│    ├──EMOTION1
│    │    ├── 21.15-26.44.lab
│    │    ├── 21.15-26.44.wav
│    │    ├── 27.51-29.98.lab
│    │    ├── 27.51-29.98.wav
│    │    ├── 30.1-32.71.lab
│    │    └── 30.1-32.71.flac
│    └──EMOTION2
│         ├── 30.1-32.71.lab
│         └── 30.1-32.71.mp3
└── SPEAKER2
    └─── EMOTION3
          ├── 30.1-32.71.lab
          └── 30.1-32.71.mp3
```

也就是`ref_data`里先放`{说话人}`文件夹, 每个说话人下再放`{情绪}`文件夹, 每个情绪文件夹下放任意个`音频-文本对`。

### 3. 在虚拟环境里输入

```bash
python tools/gen_ref.py
```

生成参考目录.

### 4. 调用 api.

```bash
python -m tools.post_api \
    --text "要输入的文本" \
    --speaker "说话人1" \
    --emotion "情绪1" \
    --streaming True
```

以上示例仅供测试.

## WebUI 推理

你可以使用以下命令来启动 WebUI:

```bash
python -m tools.webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.2" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

!!! note
    你可以使用 Gradio 环境变量, 如 `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` 来配置 WebUI.

祝大家玩得开心!
