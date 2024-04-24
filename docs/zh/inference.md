# 推理

推理支持命令行, http api, 以及 webui 三种方式.  

!!! note
    总的来说, 推理分为几个部分:  

    1. 给定一段 5-10 秒的语音, 将它用 VQGAN 编码.  
    2. 将编码后的语义 token 和对应文本输入语言模型作为例子.  
    3. 给定一段新文本, 让模型生成对应的语义 token.  
    4. 将生成的语义 token 输入 VQGAN 解码, 生成对应的语音.  

## 命令行推理

从我们的 huggingface 仓库下载所需的 `vqgan` 和 `text2semantic` 模型。
    
```bash
huggingface-cli download fishaudio/speech-lm-v1 vqgan-v1.pth --local-dir checkpoints
huggingface-cli download fishaudio/speech-lm-v1 text2semantic-400m-v0.2-4k.pth --local-dir checkpoints
```
对于中国大陆用户，可使用mirror下载。
```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/speech-lm-v1 vqgan-v1.pth --local-dir checkpoints
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/speech-lm-v1 text2semantic-400m-v0.2-4k.pth --local-dir checkpoints
```

### 1. 从语音生成 prompt: 

!!! note
    如果你打算让模型随机选择音色, 你可以跳过这一步.

```bash
python tools/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/vqgan-v1.pth"
```
你应该能得到一个 `fake.npy` 文件.

### 2. 从文本生成语义 token: 
```bash
python tools/llama/generate.py \
    --text "要转换的文本" \
    --prompt-text "你的参考文本" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/text2semantic-400m-v0.2-4k.pth" \
    --num-samples 2 \
    --compile
```

该命令会在工作目录下创建 `codes_N` 文件, 其中 N 是从 0 开始的整数.

!!! note
    您可能希望使用 `--compile` 来融合 cuda 内核以实现更快的推理 (~30 个 token/秒 -> ~500 个 token/秒).  
    对应的, 如果你不打算使用加速, 你可以注释掉 `--compile` 参数.

!!! info
    对于不支持 bf16 的 GPU, 你可能需要使用 `--half` 参数.

!!! warning
    如果你在使用自己微调的模型, 请务必携带 `--speaker` 参数来保证发音的稳定性.  
    如果你使用了 lora, 请使用 `--config-name text2semantic_finetune_lora` 来加载模型.

### 3. 从语义 token 生成人声: 
```bash
python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/vqgan-v1.pth"
```

## HTTP API 推理

运行以下命令来启动 HTTP 服务:

```bash
python -m tools.api --listen 0.0.0.0:8000
# 推荐中国大陆用户运行以下命令来启动 HTTP 服务:
HF_ENDPOINT=https://hf-mirror.com python -m tools.api --listen 0.0.0.0:8000
```

随后, 你可以在 `http://127.0.0.1:8000/docs` 中查看并测试 API.  
一般来说, 你需要先调用 `PUT /v1/models/default` 来加载模型, 然后调用 `POST /v1/models/default/invoke` 来进行推理.
具体的参数请参考 API 文档.

## WebUI 推理

在运行 WebUI 之前, 你需要先启动 HTTP 服务, 如上所述.

随后你可以使用以下命令来启动 WebUI:

```bash
python fish_speech/webui/app.py
```

或附带参数来启动 WebUI:

```bash
# 以临时环境变量的方式启动:
GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7860 python fish_speech/webui/app.py
```

祝大家玩得开心!
