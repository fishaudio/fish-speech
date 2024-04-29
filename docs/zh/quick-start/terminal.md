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
??? info "参数列表"
    `--input-path/-i: Optional[Path]`:可选参数，想要生成角色的原始语音的路径

    `--output-path/-o: Optional[str]`:可选参数，输出路径

    `--config-name/-cfg: Optional[str]`:可选参数，配置文件路径

    `--checkpoint-path/-ckpt: Optional[str]`:可选参数，checkpoint路径
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
??? info "参数列表"
    `--text: str`: 输入的文本，作为生成文本的前缀。

    `--prompt_text: Optional[str]`: 可选参数，用于指定文本的提示。

    `--prompt_tokens: Optional[Path]`: 可选参数，用于指定文本的提示的token化表示的路径。

    `--num_samples: int`: 生成文本的样本数量。

    `--max_new_tokens: int`: 每个样本生成的最大token数量。

    `--top_k: int`: Top-K采样的K值。

    `--top_p: int`: Top-P（nucleus）采样的p值。

    `--repetition_penalty: float`: 重复惩罚系数，用于惩罚生成的重复词。

    `--temperature: float`: 温度参数，控制生成文本的多样性。

    `--checkpoint_path: Path`: 模型的检查点路径。

    `-config_name: str`: 模型配置的名称。

    `--tokenizer: str`: 分词器的名称。

    `--compile/--no-compile: bool`: 是否在生成之前对一部分代码进行编译优化。

    `--use-g2p/--no-g2p: bool`: 是否使用g2p（grapheme-to-phoneme）转换。

    `--seed: int`: 随机数种子，用于确定生成的随机性。

    `--speaker: Optional[str]`: 可选参数，指定说话者的标识。

    `--order: str`: 顺序，指定生成文本的顺序。

    `--half/--no-half: bool`: 是否使用半精度浮点数进行模型推理。

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
??? info "参数列表"
    `--input-path/-i: Optional[Path]`:可选参数，语义token的numpy数组文件

    `--output-path/-o: Optional[str]`:可选参数，输出路径

    `--config-name/-cfg: Optional[str]`:可选参数，配置文件路径

    `--checkpoint-path/-ckpt: Optional[str]`:可选参数，checkpoint路径