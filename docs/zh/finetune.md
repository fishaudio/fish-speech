# 微调

显然, 当你打开这个页面的时候, 你已经对预训练模型 zero-shot 的效果不算满意. 你想要微调一个模型, 使得它在你的数据集上表现更好.  

在目前版本，你只需要微调'LLAMA'部分即可.

## LLAMA 微调
### 1. 准备数据集

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 30.1-32.71.lab
│   └── 30.1-32.71.mp3
└── SPK2
    ├── 38.79-40.85.lab
    └── 38.79-40.85.mp3
```

你需要将数据集转为以上格式, 并放到 `data` 下, 音频后缀可以为 `.mp3`, `.wav` 或 `.flac`, 标注文件后缀建议为 `.lab`.

!!! warning
    建议先对数据集进行响度匹配, 你可以使用 [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) 来完成这一步骤. 
    ```bash
    fap loudness-norm data-raw data --clean
    ```

### 2. 批量提取语义 token

确保你已经下载了 vqgan 权重, 如果没有, 请运行以下命令:

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

对于中国大陆用户, 可使用 mirror 下载.

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

随后可运行以下命令来提取语义 token:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "firefly_gan_vq" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

!!! note
    你可以调整 `--num-workers` 和 `--batch-size` 来提高提取速度, 但是请注意不要超过你的显存限制.  

该命令会在 `data` 目录下创建 `.npy` 文件, 如下所示:

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 21.15-26.44.npy
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 27.51-29.98.npy
│   ├── 30.1-32.71.lab
│   ├── 30.1-32.71.mp3
│   └── 30.1-32.71.npy
└── SPK2
    ├── 38.79-40.85.lab
    ├── 38.79-40.85.mp3
    └── 38.79-40.85.npy
```

### 3. 打包数据集为 protobuf

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

命令执行完毕后, 你应该能在 `data` 目录下看到 `protos` 文件.


### 4. 最后, 使用 LoRA 进行微调

同样的, 请确保你已经下载了 `LLAMA` 权重, 如果没有, 请运行以下命令:

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

对于中国大陆用户, 可使用 mirror 下载.

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

最后, 你可以运行以下命令来启动微调:

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
    你可以通过修改 `fish_speech/configs/text2semantic_finetune.yaml` 来修改训练参数如 `batch_size`, `gradient_accumulation_steps` 等, 来适应你的显存.

!!! note
    对于 Windows 用户, 你可以使用 `trainer.strategy.process_group_backend=gloo` 来避免 `nccl` 的问题.

训练结束后, 你可以参考 [推理](inference.md) 部分, 并携带 `--speaker SPK1` 参数来测试你的模型.

!!! info
    默认配置下, 基本只会学到说话人的发音方式, 而不包含音色, 你依然需要使用 prompt 来保证音色的稳定性.  
    如果你想要学到音色, 请将训练步数调大, 但这有可能会导致过拟合. 

训练完成后, 你需要先将 loRA 的权重转为普通权重, 然后再进行推理.

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/fish-speech-1.4 \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/fish-speech-1.4-yth-lora/
```

!!! note
    你也可以尝试其他的 checkpoint, 我们建议你使用最早的满足你要求的 checkpoint, 他们通常在 OOD 上表现更好.
