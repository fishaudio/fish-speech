## 1. 准备数据集

```
.
├── SPK1
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.mp3
│   └── 30.1-32.71.mp3
└── SPK2
    └── 38.79-40.85.mp3
```

你需要将数据集转为以上格式, 并放到 `data/demo` 下, 音频后缀可以为 `.mp3`, `.wav` 或 `.flac`.

## 2. 分割训练集和验证集

```bash
python tools/vqgan/create_train_split.py data/demo
```

该命令会在 `data/demo` 目录下创建 `data/demo/vq_train_filelist.txt` 和 `data/demo/vq_val_filelist.txt` 文件, 分别用于训练和验证.  

!!!info
    对于 VITS 格式, 你可以使用 `--filelist xxx.list` 来指定文件列表.  
    请注意, `filelist` 所指向的音频文件必须也位于 `data/demo` 文件夹下.

## 3. 启动训练

```bash
python fish_speech/train.py --config-name vqgan_finetune
```

!!! note
    你可以通过修改 `fish_speech/configs/vqgan_finetune.yaml` 来修改训练参数, 但大部分情况下, 你不需要这么做.

## 4. 测试音频
    
```bash
python tools/vqgan/inference.py -i test.wav --checkpoint-path results/vqgan_finetune/checkpoints/step_000010000.ckpt
```

你可以查看 `fake.wav` 来判断微调效果.

!!! note
    你也可以尝试其他的 checkpoint, 我们建议你使用最早的满足你要求的 checkpoint, 他们通常在 OOD 上表现更好.
