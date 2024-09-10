# 微調整

明らかに、このページを開いたとき、few-shot 事前トレーニングモデルのパフォーマンスに満足していなかったことでしょう。データセット上でのパフォーマンスを向上させるためにモデルを微調整したいと考えています。

現在のバージョンでは、「LLAMA」部分のみを微調整する必要があります。

## LLAMAの微調整
### 1. データセットの準備

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

データセットを上記の形式に変換し、「data」ディレクトリに配置する必要があります。音声ファイルの拡張子は「.mp3」、「.wav」、または「.flac」にすることができ、注釈ファイルの拡張子は「.lab」にする必要があります。

!!! warning
    データセットにラウドネス正規化を適用することをお勧めします。これを行うには、[fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) を使用できます。

    ```bash
    fap loudness-norm data-raw data --clean
    ```


### 2. セマンティックトークンのバッチ抽出

VQGANの重みをダウンロードしたことを確認してください。まだダウンロードしていない場合は、次のコマンドを実行してください。

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

次に、次のコマンドを実行してセマンティックトークンを抽出できます。

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "firefly_gan_vq" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

!!! note
    `--num-workers` と `--batch-size` を調整して抽出速度を上げることができますが、GPUメモリの制限を超えないようにしてください。  
    VITS形式の場合、`--filelist xxx.list` を使用してファイルリストを指定できます。

このコマンドは、`data`ディレクトリに`.npy`ファイルを作成します。以下のように表示されます。

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

### 3. データセットをprotobufにパックする

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

コマンドの実行が完了すると、`data`ディレクトリに`quantized-dataset-ft.protos`ファイルが表示されます。

### 4. 最後に、LoRAを使用して微調整する

同様に、`LLAMA`の重みをダウンロードしたことを確認してください。まだダウンロードしていない場合は、次のコマンドを実行してください。

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

最後に、次のコマンドを実行して微調整を開始できます。

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
    `fish_speech/configs/text2semantic_finetune.yaml` を変更して、`batch_size`、`gradient_accumulation_steps` などのトレーニングパラメータを変更し、GPUメモリに適合させることができます。

!!! note
    Windowsユーザーの場合、`trainer.strategy.process_group_backend=gloo` を使用して `nccl` の問題を回避できます。

トレーニングが完了したら、[推論](inference.md)セクションを参照し、`--speaker SPK1` を使用して音声を生成します。

!!! info
    デフォルトでは、モデルは話者の発話パターンのみを学習し、音色は学習しません。音色の安定性を確保するためにプロンプトを使用する必要があります。
    音色を学習したい場合は、トレーニングステップ数を増やすことができますが、これにより過学習が発生する可能性があります。

トレーニングが完了したら、推論を行う前にLoRAの重みを通常の重みに変換する必要があります。

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/fish-speech-1.4 \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/fish-speech-1.4-yth-lora/
```
!!! note
    他のチェックポイントを試すこともできます。要件を満たす最も早いチェックポイントを使用することをお勧めします。これらは通常、分布外（OOD）データでより良いパフォーマンスを発揮します。
