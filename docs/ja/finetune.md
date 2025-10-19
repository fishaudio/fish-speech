# ファインチューニング

このページを開いたということは、明らかに、事前学習済みモデルのゼロショット性能に満足していないということでしょう。データセットでより良い性能を発揮するようにモデルをファインチューニングしたいとお考えのはずです。

現在のバージョンでは、「LLAMA」部分のみをファインチューニングする必要があります。

## LLAMA のファインチューニング
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

データセットを上記の形式に変換し、`data` ディレクトリに配置する必要があります。音声ファイルの拡張子は `.mp3`、`.wav`、または `.flac` が使用でき、注釈ファイルの拡張子は `.lab` にすることを推奨します。

!!! info
    `.lab` 注釈ファイルには、音声の書き起こしテキストのみを含める必要があり、特別なフォーマット要件はありません。たとえば、`hi.mp3` の内容が「こんにちは、さようなら。」である場合、`hi.lab` ファイルには「こんにちは、さようなら。」という一行のテキストのみが含まれます。

!!! warning
    データセットにラウドネス正規化を適用することをお勧めします。これには [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) を使用できます。
    ```bash
    fap loudness-norm data-raw data --clean
    ```

### 2. セマンティックトークンの一括抽出

VQGANの重みをダウンロードしていることを確認してください。まだの場合は、次のコマンドを実行してください。

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

その後、次のコマンドを実行してセマンティックトークンを抽出できます。

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

!!! note
    `--num-workers` と `--batch-size` を調整して抽出速度を向上させることができますが、GPUメモリの制限を超えないように注意してください。

このコマンドは `data` ディレクトリに `.npy` ファイルを作成します。以下のようになります。

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
    └── 38.79-40.85.npy```

### 3. データセットを protobuf にパックする

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

コマンドの実行が完了すると、`data` ディレクトリに `protos` ファイルが表示されるはずです。

### 4. 最後に LoRA でファインチューニング

同様に、`LLAMA` の重みをダウンロードしていることを確認してください。まだの場合は、次のコマンドを実行してください。

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

最後に、次のコマンドを実行してファインチューニングを開始できます。

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
    `fish_speech/configs/text2semantic_finetune.yaml` を変更することで、`batch_size` や `gradient_accumulation_steps` などのトレーニングパラメータをGPUメモリに合わせて変更できます。

!!! note
    Windows ユーザーの場合、`trainer.strategy.process_group_backend=gloo` を使用して `nccl` の問題を回避できます。

トレーニングが完了したら、[推論](inference.md) のセクションを参照してモデルをテストできます。

!!! info
    デフォルト設定では、モデルは話者の発音方法のみを学習し、音色は学習しません。音色の安定性を確保するためには、依然としてプロンプトを使用する必要があります。
    音色を学習させたい場合は、トレーニングステップ数を増やしてください。ただし、これにより過学習が発生する可能性があります。

トレーニング後、推論を行う前に LoRA の重みを通常の重みに変換する必要があります。

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/openaudio-s1-mini \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/openaudio-s1-mini-yth-lora/
```

!!! note
    他のチェックポイントを試すこともできます。要件を満たす最も早いチェックポイントを使用することをお勧めします。これらは通常、OOD（分布外）データに対してより良いパフォーマンスを発揮します。
