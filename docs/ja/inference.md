# 推論

ボコーダーモデルが変更されたため、以前よりも多くのVRAMが必要です。スムーズな推論には12GBを推奨します。

推論には、コマンドライン、HTTP API、WebUIをサポートしており、お好きな方法を選択できます。

## 重みのダウンロード

まず、モデルの重みをダウンロードする必要があります：

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## コマンドライン推論

!!! note
    モデルにランダムに音色を選択させる場合は、この手順をスキップできます。

### 1. 参照音声からVQトークンを取得

```bash
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

`fake.npy` と `fake.wav` が得られるはずです。

### 2. テキストからセマンティックトークンを生成：

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "変換したいテキスト" \
    --prompt-text "参照テキスト" \
    --prompt-tokens "fake.npy" \
    --compile
```

このコマンドは、作業ディレクトリに `codes_N` ファイルを作成します（Nは0から始まる整数）。

!!! note
    より高速な推論のために `--compile` を使用してCUDAカーネルを融合することができます（約15トークン/秒 -> 約150トークン/秒, RTX 4090 GPU）。
    対応して、加速を使用しない場合は、`--compile` パラメータをコメントアウトできます。

!!! info
    bf16をサポートしないGPUの場合、`--half` パラメータの使用が必要かもしれません。

### 3. セマンティックトークンから音声を生成：

!!! warning "将来の警告"
    元のパス（tools/vqgan/inference.py）からアクセス可能なインターフェースを維持していますが、このインターフェースは後続のリリースで削除される可能性があるため、できるだけ早くコードを変更してください。

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy"
```

## HTTP API推論

推論用のHTTP APIを提供しています。以下のコマンドでサーバーを開始できます：

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> 推論を高速化したい場合は、`--compile` パラメータを追加できます。

その後、http://127.0.0.1:8080/ でAPIを表示・テストできます。

## GUI推論 
[クライアントをダウンロード](https://github.com/AnyaCoder/fish-speech-gui/releases)

## WebUI推論

以下のコマンドでWebUIを開始できます：

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

または単純に

```bash
python -m tools.run_webui
```
> 推論を高速化したい場合は、`--compile` パラメータを追加できます。

!!! note
    ラベルファイルと参照音声ファイルをメインディレクトリの `references` フォルダに事前に保存することができます（自分で作成する必要があります）。これにより、WebUIで直接呼び出すことができます。

!!! note
    `GRADIO_SHARE`、`GRADIO_SERVER_PORT`、`GRADIO_SERVER_NAME` などのGradio環境変数を使用してWebUIを設定できます。
