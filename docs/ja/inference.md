# 推論

推論は、コマンドライン、HTTP API、および Web UI をサポートしています。

!!! note
    全体として、推論は次のいくつかの部分で構成されています：

    1. VQGANを使用して、与えられた約10秒の音声をエンコードします。
    2. エンコードされたセマンティックトークンと対応するテキストを例として言語モデルに入力します。
    3. 新しいテキストが与えられた場合、モデルに対応するセマンティックトークンを生成させます。
    4. 生成されたセマンティックトークンをVITS / VQGANに入力してデコードし、対応する音声を生成します。

## モデルをダウンロード
必要な`vqgan`および`llama`モデルを Hugging Face リポジトリからダウンロードします。

```bash
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

## コマンドライン推論
### 1. 音声からプロンプトを生成する：

!!! note
    モデルにランダムに音声の音色を選ばせる場合、このステップをスキップできます。

!!! warning "将来のバージョンに関する警告"
    元のパス（tools/vqgan/infernce.py）からアクセスできるインターフェースは残していますが、このインターフェースは将来のいくつかのバージョンで削除される可能性があります。お早めにコードを変更してください。

```bash
python fish_speech/models/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

`fake.npy`ファイルが生成されるはずです。

### 2. テキストからセマンティックトークンを生成する：

!!! warning "将来のバージョンに関する警告"
    元のパス（tools/llama/generate.py）からアクセスできるインターフェースは残していますが、このインターフェースは将来のいくつかのバージョンで削除される可能性があります。お早めにコードを変更してください。

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "変換したいテキスト" \
    --prompt-text "参照テキスト" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5" \
    --num-samples 2 \
    --compile
```

このコマンドは、作業ディレクトリに`codes_N`ファイルを作成します。ここで、N は 0 から始まる整数です。

!!! note
    `--compile`を使用して CUDA カーネルを融合し、より高速な推論を実現することができます（約 30 トークン/秒 -> 約 500 トークン/秒）。
    それに対応して、加速を使用しない場合は、`--compile`パラメータをコメントアウトできます。

!!! info
    bf16 をサポートしていない GPU の場合、`--half`パラメータを使用する必要があるかもしれません。

### 3. セマンティックトークンから音声を生成する：

#### VQGAN デコーダー

!!! warning "将来のバージョンに関する警告"
    元のパス（tools/vqgan/infernce.py）からアクセスできるインターフェースは残していますが、このインターフェースは将来のいくつかのバージョンで削除される可能性があります。お早めにコードを変更してください。

```bash
python fish_speech/models/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

## HTTP API 推論

推論のための HTTP API を提供しています。次のコマンドを使用してサーバーを起動できます：

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/fish-speech-1.5" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

> 推論を高速化したい場合は、`--compile` パラメータを追加できます。

その後、`http://127.0.0.1:8080/`で API を表示およびテストできます。

以下は、`tools/api_client.py` を使用してリクエストを送信する例です。

```bash
python -m tools.api_client \
    --text "入力するテキスト" \
    --reference_audio "参照音声へのパス" \
    --reference_text "参照音声テキスト" \
    --streaming True
```

上記のコマンドは、参照音声の情報に基づいて必要な音声を合成し、ストリーミング方式で返すことを示しています。

!!! info
    使用可能なパラメータの詳細については、コマンド` python -m tools.api_client -h `を使用してください

## WebUI 推論

次のコマンドを使用して WebUI を起動できます：

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.5" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```
> 推論を高速化したい場合は、`--compile` パラメータを追加できます。

!!! note
    ラベルファイルと参照音声ファイルをメインディレクトリの `references` フォルダ（自分で作成する必要があります）に事前に保存しておくことで、WebUI で直接呼び出すことができます。

!!! note
    Gradio 環境変数（`GRADIO_SHARE`、`GRADIO_SERVER_PORT`、`GRADIO_SERVER_NAME`など）を使用して WebUI を構成できます。

お楽しみください！
