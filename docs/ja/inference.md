# 推論

推論は、コマンドライン、HTTP API、およびWeb UIをサポートしています。

!!! note
    全体として、推論は次のいくつかの部分で構成されています：

    1. VQGANを使用して、与えられた約10秒の音声をエンコードします。
    2. エンコードされたセマンティックトークンと対応するテキストを例として言語モデルに入力します。
    3. 新しいテキストが与えられた場合、モデルに対応するセマンティックトークンを生成させます。
    4. 生成されたセマンティックトークンをVITS / VQGANに入力してデコードし、対応する音声を生成します。

## コマンドライン推論

必要な`vqgan`および`llama`モデルをHugging Faceリポジトリからダウンロードします。
    
```bash
huggingface-cli download fishaudio/fish-speech-1.2 --local-dir checkpoints/fish-speech-1.2
```

### 1. 音声からプロンプトを生成する：

!!! note
    モデルにランダムに音声の音色を選ばせる場合、このステップをスキップできます。

```bash
python tools/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
```
`fake.npy`ファイルが生成されるはずです。

### 2. テキストからセマンティックトークンを生成する：
```bash
python tools/llama/generate.py \
    --text "変換したいテキスト" \
    --prompt-text "参照テキスト" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.2" \
    --num-samples 2 \
    --compile
```

このコマンドは、作業ディレクトリに`codes_N`ファイルを作成します。ここで、Nは0から始まる整数です。

!!! note
    `--compile`を使用してCUDAカーネルを融合し、より高速な推論を実現することができます（約30トークン/秒 -> 約500トークン/秒）。
    それに対応して、加速を使用しない場合は、`--compile`パラメータをコメントアウトできます。

!!! info
    bf16をサポートしていないGPUの場合、`--half`パラメータを使用する必要があるかもしれません。

!!! warning
    自分で微調整したモデルを使用している場合、発音の安定性を確保するために`--speaker`パラメータを必ず持たせてください。

### 3. セマンティックトークンから音声を生成する：

#### VQGANデコーダー（推奨されません）
```bash
python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth"
```

## HTTP API推論

推論のためのHTTP APIを提供しています。次のコマンドを使用してサーバーを起動できます：

```bash
python -m tools.api \
    --listen 0.0.0.0:8000 \
    --llama-checkpoint-path "checkpoints/fish-speech-1.2" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --decoder-config-name firefly_gan_vq

推論を高速化したい場合は、--compileパラメータを追加できます。

その後、http://127.0.0.1:8000/でAPIを表示およびテストできます。

## WebUI推論

次のコマンドを使用してWebUIを起動できます：

```bash
python -m tools.webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.2" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

!!! note
    Gradio環境変数（`GRADIO_SHARE`、`GRADIO_SERVER_PORT`、`GRADIO_SERVER_NAME`など）を使用してWebUIを構成できます。

お楽しみください！
