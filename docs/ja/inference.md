# 推論

Fish Audio S2 モデルは大きなビデオメモリを必要とします。推論には少なくとも 24GB の GPU を使用することをお勧めします。

## 重みのダウンロード

まず、モデルの重みをダウンロードする必要があります：

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## コマンドライン推論

!!! note
    モデルに音声をランダムに選択させる場合は、このステップをスキップできます。

### 1. リファレンスオーディオから VQ トークンを取得する

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

`fake.npy` と `fake.wav` が生成されるはずです。

### 2. テキストから Semantic トークンを生成する：

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "変換したいテキスト" \
    --prompt-text "リファレンステキスト" \
    --prompt-tokens "fake.npy" \
    # --compile
```

このコマンドは、作業ディレクトリに `codes_N` ファイルを作成します。ここで N は 0 から始まる整数です。

!!! note
    より高速な推論のために CUDA カーネルを融合する `--compile` を使用したい場合がありますが、私たちの sglang 推論加速最適化を使用することをお勧めします。
    同様に、加速を使用する予定がない場合は、`--compile` パラメータをコメントアウトしてください。

!!! info
    bf16 をサポートしていない GPU の場合、`--half` パラメータを使用する必要があるかもしれません。

### 3. セマンティックトークンから音声を生成する：

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

その後、`fake.wav` ファイルが取得できます。

## WebUI 推論

まもなく公開予定です。
