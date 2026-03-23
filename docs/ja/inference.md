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

### 1. Gradio WebUI

互換性を維持するため、以前の Gradio WebUI も引き続き利用可能です。

```bash
python tools/run_webui.py # 加速が必要な場合は --compile
```

### 2. Awesome WebUI

Awesome WebUI は TypeScript で開発された、より豊富な機能と優れたユーザー体験を提供する最新の Web インターフェースです。

**WebUI のビルド：**

ローカルまたはサーバーに Node.js と npm がインストールされている必要があります。

1. `awesome_webui` ディレクトリに移動します：
   ```bash
   cd awesome_webui
   ```
2. 依存関係をインストールします：
   ```bash
   npm install
   ```
3. WebUI をビルドします：
   ```bash
   npm run build
   ```

**バックエンドサーバーの起動：**

WebUI のビルドが完了したら、プロジェクトのルートに戻り、API サーバーを起動します：

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**アクセス：**

サーバーが起動したら、ブラウザから以下のアドレスにアクセスして体験できます：
`http://localhost:8888/ui`
