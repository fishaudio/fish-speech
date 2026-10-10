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

## S3-drama

S3-drama は S2-Pro と同じコマンドラインと API サーバーで動作します。モデルの種類はチェックポイントの `config.json` から自動的に判別されます。

!!! warning
    S3-drama は NVIDIA GPU でのみ動作します。CPU、Apple MPS、AMD ROCm には対応しておらず、`--compile` も使用できません。

| GPU | 動作モード | GPU メモリ |
| --- | --- | --- |
| SM89 以降（L40、H100、H20） | MoE エキスパートは FP8 のまま、CUDA グラフでデコード、GPU 1 枚 | 約 41 GB |
| それ以前の GPU（V100、A100） | MoE エキスパートを逆量子化し、複数の GPU に分散 | 合計約 70 GB |

!!! note
    V100 など SM80 より前の GPU は bfloat16 に対応していないため、S3 は自動的に float16（残差ストリームは fp32 のまま）で動作します。この経路は実機ではまだテストされていません。無音やノイズが出る場合や "non-finite logits" エラーが出る場合は、issue を作成してください。

### 重みのダウンロード

```bash
hf download fishaudio/s3-drama-it-fp8 --local-dir checkpoints/s3-drama-fp8
```

チェックポイントのディレクトリに `codec.pth` がない場合は、`checkpoints/s2-pro/codec.pth` の S2-Pro コーデックが使われます。

### コマンドライン推論

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/s3-drama-fp8 \
    --text "変換したいテキスト" \
    --prompt-audio "reference.wav" \
    --prompt-text "リファレンステキスト" \
    --output "output.wav" \
    --device cuda:0
```

`--output` を指定すると、デコードした音声がそのまま書き出されます。それ以前の GPU では、`--device cuda:0,cuda:1,cuda:2,cuda:3` のように複数のデバイスを指定してください。最後のデバイスがオーディオデコーダーとコーデックを担当します。`--device cuda` とだけ指定した場合は、認識されている GPU のうち先頭から最大 4 枚を使用します。

### API サーバー

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s3-drama-fp8 \
    --device cuda:0
```

Docker では、server イメージで S3 のチェックポイントを読み込みます。SM89 以降の GPU では以下のように GPU を 1 枚だけ公開し、それ以前の GPU では `--gpus all` を使用してください。

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s3-drama-fp8/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### S2-Pro との違い

- 長いテキストは約 300 バイト（UTF-8）、1 チャンクあたり最大 3 話者のチャンクに分割され（できるだけ文末で区切ります）、順番に生成されます。最初のチャンクと直前のチャンクがコンテキストとしてモデルに戻され、声の一貫性を保ちます。デフォルトでは 1 チャンクあたり最大 600 フレーム（約 28 秒）を生成します。
- `single_chunk`（コマンドラインでは `--single-chunk`）を指定すると、テキスト全体を一度に生成します。デフォルトの上限は 2048 フレーム（約 95 秒）です。どちらのモードでも、これより大きい `max_new_tokens` は 10,240 フレームからプロンプト長を引いた値に制限されます。
- 話者タグは `<|speaker:0|>` から `<|speaker:15|>` まで使用でき、タグのないテキストは speaker 0 として扱われます。話者ごとに別の声を使うには、各参照音声に `speaker` フィールド（`tools/api_client.py` では `--reference_speaker`）を指定してください。先頭に話者タグのない書き起こしにはそのタグが付けられます。書き起こしの先頭に直接タグ（例：`<|speaker:1|>`）を書くこともできます。参照音声の順番では話者は決まりません。
- `chunk_length` と `repetition_penalty` は効果がありません。S3 には繰り返しを考慮したサンプリングが組み込まれています。デフォルトのサンプリング設定は temperature 1.0、top-p 0.9、top-k 30 です。
- `streaming` を有効にすると、音声はチャンクごとに返されます。
- `FISH_S3_FP8_EXPERTS=0` または `1` で逆量子化エキスパートまたは FP8 エキスパートを強制でき、`FISH_S3_CUDA_GRAPH=0` で CUDA グラフによるデコードを無効にできます。

## WebUI 推論

### Awesome WebUI

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
