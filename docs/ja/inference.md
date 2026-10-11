# 推論

Fish Audio は 2 つのモデルを提供しています：**S2-Pro** と **Drama 3**。両者は同じコマンドラインと API サーバーを共有し、モデル種別は重みディレクトリの `config.json` から自動判定されるため、以下の各節は「タスク → 各モデルのコマンド」の順で構成しています。

まず[インストール](install.md)を完了し、[モデルの重み](install.md#重みのダウンロード)をダウンロードしてください。

!!! warning "Drama 3 の実行制限"
    Drama 3 は NVIDIA GPU でのみ動作します。CPU、Apple MPS、AMD ROCm には非対応で、`--compile` も使用できません。GPU メモリは[必要条件](install.md#必要条件)を参照してください。

!!! note
    V100 など SM80 より前の GPU には bfloat16 がないため、Drama 3 は float16 に切り替わります（残差ストリームは fp32 のまま）。この経路は実機では未検証です。無音・ノイズ・"non-finite logits" エラーが出た場合は issue を報告してください。

## コマンドライン推論

### S2-Pro

S2-Pro は「VQ トークン → Semantic トークン → 音声」の 3 ステップです。

!!! note
    モデルに音色をランダムに選ばせる場合は、ステップ 1 をスキップできます。

**1. リファレンスオーディオから VQ トークンを取得する**

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

`fake.npy` と `fake.wav` が得られます。

**2. テキストから Semantic トークンを生成する**

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "変換したいテキスト" \
    --prompt-text "リファレンステキスト" \
    --prompt-tokens "fake.npy" \
    # --compile
```

作業ディレクトリに `codes_N` ファイルが作成されます（N は 0 から始まる整数）。

!!! note
    `--compile` で CUDA カーネルを融合し、推論を高速化できます。ただし弊社の sglang による推論高速化の利用を推奨します。
    高速化を使わない場合は、`--compile` パラメータをコメントアウトしてください。

!!! info
    bf16 非対応の GPU では `--half` パラメータが必要な場合があります。

**3. セマンティックトークンから音声を生成する**

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

`fake.wav` ファイルが得られます。

### Drama 3

Drama 3 は 1 コマンドで完結し、`--output` がデコード済みの音声を直接書き出します：

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/drama-3 \
    --text "変換したいテキスト" \
    --prompt-audio "reference.wav" \
    --prompt-text "リファレンステキスト" \
    --output "output.wav" \
    --device cuda:0
```

古い GPU では `--device cuda:0,cuda:1,cuda:2,cuda:3` のように複数枚を指定できます。最後の 1 枚が音声デコーダと codec を担当します。`--device cuda` のみの場合は、可視な GPU のうち最大 4 枚を使用します。

## API サーバー

### S2-Pro の起動

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq
```

### Drama 3 の起動

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/drama-3 \
    --device cuda:0
```

!!! note
    Drama 3 は `--compile` に非対応です。古い GPU では `--device` に複数枚を列挙してください。最後の 1 枚が音声デコーダと codec を担当します。

両モデルで共通の起動オプション：

- `--compile`：`torch.compile` による最適化を有効化（S2-Pro のみ）
- `--half`：fp16 モードを使用
- `--api-key`：bearer トークン認証を要求
- `--workers`：ワーカープロセス数を設定
- `--max-text-length`：1 リクエストのテキスト長を制限（0 は無制限）

ヘルスチェック：

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

`{"status":"ok"}` が返れば正常です。

### API エンドポイント

| メソッド | パス | 説明 |
| --- | --- | --- |
| `POST` | `/v1/tts` | テキスト音声変換（リクエストボディは msgpack） |
| `GET` | `/v1/tts/defaults` | 読み込まれたモデルのサンプリング既定値 |
| `GET` | `/v1/health` | ヘルスチェック |
| `POST` | `/v1/vqgan/encode` | リファレンス音声 → VQ トークン |
| `POST` | `/v1/vqgan/decode` | VQ トークン → 音声 |
| `POST` | `/v1/references/add` | リファレンス音声を保存（フォーム項目 `id`、`audio`（ファイル）、`text`） |
| `GET` | `/v1/references/list` | 保存済み音声の一覧 |
| `POST` | `/v1/references/update` | リネーム（フォーム項目 `old_reference_id`、`new_reference_id`） |
| `DELETE` | `/v1/references/delete` | 削除（フォーム項目 `reference_id`） |
| `GET` | `/ui` | WebUI（Docker イメージには組み込み済み。ソース実行では要先ビルド） |

保存した音声は `--reference_id` で参照できます（音声フィールドには MIME タイプが必要です。指定しないとサーバーが認識できません）：

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
    -F id=my-speaker \
    -F "audio=@reference.wav;type=audio/wav" \
    -F text="リファレンス音声の書き起こし"

python tools/api_client.py --url http://127.0.0.1:8080/v1/tts \
    --text "Hello" --reference_id my-speaker --output demo
```

### リクエストの送信

どのモデルが使われるかはサーバー起動時に読み込んだ重みで決まり、リクエスト側にモデル指定フィールドはありません。

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "Hello from Fish Speech" \
    --output demo
```

主なパラメータ：

| パラメータ | 説明 |
| --- | --- |
| `--reference_audio` / `--reference_text` | リファレンス音声とその書き起こし。両者の数は一致させる必要があります |
| `--reference_speaker` | 各リファレンス音声の話者 ID（0–15）。`<\|speaker:N\|>` プレフィックスとして付与（Drama 3 の主な使い方） |
| `--reference_id` | 保存済みのリファレンス音声を使用（例：`my-speaker`） |
| `--format` | 出力形式：`wav`、`pcm`、`mp3`、`opus`（既定値 `wav`） |
| `--streaming` | 音声をストリームで返す |
| `--single_chunk` | 分割せず全体を一度に生成（Drama 3 のみ） |
| `--chunk_length` | チャンク長、既定値 200（S2-Pro のみ） |
| `--max_new_tokens` | チャンクあたりの最大フレーム数。未指定はサーバー既定の 1024、0 はモデルの上限 |
| `--temperature` / `--top_p` | サンプリング。未指定はモデルの既定値（0.7 / 0.7） |
| `--seed` | 乱数シードを固定（音色の固定には使えません） |
| `--api_key` | サーバーが `--api-key` を有効にしている場合は必須 |
| `--no-play` | 生成後に音声を再生しない（既定は再生。オーディオ機器のない環境では指定） |

!!! note
    `repetition_penalty` は互換性のために残されていますが、S2-Pro と Drama 3 はどちらも使用せず、内蔵の繰り返し認識サンプリングを使います。

!!! tip
    Docker でサーバーを起動する場合は [Dockerセットアップ](install.md#dockerセットアップ)を参照してください（server イメージには WebUI が組み込み済みで、起動後 `http://localhost:8080/ui` を開けます）。

## モデルの違い

| 項目 | S2-Pro | Drama 3 |
| --- | --- | --- |
| ハードウェア | GPU / CPU | NVIDIA GPU のみ、`--compile` 非対応 |
| 長文 | `chunk_length` で分割 | 約 300 バイト（UTF-8）・最大 3 話者のチャンクに自動分割し、可能な限り文末で切断。最初と最新のチャンクをコンテキストとして再入力 |
| 1 回あたりの長さ上限 | コンテキスト長 | チャンクごとに 1024 フレーム（約 47 秒）。`single_chunk` 時は 2048 フレーム（約 95 秒）。いずれも「10,240 フレーム − プロンプト長」が上限 |
| 話者 | リファレンスの順序が役割を決める | `<\|speaker:0\|>`–`<\|speaker:15\|>` タグが役割を決め、リファレンスの順序は無関係 |
| `chunk_length` | 有効 | 無効 |
| サンプリング既定値 | temperature 0.7、top-p 0.7、top-k 30 | 同じ |
| `repetition_penalty` | 無視 | 無視 |

Drama 3 には環境変数が 2 つあります：`FISH_DRAMA3_FP8_EXPERTS=0` または `1` で逆量子化 / FP8 の専門家を強制、`FISH_DRAMA3_CUDA_GRAPH=0` で CUDA graph デコードを無効化します。

## WebUI 推論

### Awesome WebUI

Awesome WebUI は TypeScript で構築されたモダンな Web インターフェースで、より豊富な機能と優れた操作性を提供します。

server Docker イメージにはビルド済みの WebUI が組み込まれており、コンテナ起動後に `http://localhost:8080/ui` を開くだけで利用できます。以下の手順はソースから実行する場合のものです。

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

ビルド後、プロジェクトルートに戻って API サーバーを起動します（`--compile` は S2-Pro のみ。Drama 3 を読み込む場合は外してください）：

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**アクセス：**

サーバー起動後、ブラウザで以下のアドレスにアクセスしてください：
`http://localhost:8888/ui`
