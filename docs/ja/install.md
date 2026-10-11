## 必要条件

| モデル | GPU メモリ | GPU |
| --- | --- | --- |
| Drama 3 | 32 GB 以上 | NVIDIA GPU のみ（CPU、Apple MPS、AMD ROCm は非対応、`--compile` も非対応） |
| S2-Pro | 24 GB 以上 | NVIDIA GPU または CPU |

- システム：Linux、WSL

!!! note
    Drama 3 の実測ピークメモリは H20 1 枚で約 41 GB です。余裕を持たせてください。V100 など SM80 より前の GPU には bfloat16 がないため、Drama 3 は float16 に切り替わります（残差ストリームは fp32 のまま）。

## システムセットアップ

Fish Audio Drama 3 と S2-Pro は同じインストール方法を共有しています。開発環境に最も適した方法を選んでください。

**前提条件**：音声処理に必要なシステム依存関係をインストールします：
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU版のインストール (CUDAバージョンを選択: cu126, cu128, cu129)
# cu126 または cu128 を使用する場合は、下記の "cu129" を両方置き換えてください。
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[cu129]"

# CPU版のみのインストール
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu]"

# デフォルトインストール (PyTorchのデフォルトインデックスを使用)
pip install -e .

# pyaudioのインストールでエラーが発生する場合は、以下のコマンドを試してください：
# conda install pyaudio
# その後、再度 pip install -e . を実行してください
```

### UV

UV は依存関係の解決とインストールがより高速です：

```bash
# GPU版のインストール (CUDAバージョンを選択: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# CPU版のみのインストール
uv sync --python 3.12 --extra cpu
```

### Intel Arc XPU サポート

Intel Arc GPU を使用する場合は、XPU サポート付きでインストールします（S2-Pro のみ。Drama 3 は NVIDIA GPU が必要です）：

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# 必要なC++標準ライブラリをインストール
conda install libstdcxx -c conda-forge

# Intel XPU対応のPyTorchをインストール
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Fish Speechのインストール
pip install -e .
```

!!! warning
    `compile` オプションは Windows と macOS ではサポートされていません。compile を使用する場合は、Triton を手動でインストールしてください。

## 重みのダウンロード

両モデルは同じコマンドラインと API サーバーを共有し、モデル種別は重みディレクトリの `config.json` から自動判定されます。同じ `checkpoints` ディレクトリに配置してください。

ダウンロードには Hugging Face の CLI が必要です：

```bash
pip install -U "huggingface_hub[cli]"
```

### Drama 3

```bash
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

重みディレクトリに `codec.pth` がない場合、`checkpoints/s2-pro/codec.pth` の S2-Pro codec が使われるため、S2-Pro も併せてダウンロードしてください。

### S2-Pro

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

重みのダウンロードが終わったら、[推論](inference.md)を参照してください。

## Dockerセットアップ

Drama 3 と S2-Pro は、用途に応じて複数の Docker デプロイ方法を提供しています。Docker Hub のビルド済みイメージを使う、Docker Compose でローカルビルドする、手動でカスタムイメージをビルドする、いずれも可能です。

GPU（デフォルト CUDA126）および CPU 向けの API サーバーイメージを提供しています。イメージには WebUI が組み込まれており、コンテナ起動後に `http://localhost:8080/ui` を開くだけで利用できます。GPU イメージは両モデルを実行でき、CPU イメージは S2-Pro のみ対応です。

### 前提条件

- Docker と Docker Compose がインストール済み
- （GPU の場合）NVIDIA Docker runtime がインストール済み（Drama 3 では必須）
- GPU メモリ：Drama 3 は 32 GB 以上、S2-Pro は 24 GB 以上を推奨

### ビルド済みイメージの実行

Drama 3：SM89 以上のカードでは以下のように 1 枚だけを公開します。古いカードでは `--gpus all` を使用してください：

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

S2-Pro：

```bash
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s2-pro \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### Docker Composeの使用

開発やカスタマイズを行う場合は、Docker Compose でローカルにビルドして実行することを推奨します：

```bash
# まず、リポジトリをクローンします
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# APIサーバーを起動（デフォルトは checkpoints/s2-pro）
docker compose --profile server up

# Drama 3 を読み込み、DEVICE で 1 枚のカードに固定（未設定なら最大 4 枚）
LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 DEVICE=cuda:0 docker compose --profile server up

# コンパイル最適化を有効にしてAPIサーバーを起動（S2-Pro のみ）
COMPILE=1 docker compose --profile server up

# CPUのみでのデプロイ（S2-Pro のみ）
BACKEND=cpu docker compose --profile server up
```

#### Docker Compose 環境変数

環境変数でデプロイをカスタマイズできます：

```bash
# .env ファイルの例
BACKEND=cuda              # または cpu
COMPILE=1                 # コンパイル最適化を有効化
API_PORT=8080             # APIサーバーのポート
UV_VERSION=0.8.15         # UV パッケージマネージャのバージョン
```

コマンドを実行するとイメージがビルドされ、コンテナが起動します。API サーバーは `http://localhost:8080`、WebUI は `http://localhost:8080/ui` でアクセスできます。

### 手動でのDockerビルド

より細かくビルドを制御したい場合：

```bash
# CUDAサポート付きのAPIサーバーイメージをビルド
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# CPUのみのイメージをビルド（マルチプラットフォーム対応）
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target server \
    -t fish-speech-server:cpu .

# 開発用イメージをビルド
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### ビルド引数

- `BACKEND`：`cuda` または `cpu`（デフォルト：`cuda`）
- `CUDA_VER`：CUDA バージョン（デフォルト：`12.6.0`）
- `UV_EXTRA`：CUDA 用の UV 拡張（デフォルト：`cu126`）
- `UBUNTU_VER`：Ubuntu バージョン（デフォルト：`24.04`）
- `PY_VER`：Python バージョン（デフォルト：`3.12`）

### ボリュームマウント

どちらの方法でも以下のディレクトリのマウントが必要です：

- `./checkpoints:/app/checkpoints` - モデル weights ディレクトリ
- `./references:/app/references` - リファレンス音声ディレクトリ

### 環境変数

- `COMPILE=1` - `torch.compile` を有効化（約 10 倍高速化）
- `LLAMA_CHECKPOINT_PATH` - メインモデルの重みパス（例：`checkpoints/drama-3`、`checkpoints/s2-pro`）
- `DECODER_CHECKPOINT_PATH` - codec の重みパス。空欄の場合はサーバーが重みディレクトリから解決します
- `DEVICE` - 推論デバイス（例：`cuda:0`）。Drama 3 は設定した 1 枚のみを使用し、未設定なら最大 4 枚
- `FISH_DRAMA3_FP8_EXPERTS=0` または `1` - Drama 3 の専門家を逆量子化 / FP8 に強制（デフォルトは GPU から自動選択）
- `FISH_DRAMA3_CUDA_GRAPH=0` - Drama 3 の CUDA graph デコードを無効化
- `API_SERVER_NAME=0.0.0.0` - API サーバーのホスト
- `API_SERVER_PORT=8080` - API サーバーのポート

!!! note
    Docker コンテナは `/app/checkpoints` からモデルの重みを読み込みます。コンテナを起動する前に、必要な重みをダウンロードしておいてください。

!!! warning
    GPU サポートには NVIDIA Docker runtime が必要です。CPU のみの場合は `--gpus all` を外し、CPU イメージを使用してください。
