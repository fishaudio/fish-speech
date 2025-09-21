## 必要条件

- GPUメモリ: 12GB (推論時)
- システム: Linux, WSL

## システムセットアップ

OpenAudioは複数のインストール方法をサポートしています。ご自身の開発環境に最も適した方法をお選びください。

**前提条件**: 音声処理のためのシステム依存関係をインストールします:
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# GPU版のインストール (CUDAバージョンを選択: cu126, cu128, cu129)
pip install -e .[cu129]

# CPU版のみのインストール
pip install -e .[cpu]

# デフォルトインストール (PyTorchのデフォルトインデックスを使用)
pip install -e .
```

### UV

UVはより高速な依存関係の解決とインストールを実現します:

```bash
# GPU版のインストール (CUDAバージョンを選択: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# CPU版のみのインストール
uv sync --python 3.12 --extra cpu
```
### Intel Arc XPU サポート

Intel Arc GPUユーザーは、以下の手順でXPUサポートをインストールしてください:

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
    `compile`オプションはWindowsとmacOSではサポートされていません。コンパイルを有効にして実行したい場合は、ご自身でTritonをインストールする必要があります。


## Dockerセットアップ

OpenAudio S1シリーズモデルは、さまざまなニーズに応えるため複数のDockerデプロイメントオプションを提供しています。Docker Hubのビルド済みイメージを使用するか、Docker Composeでローカルビルドするか、手動でカスタムイメージをビルドすることができます。

WebUIとAPIサーバーの両方について、GPU（デフォルトはCUDA 12.6）版とCPU版のDockerイメージを提供しています。Docker Hubのビルド済みイメージを使用するか、Docker Composeでローカルビルドするか、手動でカスタムイメージをビルドするかを選択できます。ローカルでビルドする場合は、以下の手順に従ってください。ビルド済みイメージを使用するだけの場合は、[推論ガイド](inference.md)を直接参照してください。

### 前提条件

- DockerとDocker Composeがインストール済みであること
- NVIDIA Dockerランタイムがインストール済みであること（GPUサポート用）
- CUDAによる推論のために、少なくとも12GBのGPUメモリがあること

### Docker Composeの使用

開発やカスタマイズのために、Docker Composeを使用してローカルでビルド・実行できます:

```bash
# まず、リポジトリをクローンします
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# CUDAでWebUIを起動
docker compose --profile webui up

# コンパイル最適化を有効にしてWebUIを起動
COMPILE=1 docker compose --profile webui up

# APIサーバーを起動
docker compose --profile server up

# コンパイル最適化を有効にしてAPIサーバーを起動
COMPILE=1 docker compose --profile server up

# CPUのみでのデプロイ
BACKEND=cpu docker compose --profile webui up
```

#### Docker Compose 環境変数

環境変数を使用してデプロイメントをカスタマイズできます:

```bash
# .env ファイルの例
BACKEND=cuda              # または cpu
COMPILE=1                 # コンパイル最適化を有効化
GRADIO_PORT=7860         # WebUIのポート
API_PORT=8080            # APIサーバーのポート
UV_VERSION=0.8.15        # UVパッケージマネージャーのバージョン
```

このコマンドはイメージをビルドし、コンテナを実行します。WebUIには`http://localhost:7860`で、APIサーバーには`http://localhost:8080`でアクセスできます。

### 手動でのDockerビルド

ビルドプロセスをカスタマイズしたい上級者向け:

```bash
# CUDAサポート付きのWebUIイメージをビルド
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

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
    --target webui \
    -t fish-speech-webui:cpu .

# 開発用イメージをビルド
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### ビルド引数

- `BACKEND`: `cuda` または `cpu` (デフォルト: `cuda`)
- `CUDA_VER`: CUDAバージョン (デフォルト: `12.6.0`)
- `UV_EXTRA`: CUDA用のUV追加パッケージ (デフォルト: `cu126`)
- `UBUNTU_VER`: Ubuntuバージョン (デフォルト: `24.04`)
- `PY_VER`: Pythonバージョン (デフォルト: `3.12`)

### ボリュームマウント

どちらの方法でも、以下のディレクトリをマウントする必要があります:

- `./checkpoints:/app/checkpoints` - モデルの重みファイル用ディレクトリ
- `./references:/app/references` - 参照音声ファイル用ディレクトリ

### 環境変数

- `COMPILE=1` - `torch.compile`を有効にして推論を高速化（約10倍）
- `GRADIO_SERVER_NAME=0.0.0.0` - WebUIサーバーのホスト
- `GRADIO_SERVER_PORT=7860` - WebUIサーバーのポート
- `API_SERVER_NAME=0.0.0.0` - APIサーバーのホスト
- `API_SERVER_PORT=8080` - APIサーバーのポート

!!! note
    Dockerコンテナは、モデルの重みが`/app/checkpoints`にマウントされることを想定しています。コンテナを起動する前に、必要なモデルの重みをダウンロードしてください。

!!! warning
    GPUサポートにはNVIDIA Dockerランタイムが必要です。CPUのみでデプロイする場合は、`--gpus all`フラグを削除し、CPU用のイメージを使用してください。
