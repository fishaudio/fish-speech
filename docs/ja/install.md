## システム要件

- GPU メモリ：12GB（推論）
- システム：Linux、WSL

## セットアップ

まず、音声処理に使用される pyaudio と sox をインストールする必要があります。

``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

pip install -e .
```

### UV

```bash
uv sync --python 3.12
```

### Intel Arc XPU 対応

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

conda install libstdcxx -c conda-forge

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

pip install -e .
```

!!! warning
    `compile` オプションは Windows と macOS でサポートされていません。compile で実行したい場合は、triton を自分でインストールする必要があります。
