# Fish Speech の紹介

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>

!!! warning
    私たちは、コードベースの違法な使用について一切の責任を負いません。お住まいの地域の DMCA（デジタルミレニアム著作権法）およびその他の関連法を参照してください。 <br/>
    このコードベースとモデルは、CC-BY-NC-SA-4.0 ライセンス下でリリースされています。

<p align="center">
   <img src="../assets/figs/diagram.png" width="75%">
</p>

## 要件

- GPU メモリ: 4GB（推論用）、8GB（ファインチューニング用）
- システム: Linux、Windows

## Windowsセットアップ

!!! info "注意"
    Windowsの専門ユーザー以外の方には、GUIを使用してプロジェクトを実行することを強くお勧めします。[GUIはこちら](https://github.com/AnyaCoder/fish-speech-gui).

プロフェッショナルなWindowsユーザーは、WSL2またはDockerを使用してコードベースを実行することを検討してください。

```bash
# Python 3.10の仮想環境を作成（virtualenvも使用可能）
conda create -n fish-speech python=3.10
conda activate fish-speech

# PyTorchをインストール
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# fish-speechをインストール
pip3 install -e .

# (アクセラレーションを有効にする) triton-windowsをインストール
pip install https://github.com/AnyaCoder/fish-speech/releases/download/v0.1.0/triton_windows-0.1.0-py3-none-any.whl
```

## Linux セットアップ

詳細については、[pyproject.toml](../../pyproject.toml)  を参照してください。
```bash
# python 3.10の仮想環境を作成します。virtualenvも使用できます。
conda create -n fish-speech python=3.10
conda activate fish-speech

# pytorchをインストールします。
pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# (Ubuntu / Debianユーザー) sox + ffmpegをインストールします。
apt install libsox-dev ffmpeg

# (Ubuntu / Debianユーザー) pyaudio をインストールします。
apt install build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0
    
# fish-speechをインストールします。
pip3 install -e .[stable]

```

## macos setup

推論をMPS上で行う場合は、`--device mps`フラグを追加してください。
推論速度の比較は[こちらのPR](https://github.com/fishaudio/fish-speech/pull/461#issuecomment-2284277772)を参考にしてください。

!!! warning
    AppleSiliconのデバイスでは、compileオプションに正式に対応していませんので、推論速度が向上する保証はありません。

```bash
# create a python 3.10 virtual environment, you can also use virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech
# install pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
# install fish-speech
pip install -e .[stable]
```

## Docker セットアップ

1. NVIDIA Container Toolkit のインストール：

    Docker で GPU を使用してモデルのトレーニングと推論を行うには、NVIDIA Container Toolkit をインストールする必要があります：

    Ubuntu ユーザーの場合：

    ```bash
    # リポジトリの追加
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    # nvidia-container-toolkit のインストール
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    # Docker サービスの再起動
    sudo systemctl restart docker
    ```

    他の Linux ディストリビューションを使用している場合は、以下のインストールガイドを参照してください：[NVIDIA Container Toolkit Install-guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。

2. fish-speech イメージのプルと実行

    ```shell
    # イメージのプル
    docker pull fishaudio/fish-speech:latest-dev
    # イメージの実行
    docker run -it \
        --name fish-speech \
        --gpus all \
        -p 7860:7860 \
        fishaudio/fish-speech:latest-dev \
        zsh
    # 他のポートを使用する場合は、-p パラメータを YourPort:7860 に変更してください
    ```

3. モデルの依存関係のダウンロード

    Docker コンテナ内のターミナルにいることを確認し、huggingface リポジトリから必要な `vqgan` と `llama` モデルをダウンロードします。

    ```bash
    huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
    ```

4. 環境変数の設定と WebUI へのアクセス

    Docker コンテナ内のターミナルで、`export GRADIO_SERVER_NAME="0.0.0.0"` と入力して、外部から Docker 内の gradio サービスにアクセスできるようにします。
    次に、Docker コンテナ内のターミナルで `python tools/run_webui.py` と入力して WebUI サービスを起動します。

    WSL または MacOS の場合は、[http://localhost:7860](http://localhost:7860) にアクセスして WebUI インターフェースを開くことができます。

    サーバーにデプロイしている場合は、localhost をサーバーの IP に置き換えてください。

## 変更履歴

- 2024/12/03: Fish-Speech を 1.5にアップデートし、より多くの言語をサポートするようになりました。オープンソース領域ではSOTA（最先端）となっています。
- 2024/09/10: Fish-Speech を Ver.1.4 に更新し、データセットのサイズを増加させ、quantizer n_groups を 4 から 8 に変更しました。
- 2024/07/02: Fish-Speech を Ver.1.2 に更新し、VITS デコーダーを削除し、ゼロショット能力を大幅に強化しました。
- 2024/05/10: Fish-Speech を Ver.1.1 に更新し、VITS デコーダーを実装して WER を減少させ、音色の類似性を向上させました。
- 2024/04/22: Fish-Speech Ver.1.0 を完成させ、VQGAN および LLAMA モデルを大幅に修正しました。
- 2023/12/28: `lora`微調整サポートを追加しました。
- 2023/12/27: `gradient checkpointing`、`causual sampling`、および`flash-attn`サポートを追加しました。
- 2023/12/19: webui および HTTP API を更新しました。
- 2023/12/18: 微調整ドキュメントおよび関連例を更新しました。
- 2023/12/17: `text2semantic`モデルを更新し、自由音素モードをサポートしました。
- 2023/12/13: ベータ版をリリースし、VQGAN モデルおよび LLAMA に基づく言語モデル（音素のみサポート）を含みます。

## 謝辞

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
