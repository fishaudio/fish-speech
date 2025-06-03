# 紹介

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
    このコードベースの違法な使用について、当方は一切の責任を負いません。お住まいの地域のDMCA（デジタルミレニアム著作権法）およびその他の関連法規をご参照ください。<br/>
    このコードベースはApache 2.0ライセンスの下でリリースされ、すべてのモデルはCC-BY-NC-SA-4.0ライセンスの下でリリースされています。

## システム要件

- GPU メモリ：12GB（推論）
- システム：Linux、Windows

## セットアップ

まず、パッケージをインストールするためのconda環境を作成する必要があります。

```bash

conda create -n fish-speech python=3.12
conda activate fish-speech

pip install sudo apt-get install portaudio19-dev # pyaudio用
pip install -e . # これにより残りのパッケージがすべてダウンロードされます。

apt install libsox-dev ffmpeg # 必要に応じて。
```

!!! warning
    `compile`オプションはWindowsとmacOSでサポートされていません。compileで実行したい場合は、tritionを自分でインストールする必要があります。

## 謝辞

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
