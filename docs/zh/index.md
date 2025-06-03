# 简介

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
    我们不对代码库的任何非法使用承担责任。请参考您所在地区有关 DMCA（数字千年版权法）和其他相关法律的规定。<br/>
    此代码库在 Apache 2.0 许可证下发布，所有模型在 CC-BY-NC-SA-4.0 许可证下发布。

## 系统要求

- GPU 内存：12GB（推理）
- 系统：Linux、Windows

## 安装

首先，我们需要创建一个 conda 环境来安装包。

```bash

conda create -n fish-speech python=3.12
conda activate fish-speech

pip install sudo apt-get install portaudio19-dev # 用于 pyaudio
pip install -e . # 这将下载所有其余的包。

apt install libsox-dev ffmpeg # 如果需要的话。
```

!!! warning
    `compile` 选项在 Windows 和 macOS 上不受支持，如果您想使用 compile 运行，需要自己安装 trition。

## 致谢

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
