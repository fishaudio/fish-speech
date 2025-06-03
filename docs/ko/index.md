# 소개

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
    코드베이스의 불법적인 사용에 대해서는 일체 책임을 지지 않습니다. 귀하의 지역의 DMCA(디지털 밀레니엄 저작권법) 및 기타 관련 법률을 참고하시기 바랍니다. <br/>
    이 코드베이스는 Apache 2.0 라이선스 하에 배포되며, 모든 모델은 CC-BY-NC-SA-4.0 라이선스 하에 배포됩니다.

## 시스템 요구사항

- GPU 메모리: 12GB (추론)
- 시스템: Linux, Windows

## 설치

먼저 패키지를 설치하기 위한 conda 환경을 만들어야 합니다.

```bash

conda create -n fish-speech python=3.12
conda activate fish-speech

pip install sudo apt-get install portaudio19-dev # pyaudio용
pip install -e . # 나머지 모든 패키지를 다운로드합니다.

apt install libsox-dev ffmpeg # 필요한 경우.
```

!!! warning
    `compile` 옵션은 Windows와 macOS에서 지원되지 않습니다. compile로 실행하려면 trition을 직접 설치해야 합니다.

## 감사의 말

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
