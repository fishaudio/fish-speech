# Introdução

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
    Não assumimos nenhuma responsabilidade pelo uso ilegal da base de código. Consulte as leis locais sobre DMCA (Digital Millennium Copyright Act) e outras leis relevantes em sua área. <br/>
    Esta base de código é lançada sob a licença Apache 2.0 e todos os modelos são lançados sob a licença CC-BY-NC-SA-4.0.

## Requisitos

- Memória GPU: 12GB (Inferência)
- Sistema: Linux, Windows

## Configuração

Primeiro, precisamos criar um ambiente conda para instalar os pacotes.

```bash

conda create -n fish-speech python=3.12
conda activate fish-speech

pip install sudo apt-get install portaudio19-dev # Para pyaudio
pip install -e . # Isso baixará todos os pacotes restantes.

apt install libsox-dev ffmpeg # Se necessário.
```

!!! warning
    A opção `compile` não é suportada no Windows e macOS, se você quiser executar com compile, precisa instalar o trition por conta própria.

## Agradecimentos

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
