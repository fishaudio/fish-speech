<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | **한국어** <br>

<a href="https://www.producthunt.com/posts/fish-speech-1-4?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-fish&#0045;speech&#0045;1&#0045;4" target="_blank">
    <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=488440&theme=light" alt="Fish&#0032;Speech&#0032;1&#0046;4 - Open&#0045;Source&#0032;Multilingual&#0032;Text&#0045;to&#0045;Speech&#0032;with&#0032;Voice&#0032;Cloning | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" />
</a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
<br>
</div>
<br>

<div align="center">
    <img src="https://count.getloli.com/get/@fish-speech?theme=asoul" /><br>
</div>
<br>

<div align="center">
    <a target="_blank" href="https://discord.gg/Es5qTB9BcN">
        <img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
    </a>
    <a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
        <img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
    </a>
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
</div>

이 코드 저장소는 Apache 2.0 라이선스 하에 배포되며, 모델은 CC-BY-NC-SA-4.0 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](../LICENSE)를 참조하십시오.

---

## 기능

1. **Zero-shot & Few-shot TTS:** 10초에서 30초의 음성 샘플을 입력하여 고품질의 TTS 출력을 생성합니다. **자세한 가이드는 [모범 사례](https://docs.fish.audio/text-to-speech/voice-clone-best-practices)를 참조하시길 바랍니다.**

2. **다국어 및 교차 언어 지원:** 다국어 걱정 없이, 텍스트를 입력창에 복사하여 붙여넣기만 하면 됩니다. 현재 영어, 일본어, 한국어, 중국어, 프랑스어, 독일어, 아랍어, 스페인어를 지원합니다.

3. **음소 의존성 제거:** 이 모델은 강력한 일반화 능력을 가지고 있으며, TTS가 음소에 의존하지 않습니다. 모든 언어 스크립트 텍스트를 손쉽게 처리할 수 있습니다.

4. **높은 정확도:** 영어 텍스트 기준 5분 기준에서 단, 2%의 문자 오류율(CER)과 단어 오류율(WER)을 달성합니다.

5. **빠른 속도:** fish-tech 가속을 통해 실시간 인자(RTF)는 Nvidia RTX 4060 노트북에서는 약 1:5, Nvidia RTX 4090에서는 1:15입니다.

6. **웹 UI 추론:** Chrome, Firefox, Edge 등 다양한 브라우저에서 호환되는 Gradio 기반의 사용하기 쉬운 웹 UI를 제공합니다.

7. **GUI 추론:** PyQt6 그래픽 인터페이스를 제공하여 API 서버와 원활하게 작동합니다. Linux, Windows 및 macOS를 지원합니다. [GUI 참조](https://github.com/AnyaCoder/fish-speech-gui).

8. **배포 친화적:** Linux, Windows, macOS에서 네이티브로 지원되는 추론 서버를 쉽게 설정할 수 있어 속도 손실을 최소화합니다.

## 면책 조항

이 코드베이스의 불법적 사용에 대해 어떠한 책임도 지지 않습니다. DMCA 및 관련 법률에 대한 로컬 법률을 참조하십시오.

## 온라인 데모

[Fish Audio](https://fish.audio)

## 로컬 추론을 위한 빠른 시작

[inference.ipynb](/inference.ipynb)

## 영상

#### V1.5 데모 영상: [Watch the video on X (Twitter).](https://x.com/FishAudio/status/1864370933496205728)

## 문서

- [English](https://speech.fish.audio/)
- [中文](https://speech.fish.audio/zh/)
- [日本語](https://speech.fish.audio/ja/)
- [Portuguese (Brazil)](https://speech.fish.audio/pt/)
- [한국어](https://speech.fish.audio/ko/)

## Samples (2024/10/02 V1.4)

- [English](https://speech.fish.audio/samples/)
- [中文](https://speech.fish.audio/zh/samples/)
- [日本語](https://speech.fish.audio/ja/samples/)
- [Portuguese (Brazil)](https://speech.fish.audio/pt/samples/)
- [한국어](https://speech.fish.audio/ko/samples/)

## Credits

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## Sponsor

<div>
  <a href="https://6block.com/">
    <img src="https://avatars.githubusercontent.com/u/60573493" width="100" height="100" alt="6Block Avatar"/>
  </a>
  <br>
  <a href="https://6block.com/">데이터 처리 후원: 6Block</a>
</div>
