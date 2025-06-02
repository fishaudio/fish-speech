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
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

이 코드베이스는 Apache License 하에 릴리스되며, 모든 모델 가중치는 CC-BY-NC-SA-4.0 License 하에 릴리스됩니다. 자세한 내용은 [LICENSE](../LICENSE)를 참조하세요.

저희는 이름을 OpenAudio로 변경했다고 발표하게 되어 기쁩니다. 이는 완전히 새로운 Text-to-Speech 모델 시리즈가 될 것입니다.

데모는 [Fish Audio Playground](https://fish.audio)에서 사용할 수 있습니다.

블로그와 기술 보고서는 [OpenAudio 웹사이트](https://openaudio.com)를 방문하세요.

## 기능
### OpenAudio-S1 (Fish-Speech의 새 버전)

1. 이 모델은 fish-speech가 가지고 있던 **모든 기능**을 가지고 있습니다.

2. OpenAudio S1은 음성 합성을 향상시키기 위한 다양한 감정, 톤, 특별한 마커를 지원합니다:
   
      (angry) (sad) (disdainful) (excited) (surprised) (satisfied) (unhappy) (anxious) (hysterical) (delighted) (scared) (worried) (indifferent) (upset) (impatient) (nervous) (guilty) (scornful) (frustrated) (depressed) (panicked) (furious) (empathetic) (embarrassed) (reluctant) (disgusted) (keen) (moved) (proud) (relaxed) (grateful) (confident) (interested) (curious) (confused) (joyful) (disapproving) (negative) (denying) (astonished) (serious) (sarcastic) (conciliative) (comforting) (sincere) (sneering) (hesitating) (yielding) (painful) (awkward) (amused)

   또한 톤 마커도 지원합니다:

   (급한 톤) (외치기) (비명지르기) (속삭이기) (부드러운 톤)

    지원되는 몇 가지 특별한 마커가 있습니다:

    (웃음) (킥킥거림) (흐느낌) (큰 소리로 우는 것) (한숨) (헐떡거림) (신음) (군중 웃음) (배경 웃음) (관객 웃음)

    또한 **하, 하, 하**를 사용하여 제어할 수도 있으며, 여러분이 직접 탐험할 수 있는 많은 다른 경우들이 있습니다.

3. OpenAudio S1은 다음 크기를 포함합니다:
-   **S1 (4B, 독점):** 전체 크기 모델.
-   **S1-mini (0.5B, 오픈소스):** S1의 증류 버전.

    S1과 S1-mini 모두 온라인 인간 피드백 강화학습(RLHF)을 통합하고 있습니다.

4. 평가

    **Seed TTS 평가 메트릭 (영어, 자동 평가, OpenAI gpt-4o-transcribe 기반, Revai/pyannote-wespeaker-voxceleb-resnet34-LM을 사용한 화자 거리):**

    -   **S1:**
        -   WER (단어 오류율): **0.008**
        -   CER (문자 오류율): **0.004**
        -   거리: **0.332**
    -   **S1-mini:**
        -   WER (단어 오류율): **0.011**
        -   CER (문자 오류율): **0.005**
        -   거리: **0.380**
    

## 면책 조항

저희는 코드베이스의 불법적인 사용에 대해 어떠한 책임도 지지 않습니다. DMCA 및 기타 관련 법률에 대한 현지 법률을 참조하세요.

## 비디오

#### 계속될 예정입니다.

## 문서

- [환경 구축](en/install.md)
- [추론](en/inference.md)

현재 모델은 **파인튜닝을 지원하지 않는다**는 점에 유의해야 합니다.

## 크레딧

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## 기술 보고서 (V1.4)
```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}
```

