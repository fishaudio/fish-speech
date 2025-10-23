<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | **한국어** | [العربية](README.ar.md) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
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
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

<div align="center">
    <a target="_blank" href="https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2">
      <img alt="TTS-Arena2 Score" src="https://img.shields.io/badge/TTS_Arena2-Rank_%231-gold?style=flat-square&logo=trophy&logoColor=white">
    </a>
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/openaudio-s1-mini">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **라이선스 고지사항**  
> 이 코드베이스는 **Apache License** 하에 릴리스되며, 모든 모델 가중치는 **CC-BY-NC-SA-4.0 License** 하에 릴리스됩니다. 자세한 내용은 [LICENSE](../LICENSE)를 참조하세요.

> [!WARNING]
> **법적 면책조항**  
> 저희는 코드베이스의 불법적인 사용에 대해 어떠한 책임도 지지 않습니다. DMCA 및 기타 관련 법률에 대한 현지 법률을 참조하세요.

## 시작하기

여기는 Fish Speech의 공식 문서입니다. 지침을 따라 쉽게 시작하세요.

- [설치](https://speech.fish.audio/ko/install/)
- [파인튜닝](https://speech.fish.audio/ko/finetune/)
- [추론](https://speech.fish.audio/ko/inference/)
- [샘플](https://speech.fish.audio/examples)

## 🎉 발표

**OpenAudio**로의 리브랜딩을 발표하게 되어 기쁩니다. Fish-Speech의 기반 위에 구축된 혁신적인 새로운 고급 Text-to-Speech 모델 시리즈를 소개합니다.

이 시리즈의 첫 번째 모델인 **OpenAudio-S1**을 출시하게 되어 자랑스럽습니다. 품질, 성능, 기능에서 상당한 개선을 제공합니다.

OpenAudio-S1은 두 가지 버전으로 제공됩니다: **OpenAudio-S1**과 **OpenAudio-S1-mini**. 두 모델 모두 [Fish Audio Playground](https://fish.audio)(**OpenAudio-S1**용)와 [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini)(**OpenAudio-S1-mini**용)에서 사용할 수 있습니다.

블로그와 기술 보고서는 [OpenAudio 웹사이트](https://openaudio.com/blogs/s1)를 방문하세요.

## 주요 특징 ✨

### **뛰어난 TTS 품질**

우리는 Seed TTS Eval Metrics를 사용하여 모델 성능을 평가했으며, 결과에 따르면 OpenAudio S1은 영어 텍스트에서 **0.008 WER**과 **0.004 CER**을 달성하여 이전 모델들보다 상당히 우수한 성능을 보입니다. (영어, 자동 평가, OpenAI gpt-4o-transcribe 기반, Revai/pyannote-wespeaker-voxceleb-resnet34-LM을 사용한 화자 거리)

| 모델 | 단어 오류율 (WER) | 문자 오류율 (CER) | 화자 거리 |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **TTS-Arena2 최고 모델** 🏆

OpenAudio S1은 텍스트 음성 변환 평가의 벤치마크인 [TTS-Arena2](https://arena.speechcolab.org/)에서 **1위**를 달성했습니다:

<div align="center">
    <img src="../docs/assets/Elo.jpg" alt="TTS-Arena2 순위" style="width: 75%;" />
</div>

### **음성 제어**
OpenAudio S1은 **음성 합성을 향상시키기 위한 다양한 감정, 톤, 특별한 마커를 지원**합니다:

- **기본 감정**:
```
(화난) (슬픈) (흥분한) (놀란) (만족한) (기쁜) 
(무서워하는) (걱정하는) (속상한) (긴장한) (좌절한) (우울한)
(공감하는) (당황한) (역겨워하는) (감동한) (자랑스러운) (편안한)
(감사하는) (자신있는) (관심있는) (호기심있는) (혼란스러운) (즐거운)
```

- **고급 감정**:
```
(경멸하는) (불행한) (불안한) (히스테리한) (무관심한) 
(조급한) (죄책감있는) (냉소적인) (공황상태인) (분노한) (마지못한)
(열성적인) (반대하는) (부정적인) (부인하는) (놀란) (진지한)
(비꼬는) (달래는) (위로하는) (진심인) (비웃는)
(망설이는) (굴복하는) (고통스러운) (어색한) (재미있어하는)
```

- **톤 마커**:
```
(급한 톤) (외치기) (비명지르기) (속삭이기) (부드러운 톤)
```

- **특별한 오디오 효과**:
```
(웃음) (킥킥거림) (흐느낌) (큰 소리로 우는 것) (한숨) (헐떡거림)
(신음) (군중 웃음) (배경 웃음) (관객 웃음)
```

또한 **하, 하, 하**를 사용하여 제어할 수도 있으며, 여러분이 직접 탐험할 수 있는 많은 다른 경우들이 있습니다.

(현재 영어, 중국어, 일본어를 지원하며, 더 많은 언어가 곧 추가될 예정입니다!)

### **두 가지 유형의 모델**

| 모델 | 크기 | 가용성 | 특징 |
|-------|------|--------------|----------|
| **S1** | 4B 매개변수 | [fish.audio](https://fish.audio)에서 사용 가능 | 모든 기능을 갖춘 플래그십 모델 |
| **S1-mini** | 0.5B 매개변수 | 허깅페이스 [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini)에서 사용 가능 | 핵심 기능을 갖춘 증류 버전 |

S1과 S1-mini 모두 온라인 인간 피드백 강화학습(RLHF)을 통합하고 있습니다.

## **기능**

1. **제로샷 및 퓨샷 TTS:** 10~30초의 음성 샘플을 입력하여 고품질 TTS 출력을 생성합니다. **자세한 가이드라인은 [음성 복제 모범 사례](https://docs.fish.audio/resources/best-practices/voice-cloning)를 참조하세요.**

2. **다국어 및 교차 언어 지원:** 다국어 텍스트를 입력 상자에 복사하여 붙여넣기만 하면 됩니다. 언어를 걱정할 필요가 없습니다. 현재 영어, 일본어, 한국어, 중국어, 프랑스어, 독일어, 아랍어, 스페인어를 지원합니다.

3. **음소 의존성 없음:** 모델은 강력한 일반화 능력을 가지고 있으며 TTS를 위해 음소에 의존하지 않습니다. 모든 언어 스크립트의 텍스트를 처리할 수 있습니다.

4. **높은 정확도:** Seed-TTS Eval에서 약 0.4%의 낮은 CER(문자 오류율)과 약 0.8%의 WER(단어 오류율)을 달성합니다.

5. **빠른 속도:** torch compile로 가속화되어 Nvidia RTX 4090 GPU에서 실시간 팩터가 약 1:7입니다.

6. **WebUI 추론:** 사용하기 쉬운 Gradio 기반 웹 UI를 제공하며 Chrome, Firefox, Edge 등 다른 브라우저와 호환됩니다.

7. **배포 용이성:** Linux 및 Windows를 기본 지원하며(macOS 지원 예정), 성능 손실을 최소화하면서 추론 서버를 쉽게 설정할 수 있습니다.

## **미디어 및 데모**

<div align="center">

### **소셜 미디어**
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Latest_Demo-black?style=for-the-badge&logo=x&logoColor=white" alt="X에서 최신 데모" />
</a>

### **인터랙티브 데모**
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Try_OpenAudio_S1-blue?style=for-the-badge" alt="OpenAudio S1 체험하기" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Try_S1_Mini-yellow?style=for-the-badge" alt="S1 Mini 체험하기" />
</a>

### **비디오 쇼케이스**

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="../docs/assets/Thumbnail.jpg" alt="OpenAudio S1 Video" style="width: 50%;" />
</a>

</div>

---

## 크레딧

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

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
