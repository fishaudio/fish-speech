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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **라이선스 고지사항**
> 이 코드베이스 및 관련 모델 가중치는 **[FISH AUDIO RESEARCH LICENSE](../LICENSE)** 하에 릴리스됩니다. 자세한 내용은 [LICENSE](../LICENSE)를 참조하세요.

> [!WARNING]
> **법적 면책조항**
> 저희는 코드베이스의 불법적인 사용에 대해 어떠한 책임도 지지 않습니다. DMCA 및 기타 관련 법률에 대한 현지 법률을 참조하세요.

## 여기서 시작하세요

여기는 Fish Speech의 공식 문서입니다. 지침을 따라 쉽게 시작하세요.

- [설치](https://speech.fish.audio/ko/install/)
- [추론](https://speech.fish.audio/ko/inference/)

## Fish Audio S2
**오픈 소스와 클로즈드 소스 모두에서 가장 뛰어난 텍스트 음성 변환 시스템**

Fish Audio S2는 [Fish Audio](https://fish.audio/)가 개발한 최신 모델로, 자연스럽고 사실적이며 감정적으로 풍부한 음성을 생성하도록 설계되었습니다. 로봇 같지 않고, 평평하지 않으며, 스튜디오 스타일의 내레이션에 제한되지 않습니다.

Fish Audio S2는 일상적인 대화와 대화에 집중하여 네이티브 멀티 화자 및 멀티 턴 생성을 가능하게 합니다. 또한 명령 제어도 지원합니다.

S2 시리즈에는 여러 모델이 포함되어 있으며, 오픈 소스 모델은 S2-Pro로 컬렉션 중 최고의 모델입니다.

실시간 체험을 위해 [Fish Audio 웹사이트](https://fish.audio/)를 방문하세요.

### 모델 변형

| 모델 | 크기 | 가용성 | 설명 |
|------|------|-------------|-------------|
| S2-Pro | 4B 매개변수 | [huggingface](https://huggingface.co/fishaudio/s2-pro) | 최고의 품질과 안정성을 갖춘 전체 기능 플래그십 모델 |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | 더 빠른 속도와 더 낮은 지연 시간을 가진 클로즈드 소스 모델 |

모델에 대한 자세한 내용은 기술 보고서를 참조하십시오.

## 주요 특징

<img src="./assets/totalability.png" width=200%>

### 자연어 기반 세밀한 인라인 제어

Fish Audio S2는 텍스트의 특정 단어 또는 구문 위치에 자연어 지시를 직접 삽입해 음성 생성을 국소적으로 제어할 수 있습니다. 고정된 사전 정의 태그에 의존하는 대신, S2는 [whisper in small voice], [professional broadcast tone], [pitch up] 같은 자유 형식 텍스트 설명을 받아 단어 수준의 개방형 표현 제어를 지원합니다.

### 다국어 지원

Fish Audio S2는 음소나 언어별 전처리 없이 고품질 다국어 텍스트 음성 변환을 지원합니다. 포함 사항:

**영어, 중국어, 일본어, 한국어, 아랍어, 독일어, 프랑스어...**

**그리고 더 많이!**

목록은 계속 확장되고 있습니다. 최신 릴리스는 [Fish Audio](https://fish.audio/)를 확인하세요.

### 네이티브 멀티 화자 생성

<img src="./assets/chattemplate.png" width=200%>

Fish Audio S2는 사용자가 여러 화자가 포함된 참조 오디오를 업로드할 수 있도록 하며, 모델은 `<|speaker:i|>` 토큰을 통해 각 화자의 특징을 처리합니다. 그런 다음 화자 ID 토큰으로 모델의 성능을 제어하여 한 번의 생성으로 여러 화자를 포함할 수 있습니다. 이전처럼 각 화자마다 별도로 참조 오디오를 업로드하고 음성을 생성할 필요가 없습니다.

### 멀티 턴 대화 생성

모델 컨텍스트의 확장 덕분에 이제 이전 정보를 활용하여 후속 생성 콘텐츠의 표현력을 높이고 콘텐츠의 자연스러움을 향상시킬 수 있습니다.

### 빠른 음성 복제

Fish Audio S2는 짧은 참조 샘플(일반적으로 10-30초)을 사용하여 정확한 음성 복제를 지원합니다. 모델은 음색, 말하기 스타일 및 감정적 경향을 캡처하여 추가 미세 조정 없이 사실적이고 일관된 복제 음성을 생성합니다.

---

## 크레딧

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## 기술 보고서
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
