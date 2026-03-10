<div align="center">
<h1>Fish Speech</h1>

[English](../en/) | [简体中文](../zh/) | [Portuguese](../pt/) | [日本語](../ja/) | **한국어** | [العربية](../ar/) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

!!! info "라이선스 공지"
    이 코드베이스 및 관련 모델 가중치는 **FISH AUDIO RESEARCH LICENSE** 하에 릴리스되었습니다. 자세한 내용은 [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE)를 참조하십시오.

!!! warning "법적 면책 조항"
    코드베이스의 불법적인 사용에 대해 당사는 어떠한 책임도 지지 않습니다. DMCA 및 기타 관련 법률에 관한 현지 규정을 참조하십시오.

## 시작하기

Fish Speech의 공식 문서입니다. 지침에 따라 쉽게 시작할 수 있습니다.

- [설치](install.md)
- [추론](inference.md)

## Fish Audio S2
**오픈 소스 및 클로즈드 소스 중 최고봉의 텍스트 음성 변환 시스템**

Fish Audio S2는 [Fish Audio](https://fish.audio/)에서 개발한 최신 모델로, 자연스럽고 사실적이며 감정이 풍부한 음성을 생성하도록 설계되었습니다. 기계적이거나 평면적이지 않으며, 스튜디오 스타일의 낭독에 국한되지 않습니다.

Fish Audio S2는 일상 대화에 중점을 두고 있으며, 네이티브 다중 화자 및 다중 턴 생성을 지원합니다. 또한 명령 제어를 지원합니다.

S2 시리즈에는 여러 모델이 포함되어 있으며, 오픈 소스 모델은 S2-Pro로, 시리즈 중에서 가장 강력한 모델입니다.

실시간 체험은 [Fish Audio 웹사이트](https://fish.audio/)를 방문해 주세요.

### 모델 변형

| 모델 | 크기 | 가용성 | 설명 |
|------|------|-------------|-------------|
| S2-Pro | 4B 매개변수 | [huggingface](https://huggingface.co/fishaudio/s2-pro) | 최고의 품질과 안정성을 갖춘 풀 기능 플래그십 모델 |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | 더 빠른 속도와 짧은 지연 시간을 갖춘 클로즈드 소스 모델 |

모델에 대한 자세한 내용은 기술 보고서를 참조하십시오.

## 하이라이트

<img src="../assets/totalability.png" width=200%>

### 자연어 제어

Fish Audio S2를 사용하면 사용자가 자연어를 사용하여 각 문장의 퍼포먼스, 부언어 정보, 감정 및 기타 음성 특성을 제어할 수 있습니다. 짧은 태그를 사용하여 모델의 퍼포먼스를 모호하게 제어하는 것뿐만 아니라 생성된 콘텐츠 전체의 품질을 크게 향상시킵니다.

### 다국어 지원

Fish Audio S2는 음소나 특정 언어의 전처리 없이도 고품질의 다국어 텍스트 음성 변환을 지원합니다. 다음을 포함합니다:

**영어, 중국어, 일본어, 한국어, 아랍어, 독일어, 프랑스어...**

**그리고 더욱 추가될 예정입니다!**

목록은 지속적으로 확대되고 있으며, 최신 릴리스는 [Fish Audio](https://fish.audio/)를 확인하십시오.

### 네이티브 다중 화자 생성

<img src="../assets/chattemplate.png" width=200%>

Fish Audio S2를 사용하면 사용자가 여러 화자가 포함된 참조 오디오를 업로드할 수 있으며, 모델은 `<|speaker:i|>` 토큰을 통해 각 화자의 특성을 처리합니다. 이후 화자 ID 토큰을 통해 모델의 퍼포먼스를 제어하여 한 번의 생성으로 여러 화자를 구현할 수 있습니다. 화자마다 개별적으로 참조 오디오를 업로드하고 음성을 생성할 필요가 더 이상 없습니다.

### 다중 턴 대화 생성

모델 컨텍스트의 확장 덕분에, 이전 컨텍스트의 정보를 사용하여 이후에 생성되는 콘텐츠의 표현력을 개선하고 콘텐츠의 자연스러움을 높일 수 있게 되었습니다.

### 빠른 음성 클로닝

Fish Audio S2는 짧은 참조 샘플(보통 10~30초)을 사용한 정확한 음성 클로닝을 지원합니다. 모델은 음색, 말하기 스타일 및 감정적 경향을 포착할 수 있으며, 추가 미세 조정 없이도 사실적이고 일관된 클로닝 음성을 생성할 수 있습니다.

---

## 감사의 인사

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
