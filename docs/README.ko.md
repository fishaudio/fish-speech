<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | **한국어** | [العربية](README.ar.md) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://arxiv.org/abs/2603.08823">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **라이선스 고지**
> 이 코드베이스 및 관련 모델 가중치는 **[FISH AUDIO RESEARCH LICENSE](../LICENSE)** 에 따라 배포됩니다. 자세한 내용은 [LICENSE](../LICENSE)를 참조하십시오.


> [!WARNING]
> **법적 면책 조항**
> 당사는 코드베이스의 불법적인 사용에 대해 어떠한 책임도 지지 않습니다. 해당 지역의 DMCA 및 기타 관련 법률을 참조하십시오.

## 빠른 시작

### 문서 입구

Fish Audio S2의 공식 문서입니다. 지침에 따라 쉽게 시작하십시오.

- [설치](https://speech.fish.audio/ko/install/)
- [명령줄 추론](https://speech.fish.audio/ko/inference/)
- [WebUI 추론](https://speech.fish.audio/ko/inference/)
- [서버 추론](https://speech.fish.audio/ko/server/)
- [Docker 배포](https://speech.fish.audio/ko/install/)

> [!IMPORTANT]
> **SGLang 서버를 사용하려면 [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md)를 참조하십시오.**

### LLM Agent 가이드

```
먼저 https://speech.fish.audio/ko/install/ 을 읽고 문서에 따라 Fish Audio S2를 설치 및 구성하십시오.
```

## Fish Audio S2 Pro
**음성 생성의 경계를 재정의하는 업계 최고의 다국어 텍스트 음성 변환(TTS) 시스템.**

Fish Audio S2 Pro는 [Fish Audio](https://fish.audio/)에서 개발한 최첨단 멀티모달 모델입니다. 전 세계 **80개 이상의 언어**를 아우르는 **1,000만 시간** 이상의 방대한 오디오 데이터로 학습되었습니다. 혁신적인 **이중 자기회귀(Dual-AR)** 아키텍처와 강화 학습(RL) 정렬 기술을 통해 S2 Pro는 극도로 자연스럽고 사실적이며 감정이 풍부한 음성을 생성하며, 오픈 소스와 클ローズ드 소스 경쟁 모두에서 선두를 달리고 있습니다.

S2 Pro의 핵심 강점은 자연어 태그(예: `[whisper]`, `[excited]`, `[angry]`)를 통해 운율과 감정을 **하위 단어 수준(Sub-word Level)**에서 매우 세밀하게 인라인 제어할 수 있다는 점입니다. 또한 다중 화자 생성 및 긴 컨텍스트의 다중 턴 대화 생성을 기본적으로 지원합니다.

지금 바로 [Fish Audio 공식 웹사이트](https://fish.audio/)에서 온라인 데모를 체험하거나, [기술 보고서](https://arxiv.org/abs/2603.08823) 및 [블로그 게시물](https://fish.audio/blog/fish-audio-open-sources-s2/)을 통해 자세히 알아보십시오.

### 모델 변체

| 모델 | 크기 | 가용성 | 설명 |
|------|------|-------------|-------------|
| S2-Pro | 4B 파라미터 | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | 최고의 품질과 안정성을 갖춘 모든 기능을 갖춘 플래그십 모델 |

모델에 대한 자세한 내용은 [기술 보고서](https://arxiv.org/abs/2411.01156)를 참조하십시오.

## 벤치마크 결과

| 벤치마크 | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER(중국어) | **0.54%** (전체 최고) |
| Seed-TTS Eval — WER(영어) | **0.99%** (전체 최고) |
| Audio Turing Test (지침 포함) | **0.515** 후험 평균 |
| EmergentTTS-Eval — 승률 | **81.88%** (전체 최고) |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — 품질 | **4.51 / 5.0** |
| 다국어 (MiniMax Testset) — 최고 WER | **24개 언어 중 11개** |
| 다국어 (MiniMax Testset) — 최고 SIM | **24개 언어 중 17개** |

Seed-TTS Eval에서 S2는 클ローズ드 소스 시스템을 포함한 모든 평가 모델 중 가장 낮은 WER을 달성했습니다: Qwen3-TTS (0.77/1.24), MiniMax Speech-02 (0.99/1.90), Seed-TTS (1.12/2.25). Audio Turing Test에서 S2의 0.515는 Seed-TTS (0.417) 대비 24%, MiniMax-Speech (0.387) 대비 33% 향상된 수치입니다. EmergentTTS-Eval에서 S2는 부차 언어학(91.61% 승률), 의문문(84.41%), 구문 복잡성(83.39%) 등의 측면에서 특히 두드러진 성과를 보였습니다.

## 하이라이트

<img src="./assets/totalability.png" width=200%>

### 자연어를 통한 초미세 인라인 제어

S2 Pro는 음성에 전례 없는 "영혼"을 부여합니다. 간단한 `[tag]` 구문을 사용하여 텍스트의 어느 위치에나 감정 지침을 정확하게 삽입할 수 있습니다.
- **15,000개 이상의 고유 태그 지원**: 고정된 사전 설정에 국한되지 않고 **자유 형식의 텍스트 설명**을 지원합니다. `[whisper in small voice]` (작은 목소리로 속삭임), `[professional broadcast tone]` (전문 방송 톤), `[pitch up]` (음높이 높임) 등을 시도해 보십시오.
- **풍부한 감정 라이브러리**:
  `[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]` `[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]` `[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]` `[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]` `[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]` `[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]` `[shocked]`

### 혁신적인 이중 자기회귀 (Dual-Autoregressive) 아키텍처

S2 Pro는 Decoder-only Transformer와 RVQ 오디오 코덱(10개 코드북, 약 21Hz 프레임 속도)으로 구성된 마스터-슬레이브 방식의 Dual-AR 아키텍처를 채택했습니다.

- **Slow AR (4B 파라미터)**: 시간 축을 따라 작동하며 핵심 의미 코드북을 예측합니다.
- **Fast AR (400M 파라미터)**: 각 타임스텝에서 나머지 9개의 잔차 코드북을 생성하여 극도로 정교한 음향 세부 사항을 복원합니다.

이러한 비대칭 설계는 오디오의 최고 충실도를 보장하는 동시에 추론 속도를 대폭 향상시킵니다.

### 강화 학습 (RL) 정렬

S2 Pro는 사후 학습 정렬을 위해 **Group Relative Policy Optimization (GRPO)** 기술을 채택했습니다. 데이터 정제 및 주석 처리에 사용된 것과 동일한 모델 세트를 보상 모델(Reward Model)로 직접 사용함으로써 사전 학습 데이터 분포와 사후 학습 목표 간의 불일치 문제를 완벽하게 해결했습니다.
- **다차원 보상 신호**: 의미 체계의 정확성, 지침 준수 능력, 음향 선호도 점수 및 음색 유사성을 종합적으로 평가하여 생성된 음성의 매초가 인간의 직관에 부합하도록 보장합니다.

### SGLang 기반의 극한 스트리밍 추론 성능

Dual-AR 아키텍처는 표준 LLM 구조와 동형이므로 S2 Pro는 Continuous Batching, Paged KV Cache, CUDA Graph 및 RadixAttention 기반 Prefix Caching을 포함한 SGLang의 모든 추론 가속 기능을 기본적으로 지원합니다.

**단일 NVIDIA H200 GPU 성능 지표:**
- **실시간 계수 (RTF)**: 0.195
- **첫 음성 지연 (TTFA)**: 약 100 ms
- **초고속 처리량**: RTF < 0.5 유지 시 처리량 3,000+ acoustic tokens/s 달성

### 강력한 다국어 지원

S2 Pro는 음소나 특정 언어 처리가 필요 없는 고품질 합성을 80개 이상의 언어에서 지원합니다.

- **1계층 (Tier 1)**: 일본어 (ja), 영어 (en), 중국어 (zh)
- **2계층 (Tier 2)**: 한국어 (ko), 스페인어 (es), 포르투갈어 (pt), 아랍어 (ar), 러시아어 (ru), 프랑스어 (fr), 독일어 (de)
- **글로벌 커버리지**: sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, xsl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo 등.

### 네이티브 다중 화자 생성

<img src="./assets/chattemplate.png" width=200%>

Fish Audio S2를 사용하면 사용자가 여러 화자가 포함된 참조 오디오를 업로드할 수 있으며, 모델은 `<|speaker:i|>` 토큰을 통해 각 화자의 특징을 처리합니다. 이후 화자 ID 토큰을 사용하여 모델의 표현을 제어함으로써 한 번의 생성에 여러 화자를 포함할 수 있습니다. 더 이상 화자마다 별도의 참조 오디오를 업로드하고 음성을 생성할 필요가 없습니다.

### 다중 턴 대화 생성

모델 컨텍스트 확장에 힘입어 이제 이전 정보의 도움을 받아 후속 생성 내용의 표현력을 높이고 콘텐츠의 자연스러움을 향상시킬 수 있습니다.

### 고속 음성 복제

Fish Audio S2는 짧은 참조 샘플(보통 10-30초)을 사용한 정확한 음성 복제를 지원합니다. 모델은 음색, 말하기 스타일 및 감정적 경향을 포착하여 추가적인 미세 조정 없이도 사실적이고 일관된 복제 음성을 생성합니다.
SGLang 서버 사용에 대해서는 [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md)를 참조하십시오.

---

## 감사의 말

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

@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823}, 
}
```
