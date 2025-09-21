# OpenAudio (구 Fish-Speech)

<div align="center">

<div align="center">

<img src="../assets/openaudio.jpg" alt="OpenAudio" style="display: block; margin: 0 auto; width: 35%;"/>

</div>

<strong>고급 텍스트-음성 변환 모델 시리즈</strong>

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

<strong>지금 체험:</strong> <a href="https://fish.audio">Fish Audio Playground</a> | <strong>자세히 알아보기:</strong> <a href="https://openaudio.com">OpenAudio 웹사이트</a>

</div>

---

!!! note "라이선스 공지"
    이 코드베이스는 **Apache 라이선스**에 따라 배포되며, 모든 모델 가중치는 **CC-BY-NC-SA-4.0 라이선스**에 따라 배포됩니다. 자세한 내용은 [코드 라이선스](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) 및 [모델 라이선스](https://spdx.org/licenses/CC-BY-NC-SA-4.0)를 참조하십시오.

!!! warning "법적 면책조항"
    코드베이스의 불법적인 사용에 대해서는 일체 책임을 지지 않습니다. 귀하의 지역의 DMCA 및 기타 관련 법률을 참고하시기 바랍니다.

## **소개**

저희는 **OpenAudio**로의 브랜드 변경을 발표하게 되어 기쁩니다. Fish-Speech를 기반으로 하여 상당한 개선과 새로운 기능을 추가한 새로운 고급 텍스트-음성 변환 모델 시리즈를 소개합니다.

**Openaudio-S1-mini**: [블로그](https://openaudio.com/blogs/s1); [동영상](https://www.youtube.com/watch?v=SYuPvd7m06A); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [동영상](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **주요 특징**

### **뛰어난 TTS 품질**

Seed TTS 평가 지표를 사용하여 모델 성능을 평가한 결과, OpenAudio S1은 영어 텍스트에서 **0.008 WER**과 **0.004 CER**을 달성하여 이전 모델보다 현저히 향상되었습니다. (영어, 자동 평가, OpenAI gpt-4o-전사 기반, 화자 거리는 Revai/pyannote-wespeaker-voxceleb-resnet34-LM 사용)

| 모델 | 단어 오류율 (WER) | 문자 오류율 (CER) | 화자 거리 |
|:-----:|:--------------------:|:-------------------------:|:----------------:|
| **S1** | **0.008** | **0.004** | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **TTS-Arena2 최고 모델**

OpenAudio S1은 [TTS-Arena2](https://arena.speechcolab.org/)에서 **#1 순위**를 달성했습니다. 이는 텍스트 음성 변환 평가의 기준입니다:

<div align="center">
    <img src="assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **음성 제어**
OpenAudio S1은 **다양한 감정, 톤, 특수 마커를 지원**하여 음성 합성을 향상시킵니다:

- **기본 감정**:
```
(화난) (슬픈) (흥미진진한) (놀란) (만족한) (기쁜) 
(무서워하는) (걱정하는) (속상한) (긴장한) (좌절한) (우울한)
(공감하는) (당황한) (역겨워하는) (감동한) (자랑스러운) (편안한)
(감사한) (자신감있는) (관심있는) (호기심있는) (혼란스러운) (즐거운)
```

- **고급 감정**:
```
(경멸하는) (불행한) (불안한) (히스테리컬한) (무관심한) 
(참을성없는) (죄책감있는) (멸시하는) (공황상태의) (격분한) (마지못한)
(열망하는) (불찬성하는) (부정적인) (부인하는) (놀란) (진지한)
(비꼬는) (화해하는) (위로하는) (진실한) (비웃는)
(주저하는) (굴복하는) (고통스러운) (어색한) (재미있어하는)
```

(현재 영어, 중국어, 일본어를 지원하며, 더 많은 언어가 곧 출시될 예정입니다!)

- **톤 마커**:
```
(서두르는 톤으로) (소리치며) (비명지르며) (속삭이며) (부드러운 톤으로)
```

- **특수 음향 효과**:
```
(웃으며) (킥킥거리며) (흐느끼며) (크게 울며) (한숨쉬며) (헐떡이며)
(신음하며) (군중 웃음소리) (배경 웃음소리) (관객 웃음소리)
```

Ha,ha,ha를 사용하여 제어할 수도 있으며, 여러분 스스로 탐구할 수 있는 다른 많은 사용법이 있습니다.

### **두 가지 모델 유형**

<div align="center">

| 모델 | 크기 | 가용성 | 특징 |
|-------|------|--------------|----------|
| **S1** | 40억 매개변수 | [fish.audio](https://fish.audio)에서 이용 가능 | 모든 기능을 갖춘 플래그십 모델 |
| **S1-mini** | 5억 매개변수 | huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini)에서 이용 가능 | 핵심 기능을 갖춘 경량화 버전 |

</div>

S1과 S1-mini 모두 온라인 인간 피드백 강화 학습(RLHF)이 통합되어 있습니다.

## **기능**

1. **제로샷 및 퓨샷 TTS:** 10~30초의 음성 샘플을 입력하여 고품질 TTS 출력을 생성합니다. **자세한 가이드라인은 [음성 복제 모범 사례](https://docs.fish.audio/text-to-speech/voice-clone-best-practices)를 참조하세요.**

2. **다국어 및 교차 언어 지원:** 다국어 텍스트를 입력 상자에 복사하여 붙여넣기만 하면 됩니다. 언어에 대해 걱정할 필요가 없습니다. 현재 영어, 일본어, 한국어, 중국어, 프랑스어, 독일어, 아랍어, 스페인어를 지원합니다.

3. **음소 의존성 없음:** 이 모델은 강력한 일반화 능력을 가지고 있으며 TTS에 음소에 의존하지 않습니다. 어떤 언어 스크립트의 텍스트도 처리할 수 있습니다.

4. **높은 정확도:** Seed-TTS Eval에서 약 0.4%의 낮은 문자 오류율(CER)과 약 0.8%의 단어 오류율(WER)을 달성합니다.

5. **빠른 속도:** torch compile 가속을 통해 Nvidia RTX 4090 실시간 계수 약 1:7.

6. **WebUI 추론:** Chrome, Firefox, Edge 및 기타 브라우저와 호환되는 사용하기 쉬운 Gradio 기반 웹 UI를 제공합니다.

7. **GUI 추론:** API 서버와 원활하게 작동하는 PyQt6 그래픽 인터페이스를 제공합니다. Linux, Windows, macOS를 지원합니다. [GUI 보기](https://github.com/AnyaCoder/fish-speech-gui).

8. **배포 친화적:** Linux, Windows (MacOS 곧 출시 예정)의 네이티브 지원으로 추론 서버를 쉽게 설정하여 속도 손실을 최소화합니다.

## **미디어 및 데모**

<!-- <div align="center"> -->

<h3><strong>소셜 미디어</strong></h3>
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-최신_데모-black?style=for-the-badge&logo=x&logoColor=white" alt="Latest Demo on X" />
</a>

<h3><strong>인터랙티브 데모</strong></h3>

<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-OpenAudio_S1_체험-blue?style=for-the-badge" alt="Try OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-S1_Mini_체험-yellow?style=for-the-badge" alt="Try S1 Mini" />
</a>

<h3><strong>동영상 쇼케이스</strong></h3>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/SYuPvd7m06A" title="OpenAudio S1 Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## **문서**

### 빠른 시작
- [환경 구축](install.md) - 개발 환경 설정
- [추론 가이드](inference.md) - 모델 실행 및 음성 생성

## **커뮤니티 및 지원**

- **Discord:** [Discord 커뮤니티](https://discord.gg/Es5qTB9BcN)에 참여하세요
- **웹사이트:** 최신 업데이트는 [OpenAudio.com](https://openaudio.com)을 방문하세요
- **온라인 체험:** [Fish Audio Playground](https://fish.audio)
