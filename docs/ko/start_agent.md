# 에이전트 시작하기

!!! note
    전체 문서는 claude3.5 Sonnet에 의해 번역되었으며, 원어민인 경우 번역에 문제가 있다고 생각되면 이슈나 풀 리퀘스트를 보내주셔서 대단히 감사합니다!

## 요구사항

- GPU 메모리: 최소 8GB(양자화 사용 시), 16GB 이상 권장
- 디스크 사용량: 10GB

## 모델 다운로드

다음 명령어로 모델을 받을 수 있습니다:

```bash
huggingface-cli download fishaudio/fish-agent-v0.1-3b --local-dir checkpoints/fish-agent-v0.1-3b
```

'checkpoints' 폴더에 파일들을 넣으세요.

또한 [inference](inference.md)에 설명된 대로 fish-speech 모델도 다운로드해야 합니다.

checkpoints에는 2개의 폴더가 있어야 합니다.

`checkpoints/fish-speech-1.4`와 `checkpoints/fish-agent-v0.1-3b`입니다.

## 환경 준비

이미 Fish-speech가 있다면 다음 명령어를 추가하여 바로 사용할 수 있습니다:
```bash
pip install cachetools
```

!!! 참고
    컴파일을 위해 Python 3.12 미만 버전을 사용해 주세요.

없다면 아래 명령어를 사용하여 환경을 구축하세요:

```bash
sudo apt-get install portaudio19-dev

pip install -e .[stable]
```

## 에이전트 데모 실행

fish-agent를 구축하려면 메인 폴더에서 아래 명령어를 사용하세요:

```bash
python -m tools.api_server --llama-checkpoint-path checkpoints/fish-agent-v0.1-3b/ --mode agent --compile
```

`--compile` 인자는 Python < 3.12에서만 지원되며, 토큰 생성 속도를 크게 향상시킵니다.

한 번에 컴파일되지 않습니다(기억해 두세요).

그런 다음 다른 터미널을 열고 다음 명령어를 사용하세요:

```bash
python -m tools.e2e_webui
```

이렇게 하면 기기에 Gradio WebUI가 생성됩니다.

모델을 처음 사용할 때는 (`--compile`이 True인 경우) 잠시 컴파일이 진행되므로 기다려 주세요.

## Gradio Webui
<p align="center">
   <img src="../../assets/figs/agent_gradio.png" width="75%">
</p>

즐거운 시간 되세요!

## 성능

테스트 결과, 4060 노트북은 겨우 실행되며 매우 부하가 큰 상태로, 초당 약 8토큰 정도만 처리합니다. 4090은 컴파일 상태에서 초당 약 95토큰을 처리하며, 이것이 저희가 권장하는 사양입니다.

# 에이전트 소개

이 데모는 초기 알파 테스트 버전으로, 추론 속도 최적화가 필요하며 수정해야 할 버그가 많이 있습니다. 버그를 발견하거나 수정하고 싶으시다면 이슈나 풀 리퀘스트를 보내주시면 매우 감사하겠습니다.
