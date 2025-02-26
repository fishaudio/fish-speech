# 추론

추론은 명령줄, HTTP API, 그리고 웹 UI에서 지원됩니다.

!!! note
    전체 추론 과정은 다음의 여러 단계로 구성됩니다:

    1. VQGAN을 사용하여 약 10초 분량의 음성을 인코딩합니다.
    2. 인코딩된 시맨틱 토큰과 해당 텍스트를 예시로 언어 모델에 입력합니다.
    3. 새로운 텍스트를 입력하면, 모델이 해당하는 시맨틱 토큰을 생성합니다.
    4. 생성된 시맨틱 토큰을 VITS / VQGAN에 입력하여 음성을 디코딩하고 생성합니다.

## 모델 다운로드
필요한 `vqgan` 및 `llama` 모델을 Hugging Face 리포지토리에서 다운로드하세요.

```bash
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

## 명령줄 추론
### 1. 음성에서 프롬프트 생성:

!!! note
    모델이 음색을 무작위로 선택하도록 하려면 이 단계를 건너뛸 수 있습니다.

!!! warning "향후 버전 경고"
    원래 경로(tools/vqgan/infernce.py)에서 접근할 수 있는 인터페이스는 유지했지만, 이 인터페이스는 향후 몇몇 버전에서 삭제될 수 있습니다. 가능한 한 빨리 코드를 변경하십시오.

```bash
python fish_speech/models/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

이 명령을 실행하면 `fake.npy` 파일을 얻게 됩니다.

### 2. 텍스트에서 시맨틱 토큰 생성:

!!! warning "향후 버전 경고"
    원래 경로(tools/llama/generate.py)에서 접근할 수 있는 인터페이스는 유지했지만, 이 인터페이스는 향후 몇몇 버전에서 삭제될 수 있습니다. 가능한 한 빨리 코드를 변경하십시오.

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "변환할 텍스트" \
    --prompt-text "참고할 텍스트" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5" \
    --num-samples 2 \
    --compile
```

이 명령을 실행하면 작업 디렉토리에 `codes_N` 파일이 생성되며, N은 0부터 시작하는 정수입니다.

!!! note
    빠른 추론을 위해 `--compile` 옵션을 사용하여 CUDA 커널을 결합할 수 있습니다 (~초당 30 토큰 -> ~초당 500 토큰).
    `--compile` 매개변수를 주석 처리하여 가속화 옵션을 사용하지 않을 수도 있습니다.

!!! info
    bf16을 지원하지 않는 GPU의 경우 `--half` 매개변수를 사용해야 할 수 있습니다.

### 3. 시맨틱 토큰에서 음성 생성:

#### VQGAN 디코더

!!! warning "향후 버전 경고"
    원래 경로(tools/vqgan/infernce.py)에서 접근할 수 있는 인터페이스는 유지했지만, 이 인터페이스는 향후 몇몇 버전에서 삭제될 수 있습니다. 가능한 한 빨리 코드를 변경하십시오.

```bash
python fish_speech/models/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

## HTTP API 추론

추론을 위한 HTTP API를 제공하고 있습니다. 아래의 명령어로 서버를 시작할 수 있습니다:

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/fish-speech-1.5" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

추론 속도를 높이고 싶다면 `--compile` 매개변수를 추가할 수 있습니다.

이후, http://127.0.0.1:8080/ 에서 API를 확인하고 테스트할 수 있습니다.

아래는 `tools/api_client.py`를 사용하여 요청을 보내는 예시입니다.

```bash
python -m tools.api_client \
    --text "입력할 텍스트" \
    --reference_audio "참고 음성 경로" \
    --reference_text "참고 음성의 텍스트 내용" \
    --streaming True
```

위 명령은 참고 음성 정보를 바탕으로 원하는 음성을 합성하고, 스트리밍 방식으로 반환합니다.

다음 예시는 여러 개의 참고 음성 경로와 텍스트를 한꺼번에 사용할 수 있음을 보여줍니다. 명령에서 공백으로 구분하여 입력합니다.

```bash
python -m tools.api_client \
    --text "입력할 텍스트" \
    --reference_audio "참고 음성 경로1" "참고 음성 경로2" \
    --reference_text "참고 음성 텍스트1" "참고 음성 텍스트2"\
    --streaming False \
    --output "generated" \
    --format "mp3"
```

위 명령어는 여러 참고 음성 정보를 바탕으로 `MP3` 형식의 음성을 합성하여, 현재 디렉토리에 `generated.mp3`로 저장합니다.

`--reference_audio`와 `--reference_text` 대신에 `--reference_id`(하나만 사용 가능)를 사용할 수 있습니다. 프로젝트 루트 디렉토리에 `references/<your reference_id>` 폴더를 만들어 해당 음성과 주석 텍스트를 넣어야 합니다. 참고 음성은 최대 90초까지 지원됩니다.

!!! info 
    제공되는 파라미터는 `python -m tools.api_client -h`를 사용하여 확인할 수 있습니다.

## GUI 추론 
[클라이언트 다운로드](https://github.com/AnyaCoder/fish-speech-gui/releases)

## WebUI 추론

다음 명령으로 WebUI를 시작할 수 있습니다:

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.5" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

> 추론 속도를 높이고 싶다면 `--compile` 매개변수를 추가할 수 있습니다.

!!! note
    라벨 파일과 참고 음성 파일을 미리 메인 디렉토리의 `references` 폴더에 저장해 두면, WebUI에서 바로 호출할 수 있습니다. (해당 폴더는 직접 생성해야 합니다.)

!!! note
    WebUI를 구성하기 위해 `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME`과 같은 Gradio 환경 변수를 사용할 수 있습니다.

즐기세요!
