# 추론

보코더 모델이 변경되어 이전보다 더 많은 VRAM이 필요하며, 원활한 추론을 위해 12GB를 권장합니다.

추론을 위해 명령줄, HTTP API, WebUI를 지원하며, 원하는 방법을 선택할 수 있습니다.

## 가중치 다운로드

먼저 모델 가중치를 다운로드해야 합니다:

```bash
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## 명령줄 추론

!!! note
    모델이 임의로 음색을 선택하도록 하려면 이 단계를 건너뛸 수 있습니다.

### 1. 참조 오디오에서 VQ 토큰 얻기

```bash
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

`fake.npy`와 `fake.wav`를 얻을 수 있습니다.

### 2. 텍스트에서 의미 토큰 생성:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "변환하고 싶은 텍스트" \
    --prompt-text "참조 텍스트" \
    --prompt-tokens "fake.npy" \
    --compile
```

이 명령은 작업 디렉토리에 `codes_N` 파일을 생성합니다. 여기서 N은 0부터 시작하는 정수입니다.

!!! note
    더 빠른 추론을 위해 `--compile`을 사용하여 CUDA 커널을 융합할 수 있습니다(약 15 토큰/초 -> 약 150 토큰/초, RTX 4090 GPU).
    이에 따라 가속을 사용하지 않으려면 `--compile` 매개변수를 주석 처리할 수 있습니다.

!!! info
    bf16을 지원하지 않는 GPU의 경우 `--half` 매개변수를 사용해야 할 수 있습니다.

### 3. 의미 토큰에서 음성 생성:

!!! warning "향후 경고"
    원래 경로(tools/vqgan/inference.py)에서 액세스 가능한 인터페이스를 유지하고 있지만, 이 인터페이스는 향후 릴리스에서 제거될 수 있으므로 가능한 한 빨리 코드를 변경해 주세요.

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy"
```

## HTTP API 추론

추론을 위한 HTTP API를 제공합니다. 다음 명령으로 서버를 시작할 수 있습니다:

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> 추론을 가속화하려면 `--compile` 매개변수를 추가할 수 있습니다.

그 후 http://127.0.0.1:8080/ 에서 API를 보고 테스트할 수 있습니다.

## GUI 추론 
[클라이언트 다운로드](https://github.com/AnyaCoder/fish-speech-gui/releases)

## WebUI 추론

다음 명령으로 WebUI를 시작할 수 있습니다:

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

또는 간단히

```bash
python -m tools.run_webui
```
> 추론을 가속화하려면 `--compile` 매개변수를 추가할 수 있습니다.

!!! note
    라벨 파일과 참조 오디오 파일을 메인 디렉토리의 `references` 폴더에 미리 저장할 수 있습니다(직접 생성해야 함). 이렇게 하면 WebUI에서 직접 호출할 수 있습니다.

!!! note
    `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME`과 같은 Gradio 환경 변수를 사용하여 WebUI를 구성할 수 있습니다.

## Docker 추론

OpenAudio는 WebUI 및 API 서버 추론을 위한 Docker 컨테이너를 제공합니다. `docker run` 명령을 직접 사용하여 컨테이너를 시작할 수 있습니다.

다음 사항을 준비해야 합니다:
- Docker 및 NVIDIA Docker 런타임 설치 (GPU 지원용)
- 모델 가중치 다운로드 ([가중치 다운로드](#가중치-다운로드) 섹션 참조)
- 참조 오디오 파일 (선택 사항, 음성 복제용)

```bash
# 모델 가중치 및 참조 오디오용 디렉토리 생성
mkdir -p checkpoints references

# 모델 가중치 다운로드 (아직 다운로드하지 않은 경우)
# hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# CUDA 지원으로 WebUI 시작 (권장, 최상의 성능)
docker run -d \
    --name fish-speech-webui \
    --gpus all \
    -p 7860:7860 \
    -v ./checkpoints:/app/checkpoints \
    -v ./references:/app/references \
    -e COMPILE=1 \
    fishaudio/fish-speech:latest-webui-cuda

# CPU 전용 추론 (느리지만 GPU 없이 작동)
docker run -d \
    --name fish-speech-webui-cpu \
    -p 7860:7860 \
    -v ./checkpoints:/app/checkpoints \
    -v ./references:/app/references \
    fishaudio/fish-speech:latest-webui-cpu
```

```bash
# CUDA 지원으로 API 서버 시작
docker run -d \
    --name fish-speech-server \
    --gpus all \
    -p 8080:8080 \
    -v ./checkpoints:/app/checkpoints \
    -v ./references:/app/references \
    -e COMPILE=1 \
    fishaudio/fish-speech:latest-server-cuda

# CPU 전용 추론
docker run -d \
    --name fish-speech-server-cpu \
    -p 8080:8080 \
    -v ./checkpoints:/app/checkpoints \
    -v ./references:/app/references \
    fishaudio/fish-speech:latest-server-cpu
```

다음 환경 변수를 사용하여 Docker 컨테이너를 사용자 정의할 수 있습니다:

- `COMPILE=1` - `torch.compile`을 활성화하여 추론 속도 향상 (약 10배, CUDA 전용)
- `GRADIO_SERVER_NAME=0.0.0.0` - WebUI 서버 호스트 (기본값: 0.0.0.0)
- `GRADIO_SERVER_PORT=7860` - WebUI 서버 포트 (기본값: 7860)
- `API_SERVER_NAME=0.0.0.0` - API 서버 호스트 (기본값: 0.0.0.0)
- `API_SERVER_PORT=8080` - API 서버 포트 (기본값: 8080)
- `LLAMA_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini` - 모델 가중치 경로
- `DECODER_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini/codec.pth` - 디코더 가중치 경로
- `DECODER_CONFIG_NAME=modded_dac_vq` - 디코더 구성 이름
```

WebUI 및 API 서버의 사용법은 위 가이드와 동일합니다.

즐기세요!
