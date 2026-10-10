# 추론

Fish Audio S2 모델은 큰 비디오 메모리(VRAM)가 필요합니다. 추론을 위해 최소 24GB 이상의 GPU를 사용하는 것을 권장합니다.

## 가중치 다운로드

먼저 모델 가중치를 다운로드해야 합니다:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## 명령줄 추론

!!! note
    모델이 음색을 무작위로 선택하게 하려면 이 단계를 건너뛸 수 있습니다.

### 1. 참조 오디오에서 VQ 토큰 가져오기

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

`fake.npy`와 `fake.wav` 파일이 생성됩니다.

### 2. 텍스트에서 Semantic 토큰 생성:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "변환하려는 텍스트" \
    --prompt-text "참조 텍스트" \
    --prompt-tokens "fake.npy" \
    # --compile
```

이 명령은 작업 디렉토리에 `codes_N` 파일을 생성합니다. 여기서 N은 0부터 시작하는 정수입니다.

!!! note
    더 빠른 추론을 위해 CUDA 커널을 병합하는 `--compile`을 사용하고 싶을 수 있지만, 당사의 sglang 추론 가속 최적화를 사용하는 것을 더 권장합니다.
    마찬가지로 가속을 사용할 계획이 없다면 `--compile` 매개변수를 주석 처리할 수 있습니다.

!!! info
    bf16을 지원하지 않는 GPU의 경우 `--half` 매개변수를 사용해야 할 수 있습니다.

### 3. 시맨틱 토큰에서 음성 생성:

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

이후 `fake.wav` 파일을 얻게 됩니다.

## S3-drama

S3-drama는 S2-Pro와 같은 명령줄 및 API 서버를 사용합니다. 모델 종류는 체크포인트의 `config.json`에서 자동으로 판별됩니다.

!!! warning
    S3-drama는 NVIDIA GPU에서만 실행됩니다. CPU, Apple MPS, AMD ROCm은 지원하지 않으며 `--compile`도 사용할 수 없습니다.

| GPU | 실행 방식 | GPU 메모리 |
| --- | --- | --- |
| SM89 이상(L40, H100, H20) | MoE 전문가를 FP8로 유지, CUDA 그래프 디코딩, GPU 1개 | 약 41 GB |
| 이전 세대 GPU(V100, A100) | MoE 전문가를 역양자화하여 여러 GPU에 분산 | 합계 약 70 GB |

!!! note
    V100 등 SM80 이전 GPU는 bfloat16을 지원하지 않으므로 S3는 자동으로 float16(잔차 스트림은 fp32 유지)으로 실행됩니다. 이 경로는 아직 실제 하드웨어에서 테스트되지 않았습니다. 무음이나 노이즈가 나오거나 "non-finite logits" 오류가 발생하면 issue를 등록해 주세요.

### 가중치 다운로드

```bash
hf download fishaudio/s3-drama-it-fp8 --local-dir checkpoints/s3-drama-fp8
```

체크포인트 디렉터리에 `codec.pth`가 없으면 `checkpoints/s2-pro/codec.pth`의 S2-Pro 코덱을 사용합니다.

### 명령줄 추론

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/s3-drama-fp8 \
    --text "변환하려는 텍스트" \
    --prompt-audio "reference.wav" \
    --prompt-text "참조 텍스트" \
    --output "output.wav" \
    --device cuda:0
```

`--output`을 지정하면 디코딩된 오디오가 바로 저장됩니다. 이전 세대 GPU에서는 `--device cuda:0,cuda:1,cuda:2,cuda:3`처럼 여러 장치를 지정하세요. 마지막 장치가 오디오 디코더와 코덱을 담당합니다. `--device cuda`만 지정하면 인식된 GPU 중 앞에서부터 최대 4개를 사용합니다.

### API 서버

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s3-drama-fp8 \
    --device cuda:0
```

Docker에서는 server 이미지로 S3 체크포인트를 불러옵니다. SM89 이상 GPU에서는 아래처럼 GPU 하나만 노출하고, 이전 세대 GPU에서는 `--gpus all`을 사용하세요.

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

Docker Compose에서는 같은 환경 변수를 설정합니다. `DEVICE=cuda:0`이면 S3가 GPU 하나만 사용하고, 설정하지 않으면 최대 4개의 GPU를 사용합니다.

```bash
LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 DEVICE=cuda:0 docker compose --profile server up
```

### S2-Pro와의 차이점

- 긴 텍스트는 약 300바이트(UTF-8), 청크당 최대 3명의 화자 단위로 나뉘어(가능하면 문장 끝에서 끊음) 순서대로 생성됩니다. 첫 번째 청크와 가장 최근 청크가 컨텍스트로 다시 입력되어 음색을 일정하게 유지합니다. 기본적으로 청크당 최대 1024프레임(약 47초)을 생성합니다.
- `single_chunk`(명령줄에서는 `--single-chunk`)를 지정하면 전체 텍스트를 한 번에 생성합니다. 이때 기본 상한 1024프레임이 전체 텍스트에 적용되므로, 긴 텍스트는 `max_new_tokens`를 늘리거나 0으로 설정해 2048프레임(약 95초)을 사용하세요. 두 모드 모두 이보다 큰 `max_new_tokens`는 10,240프레임에서 프롬프트 길이를 뺀 값으로 제한됩니다.
- 화자 태그는 `<|speaker:0|>`부터 `<|speaker:15|>`까지 사용할 수 있으며, 태그가 없는 텍스트는 speaker 0으로 처리됩니다. 화자마다 다른 목소리를 쓰려면 각 참조 오디오에 `speaker` 필드(`tools/api_client.py`에서는 `--reference_speaker`)를 지정하세요. 앞에 화자 태그가 없는 전사 텍스트에는 해당 태그가 붙습니다. 전사 텍스트 앞에 태그(예: `<|speaker:1|>`)를 직접 써도 됩니다. 참조 오디오의 순서로는 화자가 정해지지 않습니다.
- `chunk_length`는 적용되지 않습니다. `repetition_penalty`는 호환성을 위해 받기만 하며 S2-Pro와 마찬가지로 무시됩니다. 두 모델 모두 반복 인식 샘플링이 내장되어 있습니다. 기본 샘플링 설정은 S2-Pro와 마찬가지로 공식 Fish Audio SDK를 따르는 temperature 0.7, top-p 0.7, top-k 30입니다.
- `streaming`을 켜면 오디오가 청크 단위로 반환됩니다.
- `FISH_S3_FP8_EXPERTS=0` 또는 `1`로 역양자화 전문가 또는 FP8 전문가를 강제할 수 있고, `FISH_S3_CUDA_GRAPH=0`으로 CUDA 그래프 디코딩을 끌 수 있습니다.

## WebUI 추론

### Awesome WebUI

Awesome WebUI는 TypeScript 기반으로 개발된 현대적인 웹 인터페이스로, 더 풍부한 기능과 향상된 사용자 경험을 제공합니다.

server Docker 이미지에는 빌드된 WebUI가 포함되어 있어, 컨테이너 실행 후 `http://localhost:8080/ui`를 열면 바로 사용할 수 있습니다. 아래 단계는 소스에서 실행할 때의 방법입니다.

**WebUI 빌드:**

로컬 또는 서버에 Node.js와 npm이 설치되어 있어야 합니다.

1. `awesome_webui` 디렉토리로 이동합니다:
   ```bash
   cd awesome_webui
   ```
2. 의존성 설치:
   ```bash
   npm install
   ```
3. WebUI 빌드:
   ```bash
   npm run build
   ```

**백엔드 서버 실행:**

WebUI 빌드가 완료되면 프로젝트 루트로 돌아가 API 서버를 실행합니다:

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**접속:**

서버가 실행된 후 브라우저를 통해 다음 주소로 접속하면 체험할 수 있습니다:
`http://localhost:8888/ui`
