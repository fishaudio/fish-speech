# 추론

Fish Audio는 두 가지 모델을 제공합니다: **S2-Pro**와 **Drama 3**. 두 모델은 동일한 명령줄과 API 서버를 공유하고, 모델 종류는 가중치 디렉터리의 `config.json`에서 자동으로 판별되므로 아래 각 절은 "작업 → 모델별 명령" 순으로 구성되어 있습니다.

먼저 [설치](install.md)를 완료하고 [모델 가중치](install.md#가중치-다운로드)를 내려받으세요.

!!! warning "Drama 3 실행 제한"
    Drama 3는 NVIDIA GPU에서만 동작합니다. CPU, Apple MPS, AMD ROCm은 지원되지 않고 `--compile`도 사용할 수 없습니다. GPU 메모리는 [요구 사양](install.md#요구-사양)을 참고하세요.

!!! note
    V100 등 SM80 이전 GPU에는 bfloat16이 없어 Drama 3는 float16으로 전환됩니다(잔차 스트림은 fp32 유지). 이 경로는 실제 하드웨어에서 아직 검증되지 않았습니다. 무음, 노이즈 또는 "non-finite logits" 오류가 발생하면 이슈를 열어 주세요.

## 명령줄 추론

### S2-Pro

S2-Pro는 "VQ 토큰 → Semantic 토큰 → 오디오"의 3단계 파이프라인입니다.

!!! note
    모델이 임의로 목소리 톤을 고르게 하려면 1단계를 건너뛸 수 있습니다.

**1. 참조 오디오에서 VQ 토큰 가져오기**

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

`fake.npy`와 `fake.wav`를 얻게 됩니다.

**2. 텍스트에서 Semantic 토큰 생성:**

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "변환할 텍스트" \
    --prompt-text "참조 텍스트" \
    --prompt-tokens "fake.npy" \
    # --compile
```

작업 디렉터리에 `codes_N` 파일이 생성됩니다(N은 0부터 시작하는 정수).

!!! note
    `--compile`로 CUDA 커널을 융합해 추론을 빠르게 할 수 있습니다. 다만 저희는 sglang 추론 가속을 권장합니다.
    가속을 사용하지 않을 계획이라면 `--compile` 파라미터를 주석 처리하세요.

!!! info
    bf16을 지원하지 않는 GPU에서는 `--half` 파라미터가 필요할 수 있습니다.

**3. 시맨틱 토큰에서 음성 생성:**

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

그 후 `fake.wav` 파일을 얻게 됩니다.

### Drama 3

Drama 3는 한 번에 끝나며, `--output`이 디코딩된 오디오를 바로 씁니다:

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/drama-3 \
    --text "변환할 텍스트" \
    --prompt-audio "reference.wav" \
    --prompt-text "참조 텍스트" \
    --output "output.wav" \
    --device cuda:0
```

구형 GPU에서는 `--device cuda:0,cuda:1,cuda:2,cuda:3`처럼 여러 장을 지정할 수 있습니다. 마지막 장이 오디오 디코더와 codec을 담당합니다. `--device cuda`만 쓰면 보이는 GPU 중 최대 4장을 사용합니다.

## API 서버

### S2-Pro 시작

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq
```

### Drama 3 시작

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/drama-3 \
    --device cuda:0
```

!!! note
    Drama 3는 `--compile`을 지원하지 않습니다. 구형 GPU에서는 `--device`에 여러 장을 나열하세요. 마지막 장이 오디오 디코더와 codec을 담당합니다.

두 모델이 공유하는 시작 옵션:

- `--compile`: `torch.compile` 최적화 활성화(S2-Pro 전용)
- `--half`: fp16 모드 사용
- `--api-key`: bearer 토큰 인증 요구
- `--workers`: 워커 프로세스 수 설정
- `--max-text-length`: 요청 한 건의 텍스트 길이 제한(0은 제한 없음)

헬스 체크:

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

`{"status":"ok"}`가 반환되면 정상입니다.

### API 엔드포인트

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `POST` | `/v1/tts` | 텍스트 음성 변환(요청 본문은 msgpack) |
| `GET` | `/v1/tts/defaults` | 로드된 모델의 샘플링 기본값 |
| `GET` | `/v1/health` | 헬스 체크 |
| `POST` | `/v1/vqgan/encode` | 참조 오디오 → VQ 토큰 |
| `POST` | `/v1/vqgan/decode` | VQ 토큰 → 오디오 |
| `POST` | `/v1/references/add` | 참조 음성 저장(폼 필드 `id`, `audio`(파일), `text`) |
| `GET` | `/v1/references/list` | 저장된 음성 목록 |
| `POST` | `/v1/references/update` | 이름 변경(폼 필드 `old_reference_id`, `new_reference_id`) |
| `DELETE` | `/v1/references/delete` | 삭제(폼 필드 `reference_id`) |
| `GET` | `/ui` | WebUI(Docker 이미지에 포함, 소스 실행 시에는 먼저 빌드) |

음성을 저장한 뒤에는 `--reference_id`로 사용합니다(오디오 필드에 MIME 타입이 필요하며, 없으면 서버가 인식하지 못합니다):

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
    -F id=my-speaker \
    -F "audio=@reference.wav;type=audio/wav" \
    -F text="참조 오디오의 전사 텍스트"

python tools/api_client.py --url http://127.0.0.1:8080/v1/tts \
    --text "Hello" --reference_id my-speaker --output demo
```

### 요청 보내기

어떤 모델이 쓰이는지는 서버가 시작할 때 로드한 가중치로 결정되며, 요청에 모델 필드는 없습니다.

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "Hello from Fish Speech" \
    --output demo
```

주요 파라미터:

| 파라미터 | 설명 |
| --- | --- |
| `--reference_audio` / `--reference_text` | 참조 오디오와 그 전사 텍스트. 둘의 개수는 같아야 합니다 |
| `--reference_speaker` | 참조 오디오별 화자 ID(0–15). `<\|speaker:N\|>` 접두사로 기록(Drama 3의 주요 사용법) |
| `--reference_id` | 저장된 참조 음성 사용(예: `my-speaker`) |
| `--format` | 출력 형식: `wav`, `pcm`, `mp3`, `opus`(기본값 `wav`) |
| `--streaming` | 오디오를 스트리밍으로 반환 |
| `--single_chunk` | 청크로 나누지 않고 전체를 한 번에 생성(Drama 3 전용) |
| `--chunk_length` | 청크 길이, 기본값 200(S2-Pro 전용) |
| `--max_new_tokens` | 청크당 최대 프레임 수. 미지정 시 서버 기본값 1024, 0은 모델 자체 상한 |
| `--temperature` / `--top_p` | 샘플링. 미지정 시 모델 기본값(0.7 / 0.7) |
| `--seed` | 랜덤 시드 고정(음색 고정에는 사용할 수 없음) |
| `--api_key` | 서버가 `--api-key`를 사용할 때 필수 |
| `--no-play` | 생성 후 오디오를 재생하지 않음(기본은 재생. 오디오 장치가 없는 환경에서는 지정) |

!!! note
    `repetition_penalty`는 호환용으로만 남아 있으며, S2-Pro와 Drama 3 모두 사용하지 않고 내장된 반복 인식 샘플링을 씁니다.

!!! tip
    Docker로 서버를 띄우려면 [Docker 설정](install.md#docker-설정)을 참고하세요(server 이미지에는 WebUI가 포함되어 있어 컨테이너를 띄운 뒤 `http://localhost:8080/ui`를 열면 됩니다).

## 모델 차이

| 항목 | S2-Pro | Drama 3 |
| --- | --- | --- |
| 하드웨어 | GPU / CPU | NVIDIA GPU 전용, `--compile` 미지원 |
| 긴 텍스트 | `chunk_length`로 분할 | 약 300바이트(UTF-8), 청크당 최대 3화자로 자동 분할하고 가능한 문장 끝에서 절단. 첫 청크와 최신 청크를 컨텍스트로 재투입 |
| 1회 길이 상한 | 컨텍스트 길이 | 청크당 1024프레임(약 47초). `single_chunk`는 2048프레임(약 95초). 모두 "10,240프레임 − 프롬프트 길이"가 상한 |
| 화자 | 참조 순서가 역할을 결정 | `<\|speaker:0\|>`–`<\|speaker:15\|>` 태그가 역할을 결정하며 참조 순서는 무관 |
| `chunk_length` | 동작 | 동작하지 않음 |
| 샘플링 기본값 | temperature 0.7, top-p 0.7, top-k 30 | 동일 |
| `repetition_penalty` | 무시 | 무시 |

Drama 3에는 환경 변수 두 개가 있습니다: `FISH_DRAMA3_FP8_EXPERTS=0` 또는 `1`은 역양자화 / FP8 전문가를 강제하고, `FISH_DRAMA3_CUDA_GRAPH=0`은 CUDA graph 디코딩을 끕니다.

## WebUI 추론

### Awesome WebUI

Awesome WebUI는 TypeScript로 만든 현대적인 웹 인터페이스로, 더 풍부한 기능과 더 나은 사용 경험을 제공합니다.

server Docker 이미지에는 빌드된 WebUI가 포함되어 있어 컨테이너를 띄운 뒤 `http://localhost:8080/ui`를 열면 바로 됩니다. 아래 단계는 소스에서 실행할 때 해당합니다.

**WebUI 빌드:**

로컬 또는 서버에 Node.js와 npm이 설치되어 있어야 합니다.

1. `awesome_webui` 디렉터리로 이동:
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

**백엔드 서버 시작:**

빌드가 끝나면 프로젝트 루트로 돌아가 API 서버를 시작합니다(`--compile`은 S2-Pro 전용이며, Drama 3를 로드할 때는 제거하세요):

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**접속:**

서버가 시작되면 브라우저에서 다음 주소로 접속하세요:
`http://localhost:8888/ui`
