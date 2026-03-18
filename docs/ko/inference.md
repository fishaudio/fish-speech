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

## WebUI 추론

### 1. Gradio WebUI

호환성을 유지하기 위해 기존의 Gradio WebUI를 보존하고 있습니다.

```bash
python tools/run_webui.py # 가속이 필요한 경우 --compile
```

### 2. Awesome WebUI

Awesome WebUI는 TypeScript 기반으로 개발된 현대적인 웹 인터페이스로, 더 풍부한 기능과 향상된 사용자 경험을 제공합니다.

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
