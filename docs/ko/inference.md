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

준비 중입니다.
