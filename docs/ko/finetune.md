# 파인튜닝

이 페이지를 열었다는 것은, 사전 학습된 퓨샷(Few-shot) 모델의 성능에 만족하지 못했다는 의미일 것입니다. 데이터셋의 성능을 향상시키기 위해 모델을 파인튜닝하고 싶으시겠죠.

현재 버전에서는 'LLAMA' 부분만 파인튜닝하시면 됩니다.

## LLAMA 파인튜닝
### 1. 데이터셋 준비

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 30.1-32.71.lab
│   └── 30.1-32.71.mp3
└── SPK2
    ├── 38.79-40.85.lab
    └── 38.79-40.85.mp3
```

위와 같은 형식으로 데이터셋을 변환하여 `data` 디렉토리 안에 배치하세요. 오디오 파일의 확장자는 `.mp3`, `.wav`, `.flac` 중 하나여야 하며, 주석 파일은 `.lab` 확장자를 사용해야 합니다.

!!! info "데이터셋 형식"
    `.lab` 주석 파일은 오디오의 전사 내용만 포함하면 되며, 특별한 형식이 필요하지 않습니다. 예를 들어, `hi.mp3`에서 "Hello, goodbye"라는 대사를 말한다면, `hi.lab` 파일에는 "Hello, goodbye"라는 한 줄의 텍스트만 있어야 합니다.

!!! warning
    데이터셋에 대한 음량 정규화(loudness normalization)를 적용하는 것이 좋습니다. 이를 위해 [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess)를 사용할 수 있습니다.

    ```bash
    fap loudness-norm data-raw data --clean
    ```

### 2. 시맨틱 토큰 배치 추출

VQGAN 가중치를 다운로드했는지 확인하세요. 다운로드하지 않았다면 아래 명령어를 실행하세요:

```bash
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

이후 시맨틱 토큰을 추출하기 위해 아래 명령어를 실행하세요:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "firefly_gan_vq" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

!!! note
    추출 속도를 높이기 위해 `--num-workers`와 `--batch-size` 값을 조정할 수 있지만, GPU 메모리 한도를 초과하지 않도록 주의하세요.  
    VITS 형식의 경우, `--filelist xxx.list`를 사용하여 파일 목록을 지정할 수 있습니다.

이 명령을 실행하면 `data` 디렉토리 안에 `.npy` 파일이 생성됩니다. 다음과 같이 표시됩니다:

```
.
├── SPK1
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 21.15-26.44.npy
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   ├── 27.51-29.98.npy
│   ├── 30.1-32.71.lab
│   ├── 30.1-32.71.mp3
│   └── 30.1-32.71.npy
└── SPK2
    ├── 38.79-40.85.lab
    ├── 38.79-40.85.mp3
    └── 38.79-40.85.npy
```

### 3. 데이터셋을 protobuf로 패킹

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

명령이 완료되면 `data` 디렉토리 안에 `quantized-dataset-ft.protos` 파일이 생성됩니다.

### 4. 마지막으로, LoRA를 이용한 파인튜닝

마찬가지로, `LLAMA` 가중치를 다운로드했는지 확인하세요. 다운로드하지 않았다면 아래 명령어를 실행하세요:

```bash
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

마지막으로, 아래 명령어를 실행하여 파인튜닝을 시작할 수 있습니다:

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
    `batch_size`, `gradient_accumulation_steps` 등의 학습 매개변수를 GPU 메모리에 맞게 조정하려면 `fish_speech/configs/text2semantic_finetune.yaml` 파일을 수정할 수 있습니다.

!!! note
    Windows 사용자의 경우, `nccl` 문제를 피하려면 `trainer.strategy.process_group_backend=gloo`를 사용할 수 있습니다.

훈련이 완료되면 [추론](inference.md) 섹션을 참고하여 음성을 생성할 수 있습니다.

!!! info
    기본적으로 모델은 화자의 말하는 패턴만 학습하고 음색은 학습하지 않습니다. 음색의 안정성을 위해 프롬프트를 사용해야 합니다.
    음색을 학습하려면 훈련 단계를 늘릴 수 있지만, 이는 과적합의 위험을 초래할 수 있습니다.

훈련이 끝나면 LoRA 가중치를 일반 가중치로 변환한 후에 추론을 수행해야 합니다.

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/fish-speech-1.5 \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/fish-speech-1.5-yth-lora/
```

!!! note
    다른 체크포인트도 시도해 볼 수 있습니다. 요구 사항에 맞는 가장 초기 체크포인트를 사용하는 것이 좋습니다. 이들은 종종 분포 밖(OOD) 데이터에서 더 좋은 성능을 발휘합니다.
