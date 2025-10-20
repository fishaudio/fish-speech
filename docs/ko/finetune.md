# 미세 조정 (Fine-tuning)

이 페이지를 열었다는 것은, 사전 훈련된 모델의 제로샷(zero-shot) 성능에 만족하지 못했다는 의미일 것입니다. 여러분의 데이터셋에서 더 나은 성능을 내도록 모델을 미세 조정하고 싶으실 겁니다.

현재 버전에서는 'LLAMA' 부분만 미세 조정하면 됩니다.

## LLAMA 미세 조정
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

데이터셋을 위 형식으로 변환하여 `data` 폴더 아래에 배치해야 합니다. 오디오 파일 확장자는 `.mp3`, `.wav` 또는 `.flac`일 수 있으며, 주석 파일 확장자는 `.lab`을 권장합니다.

!!! info
    `.lab` 주석 파일에는 오디오의 전사 텍스트만 포함하면 되며, 특별한 형식 요구사항은 없습니다. 예를 들어 `hi.mp3`의 내용이 "안녕하세요, 안녕히 가세요."라면, `hi.lab` 파일에는 "안녕하세요, 안녕히 가세요."라는 한 줄의 텍스트만 포함하면 됩니다.

!!! warning
    데이터셋에 음량 정규화를 적용하는 것이 좋습니다. 이를 위해 [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess)를 사용할 수 있습니다.
    ```bash
    fap loudness-norm data-raw data --clean
    ```

### 2. 시맨틱 토큰 일괄 추출

VQGAN 가중치를 다운로드했는지 확인하세요. 그렇지 않은 경우 다음 명령을 실행하세요.

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

그런 다음 다음 명령을 실행하여 시맨틱 토큰을 추출할 수 있습니다.

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

!!! note
    `--num-workers`와 `--batch-size`를 조정하여 추출 속도를 높일 수 있지만, GPU 메모리 한도를 초과하지 않도록 주의하세요.

이 명령은 `data` 디렉토리에 `.npy` 파일을 생성합니다. 결과는 다음과 같습니다.

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

### 3. 데이터셋을 protobuf로 패킹하기

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

명령 실행이 완료되면 `data` 디렉토리에서 `protos` 파일을 볼 수 있어야 합니다.

### 4. 마지막으로, LoRA로 미세 조정하기

마찬가지로, `LLAMA` 가중치를 다운로드했는지 확인하세요. 그렇지 않은 경우 다음 명령을 실행하세요.

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

마지막으로, 다음 명령을 실행하여 미세 조정을 시작할 수 있습니다.

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
    `fish_speech/configs/text2semantic_finetune.yaml` 파일을 수정하여 `batch_size`, `gradient_accumulation_steps` 등 훈련 매개변수를 GPU 메모리에 맞게 조정할 수 있습니다.

!!! note
    Windows 사용자의 경우, `trainer.strategy.process_group_backend=gloo`를 사용하여 `nccl` 관련 문제를 피할 수 있습니다.

훈련이 완료되면 [추론](inference.md) 섹션을 참조하여 모델을 테스트할 수 있습니다.

!!! info
    기본 설정에서는 모델이 화자의 발음 방식만 학습하고 음색은 학습하지 않습니다. 음색 안정성을 보장하려면 여전히 프롬프트를 사용해야 합니다.
    음색을 학습시키고 싶다면 훈련 스텝 수를 늘리되, 이는 과적합(overfitting)으로 이어질 수 있습니다.

훈련 후, 추론을 수행하기 전에 LoRA 가중치를 일반 가중치로 변환해야 합니다.

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/openaudio-s1-mini \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/openaudio-s1-mini-yth-lora/
```

!!! note
    다른 체크포인트를 시도해 볼 수도 있습니다. 요구 사항을 충족하는 가장 이른 체크포인트를 사용하는 것이 좋습니다. 이러한 체크포인트는 보통 OOD(분포 외) 데이터에서 더 나은 성능을 보입니다.
