# Fine-tuning

Obviously, when you opened this page, you were not satisfied with the performance of the few-shot pre-trained model. You want to fine-tune a model to improve its performance on your dataset.

`Fish Speech` consists of two modules: `VQGAN` and `LLAMA`.

!!! info 
    You should first conduct the following test to determine if you need to fine-tune `VQGAN`:
    ```bash
    python tools/vqgan/inference.py -i test.wav
    ```
    This test will generate a `fake.wav` file. If the timbre of this file differs from the speaker's original voice, or if the quality is not high, you need to fine-tune `VQGAN`.

    Similarly, you can refer to [Inference](inference.md) to run `generate.py` and evaluate if the prosody meets your expectations. If it does not, then you need to fine-tune `LLAMA`.

## Fine-tuning VQGAN
### 1. Prepare the Dataset

```
.
├── SPK1
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.mp3
│   └── 30.1-32.71.mp3
└── SPK2
    └── 38.79-40.85.mp3
```

You need to format your dataset as shown above and place it under `data`. Audio files can have `.mp3`, `.wav`, or `.flac` extensions.

### 2. Split Training and Validation Sets

```bash
python tools/vqgan/create_train_split.py data
```

This command will create `data/vq_train_filelist.txt` and `data/vq_val_filelist.txt` in the `data/demo` directory, to be used for training and validation respectively.

!!!info
    For the VITS format, you can specify a file list using `--filelist xxx.list`.  
    Please note that the audio files in `filelist` must also be located in the `data` folder.

### 3. Start Training

```bash
python fish_speech/train.py --config-name vqgan_finetune
```

!!! note
    You can modify training parameters by editing `fish_speech/configs/vqgan_finetune.yaml`, but in most cases, this won't be necessary.

### 4. Test the Audio
    
```bash
python tools/vqgan/inference.py -i test.wav --checkpoint-path results/vqgan_finetune/checkpoints/step_000010000.ckpt
```

You can review `fake.wav` to assess the fine-tuning results.

!!! note
    You may also try other checkpoints. We suggest using the earliest checkpoint that meets your requirements, as they often perform better on out-of-distribution (OOD) data.

## Fine-tuning LLAMA
### 1. Prepare the dataset

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

You need to convert your dataset into the above format and place it under `data`. The audio file can have the extensions `.mp3`, `.wav`, or `.flac`, and the annotation file should have the extensions `.lab`.

!!! warning
    It's recommended to apply loudness normalization to the dataset. You can use [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) to do this.

    ```bash
    fap loudness-norm data-raw data --clean
    ```


### 2. Batch extraction of semantic tokens

Make sure you have downloaded the VQGAN weights. If not, run the following command:

```bash
huggingface-cli download fishaudio/fish-speech-1 vq-gan-group-fsq-2x1024.pth --local-dir checkpoints
```

You can then run the following command to extract semantic tokens:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "vqgan_pretrain" \
    --checkpoint-path "checkpoints/vq-gan-group-fsq-2x1024.pth"
```

!!! note
    You can adjust `--num-workers` and `--batch-size` to increase extraction speed, but please make sure not to exceed your GPU memory limit.  
    For the VITS format, you can specify a file list using `--filelist xxx.list`.

This command will create `.npy` files in the `data` directory, as shown below:

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

### 3. Pack the dataset into protobuf

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/quantized-dataset-ft.protos" \
    --text-extension .lab \
    --num-workers 16
```

After the command finishes executing, you should see the `quantized-dataset-ft.protos` file in the `data` directory.

!!!info
    For the VITS format, you can specify a file list using `--input xxx.list`.

### 4. Finally, start the fine-tuning

Similarly, make sure you have downloaded the `LLAMA` weights. If not, run the following command:

```bash
huggingface-cli download fishaudio/fish-speech-1 text2semantic-sft-medium-v1-4k.pth --local-dir checkpoints
```

Finally, you can start the fine-tuning by running the following command:
```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    model@model.model=dual_ar_2_codebook_medium
```

!!! note
    You can modify the training parameters such as `batch_size`, `gradient_accumulation_steps`, etc. to fit your GPU memory by modifying `fish_speech/configs/text2semantic_finetune.yaml`.

!!! note
    For Windows users, you can use `trainer.strategy.process_group_backend=gloo` to avoid `nccl` issues.

After training is complete, you can refer to the [inference](inference.md) section, and use `--speaker SPK1` to generate speech.

!!! info
    By default, the model will only learn the speaker's speech patterns and not the timbre. You still need to use prompts to ensure timbre stability.
    If you want to learn the timbre, you can increase the number of training steps, but this may lead to overfitting.

#### Fine-tuning with LoRA

!!! note
    LoRA can reduce the risk of overfitting in models, but it may also lead to underfitting on large datasets. 

If you want to use LoRA, please add the following parameter: `+lora@model.lora_config=r_8_alpha_16`. 

After training, you need to convert the LoRA weights to regular weights before performing inference.

```bash
python tools/llama/merge_lora.py \
    --llama-config dual_ar_2_codebook_medium \
    --lora-config r_8_alpha_16 \
    --llama-weight checkpoints/text2semantic-sft-medium-v1-4k.pth \
    --lora-weight results/text2semantic-finetune-medium-lora/checkpoints/step_000000200.ckpt \
    --output checkpoints/merged.ckpt
```
