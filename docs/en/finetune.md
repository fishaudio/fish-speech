# Fine-tuning

Obviously, when you opened this page, you were not satisfied with the performance of the few-shot pre-trained model. You want to fine-tune a model to improve its performance on your dataset.

`Fish Speech` consists of two modules: `VQGAN` and `LLAMA`. Currently, we only support fine-tuning the `LLAMA` model.

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

You need to convert your dataset into the above format and place it under `data/demo`. The audio file can have the extensions `.mp3`, `.wav`, or `.flac`, and the annotation file can have the extensions `.lab` or `.txt`.

!!! note
    You can modify the dataset path and mix datasets by modifying `fish_speech/configs/data/finetune.yaml`.

### 2. Batch-wise extraction of semantic tokens

Make sure you have downloaded the VQGAN weights. If not, run the following command:

```bash
huggingface-cli download fishaudio/speech-lm-v1 vqgan-v1.pth --local-dir checkpoints
```

You can then run the following command to extract semantic tokens:

```bash
python tools/vqgan/extract_vq.py data/demo \
    --num-workers 1 --batch-size 16 \
    --config-name "vqgan_pretrain" \
    --checkpoint-path "checkpoints/vqgan-v1.pth"
```

!!! note
    You can adjust `--num-workers` and `--batch-size` to increase extraction speed, but please make sure not to exceed your GPU memory limit.

This command will create `.npy` files in the `data/demo` directory, as shown below:

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
    --config "fish_speech/configs/data/finetune.yaml" \
    --output "data/quantized-dataset-ft.protos"
```

After the command finishes executing, you should see the `quantized-dataset-ft.protos` file in the `data` directory.

### 4. Start the Rust data server

Loading and shuffling the dataset is very slow and memory-consuming. Therefore, we use a Rust server to load and shuffle the data. This server is based on GRPC and can be installed using the following method:

```bash
cd data_server
cargo build --release
```

After the compilation is complete, you can start the server using the following command:

```bash
export RUST_LOG=info # Optional, for debugging
data_server/target/release/data_server \
    --files "data/quantized-dataset-ft.protos" 
```

!!! note
    You can specify multiple `--files` parameters to load multiple datasets.

### 5. Finally, start the fine-tuning

Similarly, make sure you have downloaded the `LLAMA` weights. If not, run the following command:

```bash
huggingface-cli download fishaudio/speech-lm-v1 text2semantic-400m-v0.2-4k.pth --local-dir checkpoints
```

Finally, you can start the fine-tuning by running the following command:
```bash
python fish_speech/train.py --config-name text2semantic_finetune_spk
```

!!! note
    You can modify the training parameters such as `batch_size`, `gradient_accumulation_steps`, etc. to fit your GPU memory by modifying `fish_speech/configs/text2semantic_finetune_spk.yaml`.

After training is complete, you can refer to the inference section to generate speech.

!!! info
    By default, the model will only learn the speaker's speech patterns and not the timbre. You still need to use prompts to ensure timbre stability.
    If you want to learn the timbre, you can increase the number of training steps, but this may lead to overfitting.
