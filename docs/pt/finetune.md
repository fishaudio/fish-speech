# Ajuste Fino

É óbvio que ao abrir esta página, você não deve estar muito satisfeito com o desempenho do modelo pré-treinado com poucos exemplos. Você pode querer ajustar o modelo para melhorar seu desempenho em seu conjunto de dados.

Na atual versão, a única coisa que você precisa ajustar é a parte do 'LLAMA'.

## Ajuste Fino do LLAMA
### 1. Preparando o conjunto de dados

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

Você precisa converter seu conjunto de dados para o formato acima e colocá-lo em `data`. O arquivo de áudio pode ter as extensões `.mp3`, `.wav` ou `.flac`, e o arquivo de anotação deve ter a extensão `.lab`.

!!! warning
    É recomendado aplicar normalização de volume ao conjunto de dados. Você pode usar o [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) para fazer isso.

    ```bash
    fap loudness-norm data-raw data --clean
    ```


### 2. Extração em lote de tokens semânticos

Certifique-se de ter baixado os pesos do VQGAN. Se não, execute o seguinte comando:

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

Em seguida, você pode executar o seguinte comando para extrair os tokens semânticos:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "firefly_gan_vq" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

!!! note
    Você pode ajustar `--num-workers` e `--batch-size` para aumentar a velocidade de extração, mas certifique-se de não exceder o limite de memória da sua GPU.  
    Para o formato VITS, você pode especificar uma lista de arquivos usando `--filelist xxx.list`.

Este comando criará arquivos `.npy` no diretório `data`, como mostrado abaixo:

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

### 3. Empacotar o conjunto de dados em protobuf

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

Após executar o comando, você deverá ver o arquivo `quantized-dataset-ft.protos` no diretório `data`.

### 4. E finalmente, chegamos ao ajuste fino com LoRA

Da mesma forma, certifique-se de ter baixado os pesos do `LLAMA`. Se não, execute o seguinte comando:

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

E então, execute o seguinte comando para iniciar o ajuste fino:

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
    Se quiser, você pode modificar os parâmetros de treinamento, como `batch_size`, `gradient_accumulation_steps`, etc., para se ajustar à memória da sua GPU, modificando `fish_speech/configs/text2semantic_finetune.yaml`.

!!! note
    Para usuários do Windows, é recomendado usar `trainer.strategy.process_group_backend=gloo` para evitar problemas com `nccl`.

Após concluir o treinamento, consulte a seção [inferência](inference.md), e use `--speaker SPK1` para gerar fala.

!!! info
    Por padrão, o modelo aprenderá apenas os padrões de fala do orador e não o timbre. Ainda pode ser preciso usar prompts para garantir a estabilidade do timbre.
    Se quiser que ele aprenda o timbre, aumente o número de etapas de treinamento, mas isso pode levar ao overfitting (sobreajuste).

Após o treinamento, é preciso converter os pesos do LoRA em pesos regulares antes de realizar a inferência.

```bash
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/fish-speech-1.4 \
    --lora-weight results/$project/checkpoints/step_000000010.ckpt \
    --output checkpoints/fish-speech-1.4-yth-lora/
```
!!! note
    É possível também tentar outros checkpoints. Sugerimos usar o checkpoint que melhor atenda aos seus requisitos, pois eles geralmente têm um desempenho melhor em dados fora da distribuição (OOD).
