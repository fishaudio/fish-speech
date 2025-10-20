# Ajuste Fino (Fine-tuning)

Obviamente, ao abrir esta página, você não estava satisfeito com o desempenho do modelo pré-treinado em modo zero-shot. Você deseja fazer um ajuste fino em um modelo para melhorar seu desempenho em seu conjunto de dados.

Na versão atual, você só precisa fazer o ajuste fino da parte 'LLAMA'.

## Ajuste Fino do LLAMA
### 1. Prepare o conjunto de dados

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

Você precisa converter seu conjunto de dados para o formato acima e colocá-lo no diretório `data`. O arquivo de áudio pode ter as extensões `.mp3`, `.wav` ou `.flac`, e o arquivo de anotação deve ter a extensão `.lab`.

!!! info
    O arquivo de anotação `.lab` precisa conter apenas a transcrição do áudio, sem necessidade de formatação especial. Por exemplo, se `hi.mp3` contiver "Olá, adeus.", então o arquivo `hi.lab` conterá uma única linha de texto: "Olá, adeus.".

!!! warning
    Recomenda-se aplicar a normalização de volume (loudness) ao conjunto de dados. Você pode usar o [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) para fazer isso.
    ```bash
    fap loudness-norm data-raw data --clean
    ```

### 2. Extração em lote de tokens semânticos

Certifique-se de que você baixou os pesos do VQGAN. Se não, execute o seguinte comando:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

Em seguida, você pode executar o seguinte comando para extrair os tokens semânticos:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

!!! note
    Você pode ajustar `--num-workers` e `--batch-size` para aumentar a velocidade de extração, mas certifique-se de não exceder o limite de memória da sua GPU.

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

### 3. Empacote o conjunto de dados em protobuf

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

Após a conclusão da execução do comando, você deverá ver o arquivo `protos` no diretório `data`.

### 4. Finalmente, ajuste fino com LoRA

Da mesma forma, certifique-se de que você baixou os pesos do `LLAMA`. Se não, execute o seguinte comando:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

Finalmente, você pode iniciar o ajuste fino executando o seguinte comando:

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
    Você pode modificar os parâmetros de treinamento, como `batch_size`, `gradient_accumulation_steps`, etc., para se adequar à memória da sua GPU, modificando `fish_speech/configs/text2semantic_finetune.yaml`.

!!! note
    Para usuários do Windows, você pode usar `trainer.strategy.process_group_backend=gloo` para evitar problemas com `nccl`.

Após o treinamento ser concluído, você pode consultar a seção de [inferência](inference.md) para testar seu modelo.

!!! info
    Por padrão, o modelo aprenderá apenas os padrões de fala do locutor e não o timbre. Você ainda precisará usar prompts para garantir a estabilidade do timbre.
    Se você quiser aprender o timbre, pode aumentar o número de passos de treinamento, mas isso pode levar a um sobreajuste (overfitting).

Após o treinamento, você precisa converter os pesos do LoRA para pesos regulares antes de realizar a inferência.

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/openaudio-s1-mini \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/openaudio-s1-mini-yth-lora/```
!!! note
    Você também pode tentar outros checkpoints. Sugerimos usar o checkpoint mais antigo que atenda aos seus requisitos, pois eles geralmente têm um desempenho melhor em dados fora de distribuição (OOD).
