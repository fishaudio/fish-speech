# Inferência

O modelo Fish Audio S2 requer uma grande quantidade de VRAM. Recomendamos o uso de uma GPU com pelo menos 24GB para inferência.

## Baixar Pesos

Primeiro, você precisa baixar os pesos do modelo:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## Inferência por Linha de Comando

!!! note
    Se você planeja deixar o modelo escolher aleatoriamente um timbre de voz, pode pular esta etapa.

### 1. Obter tokens VQ do áudio de referência

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

Você deve obter um `fake.npy` e um `fake.wav`.

### 2. Gerar tokens Semânticos a partir do texto:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "O texto que você deseja converter" \
    --prompt-text "Seu texto de referência" \
    --prompt-tokens "fake.npy" \
    # --compile
```

Este comando criará um arquivo `codes_N` no diretório de trabalho, onde N é um número inteiro começando em 0.

!!! note
    Você pode querer usar `--compile` para fundir kernels CUDA para uma inferência mais rápida. No entanto, recomendamos usar nossa otimização de aceleração de inferência sglang.
    Da mesma forma, se você não planeja usar aceleração, pode comentar o parâmetro `--compile`.

!!! info
    Para GPUs que não suportam bf16, você pode precisar usar o parâmetro `--half`.

### 3. Gerar vocais a partir de tokens semânticos:

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

Depois disso, você obterá um arquivo `fake.wav`.

## Inferência WebUI

Em breve.
