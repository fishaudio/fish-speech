# Inferência

Suporte para inferência por linha de comando, API HTTP e interface web (WebUI).

!!! note
    O processo de raciocínio, em geral, consiste em várias partes:

    1. Codificar cerca de 10 segundos de voz usando VQGAN.
    2. Inserir os tokens semânticos codificados e o texto correspondente no modelo de linguagem como um exemplo.
    3. Dado um novo trecho de texto, fazer com que o modelo gere os tokens semânticos correspondentes.
    4. Inserir os tokens semânticos gerados no VITS / VQGAN para decodificar e gerar a voz correspondente.

## Inferência por Linha de Comando

Baixe os modelos `vqgan` e `llama` necessários do nosso repositório Hugging Face.

```bash
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir checkpoints/fish-speech-1.4
```

### 1. Gerar prompt a partir da voz:

!!! note
    Se quiser permitir que o modelo escolha aleatoriamente um timbre de voz, pule esta etapa.

```bash
python tools/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

Você deverá obter um arquivo `fake.npy`.

### 2. Gerar tokens semânticos a partir do texto:

```bash
python tools/llama/generate.py \
    --text "O texto que você deseja converter" \
    --prompt-text "Seu texto de referência" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.4" \
    --num-samples 2 \
    --compile
```

Este comando criará um arquivo `codes_N` no diretório de trabalho, onde N é um número inteiro começando de 0.

!!! note
    Use `--compile` para fundir kernels CUDA para ter uma inferência mais rápida (~30 tokens/segundo -> ~500 tokens/segundo).
    Mas, se não planeja usar a aceleração CUDA, comente o parâmetro `--compile`.

!!! info
    Para GPUs que não suportam bf16, pode ser necessário usar o parâmetro `--half`.

### 3. Gerar vocais a partir de tokens semânticos:

#### Decodificador VQGAN

```bash
python tools/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

## Inferência por API HTTP

Fornecemos uma API HTTP para inferência. O seguinte comando pode ser usado para iniciar o servidor:

```bash
python -m tools.api \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/fish-speech-1.4" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

Para acelerar a inferência, adicione o parâmetro `--compile`.

Depois disso, é possível visualizar e testar a API em http://127.0.0.1:8080/.

Abaixo está um exemplo de envio de uma solicitação usando `tools/post_api.py`.

```bash
python -m tools.post_api \
    --text "Texto a ser inserido" \
    --reference_audio "Caminho para o áudio de referência" \
    --reference_text "Conteúdo de texto do áudio de referência" \
    --streaming True
```

O comando acima indica a síntese do áudio desejada de acordo com as informações do áudio de referência e a retorna em modo de streaming.

Caso selecione, de forma aleatória, o áudio de referência com base em `{SPEAKER}` e `{EMOTION}`, o configure de acordo com as seguintes etapas:

### 1. Crie uma pasta `ref_data` no diretório raiz do projeto.

### 2. Crie uma estrutura de diretórios semelhante à seguinte dentro da pasta `ref_data`.

```
.
├── SPEAKER1
│    ├──EMOTION1
│    │    ├── 21.15-26.44.lab
│    │    ├── 21.15-26.44.wav
│    │    ├── 27.51-29.98.lab
│    │    ├── 27.51-29.98.wav
│    │    ├── 30.1-32.71.lab
│    │    └── 30.1-32.71.flac
│    └──EMOTION2
│         ├── 30.1-32.71.lab
│         └── 30.1-32.71.mp3
└── SPEAKER2
    └─── EMOTION3
          ├── 30.1-32.71.lab
          └── 30.1-32.71.mp3
```

Ou seja, primeiro coloque as pastas `{SPEAKER}` em `ref_data`, depois coloque as pastas `{EMOTION}` em cada pasta de orador (speaker) e coloque qualquer número de `pares áudio-texto` em cada pasta de emoção.

### 3. Digite o seguinte comando no ambiente virtual

```bash
python tools/gen_ref.py

```

### 4. Chame a API.

```bash
python -m tools.post_api \
    --text "Texto a ser inserido" \
    --speaker "${SPEAKER1}" \
    --emotion "${EMOTION1}" \
    --streaming True
```

O exemplo acima é apenas para fins de teste.

## Inferência por WebUI

Para iniciar a WebUI de Inferência execute o seguinte comando:

```bash
python -m tools.webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.4" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

!!! note
    É possível usar variáveis de ambiente do Gradio, como `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME`, para configurar a WebUI.

Divirta-se!
