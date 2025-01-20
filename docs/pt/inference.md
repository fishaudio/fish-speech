# Inferência

Suporte para inferência por linha de comando, API HTTP e interface web (WebUI).

!!! note
    O processo de raciocínio, em geral, consiste em várias partes:

    1. Codificar cerca de 10 segundos de voz usando VQGAN.
    2. Inserir os tokens semânticos codificados e o texto correspondente no modelo de linguagem como um exemplo.
    3. Dado um novo trecho de texto, fazer com que o modelo gere os tokens semânticos correspondentes.
    4. Inserir os tokens semânticos gerados no VITS / VQGAN para decodificar e gerar a voz correspondente.

## Baixar modelos
Baixe os modelos `vqgan` e `llama` necessários do nosso repositório Hugging Face.

```bash
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5
```

## Inferência por Linha de Comando
### 1. Gerar prompt a partir da voz:

!!! note
    Se quiser permitir que o modelo escolha aleatoriamente um timbre de voz, pule esta etapa.

!!! warning "Aviso de Versão Futura"
    Mantivemos a interface acessível a partir do caminho original (tools/vqgan/infernce.py), mas esta interface poderá ser removida em algumas versões futuras. Por favor, altere o seu código o mais breve possível.

```bash
python fish_speech/models/vqgan/inference.py \
    -i "paimon.wav" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

Você deverá obter um arquivo `fake.npy`.

### 2. Gerar tokens semânticos a partir do texto:

!!! warning "Aviso de Versão Futura"
    Mantivemos a interface acessível a partir do caminho original (tools/llama/generate.py), mas esta interface poderá ser removida em algumas versões futuras. Por favor, altere o seu código o mais breve possível.

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "O texto que você deseja converter" \
    --prompt-text "Seu texto de referência" \
    --prompt-tokens "fake.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5" \
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

!!! warning "Aviso de Versão Futura"
    Mantivemos a interface acessível a partir do caminho original (tools/vqgan/infernce.py), mas esta interface poderá ser removida em algumas versões futuras. Por favor, altere o seu código o mais breve possível.

```bash
python fish_speech/models/vqgan/inference.py \
    -i "codes_0.npy" \
    --checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
```

## Inferência por API HTTP

Fornecemos uma API HTTP para inferência. O seguinte comando pode ser usado para iniciar o servidor:

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/fish-speech-1.5" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```

> Para acelerar a inferência, adicione o parâmetro `--compile`.

Depois disso, é possível visualizar e testar a API em http://127.0.0.1:8080/.

Abaixo está um exemplo de envio de uma solicitação usando `tools/api_client.py`.

```bash
python -m tools.api_client \
    --text "Texto a ser inserido" \
    --reference_audio "Caminho para o áudio de referência" \
    --reference_text "Conteúdo de texto do áudio de referência" \
    --streaming True
```

O comando acima indica a síntese do áudio desejada de acordo com as informações do áudio de referência e a retorna em modo de streaming.

!!! info
    Para aprender mais sobre parâmetros disponíveis, você pode usar o comando `python -m tools.api_client -h`

## Inferência por WebUI

Para iniciar a WebUI de Inferência execute o seguinte comando:

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/fish-speech-1.5" \
    --decoder-checkpoint-path "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth" \
    --decoder-config-name firefly_gan_vq
```
> Para acelerar a inferência, adicione o parâmetro `--compile`.

!!! note
    Você pode salvar antecipadamente o arquivo de rótulos e o arquivo de áudio de referência na pasta `references` do diretório principal (que você precisa criar), para que possa chamá-los diretamente na WebUI.
    
!!! note
    É possível usar variáveis de ambiente do Gradio, como `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME`, para configurar a WebUI.

Divirta-se!
