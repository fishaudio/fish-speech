# Inferência

Como o modelo vocoder foi alterado, você precisa de mais VRAM do que antes, sendo recomendado 12GB para inferência fluente.

Suportamos linha de comando, API HTTP e WebUI para inferência, você pode escolher qualquer método que preferir.

## Baixar Pesos

Primeiro você precisa baixar os pesos do modelo:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## Inferência por Linha de Comando

!!! note
    Se você planeja deixar o modelo escolher aleatoriamente um timbre de voz, pode pular esta etapa.

### 1. Obter tokens VQ do áudio de referência

```bash
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

Você deve obter um `fake.npy` e um `fake.wav`.

### 2. Gerar tokens semânticos do texto:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "O texto que você quer converter" \
    --prompt-text "Seu texto de referência" \
    --prompt-tokens "fake.npy" \
    --compile
```

Este comando criará um arquivo `codes_N` no diretório de trabalho, onde N é um inteiro começando de 0.

!!! note
    Você pode querer usar `--compile` para fundir kernels CUDA para inferência mais rápida (~30 tokens/segundo -> ~500 tokens/segundo).
    Correspondentemente, se você não planeja usar aceleração, pode comentar o parâmetro `--compile`.

!!! info
    Para GPUs que não suportam bf16, você pode precisar usar o parâmetro `--half`.

### 3. Gerar vocais a partir de tokens semânticos:

!!! warning "Aviso Futuro"
    Mantivemos a interface acessível do caminho original (tools/vqgan/inference.py), mas esta interface pode ser removida em versões subsequentes, então por favor altere seu código o mais breve possível.

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy"
```

## Inferência com API HTTP

Fornecemos uma API HTTP para inferência. Você pode usar o seguinte comando para iniciar o servidor:

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> Se você quiser acelerar a inferência, pode adicionar o parâmetro `--compile`.

Depois disso, você pode visualizar e testar a API em http://127.0.0.1:8080/.

## Inferência GUI 
[Baixar cliente](https://github.com/AnyaCoder/fish-speech-gui/releases)

## Inferência WebUI

Você pode iniciar o WebUI usando o seguinte comando:

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

Ou simplesmente

```bash
python -m tools.run_webui
```
> Se você quiser acelerar a inferência, pode adicionar o parâmetro `--compile`.

!!! note
    Você pode salvar o arquivo de rótulo e o arquivo de áudio de referência antecipadamente na pasta `references` no diretório principal (que você precisa criar), para que possa chamá-los diretamente no WebUI.

!!! note
    Você pode usar variáveis de ambiente do Gradio, como `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` para configurar o WebUI.
