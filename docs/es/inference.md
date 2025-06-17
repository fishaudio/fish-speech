# Inferencia

Como el modelo vocoder ha cambiado, ahora necesitas más VRAM que antes; se recomienda tener 12GB para una inferencia fluida

Se admite inferencia por línea de comandos, API HTTP y WebUI, puedes usar el método que prefieras.

## Descargar pesos del modelo

Primero necesitas descargar los pesos del modelo:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

## Inferencia por Línea de Comandos

!!! nota
    Si planeas permitir que el modelo elija aleatoriamente un timbre de voz, puedes omitir este paso.

### 1. Obtener tokens VQ desde un audio de referencia

```bash
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

Deberías obtener un archivo `fake.npy` y un `fake.wav`.

### 2. Generar tokens semánticos a partir de texto:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    --compile
```

Este comando creará un archivo `codes_N` en el directorio de trabajo, donde N es un número entero empezando desde 0.

!!! nota
    Puedes usar `--compile` para fusionar kernels de CUDA y acelerar la inferencia (~15 tokens/segundo → ~150 tokens/segundo en una GPU RTX 4090).
    Si no planeas usar aceleración, puedes comentar el parámetro `--compile`.

!!! info
    Para GPUs que no soportan bf16, puede que necesites usar el parámetro `--half`.

### 3. Generar audio vocal a partir de los tokens semánticos:

!!! advertencia "Advertencia futura"
    La interfaz original (tools/vqgan/inference.py) todavía está disponible, pero podría eliminarse en futuras versiones. Se recomienda actualizar tu código lo antes posible.

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

## Inferencia por API HTTP

Proveemos una API HTTP para realizar inferencia. Puedes iniciar el servidor con el siguiente comando:

```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> Para acelerar la inferencia, puedes agregar el parámetro `--compile`.

Después de esto, puedes visualizar y probar la API en http://127.0.0.1:8080/.

## Inferencia con Interfaz Gráfica (GUI) 
[Descargar cliente](https://github.com/AnyaCoder/fish-speech-gui/releases)

## Inferencia mediante WebUI

Puedes iniciar la WebUI con el siguiente comando:

```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

O simplemente:

```bash
python -m tools.run_webui
```
> Para acelerar la inferencia, puedes agregar el parámetro `--compile`.


!!! nota
    Puedes guardar el archivo de etiquetas y el audio de referencia previamente en la carpeta `references` (que debes crear tú mismo) dentro del directorio principal, para poder llamarlos directamente desde la WebUI.

!!! nota
    Puedes usar variables de entorno de Gradio, como `GRADIO_SHARE`, `GRADIO_SERVER_PORT`, `GRADIO_SERVER_NAME` para configurar la WebUI.

¡Disfruta!
