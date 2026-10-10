# Inferencia

El modelo Fish Audio S2 requiere una gran cantidad de VRAM. Recomendamos usar una GPU con al menos 24GB para la inferencia.

## Descargar Pesos

Primero, necesitas descargar los pesos del modelo:

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

## Inferencia por Línea de Comandos

!!! note
Si planeas dejar que el modelo elija aleatoriamente el timbre de voz, puedes omitir este paso.

### 1. Obtener tokens VQ a partir de audio de referencia

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

Deberías obtener un `fake.npy` y un `fake.wav`.

### 2. Generar tokens semánticos a partir de texto:

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "El texto que quieres convertir" \
    --prompt-text "Tu texto de referencia" \
    --prompt-tokens "fake.npy" \
    # --compile
```

Este comando creará un archivo `codes_N` en el directorio de trabajo, donde N es un entero que comienza desde 0.

!!! note
Puede que quieras usar `--compile` para fusionar kernels CUDA y acelerar la inferencia. Sin embargo, recomendamos usar nuestra optimización de aceleración de inferencia con sglang.
Correspondientemente, si no planeas usar aceleración, puedes comentar el parámetro `--compile`.

!!! info
Para GPUs que no soportan bf16, puede que necesites usar el parámetro `--half`.

### 3. Generar audio a partir de tokens semánticos:

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

Después de eso, obtendrás un archivo `fake.wav`.

## S3-drama

S3-drama usa la misma línea de comandos y el mismo servidor API que S2-Pro; el tipo de modelo se detecta a partir del `config.json` del checkpoint.

!!! warning
    S3-drama solo funciona en GPUs NVIDIA. No se admiten CPU, Apple MPS ni AMD ROCm, y tampoco `--compile`.

| GPU | Modo | Memoria de GPU |
| --- | --- | --- |
| SM89 o posterior (L40, H100, H20) | Expertos MoE en FP8, decodificación con CUDA graphs, una sola GPU | unos 41 GB |
| GPUs anteriores (V100, A100) | Expertos MoE descuantizados y repartidos entre varias GPUs | unos 70 GB en total |

### Descargar pesos

```bash
hf download fishaudio/s3-drama-it-fp8 --local-dir checkpoints/s3-drama-fp8
```

Si el directorio del checkpoint no tiene `codec.pth`, se usa el códec de S2-Pro en `checkpoints/s2-pro/codec.pth`.

### Inferencia por línea de comandos

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/s3-drama-fp8 \
    --text "El texto que quieres convertir" \
    --prompt-audio "reference.wav" \
    --prompt-text "Tu texto de referencia" \
    --output "output.wav" \
    --device cuda:0
```

`--output` escribe directamente el audio decodificado. En GPUs anteriores, indica varios dispositivos, por ejemplo `--device cuda:0,cuda:1,cuda:2,cuda:3`; el último se encarga del decodificador de audio y del códec. Con solo `--device cuda` se usan hasta las cuatro primeras GPUs visibles.

### Servidor API

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s3-drama-fp8 \
    --device cuda:0
```

Con Docker, ejecuta la imagen del servidor apuntando al checkpoint de S3. En tarjetas SM89 o posteriores, expón una sola GPU como abajo; en tarjetas anteriores, usa `--gpus all`:

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s3-drama-fp8 \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s3-drama-fp8/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### Diferencias con S2-Pro

- Los textos largos se dividen en fragmentos de unos 300 bytes (UTF-8), preferiblemente al final de las frases, y los fragmentos se generan uno tras otro. El primer fragmento y el más reciente se vuelven a pasar como contexto para mantener la voz coherente. Por defecto, cada fragmento produce como máximo 600 frames (unos 28 segundos).
- Activa `single_chunk` (`--single-chunk` en la línea de comandos) para generar todo el texto de una vez, por defecto como máximo 2048 frames (unos 95 segundos). En ambos modos, un `max_new_tokens` mayor se limita a 10.240 frames menos la longitud del prompt.
- Solo se acepta la etiqueta de hablante `<|speaker:0|>`.
- `chunk_length` y `repetition_penalty` no tienen efecto; S3 incorpora muestreo con detección de repeticiones. El muestreo por defecto es temperature 1.0, top-p 0.9, top-k 30.
- Con `streaming` activado, el audio se devuelve fragmento a fragmento.
- `FISH_S3_FP8_EXPERTS=0` o `1` fuerza expertos descuantizados o en FP8, y `FISH_S3_CUDA_GRAPH=0` desactiva la decodificación con CUDA graphs.

## Inferencia con WebUI

Próximamente.
