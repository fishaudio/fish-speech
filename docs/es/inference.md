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

## Inferencia con WebUI

Próximamente.
