# Fine-tuning

!!! warning
Recomendamos encarecidamente no realizar fine-tuning sobre un modelo entrenado con RL. Ajustar un modelo después de RL puede cambiar la distribución del modelo, lo que puede llevar a una degradación del rendimiento.

En la versión actual, solo necesitas hacer fine-tuning de la parte ‘LLAMA’.

## Fine-tuning LLAMA

### 1. Preparar el dataset

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

Necesitas convertir tu dataset al formato anterior y colocarlo dentro de `data`. El archivo de audio puede tener extensiones `.mp3`, `.wav` o `.flac`, y el archivo de anotación debe tener la extensión `.lab`.

!!! info
El archivo de anotación `.lab` solo necesita contener la transcripción del audio, sin ningún formato especial. Por ejemplo, si `hi.mp3` dice "Hello, goodbye," entonces el archivo `hi.lab` contendría una única línea de texto: "Hello, goodbye."

!!! warning
Se recomienda aplicar normalización de loudness al dataset. Puedes usar [fish-audio-preprocess](https://github.com/fishaudio/audio-preprocess) para hacerlo.

````
```bash
fap loudness-norm data-raw data --clean
```
````

### 2. Extracción por lotes de tokens semánticos

Asegúrate de haber descargado los pesos de VQGAN. Si no, ejecuta el siguiente comando:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

Luego puedes ejecutar el siguiente comando para extraer los tokens semánticos:

```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

!!! note
Puedes ajustar `--num-workers` y `--batch-size` para aumentar la velocidad de extracción, pero asegúrate de no exceder el límite de memoria de tu GPU.

Este comando creará archivos `.npy` en el directorio `data`, como se muestra a continuación:

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

### 3. Empaquetar el dataset en protobuf

```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

Después de que el comando termine de ejecutarse, deberías ver el archivo `protos` en el directorio `data`.

### 4. Finalmente, fine-tuning con LoRA

De manera similar, asegúrate de haber descargado los pesos de `LLAMA`. Si no, ejecuta el siguiente comando:

```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

Finalmente, puedes comenzar el fine-tuning ejecutando el siguiente comando:

```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

!!! note
Puedes modificar parámetros de entrenamiento como `batch_size`, `gradient_accumulation_steps`, etc., para ajustarlos a la memoria de tu GPU editando `fish_speech/configs/text2semantic_finetune.yaml`.

!!! note
Para usuarios de Windows, puedes usar `trainer.strategy.process_group_backend=gloo` para evitar problemas con `nccl`.

Una vez que el entrenamiento esté completo, puedes consultar la sección de [inference](inference.md) para probar tu modelo.

!!! info
Por defecto, el modelo solo aprenderá los patrones de habla del hablante y no el timbre. Aún necesitas usar prompts para asegurar la estabilidad del timbre.
Si quieres aprender el timbre, puedes aumentar el número de pasos de entrenamiento, pero esto puede llevar a overfitting.

Después del entrenamiento, necesitas convertir los pesos LoRA a pesos normales antes de realizar inferencia.

```bash
python tools/llama/merge_lora.py \
	--lora-config r_8_alpha_16 \
	--base-weight checkpoints/openaudio-s1-mini \
	--lora-weight results/$project/checkpoints/step_000000010.ckpt \
	--output checkpoints/openaudio-s1-mini-yth-lora/
```

!!! note
También puedes probar otros checkpoints. Sugerimos usar el checkpoint más temprano que cumpla con tus requisitos, ya que suelen rendir mejor en datos fuera de distribución (OOD).
