# Inferencia

Fish Audio ofrece dos modelos: **S2-Pro** y **Drama 3**. Comparten la misma línea de comandos y el mismo servidor API, y el tipo de modelo se lee del `config.json` del directorio de pesos, por lo que cada sección siguiente está organizada como «tarea → comando de cada modelo».

Antes de empezar, completa la [Instalación](install.md) y descarga los [pesos del modelo](install.md#descargar-pesos).

!!! warning "Límites de ejecución de Drama 3"
    Drama 3 solo funciona en GPUs NVIDIA: no es compatible con CPU, Apple MPS ni AMD ROCm, y tampoco con `--compile`. Consulta los [Requisitos](install.md#requisitos) para conocer la memoria GPU necesaria.

!!! note
    La V100 y otras GPUs anteriores a SM80 no tienen bfloat16, por lo que Drama 3 cambia a float16 (manteniendo el flujo residual en fp32). Esta ruta aún no se ha probado en hardware real; si obtienes silencio, ruido o un error "non-finite logits", abre una incidencia.

## Inferencia por Línea de Comandos

### S2-Pro

S2-Pro usa un pipeline de tres pasos: «tokens VQ → tokens semánticos → audio».

!!! note
    Si quieres que el modelo elija el timbre al azar, puedes omitir el paso 1.

**1. Obtener tokens VQ a partir de audio de referencia**

```bash
python fish_speech/models/dac/inference.py \
    -i "test.wav" \
    --checkpoint-path "checkpoints/s2-pro/codec.pth"
```

Deberías obtener un `fake.npy` y un `fake.wav`.

**2. Generar tokens semánticos a partir de texto:**

```bash
python fish_speech/models/text2semantic/inference.py \
    --text "El texto que quieres convertir" \
    --prompt-text "Tu texto de referencia" \
    --prompt-tokens "fake.npy" \
    # --compile
```

Este comando creará un archivo `codes_N` en el directorio de trabajo, donde N es un entero que empieza en 0.

!!! note
    Puede que quieras usar `--compile` para fusionar kernels CUDA y acelerar la inferencia. Sin embargo, te recomendamos nuestra optimización de aceleración sglang.
    Si no planeas usar aceleración, puedes comentar el parámetro `--compile`.

!!! info
    En GPUs que no admiten bf16, puede que necesites el parámetro `--half`.

**3. Generar audio a partir de tokens semánticos:**

```bash
python fish_speech/models/dac/inference.py \
    -i "codes_0.npy" \
```

Después de esto obtendrás un archivo `fake.wav`.

### Drama 3

Drama 3 lo hace en un solo comando; `--output` escribe directamente el audio decodificado:

```bash
python fish_speech/models/text2semantic/inference.py \
    --checkpoint-path checkpoints/drama-3 \
    --text "El texto que quieres convertir" \
    --prompt-audio "reference.wav" \
    --prompt-text "Tu texto de referencia" \
    --output "output.wav" \
    --device cuda:0
```

En GPUs más antiguas puedes indicar varios dispositivos, por ejemplo `--device cuda:0,cuda:1,cuda:2,cuda:3`; el último aloja el decodificador de audio y el codec. Si escribes solo `--device cuda`, se usan hasta las cuatro primeras GPUs visibles.

## Servidor API

### Iniciar S2-Pro

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq
```

### Iniciar Drama 3

```bash
python tools/api_server.py \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path checkpoints/drama-3 \
    --device cuda:0
```

!!! note
    Drama 3 no admite `--compile`. En GPUs más antiguas, enumera varios dispositivos con `--device`; el último aloja el decodificador de audio y el codec.

Opciones de arranque comunes a ambos modelos:

- `--compile`: activa la optimización `torch.compile` (solo S2-Pro)
- `--half`: usa el modo fp16
- `--api-key`: exige autenticación con bearer token
- `--workers`: establece el número de procesos de trabajo
- `--max-text-length`: limita la longitud del texto de cada petición; 0 significa sin límite

Comprobación de salud:

```bash
curl -X GET http://127.0.0.1:8080/v1/health
```

La respuesta `{"status":"ok"}` indica que el servicio funciona correctamente.

### Endpoints de la API

| Método | Ruta | Descripción |
| --- | --- | --- |
| `POST` | `/v1/tts` | Texto a voz; el cuerpo de la petición es msgpack |
| `GET` | `/v1/tts/defaults` | Valores de muestreo por defecto del modelo cargado |
| `GET` | `/v1/health` | Comprobación de salud |
| `POST` | `/v1/vqgan/encode` | Audio de referencia → tokens VQ |
| `POST` | `/v1/vqgan/decode` | Tokens VQ → audio |
| `POST` | `/v1/references/add` | Guarda una voz de referencia; campos `id`, `audio` (archivo), `text` |
| `GET` | `/v1/references/list` | Lista las voces guardadas |
| `POST` | `/v1/references/update` | Renombra; campos `old_reference_id`, `new_reference_id` |
| `DELETE` | `/v1/references/delete` | Elimina; campo `reference_id` |
| `GET` | `/ui` | WebUI (incluida en la imagen Docker; hay que compilarla al ejecutar desde el código fuente) |

Una vez guardada la voz, úsala con `--reference_id` (el campo de audio necesita su tipo MIME; de lo contrario el servidor no lo reconoce):

```bash
curl -X POST http://127.0.0.1:8080/v1/references/add \
    -F id=my-speaker \
    -F "audio=@reference.wav;type=audio/wav" \
    -F text="Transcripción del audio de referencia"

python tools/api_client.py --url http://127.0.0.1:8080/v1/tts \
    --text "Hello" --reference_id my-speaker --output demo
```

### Enviar una petición

El modelo que se usa depende de los pesos que el servidor cargó al iniciarse; no hay ningún campo de modelo en la petición.

```bash
python tools/api_client.py \
    --url http://127.0.0.1:8080/v1/tts \
    --text "Hello from Fish Speech" \
    --output demo
```

Parámetros habituales:

| Parámetro | Descripción |
| --- | --- |
| `--reference_audio` / `--reference_text` | Audio de referencia y su transcripción; ambos necesitan el mismo número de valores |
| `--reference_speaker` | ID de hablante (0–15) de cada audio de referencia, escrito como prefijo `<\|speaker:N\|>` (el uso principal en Drama 3) |
| `--reference_id` | Usa una voz de referencia guardada, por ejemplo `my-speaker` |
| `--format` | Formato de salida: `wav`, `pcm`, `mp3` u `opus`; por defecto `wav` |
| `--streaming` | Devuelve el audio en streaming |
| `--single_chunk` | Genera todo el texto de una vez, sin dividirlo en bloques (solo Drama 3) |
| `--chunk_length` | Longitud del bloque, 200 por defecto (solo S2-Pro) |
| `--max_new_tokens` | Máximo de frames por bloque; si se deja vacío se usa el valor del servidor, 1024; 0 usa el límite propio del modelo |
| `--temperature` / `--top_p` | Muestreo; si se dejan vacíos se usan los valores por defecto del modelo (0.7 / 0.7) |
| `--seed` | Fija la semilla aleatoria (no sirve para fijar el timbre) |
| `--api_key` | Obligatorio cuando el servidor se ejecuta con `--api-key` |
| `--no-play` | No reproduce el audio al terminar (por defecto sí lo reproduce; necesario en máquinas sin dispositivo de audio) |

!!! note
    `repetition_penalty` se mantiene solo por compatibilidad; tanto S2-Pro como Drama 3 lo ignoran y usan muestreo con detección de repetición incorporado.

!!! tip
    Para iniciar el servidor con Docker, consulta [Configuración con Docker](install.md#configuración-con-docker) (la imagen del servidor ya incluye la WebUI: abre `http://localhost:8080/ui` cuando esté en marcha).

## Diferencias entre modelos

| Elemento | S2-Pro | Drama 3 |
| --- | --- | --- |
| Hardware | GPU / CPU | Solo NVIDIA GPU, sin `--compile` |
| Texto largo | Dividido según `chunk_length` | Se divide automáticamente en bloques de unos 300 bytes (UTF-8) con un máximo de tres hablantes, cortando al final de frase cuando es posible; el primer bloque y el más reciente se reintroducen como contexto |
| Límite por pasada | Longitud de contexto | 1024 frames por bloque (unos 47 segundos); 2048 frames (unos 95 segundos) con `single_chunk`; ambos están limitados a 10.240 frames menos la longitud del prompt |
| Hablantes | El orden de las referencias decide el papel | Las etiquetas `<\|speaker:0\|>`–`<\|speaker:15\|>` deciden el papel; el orden de las referencias no importa |
| `chunk_length` | Surte efecto | No surte efecto |
| Muestreo por defecto | temperature 0.7, top-p 0.7, top-k 30 | Igual |
| `repetition_penalty` | Ignorado | Ignorado |

Drama 3 tiene además dos variables de entorno: `FISH_DRAMA3_FP8_EXPERTS=0` o `1` fuerza expertos descuantizados o en FP8, y `FISH_DRAMA3_CUDA_GRAPH=0` desactiva la decodificación con CUDA graphs.

## Inferencia con WebUI

### Awesome WebUI

Awesome WebUI es una interfaz web moderna construida con TypeScript, que ofrece funciones más ricas y una mejor experiencia de uso.

Las imágenes Docker del servidor ya incluyen la WebUI compilada: abre `http://localhost:8080/ui` cuando el contenedor esté en marcha. Los pasos siguientes son para ejecutar desde el código fuente.

**Compilar la WebUI:**

Necesitas tener Node.js y npm instalados en tu máquina o servidor.

1. Entra en el directorio `awesome_webui`:
   ```bash
   cd awesome_webui
   ```
2. Instala las dependencias:
   ```bash
   npm install
   ```
3. Compila la WebUI:
   ```bash
   npm run build
   ```

**Iniciar el servidor backend:**

Después de compilar la WebUI, vuelve a la raíz del proyecto e inicia el servidor API (`--compile` es solo para S2-Pro; quítalo al cargar Drama 3):

```bash
python tools/api_server.py --listen 0.0.0.0:8888 --compile
```

**Acceso:**

Cuando el servidor esté en marcha, puedes acceder desde tu navegador a:
`http://localhost:8888/ui`
