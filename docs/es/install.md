## Requisitos

| Modelo | Memoria GPU | GPU |
| --- | --- | --- |
| Drama 3 | al menos 32 GB | solo NVIDIA GPU: no compatible con CPU, Apple MPS ni AMD ROCm, y sin `--compile` |
| S2-Pro | al menos 24 GB | NVIDIA GPU o CPU |

- Sistema: Linux, WSL

!!! note
    El pico de memoria medido de Drama 3 en una sola H20 es de unos 41 GB, así que deja algo de margen. La V100 y otras GPUs anteriores a SM80 no tienen bfloat16, por lo que Drama 3 cambia a float16 (manteniendo el flujo residual en fp32).

## Configuración del sistema

Fish Audio Drama 3 y S2-Pro comparten la misma instalación. Elige la opción que mejor se adapte a tu entorno de desarrollo.

**Prerrequisitos**: instala las dependencias del sistema necesarias para el procesamiento de audio:
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Instalación con GPU (elige tu versión de CUDA: cu126, cu128, cu129)
# Reemplaza ambos valores "cu129" si usas cu126 o cu128.
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[cu129]"

# Instalación solo CPU
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu]"

# Instalación por defecto (usa el índice por defecto de PyTorch)
pip install -e .

# Si encuentras un error durante la instalación debido a pyaudio, considera usar el siguiente comando:
# conda install pyaudio
# Luego ejecuta pip install -e . nuevamente
```

### UV

UV ofrece una resolución e instalación de dependencias más rápidas:

```bash
# Instalación con GPU (elige tu versión de CUDA: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# Instalación solo CPU
uv sync --python 3.12 --extra cpu
```

### Soporte para Intel Arc XPU

Si usas una GPU Intel Arc, instala con soporte XPU (solo S2-Pro; Drama 3 necesita una GPU NVIDIA):

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Instalar la biblioteca estándar de C++ requerida
conda install libstdcxx -c conda-forge

# Instalar PyTorch con soporte Intel XPU
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Instalar Fish Speech
pip install -e .
```

!!! warning
    La opción `compile` no está soportada en Windows y macOS. Si quieres ejecutar con compile, debes instalar Triton manualmente.

## Descargar Pesos

Ambos modelos comparten la misma línea de comandos y el mismo servidor API; el tipo de modelo se lee del `config.json` del directorio de pesos. Guárdalos en el mismo directorio `checkpoints`.

Para descargar necesitas la CLI de Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
```

### Drama 3

```bash
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

Si el directorio de pesos no tiene `codec.pth`, se usa el codec de S2-Pro en `checkpoints/s2-pro/codec.pth`, así que descarga también S2-Pro.

### S2-Pro

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

Una vez descargados los pesos, consulta [Inferencia](inference.md).

## Configuración con Docker

Drama 3 y S2-Pro ofrecen varias formas de despliegue con Docker. Puedes usar imágenes preconstruidas de Docker Hub, compilar localmente con Docker Compose o crear imágenes personalizadas manualmente.

Ofrecemos imágenes del servidor API para GPU (CUDA126 por defecto) y CPU. Las imágenes ya incluyen la WebUI: abre `http://localhost:8080/ui` cuando el contenedor esté en marcha. La imagen GPU ejecuta ambos modelos; la imagen CPU solo admite S2-Pro.

### Prerrequisitos

- Docker y Docker Compose instalados
- (Para GPU) NVIDIA Docker runtime instalado (obligatorio para Drama 3)
- Memoria GPU: al menos 32 GB para Drama 3, 24 GB recomendados para S2-Pro

### Ejecutar la imagen preconstruida

Drama 3: en tarjetas SM89 o más recientes expón una sola tarjeta como se indica abajo; en tarjetas más antiguas usa `--gpus all`:

```bash
docker run --gpus device=0 \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

S2-Pro:

```bash
docker run --gpus all \
    -v ./checkpoints:/app/checkpoints \
    -e LLAMA_CHECKPOINT_PATH=checkpoints/s2-pro \
    -e DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth \
    -p 8080:8080 \
    fishaudio/fish-speech:server-cuda
```

### Usar docker compose

Para desarrollo o personalización, te recomendamos usar Docker Compose para compilar y ejecutar localmente:

```bash
# Clonar el repositorio primero
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Iniciar servidor API (por defecto carga checkpoints/s2-pro)
docker compose --profile server up

# Cargar Drama 3 y fijarlo en una sola tarjeta con DEVICE (hasta cuatro si no se define)
LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 DEVICE=cuda:0 docker compose --profile server up

# Iniciar servidor API con optimización compile (solo S2-Pro)
COMPILE=1 docker compose --profile server up

# Para despliegue solo CPU (solo S2-Pro)
BACKEND=cpu docker compose --profile server up
```

#### Variables de entorno para Docker Compose

Puedes personalizar el despliegue con variables de entorno:

```bash
# Ejemplo de archivo .env
BACKEND=cuda              # o cpu
COMPILE=1                 # Activar optimización compile
API_PORT=8080             # Puerto del servidor API
UV_VERSION=0.8.15         # Versión del gestor de paquetes UV
```

El comando compilará la imagen e iniciará el contenedor. Puedes acceder al servidor API en `http://localhost:8080` y a la WebUI en `http://localhost:8080/ui`.

### Construcción manual con Docker

Para quienes quieran controlar el proceso de compilación con más detalle:

```bash
# Construir imagen del servidor API con soporte CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# Construir imágenes solo CPU (soporta múltiples plataformas)
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target server \
    -t fish-speech-server:cpu .

# Construir imagen de desarrollo
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### Argumentos de build

- `BACKEND`: `cuda` o `cpu` (por defecto: `cuda`)
- `CUDA_VER`: versión de CUDA (por defecto: `12.6.0`)
- `UV_EXTRA`: extra UV para CUDA (por defecto: `cu126`)
- `UBUNTU_VER`: versión de Ubuntu (por defecto: `24.04`)
- `PY_VER`: versión de Python (por defecto: `3.12`)

### Montajes de volumen

Ambos métodos requieren montar estos directorios:

- `./checkpoints:/app/checkpoints` - Directorio de pesos del modelo
- `./references:/app/references` - Directorio de archivos de audio de referencia

### Variables de entorno

- `COMPILE=1` - Activar torch.compile para una inferencia más rápida (~10x)
- `LLAMA_CHECKPOINT_PATH` - Ruta de los pesos del modelo principal, por ejemplo `checkpoints/drama-3` o `checkpoints/s2-pro`
- `DECODER_CHECKPOINT_PATH` - Ruta de los pesos del codec; si se deja vacío, el servidor lo resuelve desde el directorio de pesos
- `DEVICE` - Dispositivo de inferencia, por ejemplo `cuda:0`; Drama 3 usa solo esa tarjeta cuando se define y hasta cuatro si se deja vacío
- `FISH_DRAMA3_FP8_EXPERTS=0` o `1` - Fuerza el uso de expertos descuantizados o en FP8 en Drama 3 (por defecto se elige según la GPU)
- `FISH_DRAMA3_CUDA_GRAPH=0` - Desactiva la decodificación con CUDA graphs de Drama 3
- `API_SERVER_NAME=0.0.0.0` - Dirección del servidor API
- `API_SERVER_PORT=8080` - Puerto del servidor API

!!! note
    Los contenedores Docker leen los pesos del modelo desde `/app/checkpoints`. Asegúrate de descargar los pesos necesarios antes de iniciar los contenedores.

!!! warning
    El soporte de GPU requiere NVIDIA Docker runtime. Para despliegues solo CPU, quita `--gpus all` y usa imágenes de CPU.
