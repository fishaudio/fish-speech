## Requisitos

* Memoria GPU: 24GB (Inferencia)
* Sistema: Linux, WSL

## Configuración del Sistema

Fish Audio S2 soporta múltiples métodos de instalación. Elige el que mejor se adapte a tu entorno de desarrollo.

**Prerequisitos**: Instala dependencias del sistema para procesamiento de audio:

```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Instalación con GPU (elige tu versión de CUDA: cu126, cu128, cu129)
pip install -e .[cu129]

# Instalación solo CPU
pip install -e .[cpu]

# Instalación por defecto (usa el índice de PyTorch por defecto)
pip install -e .

# Si encuentras un error durante la instalación debido a pyaudio, considera usar:
# conda install pyaudio
# Luego ejecuta pip install -e . nuevamente
```

### UV

UV proporciona resolución de dependencias e instalación más rápida:

```bash
# Instalación con GPU (elige tu versión de CUDA: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# Instalación solo CPU
uv sync --python 3.12 --extra cpu
```

### Soporte Intel Arc XPU

Para usuarios con GPU Intel Arc, instala con soporte XPU:

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Instalar librería estándar de C++ requerida
conda install libstdcxx -c conda-forge

# Instalar PyTorch con soporte Intel XPU
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Instalar Fish Speech
pip install -e .
```

!!! warning
La opción `compile` no está soportada en Windows ni macOS. Si quieres usar compile, necesitas instalar Triton manualmente.

## Configuración con Docker

La serie de modelos Fish Audio S2 ofrece múltiples opciones de despliegue con Docker para distintos escenarios. Puedes usar imágenes preconstruidas desde Docker Hub, construir localmente con Docker Compose o crear imágenes personalizadas manualmente.

Proveemos imágenes Docker tanto para WebUI como para servidor API en GPU (CUDA126 por defecto) y CPU. Puedes usar las imágenes preconstruidas desde Docker Hub, construir localmente con Docker Compose o crear imágenes personalizadas manualmente. Si quieres construir localmente, sigue las instrucciones de abajo. Si solo quieres usar imágenes preconstruidas, sigue la [guía de inferencia](inference.md).

### Prerrequisitos

* Docker y Docker Compose instalados
* NVIDIA Docker runtime (para soporte GPU)
* Al menos 24GB de memoria GPU para inferencia con CUDA

# Usar docker compose

Para desarrollo o personalización, puedes usar Docker Compose para construir y ejecutar localmente:

```bash
# Clonar el repositorio primero
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Iniciar WebUI con CUDA
docker compose --profile webui up

# Iniciar WebUI con optimización compile
COMPILE=1 docker compose --profile webui up

# Iniciar servidor API
docker compose --profile server up

# Iniciar servidor API con optimización compile  
COMPILE=1 docker compose --profile server up

# Para despliegue solo CPU
BACKEND=cpu docker compose --profile webui up
```

#### Variables de Entorno para Docker Compose

Puedes personalizar el despliegue usando variables de entorno:

```bash
# Ejemplo de archivo .env
BACKEND=cuda              # o cpu
COMPILE=1                 # Habilitar optimización compile
GRADIO_PORT=7860         # Puerto WebUI
API_PORT=8080            # Puerto servidor API
UV_VERSION=0.8.15        # Versión del gestor de paquetes UV
```

El comando construirá la imagen y ejecutará el contenedor. Puedes acceder a la WebUI en `http://localhost:7860` y al servidor API en `http://localhost:8080`.

### Build manual con Docker

Para usuarios avanzados que quieran personalizar el proceso de build:

```bash
# Construir imagen WebUI con soporte CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

# Construir imagen servidor API con soporte CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# Construir imágenes solo CPU (multi-plataforma)
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target webui \
    -t fish-speech-webui:cpu .

# Construir imagen de desarrollo
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### Argumentos de Build

* `BACKEND`: `cuda` o `cpu` (default: `cuda`)
* `CUDA_VER`: versión de CUDA (default: `12.6.0`)
* `UV_EXTRA`: extra de UV para CUDA (default: `cu126`)
* `UBUNTU_VER`: versión de Ubuntu (default: `24.04`)
* `PY_VER`: versión de Python (default: `3.12`)

### Montaje de Volúmenes

Ambos métodos requieren montar estos directorios:

* `./checkpoints:/app/checkpoints` - Directorio de pesos del modelo
* `./references:/app/references` - Directorio de audios de referencia

### Variables de Entorno

* `COMPILE=1` - Habilita torch.compile para inferencia más rápida (~10x)
* `GRADIO_SERVER_NAME=0.0.0.0` - Host del servidor WebUI
* `GRADIO_SERVER_PORT=7860` - Puerto del servidor WebUI
* `API_SERVER_NAME=0.0.0.0` - Host del servidor API
* `API_SERVER_PORT=8080` - Puerto del servidor API

!!! note
Los contenedores Docker esperan que los pesos del modelo estén montados en `/app/checkpoints`. Asegúrate de descargar los pesos necesarios antes de iniciar los contenedores.

!!! warning
El soporte GPU requiere NVIDIA Docker runtime. Para despliegue solo CPU, elimina el flag `--gpus all` y usa imágenes CPU.
