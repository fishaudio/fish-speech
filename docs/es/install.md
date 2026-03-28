## Requisitos

* Memoria GPU: 24GB (Inferencia)
* Sistema: Linux, WSL

## Configuración del sistema

Fish Audio S2 admite múltiples métodos de instalación. Elige el que mejor se adapte a tu entorno de desarrollo.

**Prerrequisitos**: Instala las dependencias del sistema para el procesamiento de audio:

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

# Instalación por defecto (usa el índice por defecto de PyTorch)
pip install -e .

# Si encuentras un error durante la instalación debido a pyaudio, considera usar el siguiente comando:
# conda install pyaudio
# Luego ejecuta pip install -e . nuevamente
```

### UV

UV proporciona una resolución de dependencias e instalación más rápida:

```bash
# Instalación con GPU (elige tu versión de CUDA: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# Instalación solo CPU
uv sync --python 3.12 --extra cpu
```

### Soporte para Intel Arc XPU

Para usuarios de GPU Intel Arc, instala con soporte XPU:

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
La opción `compile` no es compatible con Windows ni macOS. Si quieres ejecutar con compile, necesitas instalar Triton manualmente.

## Configuración con Docker

El modelo de la serie Fish Audio S2 ofrece múltiples opciones de despliegue con Docker para adaptarse a diferentes necesidades. Puedes usar imágenes preconstruidas desde Docker Hub, construir localmente con Docker Compose o crear imágenes personalizadas manualmente.

Proporcionamos imágenes Docker tanto para WebUI como para el servidor API en GPU (CUDA126 por defecto) y CPU. Puedes usar imágenes preconstruidas desde Docker Hub, construir localmente con Docker Compose o crear imágenes personalizadas manualmente. Si quieres construir localmente, sigue las instrucciones a continuación. Si solo quieres usar imágenes preconstruidas, sigue la [guía de inferencia](inference.md).

### Prerrequisitos

* Docker y Docker Compose instalados
* Runtime de NVIDIA Docker (para soporte GPU)
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

#### Variables de entorno para Docker Compose

Puedes personalizar el despliegue usando variables de entorno:

```bash
# Ejemplo de archivo .env
BACKEND=cuda              # o cpu
COMPILE=1                 # Habilitar optimización compile
GRADIO_PORT=7860         # Puerto de WebUI
API_PORT=8080            # Puerto del servidor API
UV_VERSION=0.8.15        # Versión del gestor de paquetes UV
CUDA_VER=12.9.0          # Versión base de imagen CUDA (ej. 12.6.0 para drivers más antiguos)
UV_EXTRA=cu129           # Variante CUDA de PyTorch (cu126, cu128, cu129) — debe coincidir con CUDA_VER
```

El comando construirá la imagen y ejecutará el contenedor. Puedes acceder a la WebUI en `http://localhost:7860` y al servidor API en `http://localhost:8080`.

### Construcción manual con Docker

Para usuarios avanzados que quieran personalizar el proceso de build:

```bash
# Construir imagen WebUI con soporte CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.9.0 \
    --build-arg UV_EXTRA=cu129 \
    --target webui \
    -t fish-speech-webui:cuda .

# Construir imagen del servidor API con soporte CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.9.0 \
    --build-arg UV_EXTRA=cu129 \
    --target server \
    -t fish-speech-server:cuda .

# Construir imágenes solo CPU (soporta múltiples plataformas)
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

#### Argumentos de build

* `BACKEND`: `cuda` o `cpu` (por defecto: `cuda`)
* `CUDA_VER`: versión de CUDA (por defecto: `12.6.0`)
* `UV_EXTRA`: extra de UV para CUDA (por defecto: `cu126`)
* `UBUNTU_VER`: versión de Ubuntu (por defecto: `24.04`)
* `PY_VER`: versión de Python (por defecto: `3.12`)

### Montajes de volumen

Ambos métodos requieren montar estos directorios:

* `./checkpoints:/app/checkpoints` - Directorio de pesos del modelo
* `./references:/app/references` - Directorio de archivos de audio de referencia

### Variables de entorno

* `COMPILE=1` - Habilitar torch.compile para inferencia más rápida (~10x de mejora)
* `GRADIO_SERVER_NAME=0.0.0.0` - Host del servidor WebUI
* `GRADIO_SERVER_PORT=7860` - Puerto del servidor WebUI
* `API_SERVER_NAME=0.0.0.0` - Host del servidor API
* `API_SERVER_PORT=8080` - Puerto del servidor API

!!! note
Los contenedores Docker esperan que los pesos del modelo estén montados en `/app/checkpoints`. Asegúrate de descargar los pesos del modelo necesarios antes de iniciar los contenedores.

!!! warning
El soporte GPU requiere el runtime de NVIDIA Docker. Para despliegues solo CPU, elimina el flag `--gpus all` y usa imágenes de CPU.
