## Requisitos

- Memória da GPU: 12GB (Inferência)
- Sistema: Linux, WSL

## Configuração do Sistema

O OpenAudio suporta múltiplos métodos de instalação. Escolha o que melhor se adapta ao seu ambiente de desenvolvimento.

**Pré-requisitos**: Instale as dependências de sistema para processamento de áudio:
``` bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Instalação com GPU (escolha a sua versão do CUDA: cu126, cu128, cu129)
pip install -e .[cu129]

# Instalação apenas para CPU
pip install -e .[cpu]

# Instalação padrão (usa o índice padrão do PyTorch)
pip install -e .
```

### UV

O UV oferece uma resolução e instalação de dependências mais rápidas:

```bash
# Instalação com GPU (escolha a sua versão do CUDA: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# Instalação apenas para CPU
uv sync --python 3.12 --extra cpu
```
### Suporte para Intel Arc XPU

Para utilizadores de GPUs Intel Arc, instale o suporte XPU da seguinte forma:

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Instalar a biblioteca padrão C++ necessária
conda install libstdcxx -c conda-forge

# Instalar o PyTorch com suporte para Intel XPU
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

# Instalar o Fish Speech
pip install -e .
```

!!! warning
    A opção `compile` não é suportada no Windows e macOS. Se desejar executar com compilação, terá de instalar o Triton manualmente.


## Configuração do Docker

O modelo da série OpenAudio S1 oferece múltiplas opções de implementação com Docker para satisfazer diferentes necessidades. Pode usar imagens pré-construídas do Docker Hub, construir localmente com o Docker Compose, ou construir manualmente imagens personalizadas.

Fornecemos imagens Docker para a WebUI e o servidor API, tanto para GPU (CUDA 12.6 por defeito) como para CPU. Pode usar as imagens pré-construídas do Docker Hub, construir localmente com o Docker Compose, ou construir manualmente imagens personalizadas. Se quiser construir localmente, siga as instruções abaixo. Se apenas quiser usar as imagens pré-construídas, siga diretamente o [guia de inferência](inference.md).

### Pré-requisitos

- Docker e Docker Compose instalados
- NVIDIA Docker runtime instalado (para suporte de GPU)
- Pelo menos 12GB de memória de GPU para inferência com CUDA

### Usar o Docker Compose

Para desenvolvimento ou personalização, pode usar o Docker Compose para construir e executar localmente:

```bash
# Primeiro, clone o repositório
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Iniciar a WebUI com CUDA
docker compose --profile webui up

# Iniciar a WebUI com otimização de compilação
COMPILE=1 docker compose --profile webui up

# Iniciar o servidor API
docker compose --profile server up

# Iniciar o servidor API com otimização de compilação
COMPILE=1 docker compose --profile server up

# Implementação apenas com CPU
BACKEND=cpu docker compose --profile webui up
```

#### Variáveis de Ambiente para o Docker Compose

Pode personalizar a implementação usando variáveis de ambiente:

```bash
# Exemplo de ficheiro .env
BACKEND=cuda              # ou cpu
COMPILE=1                 # Ativar otimização de compilação
GRADIO_PORT=7860         # Porta da WebUI
API_PORT=8080            # Porta do servidor API
UV_VERSION=0.8.15        # Versão do gestor de pacotes UV
```

O comando irá construir a imagem e executar o contentor. Pode aceder à WebUI em `http://localhost:7860` e ao servidor API em `http://localhost:8080`.

### Construção Manual com Docker

Para utilizadores avançados que desejam personalizar o processo de construção:

```bash
# Construir imagem da WebUI com suporte CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target webui \
    -t fish-speech-webui:cuda .

# Construir imagem do servidor API com suporte CUDA
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --build-arg CUDA_VER=12.6.0 \
    --build-arg UV_EXTRA=cu126 \
    --target server \
    -t fish-speech-server:cuda .

# Construir imagem apenas para CPU (suporta multiplataforma)
docker build \
    --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cpu \
    --target webui \
    -t fish-speech-webui:cpu .

# Construir imagem de desenvolvimento
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### Argumentos de Construção

- `BACKEND`: `cuda` ou `cpu` (padrão: `cuda`)
- `CUDA_VER`: Versão do CUDA (padrão: `12.6.0`)
- `UV_EXTRA`: Pacote extra do UV para CUDA (padrão: `cu126`)
- `UBUNTU_VER`: Versão do Ubuntu (padrão: `24.04`)
- `PY_VER`: Versão do Python (padrão: `3.12`)

### Montagem de Volumes

Ambos os métodos requerem a montagem dos seguintes diretórios:

- `./checkpoints:/app/checkpoints` - Diretório dos pesos do modelo
- `./references:/app/references` - Diretório dos ficheiros de áudio de referência

### Variáveis de Ambiente

- `COMPILE=1` - Ativa o `torch.compile` para uma inferência mais rápida (cerca de 10x)
- `GRADIO_SERVER_NAME=0.0.0.0` - Anfitrião do servidor WebUI
- `GRADIO_SERVER_PORT=7860` - Porta do servidor WebUI
- `API_SERVER_NAME=0.0.0.0` - Anfitrião do servidor API
- `API_SERVER_PORT=8080` - Porta do servidor API

!!! note
    Os contentores Docker esperam que os pesos do modelo sejam montados em `/app/checkpoints`. Certifique-se de que descarregou os pesos do modelo necessários antes de iniciar os contentores.

!!! warning
    O suporte para GPU requer o NVIDIA Docker runtime. Para implementações apenas com CPU, remova a flag `--gpus all` e use as imagens de CPU.
