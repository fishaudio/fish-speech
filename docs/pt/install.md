## Requisitos

| Modelo | Memória GPU | GPU |
| --- | --- | --- |
| Drama 3 | pelo menos 32 GB | apenas NVIDIA GPU: sem CPU, Apple MPS ou AMD ROCm e sem `--compile` |
| S2-Pro | pelo menos 24 GB | NVIDIA GPU ou CPU |

- Sistema: Linux, WSL

!!! note
    O pico de memória medido do Drama 3 num único H20 é cerca de 41 GB, por isso deixe alguma margem. A V100 e outras GPUs anteriores a SM80 não têm bfloat16, pelo que o Drama 3 passa a float16 (mantendo o fluxo residual em fp32).

## Configuração do Sistema

O Fish Audio Drama 3 e o S2-Pro partilham a mesma instalação. Escolha a opção que melhor se adapta ao seu ambiente de desenvolvimento.

**Pré-requisitos**: instale as dependências de sistema necessárias para o processamento de áudio:
```bash
apt install portaudio19-dev libsox-dev ffmpeg
```

### Conda

```bash
conda create -n fish-speech python=3.12
conda activate fish-speech

# Instalação com GPU (escolha a sua versão do CUDA: cu126, cu128, cu129)
# Substitua os dois valores "cu129" ao usar cu126 ou cu128.
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e ".[cu129]"

# Instalação apenas para CPU
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu]"

# Instalação padrão (usa o índice padrão do PyTorch)
pip install -e .

# Se encontrar um erro durante a instalação devido ao pyaudio, considere usar o seguinte comando:
# conda install pyaudio
# De seguida, execute pip install -e . novamente
```

### UV

O UV resolução e instalação de dependências mais rápidas:

```bash
# Instalação com GPU (escolha a sua versão do CUDA: cu126, cu128, cu129)
uv sync --python 3.12 --extra cu129

# Instalação apenas para CPU
uv sync --python 3.12 --extra cpu
```

### Suporte para Intel Arc XPU

Se usa uma GPU Intel Arc, instale com suporte XPU (apenas S2-Pro; o Drama 3 precisa de uma NVIDIA GPU):

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
    A opção `compile` não é suportada no Windows e no macOS. Se quiser usar compile, tem de instalar o Triton manualmente.

## Baixar Pesos

Os dois modelos partilham a mesma linha de comandos e o mesmo servidor API; o tipo de modelo é lido do `config.json` do diretório de pesos. Guarde-os no mesmo diretório `checkpoints`.

Para descarregar precisa da CLI da Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"
```

### Drama 3

```bash
hf download fishaudio/drama-3 --local-dir checkpoints/drama-3
```

Se o diretório de pesos não tiver `codec.pth`, é usado o codec do S2-Pro em `checkpoints/s2-pro/codec.pth`, por isso descarregue também o S2-Pro.

### S2-Pro

```bash
hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
```

Depois de descarregar os pesos, consulte [Inferência](inference.md).

## Configuração do Docker

O Drama 3 e o S2-Pro oferecem várias formas de implementação com Docker. Pode usar imagens pré-construídas do Docker Hub, construir localmente com Docker Compose ou criar imagens personalizadas manualmente.

Fornecemos imagens do servidor API para GPU (CUDA126 por predefinição) e CPU. As imagens já incluem a WebUI: abra `http://localhost:8080/ui` depois de o contentor arrancar. A imagem GPU corre os dois modelos; a imagem CPU só suporta o S2-Pro.

### Pré-requisitos

- Docker e Docker Compose instalados
- (Para GPU) NVIDIA Docker runtime instalado (obrigatório para o Drama 3)
- Memória GPU: pelo menos 32 GB para o Drama 3, 24 GB recomendados para o S2-Pro

### Executar a imagem pré-construída

Drama 3: em placas SM89 ou mais recentes exponha apenas uma placa, como abaixo; em placas mais antigas use `--gpus all`:

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

### Usar o Docker Compose

Para desenvolvimento ou personalização, recomenda-se usar o Docker Compose para construir e executar localmente:

```bash
# Primeiro, clone o repositório
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Iniciar o servidor API (por predefinição carrega checkpoints/s2-pro)
docker compose --profile server up

# Carregar o Drama 3 e fixá-lo numa única placa com DEVICE (até quatro quando não definido)
LLAMA_CHECKPOINT_PATH=checkpoints/drama-3 DEVICE=cuda:0 docker compose --profile server up

# Iniciar o servidor API com otimização de compilação (apenas S2-Pro)
COMPILE=1 docker compose --profile server up

# Implementação apenas com CPU (apenas S2-Pro)
BACKEND=cpu docker compose --profile server up
```

#### Variáveis de Ambiente para o Docker Compose

Pode personalizar a implementação com variáveis de ambiente:

```bash
# Exemplo de ficheiro .env
BACKEND=cuda              # ou cpu
COMPILE=1                 # Ativar otimização de compilação
API_PORT=8080             # Porta do servidor API
UV_VERSION=0.8.15         # Versão do gestor de pacotes UV
```

O comando constrói a imagem e inicia o contentor. Pode aceder ao servidor API em `http://localhost:8080` e à WebUI em `http://localhost:8080/ui`.

### Construção Manual com Docker

Para quem quer controlar o processo de construção em detalhe:

```bash
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
    --target server \
    -t fish-speech-server:cpu .

# Construir imagem de desenvolvimento
docker build \
    --platform linux/amd64 \
    -f docker/Dockerfile \
    --build-arg BACKEND=cuda \
    --target dev \
    -t fish-speech-dev:cuda .
```

#### Argumentos de Construção

- `BACKEND`: `cuda` ou `cpu` (predefinição: `cuda`)
- `CUDA_VER`: versão do CUDA (predefinição: `12.6.0`)
- `UV_EXTRA`: extra UV para CUDA (predefinição: `cu126`)
- `UBUNTU_VER`: versão do Ubuntu (predefinição: `24.04`)
- `PY_VER`: versão do Python (predefinição: `3.12`)

### Montagem de Volumes

Ambos os métodos precisam de montar estes diretórios:

- `./checkpoints:/app/checkpoints` - Diretório dos pesos do modelo
- `./references:/app/references` - Diretório dos áudios de referência

### Variáveis de Ambiente

- `COMPILE=1` - Ativa o torch.compile para inferência mais rápida (~10x)
- `LLAMA_CHECKPOINT_PATH` - Caminho dos pesos do modelo principal, por exemplo `checkpoints/drama-3` ou `checkpoints/s2-pro`
- `DECODER_CHECKPOINT_PATH` - Caminho dos pesos do codec; se vazio, o servidor resolve a partir do diretório de pesos
- `DEVICE` - Dispositivo de inferência, por exemplo `cuda:0`; o Drama 3 usa apenas essa placa quando definido e até quatro quando vazio
- `FISH_DRAMA3_FP8_EXPERTS=0` ou `1` - Força o Drama 3 a usar especialistas desquantizados ou em FP8 (por predefinição é escolhido segundo a GPU)
- `FISH_DRAMA3_CUDA_GRAPH=0` - Desativa a descodificação com CUDA graphs do Drama 3
- `API_SERVER_NAME=0.0.0.0` - Endereço do servidor API
- `API_SERVER_PORT=8080` - Porta do servidor API

!!! note
    Os contentores Docker leem os pesos dos modelos de `/app/checkpoints`. Descarregue os pesos necessários antes de iniciar os contentores.

!!! warning
    O suporte a GPU requer o NVIDIA Docker runtime. Para implementação apenas com CPU, remova `--gpus all` e use as imagens de CPU.
