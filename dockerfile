FROM nvcr.io/nvidia/pytorch:24.02-py3

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl build-essential ffmpeg libsm6 libxext6 libjpeg-dev \
    zlib1g-dev aria2 zsh openssh-server sudo python3.10-venv protobuf-compiler && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install zsh
RUN sh -c "$(curl https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" "" --unattended

# Set zsh as default shell
RUN chsh -s /usr/bin/zsh
ENV SHELL=/usr/bin/zsh

# Setup torchaudio
RUN git clone https://github.com/pytorch/audio --recursive --depth 1 && \
    cd audio && pip install -v --no-use-pep517 . && \
    cd .. && rm -rf audio && python -c "import torchaudio; print(torchaudio.__version__)"

# Setup flash-attn
RUN pip3 install --upgrade pip && \
    pip3 install ninja packaging && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE pip3 install git+https://github.com/Dao-AILab/flash-attention.git

# Test flash-attn
RUN python3 -c "from flash_attn import flash_attn_varlen_func"

# Project Env
WORKDIR /exp
COPY pyproject.toml ./
COPY data_server ./data_server
COPY fish_speech ./fish_speech

# Setup rust-data-server
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    cd data_server && $HOME/.cargo/bin/cargo build --release && cp target/release/data_server /usr/local/bin/ && \
    cd .. && rm -rf data_server && data_server --help

RUN pip3 install -e . && pip uninstall -y fish-speech && rm -rf fish_speech

CMD /bin/zsh
