FROM nvcr.io/nvidia/pytorch:23.09-py3

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl build-essential ffmpeg libsm6 libxext6 libjpeg-dev \
    zlib1g-dev aria2 zsh openssh-server sudo python3.10-venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install s5cmd
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.2.0/s5cmd_2.2.0_Linux-64bit.tar.gz | tar xvz -C /tmp && \
    mv /tmp/s5cmd /usr/local/bin/s5cmd && s5cmd --help

# Install code server and zsh
RUN wget -c https://github.com/coder/code-server/releases/download/v4.5.1/code-server_4.5.1_amd64.deb && \
    dpkg -i ./code-server_4.5.1_amd64.deb && \
    code-server --install-extension ms-python.python && \
    rm ./code-server_4.5.1_amd64.deb && \
    sh -c "$(curl https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" "" --unattended

# Set zsh as default shell
RUN chsh -s /usr/bin/zsh
ENV SHELL=/usr/bin/zsh

# Setup flash-attn
RUN pip3 install --upgrade pip && \
    pip3 install ninja packaging && \
    pip3 install git+https://github.com/Dao-AILab/flash-attention.git

# Setup rust-data-server
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    cd data_server && cargo build --release

# Project Env
WORKDIR /exp

COPY . .
RUN pip3 install -e .

CMD /bin/zsh
