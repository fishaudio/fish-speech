FROM python:3.10.14-bookworm

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl build-essential ffmpeg libsm6 libxext6 libjpeg-dev \
    zlib1g-dev aria2 zsh openssh-server sudo protobuf-compiler cmake libsox-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install oh-my-zsh so your terminal looks nice
RUN sh -c "$(curl https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)" "" --unattended

# Set zsh as default shell
RUN chsh -s /usr/bin/zsh
ENV SHELL=/usr/bin/zsh

# Setup torchaudio
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project Env
WORKDIR /exp
COPY . .
RUN pip3 install -e .

CMD /bin/zsh
