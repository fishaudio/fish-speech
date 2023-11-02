# Fish Speech

This repo is still under construction. Please check back later.

## Setup
```bash
# Basic environment setup
conda create -n fish-speech python=3.10
conda activate fish-speech
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install flash-attn (for linux)
pip3 install ninja && MAX_JOBS=4 pip3 install flash-attn --no-build-isolation

# Install fish-speech
pip3 install -e .
```
