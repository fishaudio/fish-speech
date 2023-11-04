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

## Credits
- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
