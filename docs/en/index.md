# Introduction

!!! warning
    We assume no responsibility for any illegal use of the codebase. Please refer to the local laws regarding DMCA (Digital Millennium Copyright Act) and other relevant laws in your area.

This codebase is released under the `BSD-3-Clause` license, and all models are released under the CC-BY-NC-SA-4.0 license.

<p align="center">
<img src="/assets/figs/diagram.png" width="75%">
</p>

## Requirements
- GPU Memory: 2GB (for inference), 16GB (for fine-tuning)
- System: Linux (full functionality), Windows (inference only, no support for `flash-attn`, no support for `torch.compile`)

Therefore, we strongly recommend Windows users to use WSL2 or docker to run the codebase.

## Setup
```bash
# Create a python 3.10 virtual environment, you can also use virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# Install pytorch nightly
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

# Install flash-attn (for Linux)
pip3 install ninja && MAX_JOBS=4 pip3 install flash-attn --no-build-isolation

# Install fish-speech
pip3 install -e .
```

## Changelog

- 2023/12/28: Added `lora` fine-tuning support.
- 2023/12/27: Add `gradient checkpointing`, `causual sampling`, and `flash-attn` support.
- 2023/12/19: Updated webui and HTTP API.
- 2023/12/18: Updated fine-tuning documentation and related examples.
- 2023/12/17: Updated `text2semantic` model, supporting phoneme-free mode.
- 2023/12/13: Beta version released, includes VQGAN model and a language model based on LLAMA (phoneme support only).

## Acknowledgements
- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
