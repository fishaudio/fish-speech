# Fish Speech

**Documentation is under construction**

[中文文档](README.zh.md)

This codebase is released under BSD-3-Clause License, and all models are released under CC-BY-NC-SA-4.0 License. Please refer to [LICENSE](LICENSE) for more details. 

## Disclaimer
We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Requirements
- GPU memory: 4GB (for inference), 24GB (for finetuning)
- System: Linux (full functionality), Windows (inference only, flash-attn is not supported, torch.compile is not supported)

Therefore, we strongly recommend to use WSL2 or docker to run the codebase for Windows users.

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

## Inference (CLI)
Download required `vqgan` and `text2semantic` model from our huggingface repo.

```bash
TODO
```

Generate semantic tokens from text:
```bash
python tools/llama/generate.py
```

You may want to use `--compile` to fuse cuda kernels faster inference (~25 tokens/sec -> ~300 tokens/sec).

Generate vocals from semantic tokens:
```bash
python tools/vqgan/inference.py -i codes_0.npy
```

## Rust Data Server
Since loading and shuffle the dataset is very slow and memory consuming, we use a rust server to load and shuffle the dataset. The server is based on GRPC and can be installed by

```bash
cd data_server
cargo build --release
```

## Credits
- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)

