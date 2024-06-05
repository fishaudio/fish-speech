# Introduction

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/lengyue233/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/lengyue233/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>

!!! warning
We assume no responsibility for any illegal use of the codebase. Please refer to the local laws regarding DMCA (Digital Millennium Copyright Act) and other relevant laws in your area.

This codebase is released under the `BSD-3-Clause` license, and all models are released under the CC-BY-NC-SA-4.0 license.

<p align="center">
<img src="/assets/figs/diagram.png" width="75%">
</p>

## Requirements

- GPU Memory: 4GB (for inference), 16GB (for fine-tuning)
- System: Linux, Windows

~~We recommend Windows users to use WSL2 or docker to run the codebase, or use the integrated environment developed by the community.~~

## Windows Setup

Windows professional users may consider WSL2 or Docker to run the codebase.

Non-professional Windows users can consider the following methods to run the codebase without a Linux environment (with model compilation capabilities aka `torch.compile`):

0. Extract the project zip file.
1. Click `install_env.bat` to install the environment.

   1. You can decide whether to use a mirror site for downloading by editing the `USE_MIRROR` item in `install_env.bat`.
   2. The default is `preview`, using a mirror site and the latest development version of torch (the only way to activate the compilation method).
   3. `false` uses the original site to download the environment. `true` uses the mirror site to download the stable version of torch and other environments.

2. (Optional, this step is to activate the model compilation environment)

   1. Use the following links to download the `LLVM` compiler.

      - [LLVM-17.0.6 (original site download)](https://huggingface.co/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true)
      - [LLVM-17.0.6 (mirror site download)](https://hf-mirror.com/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true)
      - After downloading `LLVM-17.0.6-win64.exe`, double-click to install, choose the appropriate installation location, and most importantly, check `Add Path to Current User` to add the environment variable.
      - Confirm the installation is complete.

   2. Download and install the Microsoft Visual C++ Redistributable Package to resolve potential .dll missing issues.
      - [MSVC++ 14.40.33810.0 download](https://aka.ms/vs/17/release/vc_redist.x64.exe)

3. Double-click `start.bat` to enter the Fish-Speech training and inference configuration WebUI page.

   - Want to go directly to the inference page? Edit the `API_FLAGS.txt` in the project root directory, and modify the first three lines as follows:

   ```text
   --infer
   # --api
   # --listen ...
   ...
   ```

   - Want to start the API server? Edit the API_FLAGS.txt in the project root directory, and modify the first three lines as follows:

   ```text
   # --infer
   --api
   --listen ...
   ...
   ```

4. (Optional) Double-click run_cmd.bat to enter the conda/python command line environment of this project.

## Linux Setup

```bash
# Create a python 3.10 virtual environment, you can also use virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# Install pytorch
pip3 install torch torchvision torchaudio

# Install fish-speech
pip3 install -e .

# (Ubuntu / Debian User) Install sox
apt install libsox-dev
```

## Changelog

- 2024/05/10: Updated Fish-Speech to 1.1 version, implement VITS decoder to reduce WER and improve timbre similarity.
- 2024/04/22: Finished Fish-Speech 1.0 version, significantly modified VQGAN and LLAMA models.
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
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
