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
<img src="/docs/assets/figs/diagram.png" width="75%">
</p>

## Requirements

- GPU Memory: 4GB (for inference), 16GB (for fine-tuning)
- System: Linux, Windows

## Windows Setup

Windows professional users may consider WSL2 or Docker to run the codebase.

Non-professional Windows users can consider the following methods to run the codebase without a Linux environment (with model compilation capabilities aka `torch.compile`):

<ol>
   <li>Unzip the project package.</li>
   <li>Click <code>install_env.bat</code> to install the environment.
      <ul>
            <li>You can decide whether to use a mirror site for downloads by editing the <code>USE_MIRROR</code> item in <code>install_env.bat</code>.</li>
            <li><code>USE_MIRROR=false</code> downloads the latest stable version of <code>torch</code> from the original site. <code>USE_MIRROR=true</code> downloads the latest version of <code>torch</code> from a mirror site. The default is <code>true</code>.</li>
            <li>You can decide whether to enable the compiled environment download by editing the <code>INSTALL_TYPE</code> item in <code>install_env.bat</code>.</li>
            <li><code>INSTALL_TYPE=preview</code> downloads the preview version with the compiled environment. <code>INSTALL_TYPE=stable</code> downloads the stable version without the compiled environment.</li>
      </ul>
   </li>
   <li>If step 2 has <code>USE_MIRROR=preview</code>, execute this step (optional, for activating the compiled model environment):
      <ol>
            <li>Download the LLVM compiler using the following links:
               <ul>
                  <li><a href="https://huggingface.co/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6 (original site download)</a></li>
                  <li><a href="https://hf-mirror.com/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6 (mirror site download)</a></li>
                  <li>After downloading <code>LLVM-17.0.6-win64.exe</code>, double-click to install it, choose an appropriate installation location, and most importantly, check <code>Add Path to Current User</code> to add to the environment variables.</li>
                  <li>Confirm the installation is complete.</li>
               </ul>
            </li>
            <li>Download and install the Microsoft Visual C++ Redistributable package to resolve potential .dll missing issues.
               <ul>
                  <li><a href="https://aka.ms/vs/17/release/vc_redist.x64.exe">MSVC++ 14.40.33810.0 Download</a></li>
               </ul>
            </li>
            <li>Download and install Visual Studio Community Edition to obtain MSVC++ build tools, resolving LLVM header file dependencies.
               <ul>
                  <li><a href="https://visualstudio.microsoft.com/zh-hans/downloads/">Visual Studio Download</a></li>
                  <li>After installing Visual Studio Installer, download Visual Studio Community 2022.</li>
                  <li>Click the <code>Modify</code> button as shown below, find the <code>Desktop development with C++</code> option, and check it for download.</li>
                  <p align="center">
                     <img src="/docs/assets/figs/VS_1.jpg" width="75%">
                  </p>
               </ul>
            </li>
      </ol>
   </li>
   <li>Double-click <code>start.bat</code> to enter the Fish-Speech training inference configuration WebUI page.
      <ul>
            <li>(Optional) Want to go directly to the inference page? Edit the <code>API_FLAGS.txt</code> in the project root directory and modify the first three lines as follows:
               <pre><code>--infer
# --api
# --listen ...
...</code></pre>
            </li>
            <li>(Optional) Want to start the API server? Edit the <code>API_FLAGS.txt</code> in the project root directory and modify the first three lines as follows:
               <pre><code># --infer
--api
--listen ...
...</code></pre>
            </li>
      </ul>
   </li>
   <li>(Optional) Double-click <code>run_cmd.bat</code> to enter the conda/python command line environment of this project.</li>
</ol>

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

- 2024/07/02: Updated Fish-Speech to 1.2 version, remove VITS Decoder, and greatly enhanced zero-shot ability.
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
