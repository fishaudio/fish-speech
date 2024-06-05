# 介绍

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
我们不对代码库的任何非法使用承担任何责任. 请参阅您当地关于 DMCA (数字千年法案) 和其他相关法律法规.

此代码库根据 `BSD-3-Clause` 许可证发布, 所有模型根据 CC-BY-NC-SA-4.0 许可证发布.

<p align="center">
<img src="/assets/figs/diagram.png" width="75%">
</p>

## 要求

- GPU 内存: 4GB (用于推理), 16GB (用于微调)
- 系统: Linux, Windows

~~我们建议 Windows 用户使用 WSL2 或 docker 来运行代码库, 或者使用由社区开发的整合环境.~~

## Windows 配置

Windows 专业用户可以考虑 WSL2 或 docker 来运行代码库。

Windows 非专业用户可考虑以下为免 Linux 环境的基础运行方法（附带模型编译功能，即 `torch.compile`）：

0. 解压项目压缩包。
1. 点击`install_env.bat`安装环境。
   - 可以通过编辑`install_env.bat`的`USE_MIRROR`项来决定是否使用镜像站下载。
   - 默认为`preview`, 使用镜像站且使用最新开发版本 torch（唯一激活编译方式）。
   - `false`使用原始站下载环境。`true`为从镜像站下载稳定版本 torch 和其余环境。
2. (可跳过，此步为激活编译模型环境)

   1. 使用如下链接下载`LLVM`编译器。
      - [LLVM-17.0.6 (原始站点下载)](https://huggingface.co/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true)
      - [LLVM-17.0.6 (镜像站点下载)](https://hf-mirror.com/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true)
      - 下载完`LLVM-17.0.6-win64.exe`后，双击进行安装，选择合适的安装位置，最重要的是勾选`Add Path to Current User`添加环境变量。
      - 确认安装完成。
   2. 下载安装`Microsoft Visual C++ 可再发行程序包`, 解决潜在`.dll`丢失问题。
      - [MSVC++ 14.40.33810.0 下载](https://aka.ms/vs/17/release/vc_redist.x64.exe)

3. 双击`start.bat`, 进入 Fish-Speech 训练推理配置 WebUI 页面。

   - 想直接进入推理页面？编辑项目根目录下的`API_FLAGS.txt`, 前三行修改成如下格式:

   ```text
   --infer
   # --api
   # --listen ...
   ...
   ```

   - 想启动 API 服务器？编辑项目根目录下的`API_FLAGS.txt`, 前三行修改成如下格式:

   ```text
   # --infer
   --api
   --listen ...
   ...
   ```

4. (可选)双击`run_cmd.bat`进入本项目的 conda/python 命令行环境

## Linux 配置

```bash
# 创建一个 python 3.10 虚拟环境, 你也可以用 virtualenv
conda create -n fish-speech python=3.10
conda activate fish-speech

# 安装 pytorch
pip3 install torch torchvision torchaudio

# 安装 fish-speech
pip3 install -e .

# (Ubuntu / Debian 用户) 安装 sox
apt install libsox-dev
```

## 更新日志

- 2024/05/10: 更新了 Fish-Speech 到 1.1 版本，引入了 VITS Decoder 来降低口胡和提高音色相似度.
- 2024/04/22: 完成了 Fish-Speech 1.0 版本, 大幅修改了 VQGAN 和 LLAMA 模型.
- 2023/12/28: 添加了 `lora` 微调支持.
- 2023/12/27: 添加了 `gradient checkpointing`, `causual sampling` 和 `flash-attn` 支持.
- 2023/12/19: 更新了 Webui 和 HTTP API.
- 2023/12/18: 更新了微调文档和相关例子.
- 2023/12/17: 更新了 `text2semantic` 模型, 支持无音素模式.
- 2023/12/13: 测试版发布, 包含 VQGAN 模型和一个基于 LLAMA 的语言模型 (只支持音素).

## 致谢

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
