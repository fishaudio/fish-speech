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
<img src="/docs/assets/figs/diagram.png" width="75%">
</p>

## 要求

- GPU 内存: 4GB (用于推理), 16GB (用于微调)
- 系统: Linux, Windows

## Windows 配置

Windows 专业用户可以考虑 WSL2 或 docker 来运行代码库。

Windows 非专业用户可考虑以下为免 Linux 环境的基础运行方法（附带模型编译功能，即 `torch.compile`）：

<ol>
   <li>解压项目压缩包。</li>
   <li>点击 install_env.bat 安装环境。
      <ul>
            <li>可以通过编辑 install_env.bat 的 <code>USE_MIRROR</code> 项来决定是否使用镜像站下载。</li>
            <li><code>USE_MIRROR=false</code> 使用原始站下载最新稳定版 <code>torch</code> 环境。<code>USE_MIRROR=true</code> 为从镜像站下载最新 <code>torch</code> 环境。默认为 <code>true</code>。</li>
            <li>可以通过编辑 install_env.bat 的 <code>INSTALL_TYPE</code> 项来决定是否启用可编译环境下载。</li>
            <li><code>INSTALL_TYPE=preview</code> 下载开发版编译环境。<code>INSTALL_TYPE=stable</code> 下载稳定版不带编译环境。</li>
      </ul>
   </li>
   <li>若第2步 INSTALL_TYPE=preview 则执行这一步（可跳过，此步为激活编译模型环境）
      <ol>
            <li>使用如下链接下载 LLVM 编译器。
               <ul>
                  <li><a href="https://huggingface.co/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6（原站站点下载）</a></li>
                  <li><a href="https://hf-mirror.com/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6（镜像站点下载）</a></li>
                  <li>下载完 LLVM-17.0.6-win64.exe 后，双击进行安装，选择合适的安装位置，最重要的是勾选 <code>Add Path to Current User</code> 添加环境变量。</li>
                  <li>确认安装完成。</li>
               </ul>
            </li>
            <li>下载安装 Microsoft Visual C++ 可再发行程序包，解决潜在 .dll 丢失问题。
               <ul>
                  <li><a href="https://aka.ms/vs/17/release/vc_redist.x64.exe">MSVC++ 14.40.33810.0 下载</a></li>
               </ul>
            </li>
            <li>下载安装 Visual Studio 社区版以获取 MSVC++ 编译工具, 解决 LLVM 的头文件依赖问题。
               <ul>
                  <li><a href="https://visualstudio.microsoft.com/zh-hans/downloads/">Visual Studio 下载</a></li>
                  <li>安装好Visual Studio Installer之后，下载Visual Studio Community 2022</li>
                  <li>如下图点击<code>修改</code>按钮，找到<code>使用C++的桌面开发</code>项，勾选下载</li>
                  <p align="center">
                     <img src="/docs/assets/figs/VS_1.jpg" width="75%">
                  </p>
               </ul>
            </li>
      </ol>
   </li>
   <li>双击 start.bat, 进入 Fish-Speech 训练推理配置 WebUI 页面。
      <ul>
            <li>(可选) 想直接进入推理页面？编辑项目根目录下的 <code>API_FLAGS.txt</code>, 前三行修改成如下格式:
               <pre><code>--infer
# --api
# --listen ...
...</code></pre>
            </li>
            <li>(可选) 想启动 API 服务器？编辑项目根目录下的 <code>API_FLAGS.txt</code>, 前三行修改成如下格式:
               <pre><code># --infer
--api
--listen ...
...</code></pre>
            </li>
      </ul>
   </li>
   <li>（可选）双击 <code>run_cmd.bat</code> 进入本项目的 conda/python 命令行环境</li>
</ol>

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

- 2024/07/02: 更新了 Fish-Speech 到 1.2 版本，移除 VITS Decoder，同时极大幅度提升zero-shot能力.
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
