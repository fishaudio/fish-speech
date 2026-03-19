<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | **简体中文** | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [العربية](README.ar.md) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
<br>
</div>
<br>

<div align="center">
    <img src="https://count.getloli.com/get/@fish-speech?theme=asoul" /><br>
</div>

<br>

<div align="center">
    <a target="_blank" href="https://discord.gg/Es5qTB9BcN">
        <img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
    </a>
    <a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
        <img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
    </a>
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

<div align="center">
    <a target="_blank" href="https://huggingface.co/fishaudio/s2">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://arxiv.org/abs/2603.08823">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **许可证声明**
> 此代码库及其相关的模型权重均在 **[FISH AUDIO RESEARCH LICENSE](../LICENSE)** 下发布。更多详情请参考 [LICENSE](../LICENSE)。


> [!WARNING]
> **法律免责声明**
> 我们不对代码库的任何非法使用承担责任。请参考您当地关于 DMCA 和其他相关法律的法规。

## 快速开始

### 文档入口

这里是 Fish Audio S2 的官方文档，请按照说明轻松入门。

- [安装](https://speech.fish.audio/zh/install/)
- [命令行推理](https://speech.fish.audio/zh/inference/)
- [WebUI 推理](https://speech.fish.audio/zh/inference/)
- [服务端推理](https://speech.fish.audio/zh/server/)
- [Docker 部署](https://speech.fish.audio/zh/install/)

> [!IMPORTANT]
> **如需使用 SGLang Server，请参考 [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md)。**

### LLM Agent 指南

```
请先阅读 https://speech.fish.audio/zh/install/ ，并按文档安装和配置 Fish Audio S2。
```

## Fish Audio S2 Pro
**行业顶尖的多语言文本转语音 (TTS) 系统，重新定义声音生成的边界。**

Fish Audio S2 Pro 是 [Fish Audio](https://fish.audio/) 开发的最先进的多模态模型。S2 Pro 训练自超过 **1000 万小时** 的海量音频数据，覆盖全球 **80 多种语言**。通过创新的 **双自回归 (Dual-AR)** 架构与强化学习 (RL) 对齐技术，S2 Pro 能生成极具自然感、真实感且情感饱满的语音，在开源与闭源竞争中均处于领先地位。

S2 Pro 的杀手锏在于支持通过自然语言标签（如 `[whisper]`、`[excited]`、`[angry]`）对韵律与情绪进行 **亚词级（Sub-word Level）** 的极细粒度行内控制，同时原生支持多说话人与超长上下文的多轮对话生成。

立即访问 [Fish Audio 官网](https://fish.audio/) 体验在线演示，或阅读我们的[技术报告](https://arxiv.org/abs/2603.08823)与[博客文章](https://fish.audio/blog/fish-audio-open-sources-s2/)深入了解。

### 模型变体

| 模型 | 大小 | 可用性 | 描述 |
|------|------|-------------|-------------|
| S2-Pro | 4B 参数 | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | 功能齐全的旗舰模型，具有最高质量和稳定性 |

有关模型的更多详情，请参见[技术报告](https://arxiv.org/abs/2411.01156)。

## 基准测试结果

| 基准 | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER（中文） | **0.54%**（总体最佳） |
| Seed-TTS Eval — WER（英文） | **0.99%**（总体最佳） |
| Audio Turing Test（含指令） | **0.515** 后验均值 |
| EmergentTTS-Eval — 胜率 | **81.88%**（总体最高） |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — 质量 | **4.51 / 5.0** |
| 多语言（MiniMax Testset）— 最佳 WER | **24** 种语言中的 **11** 种 |
| 多语言（MiniMax Testset）— 最佳 SIM | **24** 种语言中的 **17** 种 |

在 Seed-TTS Eval 上，S2 在所有已评估模型（包括闭源系统）中实现了最低 WER：Qwen3-TTS（0.77/1.24）、MiniMax Speech-02（0.99/1.90）、Seed-TTS（1.12/2.25）。在 Audio Turing Test 上，S2 的 0.515 相比 Seed-TTS（0.417）提升 24%，相比 MiniMax-Speech（0.387）提升 33%。在 EmergentTTS-Eval 中，S2 在副语言学（91.61% 胜率）、疑问句（84.41%）和句法复杂度（83.39%）等维度表现尤为突出。

## 亮点

<img src="./assets/totalability.png" width=200%>

### 通过自然语言进行极细粒度行内控制

S2 Pro 赋予了语音前所未有的“灵性”。通过简单的 `[tag]` 语法，你可以在文本的任何位置精准嵌入情感指令。
- **15,000+ 独特标签支持**：不局限于固定的预设，支持 **自由格式的文本描述**。你可以尝试 `[whisper in small voice]` (低声耳语), `[professional broadcast tone]` (专业播音腔), 或 `[pitch up]` (提高音调)。
- **丰富的情绪库**：
  `[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]` `[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]` `[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]` `[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]` `[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]` `[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]` `[shocked]`

### 创新的双自回归 (Dual-Autoregressive) 架构

S2 Pro 采用了主从式 Dual-AR 架构，由 Decoder-only Transformer 与 RVQ 音频编解码器（10 个码本，约 21 Hz 帧率）组成：

- **Slow AR (4B 参数)**：沿时间轴工作，预测核心的语义码本。
- **Fast AR (400M 参数)**：在每个时间步生成剩余 9 个残差码本，细腻还原极致的音频细节。

这种非对称设计在保证音频极致保真度的同时，大幅提升了推理速度。

### 强化学习对齐 (RL Alignment)

S2 Pro 采用了 **Group Relative Policy Optimization (GRPO)** 技术进行后训练对齐。我们将用于数据清洗与标注的同一套模型直接作为奖励模型 (Reward Model)，完美解决了预训练数据分布与后训练目标之间的不匹配问题。
- **多维奖励信号**：综合评估语义准确性、指令遵循能力、声学偏好评分以及音色相似度，确保生成的每一秒语音都符合人类直觉。

### 极致的流式推理性能 (基于 SGLang)

由于 Dual-AR 架构与标准 LLM 结构同构，S2 Pro 原生支持 SGLang 的所有推理加速特性，包括连续批处理 (Continuous Batching)、分页 KV Cache、CUDA Graph 与基于 RadixAttention 的前缀缓存。

**单张 NVIDIA H200 GPU 性能表现：**
- **实时因子 (RTF)**：0.195
- **首音延迟 (TTFA)**：约 100 ms
- **极速吞吐**：在保持 RTF < 0.5 时，吞吐量达到 3,000+ acoustic tokens/s

### 强大的多语言支持

S2 Pro 支持 80 多种语言，无需音素或特定语言的处理即可实现高质量合成：

- **第一梯队 (Tier 1)**：日语 (ja), 英语 (en), 中文 (zh)
- **第二梯队 (Tier 2)**：韩语 (ko), 西班牙语 (es), 葡萄牙语 (pt), 阿拉伯语 (ar), 俄语 (ru), 法语 (fr), 德语 (de)
- **全球覆盖**：sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, xsl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo 等。

### 原生多说话人生成

<img src="./assets/chattemplate.png" width=200%>

Fish Audio S2 允许用户上传包含多个说话人的参考音频，模型将通过 `<|speaker:i|>` 令牌处理每个说话人的特征。之后您可以通过说话人 ID 令牌控制模型的表现，从而实现一次生成中包含多个说话人。再也不需要像以前那样针对每个说话人都单独上传参考音频与生成语音了。

### 多轮对话生成

得益于模型上下文的扩展，我们的模型现在可以借助上文的信息提高后续生成内容的表现力，从而提升内容的自然度。

### 快速语音克隆

Fish Audio S2 支持使用短参考样本（通常为 10-30 秒）进行准确的语音克隆。模型可以捕捉音色、说话风格和情感倾向，无需额外微调即可生成逼真且一致的克隆语音。
如需使用 SGLang Server，请参考 [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md) 。

---

## 致谢

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## 技术报告

```bibtex
@misc{fish-speech-v1.4,
      title={Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis},
      author={Shijia Liao and Yuxuan Wang and Tianyu Li and Yifan Cheng and Ruoyi Zhang and Rongzhi Zhou and Yijin Xing},
      year={2024},
      eprint={2411.01156},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.01156},
}

@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823}, 
}
```
