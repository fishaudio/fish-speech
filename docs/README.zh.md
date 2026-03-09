<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | **简体中文** | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [العربية](README.ar.md) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</a>
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
    <a target="_blank" href="https://huggingface.co/spaces/TTS-AGI/TTS-Arena-V2">
      <img alt="TTS-Arena2 Score" src="https://img.shields.io/badge/TTS_Arena2-Rank_%231-gold?style=flat-square&logo=trophy&logoColor=white">
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **许可证声明**
> 此代码库在 **Apache License** 下发布，所有模型权重在 **CC-BY-NC-SA-4.0 License** 下发布。更多详情请参考 [LICENSE](../LICENSE)。

> [!WARNING]
> **法律免责声明**
> 我们不对代码库的任何非法使用承担责任。请参考您当地关于 DMCA 和其他相关法律的法规。

## 从这里开始

这里是 Fish Speech 的官方文档，请按照说明轻松入门。

- [安装](https://speech.fish.audio/zh/install/)
- [推理](https://speech.fish.audio/zh/inference/)

## Fish Audio S2
**开源和闭源中最出色的文本转语音系统**

Fish Audio S2 是由 [Fish Audio](https://fish.audio/) 开发的最新模型，旨在生成听起来自然、真实且情感丰富的语音——不机械、不平淡，也不局限于录音室风格的朗读。

Fish Audio S2 专注于日常对话，支持原生多说话人和多轮生成。同时支持指令控制。

S2 系列包含多个模型，开源模型为 S2-Pro，是该系列中性能最强的模型。

请访问 [Fish Audio 网站](https://fish.audio/) 以获取实时体验。

### 模型变体

| 模型 | 大小 | 可用性 | 描述 |
|------|------|-------------|-------------|
| S2-Pro | 4B 参数 | [huggingface](https://huggingface.co/fishaudio/s2-pro) | 功能齐全的旗舰模型，具有最高质量和稳定性 |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | 我们的闭源模型，具有更快的速度和更低的延迟 |

有关模型的更多详情，请参见技术报告。

## 亮点

<img src="./assets/totalability.png" width=200%>

### 自然语言控制

Fish Audio S2 允许用户使用自然语言去控制每一句内容的表现，副语言信息，情绪以及更多语音特征，而不单单局限于使用简短的标签去模糊地控制模型的表现，这极大地提高了生成内容整体的质量。

### 多语言支持

Fish Audio S2 支持高质量的多语言文本转语音，无需音素或特定语言的预处理。包括：

**英语、中文、日语、韩语、阿拉伯语、德语、法语...**

**以及更多！**

列表正在不断扩大，请查看 [Fish Audio](https://fish.audio/) 获取最新发布。

### 原生多说话人生成

<img src="./assets/chattemplate.png" width=200%>

Fish Audio S2 允许用户上传包含多个说话人的参考音频，模型将通过 `<|speaker:i|>` 令牌处理每个说话人的特征。之后您可以通过说话人 ID 令牌控制模型的表现，从而实现一次生成中包含多个说话人。再也不需要像以前那样针对每个说话人都单独上传参考音频与生成语音了。

### 多轮对话生成

得益于模型上下文的扩展，我们的模型现在可以借助上文的信息提高后续生成内容的表现力，从而提升内容的自然度。

### 快速语音克隆

Fish Audio S2 支持使用短参考样本（通常为 10-30 秒）进行准确的语音克隆。模型可以捕捉音色、说话风格和情感倾向，无需额外微调即可生成逼真且一致的克隆语音。

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
```

