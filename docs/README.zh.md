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
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
    <a target="_blank" href="https://huggingface.co/fishaudio/openaudio-s1-mini">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **许可证声明**  
> 此代码库在 **Apache License** 下发布，所有模型权重在 **CC-BY-NC-SA-4.0 License** 下发布。更多详情请参考 [LICENSE](../LICENSE)。

> [!WARNING]
> **法律免责声明**  
> 我们不对代码库的任何非法使用承担责任。请参考您当地关于 DMCA 和其他相关法律的法规。

## 开始使用

这里是 Fish Speech 的官方文档，按照说明轻松开始使用。

- [安装](https://speech.fish.audio/zh/install/)
- [微调](https://speech.fish.audio/zh/finetune/)
- [推理](https://speech.fish.audio/zh/inference/)
- [示例](https://speech.fish.audio/examples)

## 🎉 公告

我们很高兴地宣布，我们已将品牌重塑为 **OpenAudio** —— 推出基于 Fish-Speech 基础构建的革命性新一代高级文本转语音模型系列。

我们自豪地发布 **OpenAudio-S1** 作为该系列的第一个模型，在质量、性能和功能方面都有显著改进。

OpenAudio-S1 提供两个版本：**OpenAudio-S1** 和 **OpenAudio-S1-mini**。两个模型现在都可以在 [Fish Audio Playground](https://fish.audio)（**OpenAudio-S1**）和 [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini)（**OpenAudio-S1-mini**）上使用。

请访问 [OpenAudio 网站](https://openaudio.com/blogs/s1) 获取博客和技术报告。

## 亮点 ✨

### **出色的 TTS 质量**

我们使用 Seed TTS 评估指标来评估模型性能，结果显示 OpenAudio S1 在英语文本上达到了 **0.008 WER** 和 **0.004 CER**，这比以前的模型显著更好。（英语，自动评估，基于 OpenAI gpt-4o-transcribe，使用 Revai/pyannote-wespeaker-voxceleb-resnet34-LM 进行说话人距离计算）

| 模型 | 词错误率 (WER) | 字符错误率 (CER) | 说话人距离 |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **TTS-Arena2 最佳模型** 🏆

OpenAudio S1 在 [TTS-Arena2](https://arena.speechcolab.org/) 上取得了 **第一名**，这是文本转语音评估的基准：

<div align="center">
    <img src="../docs/assets/Elo.jpg" alt="TTS-Arena2 排名" style="width: 75%;" />
</div>

### **语音控制**

OpenAudio S1 **支持多种情感、语调和特殊标记** 来增强语音合成：

- **基础情感**：
```
(生气) (伤心) (兴奋) (惊讶) (满意) (高兴) 
(害怕) (担心) (沮丧) (紧张) (挫败) (郁闷)
(同情) (尴尬) (厌恶) (感动) (自豪) (放松)
(感激) (自信) (感兴趣) (好奇) (困惑) (快乐)
```

- **高级情感**：
```
(鄙视) (不开心) (焦虑) (歇斯底里) (冷漠) 
(不耐烦) (内疚) (轻蔑) (恐慌) (愤怒) (不情愿)
(热衷) (不赞成) (消极) (否认) (震惊) (严肃)
(讽刺) (安抚) (安慰) (真诚) (冷笑)
(犹豫) (屈服) (痛苦) (尴尬) (觉得有趣)
```

- **语调标记**：
```
(急促的语调) (喊叫) (尖叫) (耳语) (柔和的语调)
```

- **特殊音频效果**：
```
(笑声) (轻笑) (抽泣) (大声哭泣) (叹息) (喘息)
(呻吟) (人群笑声) (背景笑声) (观众笑声)
```

您也可以使用 哈,哈,哈 来控制，还有许多其他情况等待您自己探索。

(目前支持英语、中文和日语，更多语言即将推出！)

### **两种类型的模型**

| 模型 | 大小 | 可用性 | 特性 |
|-------|------|--------------|----------|
| **S1** | 4B 参数 | 在 [fish.audio](https://fish.audio) 上可用 | 功能齐全的旗舰模型 |
| **S1-mini** | 0.5B 参数 | 在 Hugging Face [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) 上可用 | 具有核心功能的精简版本 |

S1 和 S1-mini 都集成了在线人类反馈强化学习（RLHF）。

## **功能**

1. **零样本和少样本 TTS：** 输入 10 到 30 秒的语音样本以生成高质量的 TTS 输出。**详细指南请参见 [语音克隆最佳实践](https://docs.fish.audio/resources/best-practices/voice-cloning)。**

2. **多语言和跨语言支持：** 只需将多语言文本复制并粘贴到输入框中——无需担心语言问题。目前支持英语、日语、韩语、中文、法语、德语、阿拉伯语和西班牙语。

3. **无音素依赖：** 模型具有强大的泛化能力，不依赖音素进行 TTS。它可以处理任何语言脚本的文本。

4. **高准确性：** 在 Seed-TTS Eval 上实现约 0.4% 的低 CER（字符错误率）和约 0.8% 的 WER（词错误率）。

5. **快速：** 通过 torch compile 加速，在 Nvidia RTX 4090 GPU 上的实时因子约为 1:7。

6. **WebUI 推理：** 提供简单易用的、基于 Gradio 的 Web UI，兼容 Chrome、Firefox、Edge 等浏览器。

7. **易于部署：** 轻松设置推理服务器，原生支持 Linux 和 Windows（即将支持 macOS），最大限度地减少性能损失。

## **媒体和演示**

<div align="center">

### **社交媒体**
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-最新演示-black?style=for-the-badge&logo=x&logoColor=white" alt="X 上的最新演示" />
</a>

### **交互式演示**
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-试用_OpenAudio_S1-blue?style=for-the-badge" alt="试用 OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-试用_S1_Mini-yellow?style=for-the-badge" alt="试用 S1 Mini" />
</a>

### **视频展示**

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="../docs/assets/Thumbnail.jpg" alt="OpenAudio S1 Video" style="width: 50%;" />
</a>

</div>

---

## 致谢

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## 技术报告 (V1.4)
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
