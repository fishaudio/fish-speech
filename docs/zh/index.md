# OpenAudio (原 Fish-Speech)

<div align="center">

<div align="center">

<img src="../assets/openaudio.jpg" alt="OpenAudio" style="display: block; margin: 0 auto; width: 35%;"/>

</div>

<strong>先进的文字转语音模型系列</strong>

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>·

<strong>立即试用：</strong> <a href="https://fish.audio">Fish Audio Playground</a> | <strong>了解更多：</strong> <a href="https://openaudio.com">OpenAudio 网站</a>

</div>

---

!!! note "许可声明"
    本代码库在 **Apache 许可证**下发布，所有模型权重在 **CC-BY-NC-SA-4.0 许可证**下发布。更多详情请参阅 [代码许可证](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) 和 [模型许可证](https://spdx.org/licenses/CC-BY-NC-SA-4.0)。

!!! warning "法律免责声明"
    我们不对代码库的任何非法使用承担责任。请参考您所在地区有关 DMCA 和其他相关法律的规定。

## **介绍**

我们很高兴地宣布，我们已经更名为 **OpenAudio** - 推出全新的先进文字转语音模型系列，在 Fish-Speech 的基础上进行了重大改进并增加了新功能。

**Openaudio-S1-mini**: [博客](https://openaudio.com/blogs/s1); [视频](https://www.youtube.com/watch?v=SYuPvd7m06A); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [视频](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **亮点**

### **优秀的 TTS 质量**

我们使用 Seed TTS 评估指标来评估模型性能，结果显示 OpenAudio S1 在英文文本上达到了 **0.008 WER** 和 **0.004 CER**，明显优于以前的模型。（英语，自动评估，基于 OpenAI gpt-4o-转录，说话人距离使用 Revai/pyannote-wespeaker-voxceleb-resnet34-LM）

| 模型 | 词错误率 (WER) | 字符错误率 (CER) | 说话人距离 |
|:-----:|:--------------------:|:-------------------------:|:----------------:|
| **S1** | **0.008** | **0.004** | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **TTS-Arena2 最佳模型**

OpenAudio S1 在 [TTS-Arena2](https://arena.speechcolab.org/) 上获得了 **#1 排名**，这是文字转语音评估的基准：

<div align="center">
    <img src="../assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **语音控制**
OpenAudio S1 **支持多种情感、语调和特殊标记**来增强语音合成效果：

- **基础情感**：
```
(生气) (伤心) (兴奋) (惊讶) (满意) (高兴) 
(害怕) (担心) (沮丧) (紧张) (失望) (沮丧)
(共情) (尴尬) (厌恶) (感动) (自豪) (放松)
(感激) (自信) (感兴趣) (好奇) (困惑) (快乐)
```

- **高级情感**：
```
(鄙视) (不高兴) (焦虑) (歇斯底里) (漠不关心) 
(不耐烦) (内疚) (轻蔑) (恐慌) (愤怒) (不情愿)
(渴望) (不赞成) (否定) (否认) (惊讶) (严肃)
(讽刺) (和解) (安慰) (真诚) (冷笑)
(犹豫) (让步) (痛苦) (尴尬) (开心)
```

（现在支持英语、中文和日语，更多语言即将推出！）

- **语调标记**：
```
(匆忙的语调) (大喊) (尖叫) (耳语) (轻声)
```

- **特殊音效**：
```
(笑) (轻笑) (抽泣) (大哭) (叹气) (喘气)
(呻吟) (群体笑声) (背景笑声) (观众笑声)
```

您还可以使用 Ha,ha,ha 来控制，还有许多其他用法等待您自己探索。

### **两种模型类型**

我们提供两种模型变体以满足不同需求：

- **OpenAudio S1 (40亿参数)**：我们功能齐全的旗舰模型，可在 [fish.audio](https://fish.audio) 上使用，提供最高质量的语音合成和所有高级功能。

- **OpenAudio S1-mini (5亿参数)**：具有核心功能的蒸馏版本，可在 [Hugging Face Space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) 上使用，针对更快推理进行优化，同时保持出色的质量。

S1 和 S1-mini 都集成了在线人类反馈强化学习 (RLHF)。

## **功能特性**

1. **零样本和少样本 TTS：** 输入 10 到 30 秒的语音样本即可生成高质量的 TTS 输出。**详细指南请参见 [语音克隆最佳实践](https://docs.fish.audio/text-to-speech/voice-clone-best-practices)。**

2. **多语言和跨语言支持：** 只需复制粘贴多语言文本到输入框即可——无需担心语言问题。目前支持英语、日语、韩语、中文、法语、德语、阿拉伯语和西班牙语。

3. **无音素依赖：** 该模型具有强大的泛化能力，不依赖音素进行 TTS。它可以处理任何语言文字的文本。

4. **高度准确：** 在 Seed-TTS Eval 中实现低字符错误率 (CER) 约 0.4% 和词错误率 (WER) 约 0.8%。

5. **快速：** 通过 torch compile 加速，在 Nvidia RTX 4090 GPU 上实时因子 (RTF) 约为 1:7。

6. **WebUI 推理：** 具有易于使用的基于 Gradio 的网络界面，兼容 Chrome、Firefox、Edge 和其他浏览器。

7. **GUI 推理：** 提供与 API 服务器无缝配合的 PyQt6 图形界面。支持 Linux、Windows 和 macOS。[查看 GUI](https://github.com/AnyaCoder/fish-speech-gui)。

8. **部署友好：** 轻松设置推理服务器，原生支持 Linux、Windows（MacOS 即将推出），最小化速度损失。

## **媒体和演示**

<!-- <div align="center"> -->

<h3><strong>社交媒体</strong></h3>
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-最新演示-black?style=for-the-badge&logo=x&logoColor=white" alt="Latest Demo on X" />
</a>

<h3><strong>互动演示</strong></h3>

<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-试用_OpenAudio_S1-blue?style=for-the-badge" alt="Try OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-试用_S1_Mini-yellow?style=for-the-badge" alt="Try S1 Mini" />
</a>

<h3><strong>视频展示</strong></h3>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/SYuPvd7m06A" title="OpenAudio S1 Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## **文档**

### 快速开始
- [构建环境](install.md) - 设置您的开发环境
- [推理指南](inference.md) - 运行模型并生成语音

## **社区和支持**

- **Discord：** 加入我们的 [Discord 社区](https://discord.gg/Es5qTB9BcN)
- **网站：** 访问 [OpenAudio.com](https://openaudio.com) 获取最新更新
- **在线试用：** [Fish Audio Playground](https://fish.audio)

## 模型

OpenAudio S1 是 OpenAudio 系列的第一个模型。它是一个双解码器 VQ-GAN 声码器，可以从 VQ 码元重建音频。
