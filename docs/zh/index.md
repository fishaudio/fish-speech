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
</div>

<strong>立即试用：</strong> <a href="https://fish.audio">Fish Audio Playground</a> | <strong>了解更多：</strong> <a href="https://openaudio.com">OpenAudio 网站</a>

</div>

---

!!! warning "法律声明"
    我们不对代码库的任何非法使用承担责任。请参考您所在地区有关 DMCA（数字千年版权法）和其他相关法律的规定。
    
    **许可证：** 此代码库在 Apache 2.0 许可证下发布，所有模型在 CC-BY-NC-SA-4.0 许可证下发布。

## **介绍**

我们很高兴地宣布，我们已经更名为 **OpenAudio** - 推出全新的先进文字转语音模型系列，在 Fish-Speech 的基础上进行了重大改进并增加了新功能。

**Openaudio-S1-mini**: [视频](即将上传); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [视频](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **亮点** ✨

### **情感控制**
OpenAudio S1 **支持多种情感、语调和特殊标记**来增强语音合成效果：

- **基础情感**：
```
(angry) (sad) (excited) (surprised) (satisfied) (delighted)
(scared) (worried) (upset) (nervous) (frustrated) (depressed)
(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)
(grateful) (confident) (interested) (curious) (confused) (joyful)
```

- **高级情感**：
```
(disdainful) (unhappy) (anxious) (hysterical) (indifferent) 
(impatient) (guilty) (scornful) (panicked) (furious) (reluctant)
(keen) (disapproving) (negative) (denying) (astonished) (serious)
(sarcastic) (conciliative) (comforting) (sincere) (sneering)
(hesitating) (yielding) (painful) (awkward) (amused)
```

- **语调标记**：
```
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)
```

- **特殊音效**：
```
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting)
(groaning) (crowd laughing) (background laughter) (audience laughing)
```

您还可以使用 Ha,ha,ha 来控制，还有许多其他用法等待您自己探索。

### **卓越的 TTS 质量**

我们使用 Seed TTS 评估指标来评估模型性能，结果显示 OpenAudio S1 在英文文本上达到了 **0.008 WER** 和 **0.004 CER**，明显优于以前的模型。（英语，自动评估，基于 OpenAI gpt-4o-转录，说话人距离使用 Revai/pyannote-wespeaker-voxceleb-resnet34-LM）

| 模型 | 词错误率 (WER) | 字符错误率 (CER) | 说话人距离 |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **两种模型类型**

| 模型 | 规模 | 可用性 | 特性 |
|-------|------|--------------|----------|
| **S1** | 40亿参数 | 在 [fish.audio](fish.audio) 上可用 | 功能齐全的旗舰模型 |
| **S1-mini** | 5亿参数 | 在 huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) 上可用 | 具有核心功能的蒸馏版本 |

S1 和 S1-mini 都集成了在线人类反馈强化学习 (RLHF)。

## **功能特性**

1. **零样本和少样本 TTS：** 输入 10 到 30 秒的语音样本即可生成高质量的 TTS 输出。**详细指南请参见 [语音克隆最佳实践](https://docs.fish.audio/text-to-speech/voice-clone-best-practices)。**

2. **多语言和跨语言支持：** 只需复制粘贴多语言文本到输入框即可——无需担心语言问题。目前支持英语、日语、韩语、中文、法语、德语、阿拉伯语和西班牙语。

3. **无音素依赖：** 该模型具有强大的泛化能力，不依赖音素进行 TTS。它可以处理任何语言文字的文本。

4. **高度准确：** 在 Seed-TTS Eval 中实现低字符错误率 (CER) 约 0.4% 和词错误率 (WER) 约 0.8%。

5. **快速：** 通过 fish-tech 加速，在 Nvidia RTX 4060 笔记本电脑上实时因子约为 1:5，在 Nvidia RTX 4090 上约为 1:15。

6. **WebUI 推理：** 具有易于使用的基于 Gradio 的网络界面，兼容 Chrome、Firefox、Edge 和其他浏览器。

7. **GUI 推理：** 提供与 API 服务器无缝配合的 PyQt6 图形界面。支持 Linux、Windows 和 macOS。[查看 GUI](https://github.com/AnyaCoder/fish-speech-gui)。

8. **部署友好：** 轻松设置推理服务器，原生支持 Linux、Windows 和 MacOS，最小化速度损失。

## **免责声明**

我们不对代码库的任何非法使用承担责任。请参考您当地关于 DMCA 和其他相关法律的规定。

## **媒体和演示**

#### 🚧 即将推出
视频演示和教程正在开发中。

## **文档**

### 快速开始
- [构建环境](install.md) - 设置您的开发环境
- [推理指南](inference.md) - 运行模型并生成语音

## **社区和支持**

- **Discord：** 加入我们的 [Discord 社区](https://discord.gg/Es5qTB9BcN)
- **网站：** 访问 [OpenAudio.com](https://openaudio.com) 获取最新更新
- **在线试用：** [Fish Audio Playground](https://fish.audio)
