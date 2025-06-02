<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | **简体中文** | [Portuguese](README.pt-BR.md) | [日本語](README.ja.md) | [한국어](README.ko.md) <br>

<a href="https://www.producthunt.com/posts/fish-speech-1-4?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-fish&#0045;speech&#0045;1&#0045;4" target="_blank">
    <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=488440&theme=light" alt="Fish&#0032;Speech&#0032;1&#0046;4 - Open&#0045;Source&#0032;Multilingual&#0032;Text&#0045;to&#0045;Speech&#0032;with&#0032;Voice&#0032;Cloning | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" />
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
    <a target="_blank" href="https://huggingface.co/spaces/fishaudio/fish-speech-1">
        <img alt="Huggingface" src="https://img.shields.io/badge/🤗%20-space%20demo-yellow"/>
    </a>
    <a target="_blank" href="https://pd.qq.com/s/bwxia254o">
      <img alt="QQ Channel" src="https://img.shields.io/badge/QQ-blue?logo=tencentqq">
    </a>
</div>

此代码库在 Apache License 下发布，所有模型权重在 CC-BY-NC-SA-4.0 License 下发布。更多详情请参考 [LICENSE](../LICENSE)。

我们很高兴地宣布，我们已将名字更改为 OpenAudio，这将是一个全新的文本转语音模型系列。

演示可在 [Fish Audio Playground](https://fish.audio) 获得。

访问 [OpenAudio 网站](https://openaudio.com) 获取博客和技术报告。

## 特性
### OpenAudio-S1 (Fish-Speech 的新版本)

1. 此模型具有 fish-speech 的**所有功能**。

2. OpenAudio S1 支持多种情感、语调和特殊标记来增强语音合成：
   
      (angry) (sad) (disdainful) (excited) (surprised) (satisfied) (unhappy) (anxious) (hysterical) (delighted) (scared) (worried) (indifferent) (upset) (impatient) (nervous) (guilty) (scornful) (frustrated) (depressed) (panicked) (furious) (empathetic) (embarrassed) (reluctant) (disgusted) (keen) (moved) (proud) (relaxed) (grateful) (confident) (interested) (curious) (confused) (joyful) (disapproving) (negative) (denying) (astonished) (serious) (sarcastic) (conciliative) (comforting) (sincere) (sneering) (hesitating) (yielding) (painful) (awkward) (amused) PS:中文也支持

   同时支持语调标记：

   (急促的语调) (大喊) (尖叫) (低语) (温柔的语调)

    还有一些特殊标记得到支持：

    (笑声) (轻笑) (抽泣) (大声哭泣) (叹气) (喘气) (呻吟) (人群笑声) (背景笑声) (观众笑声)

    您也可以使用 **哈，哈，哈** 来控制，还有许多其他情况等待您自己探索。

3. OpenAudio S1 包含以下规模：
-   **S1 (4B, 专有)：** 完整规模的模型。
-   **S1-mini (0.5B, 开源)：** S1 的蒸馏版本。

    S1 和 S1-mini 都结合了在线人类反馈强化学习（RLHF）。

4. 评估

    **Seed TTS 评估指标（英语，自动评估，基于 OpenAI gpt-4o-transcribe，使用 Revai/pyannote-wespeaker-voxceleb-resnet34-LM 的说话人距离）：**

    -   **S1：**
        -   WER（词错误率）：**0.008**
        -   CER（字符错误率）：**0.004**
        -   距离：**0.332**
    -   **S1-mini：**
        -   WER（词错误率）：**0.011**
        -   CER（字符错误率）：**0.005**
        -   距离：**0.380**
    

## 免责声明

我们不对代码库的任何非法使用承担责任。请参考您当地关于 DMCA 和其他相关法律的规定。

## 视频

#### 待续。

## 文档

- [构建环境](zh/install.md)
- [推理](zh/inference.md)

需要注意的是，当前模型**不支持微调**。

## 致谢

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

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
