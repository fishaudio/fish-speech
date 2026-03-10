<div align="center">
<h1>Fish Speech</h1>

**English** | [简体中文](../zh/) | [Portuguese](../pt/) | [日本語](../ja/) | [한국어](../ko/) | [العربية](../ar/) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
<a href="https://trendshift.io/repositories/7014" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/7014" alt="fishaudio%2Ffish-speech | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
</a>
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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

!!! info "License Notice"
    This codebase and its associated model weights are released under **FISH AUDIO RESEARCH LICENSE**. Please refer to [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) for more details.

!!! warning "Legal Disclaimer"
    We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Get Started

This is the official documentation for Fish Speech. Please follow the instructions to get started easily.

- [Installation](install.md)
- [Command Line Inference](inference.md#command-line-inference)
- [WebUI Inference](inference.md#webui-inference)
- [Server Inference](server.md)
- [Docker Setup](install.md#docker-setup)

!!! note
    For SGLang server, please read [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).

### For LLM Agent

```text
Install and configure Fish-Audio S2 by following the instructions here: https://speech.fish.audio/install/
```

## Fish Audio S2
**The best text-to-speech system in both open-source and closed-source**

Fish Audio S2 is the latest model developed by [Fish Audio](https://fish.audio/), designed to generate speech that sounds natural, authentic, and emotionally rich—not mechanical, flat, or confined to studio-style reading.

Fish Audio S2 focuses on everyday conversations, supporting native multi-speaker and multi-round generation. It also supports instruction control.

The S2 series includes multiple models. The open-source model is S2-Pro, which is the most powerful model in the series.

Please visit the [Fish Audio website](https://fish.audio/) for a real-time experience.

### Model Variants

| Model | Size | Availability | Description |
|------|------|-------------|-------------|
| S2-Pro | 4B Parameters | [huggingface](https://huggingface.co/fishaudio/s2-pro) | Full-featured flagship model with the highest quality and stability |

For more details on the models, please see the technical report.

## Highlights

<img src="../assets/totalability.png" width=200%>

### Natural Language Control

Fish Audio S2 allows users to use natural language to control the performance, paralinguistic information, emotions, and more voice characteristics of each sentence, instead of just using short tags to vaguely control the model's performance. This greatly improves the overall quality of the generated content.

### Multilingual Support

Fish Audio S2 supports high-quality multilingual text-to-speech without the need for phonemes or language-specific preprocessing. Including:

**English, Chinese, Japanese, Korean, Arabic, German, French...**

**And more!**

The list is constantly expanding, please check [Fish Audio](https://fish.audio/) for the latest releases.

### Native Multi-speaker Generation

<img src="../assets/chattemplate.png" width=200%>

Fish Audio S2 allows users to upload reference audio containing multiple speakers, and the model will process each speaker's characteristics through the `<|speaker:i|>` token. You can then control the model's performance via speaker ID tokens, achieving multiple speakers in a single generation. No more need to upload reference audio and generate speech for each speaker individually.

### Multi-round Dialogue Generation

Thanks to the expansion of the model's context, our model can now use the information from the previous context to improve the expressiveness of the subsequent generated content, thereby enhancing the naturalness of the content.

### Fast Voice Cloning

Fish Audio S2 supports accurate voice cloning using short reference samples (typically 10-30 seconds). The model can capture timbre, speaking style, and emotional tendency, generating realistic and consistent cloned voices without additional fine-tuning.
Please refer to https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md to use the SGLang server.

---

## Acknowledgements

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Technical Report

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
