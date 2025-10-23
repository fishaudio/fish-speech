<div align="center">
<h1>Fish Speech</h1>

**English** | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) | [العربية](docs/README.ar.md) <br>

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
> **License Notice**  
> This codebase is released under **Apache License** and all model weights are released under **CC-BY-NC-SA-4.0 License**. Please refer to [LICENSE](LICENSE) for more details.

> [!WARNING]
> **Legal Disclaimer**  
> We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Start Here

Here are the official documents for Fish Speech, follow the instructions to get started easily.

- [Installation](https://speech.fish.audio/install/)
- [Finetune](https://speech.fish.audio/finetune/)
- [Inference](https://speech.fish.audio/inference/)
- [Samples](https://speech.fish.audio/examples)

## 🎉 Announcement

We are excited to announce that we have rebranded to **OpenAudio** — introducing a revolutionary new series of advanced Text-to-Speech models that builds upon the foundation of Fish-Speech.

We are proud to release **OpenAudio-S1** as the first model in this series, delivering significant improvements in quality, performance, and capabilities.

OpenAudio-S1 comes in two versions: **OpenAudio-S1** and **OpenAudio-S1-mini**. Both models are now available on [Fish Audio Playground](https://fish.audio) (for **OpenAudio-S1**) and [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini) (for **OpenAudio-S1-mini**).

Visit the [OpenAudio website](https://openaudio.com/blogs/s1) for blog & tech report.

## Highlights ✨

### **Excellent TTS quality**

We use Seed TTS Eval Metrics to evaluate the model performance, and the results show that OpenAudio S1 achieves **0.008 WER** and **0.004 CER** on English text, which is significantly better than previous models. (English, auto eval, based on OpenAI gpt-4o-transcribe, speaker distance using Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| Model | Word Error Rate (WER) | Character Error Rate (CER) | Speaker Distance |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **Best Model in TTS-Arena2** 🏆

OpenAudio S1 has achieved the **#1 ranking** on [TTS-Arena2](https://arena.speechcolab.org/), the benchmark for text-to-speech evaluation:

<div align="center">
    <img src="docs/assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **Speech Control**

OpenAudio S1 **supports a variety of emotional, tone, and special markers** to enhance speech synthesis:

- **Basic emotions**:
```
(angry) (sad) (excited) (surprised) (satisfied) (delighted) 
(scared) (worried) (upset) (nervous) (frustrated) (depressed)
(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)
(grateful) (confident) (interested) (curious) (confused) (joyful)
```

- **Advanced emotions**:
```
(disdainful) (unhappy) (anxious) (hysterical) (indifferent) 
(impatient) (guilty) (scornful) (panicked) (furious) (reluctant)
(keen) (disapproving) (negative) (denying) (astonished) (serious)
(sarcastic) (conciliative) (comforting) (sincere) (sneering)
(hesitating) (yielding) (painful) (awkward) (amused)
```

- **Tone markers**:
```
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)
```

- **Special audio effects**:
```
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting)
(groaning) (crowd laughing) (background laughter) (audience laughing)
```

You can also use Ha,ha,ha to control, there's many other cases waiting to be explored by yourself.

(Support for English, Chinese and Japanese now, and more languages is coming soon!)

### **Two Type of Models**

| Model | Size | Availability | Features |
|-------|------|--------------|----------|
| **S1** | 4B parameters | Avaliable on [fish.audio](https://fish.audio/) | Full-featured flagship model |
| **S1-mini** | 0.5B parameters | Avaliable on huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) | Distilled version with core capabilities |

Both S1 and S1-mini incorporate online Reinforcement Learning from Human Feedback (RLHF).

## **Features**

1. **Zero-shot & Few-shot TTS:** Input a 10 to 30-second vocal sample to generate high-quality TTS output. **For detailed guidelines, see [Voice Cloning Best Practices](https://docs.fish.audio/resources/best-practices/voice-cloning).**

2. **Multilingual & Cross-lingual Support:** Simply copy and paste multilingual text into the input box—no need to worry about the language. Currently supports English, Japanese, Korean, Chinese, French, German, Arabic, and Spanish.

3. **No Phoneme Dependency:** The model has strong generalization capabilities and does not rely on phonemes for TTS. It can handle text in any language script.

4. **Highly Accurate:** Achieves a low CER (Character Error Rate) of around 0.4% and WER (Word Error Rate) of around 0.8% for Seed-TTS Eval.

5. **Fast:** Accelerated by torch compile, the real-time factor is approximately 1:7 on an Nvidia RTX 4090 GPU.

6. **WebUI Inference:** Features an easy-to-use, Gradio-based web UI compatible with Chrome, Firefox, Edge, and other browsers.

7. **Deploy-Friendly:** Easily set up an inference server with native support for Linux and Windows (macOS support coming soon), minimizing performance loss.

## **Media & Demos**

<div align="center">

### **Social Media**
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Latest_Demo-black?style=for-the-badge&logo=x&logoColor=white" alt="Latest Demo on X" />
</a>

### **Interactive Demos**
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Try_OpenAudio_S1-blue?style=for-the-badge" alt="Try OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Try_S1_Mini-yellow?style=for-the-badge" alt="Try S1 Mini" />
</a>

### **Video Showcases**

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="docs/assets/Thumbnail.jpg" alt="OpenAudio S1 Video" style="width: 50%;" />
</a>

</div>

---

## Credits

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Tech Report (V1.4)
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
