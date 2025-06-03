# OpenAudio (formerly Fish-Speech)

<div align="center">

<div align="center">

<img src="../assets/openaudio.jpg" alt="OpenAudio" style="display: block; margin: 0 auto; width: 35%;"/>

</div>

<strong>Advanced Text-to-Speech Model Series</strong>

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

<strong>Try it now:</strong> <a href="https://fish.audio">Fish Audio Playground</a> | <strong>Learn more:</strong> <a href="https://openaudio.com">OpenAudio Website</a>

</div>

---

!!! warning "Legal Notice"
    We assume no responsibility for any illegal use of the codebase. Please refer to the local laws regarding DMCA (Digital Millennium Copyright Act) and other relevant laws in your area.
    
    **License:** This codebase is released under Apache 2.0 license and all models are released under the CC-BY-NC-SA-4.0 license.

## **Introduction**

We are excited to announce that we have rebranded to **OpenAudio** - introducing a brand new series of advanced Text-to-Speech models that builds upon the foundation of Fish-Speech with significant improvements and new capabilities.

**Openaudio-S1-mini**: [Video](To Be Uploaded); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [Video](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **Highlights** ✨

### **Emotion Control**
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

### **Excellent TTS quality**

We use Seed TTS Eval Metrics to evaluate the model performance, and the results show that OpenAudio S1 achieves **0.008 WER** and **0.004 CER** on English text, which is significantly better than previous models. (English, auto eval, based on OpenAI gpt-4o-transcribe, speaker distance using Revai/pyannote-wespeaker-voxceleb-resnet34-LM)

| Model | Word Error Rate (WER) | Character Error Rate (CER) | Speaker Distance |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **Two Type of Models**

| Model | Size | Availability | Features |
|-------|------|--------------|----------|
| **S1** | 4B parameters | Avaliable on [fish.audio](fish.audio) | Full-featured flagship model |
| **S1-mini** | 0.5B parameters | Avaliable on huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) | Distilled version with core capabilities |

Both S1 and S1-mini incorporate online Reinforcement Learning from Human Feedback (RLHF).

## **Features**

1. **Zero-shot & Few-shot TTS:** Input a 10 to 30-second vocal sample to generate high-quality TTS output. **For detailed guidelines, see [Voice Cloning Best Practices](https://docs.fish.audio/text-to-speech/voice-clone-best-practices).**

2. **Multilingual & Cross-lingual Support:** Simply copy and paste multilingual text into the input box—no need to worry about the language. Currently supports English, Japanese, Korean, Chinese, French, German, Arabic, and Spanish.

3. **No Phoneme Dependency:** The model has strong generalization capabilities and does not rely on phonemes for TTS. It can handle text in any language script.

4. **Highly Accurate:** Achieves a low CER (Character Error Rate) of around 0.4% and WER (Word Error Rate) of around 0.8% for Seed-TTS Eval.

5. **Fast:** With fish-tech acceleration, the real-time factor is approximately 1:5 on an Nvidia RTX 4060 laptop and 1:15 on an Nvidia RTX 4090.

6. **WebUI Inference:** Features an easy-to-use, Gradio-based web UI compatible with Chrome, Firefox, Edge, and other browsers.

7. **GUI Inference:** Offers a PyQt6 graphical interface that works seamlessly with the API server. Supports Linux, Windows, and macOS. [See GUI](https://github.com/AnyaCoder/fish-speech-gui).

8. **Deploy-Friendly:** Easily set up an inference server with native support for Linux, Windows (MacOS comming soon), minimizing speed loss.

## **Disclaimer**

We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## **Media & Demos**

#### 🚧 Coming Soon
Video demonstrations and tutorials are currently in development.

## **Documentation**

### Quick Start
- [Build Environment](en/install.md) - Set up your development environment
- [Inference Guide](en/inference.md) - Run the model and generate speech


## **Community & Support**

- **Discord:** Join our [Discord community](https://discord.gg/Es5qTB9BcN)
- **Website:** Visit [OpenAudio.com](https://openaudio.com) for latest updates
- **Try Online:** [Fish Audio Playground](https://fish.audio)
