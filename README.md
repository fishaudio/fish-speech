<div align="center">
<h1>Fish Speech</h1>

**English** | [简体中文](docs/README.zh.md) | [Portuguese](docs/README.pt-BR.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md) <br>

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

This codebase is released under Apache License and all model weights are released under CC-BY-NC-SA-4.0 License. Please refer to [LICENSE](LICENSE) for more details.

---
## Fish Agent
We are very excited to announce that we have made our self-research agent demo open source, you can now try our agent demo for instant English and Chinese chat locally by following the [docs](https://speech.fish.audio/start_agent/).

You should mention that the content is released under a **CC BY-NC-SA 4.0 licence**. And the demo is an early alpha test version, the inference speed needs to be optimised, and there are a lot of bugs waiting to be fixed. If you've found a bug or want to fix it, we'd be very happy to receive an issue or a pull request.

## Features
### Fish Speech

1. **Zero-shot & Few-shot TTS:** Input a 10 to 30-second vocal sample to generate high-quality TTS output. **For detailed guidelines, see [Voice Cloning Best Practices](https://docs.fish.audio/text-to-speech/voice-clone-best-practices).**

2. **Multilingual & Cross-lingual Support:** Simply copy and paste multilingual text into the input box—no need to worry about the language. Currently supports English, Japanese, Korean, Chinese, French, German, Arabic, and Spanish.

3. **No Phoneme Dependency:** The model has strong generalization capabilities and does not rely on phonemes for TTS. It can handle text in any language script.

4. **Highly Accurate:** Achieves a low CER (Character Error Rate) and WER (Word Error Rate) of around 2% for 5-minute English texts.

5. **Fast:** With fish-tech acceleration, the real-time factor is approximately 1:5 on an Nvidia RTX 4060 laptop and 1:15 on an Nvidia RTX 4090.

6. **WebUI Inference:** Features an easy-to-use, Gradio-based web UI compatible with Chrome, Firefox, Edge, and other browsers.

7. **GUI Inference:** Offers a PyQt6 graphical interface that works seamlessly with the API server. Supports Linux, Windows, and macOS. [See GUI](https://github.com/AnyaCoder/fish-speech-gui).

8. **Deploy-Friendly:** Easily set up an inference server with native support for Linux, Windows and MacOS, minimizing speed loss.

### Fish Agent
1. **Completely End to End:** Automatically integrates ASR and TTS parts, no need to plug-in other models, i.e., true end-to-end, not three-stage (ASR+LLM+TTS).

2. **Timbre Control:** Can use reference audio to control the speech timbre.

3. **Emotional:** The model can generate speech with strong emotion.

## Disclaimer

We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Online Demo

[Fish Audio](https://fish.audio)

[Fish Agent](https://fish.audio/demo/live)

## Quick Start for Local Inference 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/fishaudio/fish-speech/blob/main/inference.ipynb)

## Videos

#### V1.5 Demo Video: [Watch the video on X (Twitter).](https://x.com/FishAudio/status/1864370933496205728)

## Documents

- [English](https://speech.fish.audio/)
- [中文](https://speech.fish.audio/zh/)
- [日本語](https://speech.fish.audio/ja/)
- [Portuguese (Brazil)](https://speech.fish.audio/pt/)

## Samples (2024/10/02 V1.4)

- [English](https://speech.fish.audio/samples/)
- [中文](https://speech.fish.audio/zh/samples/)
- [日本語](https://speech.fish.audio/ja/samples/)
- [Portuguese (Brazil)](https://speech.fish.audio/pt/samples/)

## Credits

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

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

## Sponsor

<div>
  <a href="https://6block.com/">
    <img src="https://avatars.githubusercontent.com/u/60573493" width="100" height="100" alt="6Block Avatar"/>
  </a>
  <br>
  <a href="https://6block.com/">Data Processing sponsor by 6Block</a>
</div>
