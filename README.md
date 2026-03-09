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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **License Notice**  
> This codebase and its associated model weights are released under **[FISH AUDIO RESEARCH LICENSE](LICENSE)**. Please refer to [LICENSE](LICENSE) for more details.

> [!WARNING]
> **Legal Disclaimer**  
> We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Start Here

Here are the official documents for Fish Speech, follow the instructions to get started easily.

- [Installation](https://speech.fish.audio/install/)
- [Inference](https://speech.fish.audio/inference/)

## Fish Audio S2  
**Best Text-to-speech system among both open source and closed source**

Fish Audio S2 is the latest model developed by [Fish Audio](https://fish.audio/), designed to generate speech that sounds natural, realistic, and emotionally rich — not robotic, not flat, and not constrained to studio-style narration.

Fish Audio S2 focuses on daily conversation and dialogue, which enables native multi-speaker and multi-turn generation. Also supports instruction control.

The S2 series contains several models, the open-sourced model is S2-Pro, which is best model in the collection. 

Visit the [Fish Audio website](https://fish.audio/) for live playground.

### Model Variants

| Model | Size | Availability | Description |
|------|------|-------------|-------------|
| S2-Pro | 4B parameters | [huggingface](https://huggingface.co/fishaudio/s2-pro) | Full-featured flagship model with maximum quality and stability |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | Our closed source model with faster speed and lower latency |

More details of the model can be found in the technical report.

## Highlights

<img src="./docs/assets/totalability.png" width=200%>

### Natural language control

Fish Audio S2 allows users to use natural language to control the performance, paralinguistic information, emotions, and more voice features of each sentence, rather than just being limited to using short tags to vaguely control the model's performance. This greatly improves the overall quality of the generated content.

### Multilingual Support

Fish Audio S2 supports high-quality multilingual text-to-speech without requiring phonemes or language-specific preprocessing. Including:

**English, Chinese, Japanese, Korean, Arabics, German, French...**

**AND MORE!**

The list is constantly expanding, check [Fish Audio](https://fish.audio/) for the latest releases.

### Native multi-speaker generation

<img src="./docs/assets/chattemplate.png" width=200%>

Fish Audio S2 allows users to upload reference audio with multi-speaker, the model will deal with every speaker's feature via `<|speaker:i|>` token. Then you can control the model's performance with the speaker id token, allowing a single generation to include multiple speakers. You no longer need to upload reference audio separately for each speaker.

### Multi-turn generation

Thanks to the expansion of the model context, our model can now use previous information to improve the expressiveness of subsequent generated content, thereby increasing the naturalness of the content.

### Rapid Voice Cloning

Fish Audio S2 supports accurate voice cloning using a short reference sample (typically 10–30 seconds). The model captures timbre, speaking style, and emotional tendencies, producing realistic and consistent cloned voices without additional fine-tuning.

---

## Credits

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## Tech Report
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
