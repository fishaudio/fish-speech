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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **License Notice**  
> This codebase and its associated model weights are released under **[FISH AUDIO RESEARCH LICENSE](LICENSE)**. Please refer to [LICENSE](LICENSE) for more details. We will take action against any violation of the license.

> [!WARNING]
> **Legal Disclaimer**  
> We do not hold any responsibility for any illegal usage of the codebase. Please refer to your local laws about DMCA and other related laws.

## Quick Start

### For Human

Here are the official documents for Fish Audio S2, follow the instructions to get started easily.

- [Installation](https://speech.fish.audio/install/)
- [Command Line Inference](https://speech.fish.audio/inference/#command-line-inference)
- [WebUI Inference](https://speech.fish.audio/inference/#webui-inference)
- [Server Inference](https://speech.fish.audio/server/)
- [Docker Setup](https://speech.fish.audio/install/#docker-setup)

> [!IMPORTANT]
> **For SGLang server, please read [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md).**

### For LLM Agent

```
Install and configure Fish-Audio S2 by following the instructions here: https://speech.fish.audio/install/
```

## Fish Audio S2  
**Best text-to-speech system among both open source and closed source**

Fish Audio S2 is the latest model developed by [Fish Audio](https://fish.audio/). Trained on over 10 million hours of audio across approximately 50 languages, S2 combines reinforcement learning alignment with a Dual-Autoregressive architecture to generate speech that sounds natural, realistic, and emotionally rich.

S2 supports fine-grained inline control of prosody and emotion using natural-language tags like `[laugh]`, `[whispers]`, and `[super happy]`, as well as native multi-speaker and multi-turn generation.

Visit the [Fish Audio website](https://fish.audio/) for live playground. Read the [blog post](https://fish.audio/blog/fish-audio-open-sources-s2/) for more details.

### Model Variants

| Model | Size | Availability | Description |
|------|------|-------------|-------------|
| S2-Pro | 4B parameters | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | Full-featured flagship model with maximum quality and stability |

More details of the model can be found in the [technical report](https://arxiv.org/abs/2411.01156).

## Benchmark Results

| Benchmark | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER (Chinese) | **0.54%** (best overall) |
| Seed-TTS Eval — WER (English) | **0.99%** (best overall) |
| Audio Turing Test (with instruction) | **0.515** posterior mean |
| EmergentTTS-Eval — Win Rate | **81.88%** (highest overall) |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — Quality | **4.51 / 5.0** |
| Multilingual (MiniMax Testset) — Best WER | **11 of 24** languages |
| Multilingual (MiniMax Testset) — Best SIM | **17 of 24** languages |

On Seed-TTS Eval, S2 achieves the lowest WER among all evaluated models including closed-source systems: Qwen3-TTS (0.77/1.24), MiniMax Speech-02 (0.99/1.90), Seed-TTS (1.12/2.25). On the Audio Turing Test, 0.515 surpasses Seed-TTS (0.417) by 24% and MiniMax-Speech (0.387) by 33%. On EmergentTTS-Eval, S2 achieves particularly strong results in paralinguistics (91.61% win rate), questions (84.41%), and syntactic complexity (83.39%).

## Highlights

<img src="./docs/assets/totalability.png" width=200%>

### Fine-Grained Inline Control via Natural Language

S2 enables localized control over speech generation by embedding natural-language instructions directly at specific word or phrase positions within the text. Rather than relying on a fixed set of predefined tags, S2 accepts free-form textual descriptions — such as `[whisper in small voice]`, `[professional broadcast tone]`, or `[pitch up]` — allowing open-ended expression control at the word level.

### Dual-Autoregressive Architecture

S2 builds on a decoder-only transformer combined with an RVQ-based audio codec (10 codebooks, ~21 Hz frame rate). The Dual-AR architecture splits generation into two stages:

- **Slow AR** operates along the time axis and predicts the primary semantic codebook.
- **Fast AR** generates the remaining 9 residual codebooks at each time step, reconstructing fine-grained acoustic detail.

This asymmetric design — 4B parameters along the time axis, 400M parameters along the depth axis — keeps inference efficient while preserving audio fidelity.

### Reinforcement Learning Alignment

S2 uses Group Relative Policy Optimization (GRPO) for post-training alignment. The same models used to filter and annotate training data are directly reused as reward models during RL — eliminating distribution mismatch between pre-training data and post-training objectives. The reward signal combines semantic accuracy, instruction adherence, acoustic preference scoring, and timbre similarity.

### Production Streaming via SGLang

Because the Dual-AR architecture is structurally isomorphic to standard autoregressive LLMs, S2 directly inherits all LLM-native serving optimizations from SGLang — including continuous batching, paged KV cache, CUDA graph replay, and RadixAttention-based prefix caching.

On a single NVIDIA H200 GPU:

- **Real-Time Factor (RTF):** 0.195
- **Time-to-first-audio:** ~100 ms
- **Throughput:** 3,000+ acoustic tokens/s while maintaining RTF below 0.5

### Multilingual Support

S2 supports high-quality multilingual text-to-speech without requiring phonemes or language-specific preprocessing. Including:

**English, Chinese, Japanese, Korean, Arabics, German, French...**

**AND MORE!**

The list is constantly expanding, check [Fish Audio](https://fish.audio/) for the latest releases.

### Native Multi-Speaker Generation

<img src="./docs/assets/chattemplate.png" width=200%>

Fish Audio S2 allows users to upload reference audio with multi-speaker, the model will deal with every speaker's feature via `<|speaker:i|>` token. Then you can control the model's performance with the speaker id token, allowing a single generation to include multiple speakers. You no longer need to upload reference audio separately for each speaker.

### Multi-Turn Generation

Thanks to the expansion of the model context, our model can now use previous information to improve the expressiveness of subsequent generated content, thereby increasing the naturalness of the content.

### Rapid Voice Cloning

Fish Audio S2 supports accurate voice cloning using a short reference sample (typically 10–30 seconds). The model captures timbre, speaking style, and emotional tendencies, producing realistic and consistent cloned voices without additional fine-tuning.
Please refer to https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md to use the SGLang server.
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
