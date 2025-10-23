<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | **日本語** | [한국어](README.ko.md) | [العربية](README.ar.md) <br>

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
> **ライセンス注意事項**  
> このコードベースは**Apache License**の下でリリースされ、すべてのモデルウェイトは**CC-BY-NC-SA-4.0 License**の下でリリースされています。詳細については[LICENSE](../LICENSE)をご参照ください。

> [!WARNING]
> **法的免責事項**  
> 私たちはコードベースの不法な使用について一切の責任を負いません。DMCA及びその他の関連法律について、現地の法律をご参照ください。

## はじめに

こちらは Fish Speech の公式ドキュメントです。手順に従って簡単に始めることができます。

- [インストール](https://speech.fish.audio/ja/install/)
- [ファインチューニング](https://speech.fish.audio/ja/finetune/)
- [推論](https://speech.fish.audio/ja/inference/)
- [サンプル](https://speech.fish.audio/examples)

## 🎉 発表

**OpenAudio**へのリブランドを発表できることを嬉しく思います。Fish-Speechの基盤を元に構築された、革新的な新しい高度Text-to-Speechモデルシリーズを紹介します。

このシリーズの最初のモデルとして**OpenAudio-S1**をリリースできることを誇りに思います。品質、性能、機能において大幅な改善を実現しました。

OpenAudio-S1には2つのバージョンがあります：**OpenAudio-S1**と**OpenAudio-S1-mini**。両モデルとも[Fish Audio Playground](https://fish.audio)（**OpenAudio-S1**用）と[Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini)（**OpenAudio-S1-mini**用）で利用可能です。

ブログと技術レポートについては[OpenAudioウェブサイト](https://openaudio.com/blogs/s1)をご覧ください。

## ハイライト ✨

### **優秀なTTS品質**

Seed TTS Eval Metricsを使用してモデル性能を評価した結果、OpenAudio S1は英語テキストで**0.008 WER**と**0.004 CER**を達成し、これは従来のモデルより大幅に優れています。（英語、自動評価、OpenAI gpt-4o-transcribeベース、Revai/pyannote-wespeaker-voxceleb-resnet34-LMを使用した話者距離）

| モデル | 単語誤り率 (WER) | 文字誤り率 (CER) | 話者距離 |
|-------|------------------|------------------|----------|
| **S1** | **0.008** | **0.004** | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **TTS-Arena2でのベストモデル** 🏆

OpenAudio S1は、テキスト音声変換評価のベンチマークである[TTS-Arena2](https://arena.speechcolab.org/)で**1位**を獲得しました：

<div align="center">
    <img src="assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **音声制御**

OpenAudio S1は**音声合成を強化するための様々な感情、トーン、特別なマーカーをサポート**しています：

- **基本感情**:
```
(怒った) (悲しい) (興奮した) (驚いた) (満足した) (喜んだ) 
(恐れた) (心配した) (動揺した) (緊張した) (イライラした) (憂鬱な)
(共感的な) (恥ずかしい) (嫌悪した) (感動した) (誇らしい) (リラックスした)
(感謝する) (自信のある) (興味のある) (好奇心のある) (混乱した) (喜びに満ちた)
```

- **高度な感情**:
```
(軽蔑的な) (不幸な) (不安な) (ヒステリックな) (無関心な) 
(せっかちな) (罪悪感のある) (軽蔑した) (パニックした) (激怒した) (しぶしぶの)
(熱心な) (不賛成の) (否定的な) (否認する) (驚愕した) (真剣な)
(皮肉な) (宥める) (慰める) (誠実な) (冷笑する)
(躊躇する) (屈服する) (苦痛な) (気まずい) (面白がる)
```

- **トーンマーカー**:
```
(急いだトーン) (叫ぶ) (悲鳴) (囁く) (柔らかいトーン)
```

- **特別な音響効果**:
```
(笑う) (くすくす笑う) (すすり泣く) (大声で泣く) (ため息) (息切れ)
(うめく) (群衆の笑い声) (背景の笑い声) (聴衆の笑い声)
```

また、「ハ、ハ、ハ」を使って制御することもでき、あなた自身が探索できる多くの他のケースがあります。

（現在、英語、中国語、日本語をサポートしており、より多くの言語が近日公開予定です！）

### **2種類のモデル**

| モデル | サイズ | 利用可能性 | 機能 |
|-------|------|--------------|----------|
| **S1** | 4Bパラメータ | [fish.audio](https://fish.audio/)で利用可能 | フル機能のフラッグシップモデル |
| **S1-mini** | 0.5Bパラメータ | huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini)で利用可能 | コア機能を持つ蒸留版 |

S1とS1-miniの両方がオンライン人間フィードバック強化学習（RLHF）を組み込んでいます。

## **機能**

1. **ゼロショット・少数ショットTTS：** 10〜30秒の音声サンプルを入力して高品質のTTS出力を生成します。**詳細なガイドラインについては、[Voice Cloning Best Practices](https://docs.fish.audio/resources/best-practices/voice-cloning)をご覧ください。**

2. **多言語・言語横断サポート：** 多言語テキストを入力ボックスにコピー&ペーストするだけで、言語を気にする必要はありません。現在、英語、日本語、韓国語、中国語、フランス語、ドイツ語、アラビア語、スペイン語をサポートしています。

3. **音素依存なし：** モデルは強い汎化能力を持ち、TTSに音素に依存しません。どの言語の文字体系のテキストも処理できます。

4. **高精度：** Seed-TTS Evalで約0.4%の低いCER（文字誤り率）と約0.8%のWER（単語誤り率）を達成します。

5. **高速：** torch compileによる加速により、Nvidia RTX 4090 GPUで約1:7のリアルタイム係数を実現します。

6. **WebUI推論：** 使いやすいGradioベースのWeb UIを搭載し、Chrome、Firefox、Edgeなどのブラウザと互換性があります。

7. **デプロイフレンドリー：** Linux と Windows をネイティブサポートし（macOS サポートも近日対応予定）、パフォーマンスの低下を最小限に抑えながら、推論サーバーを簡単にセットアップできます。

## **メディア・デモ**

<div align="center">

### **ソーシャルメディア**
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-Latest_Demo-black?style=for-the-badge&logo=x&logoColor=white" alt="Latest Demo on X" />
</a>

### **インタラクティブデモ**
<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-Try_OpenAudio_S1-blue?style=for-the-badge" alt="Try OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-Try_S1_Mini-yellow?style=for-the-badge" alt="Try S1 Mini" />
</a>

### **ビデオショーケース**

<a href="https://www.youtube.com/watch?v=SYuPvd7m06A" target="_blank">
    <img src="assets/Thumbnail.jpg" alt="OpenAudio S1 Video" style="width: 50%;" />
</a>

</div>

---

## クレジット

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## 技術レポート (V1.4)
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
