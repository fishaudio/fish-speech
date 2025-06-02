<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | **日本語** | [한국어](README.ko.md) <br>

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

このコードベースはApache Licenseの下でリリースされ、すべてのモデルウェイトはCC-BY-NC-SA-4.0 Licenseの下でリリースされています。詳細については[LICENSE](../LICENSE)をご参照ください。

私たちは名前をOpenAudioに変更したことをお知らせでき、嬉しく思います。これは全く新しいText-to-Speechモデルシリーズになります。

デモは[Fish Audio Playground](https://fish.audio)で利用可能です。

ブログと技術レポートについては[OpenAudioウェブサイト](https://openaudio.com)をご覧ください。

## 機能
### OpenAudio-S1 (Fish-Speechの新バージョン)

1. このモデルはfish-speechが持っていた**すべての機能**を持っています。

2. OpenAudio S1は音声合成を強化するための様々な感情、トーン、特別なマーカーをサポートしています：
   
      (angry) (sad) (disdainful) (excited) (surprised) (satisfied) (unhappy) (anxious) (hysterical) (delighted) (scared) (worried) (indifferent) (upset) (impatient) (nervous) (guilty) (scornful) (frustrated) (depressed) (panicked) (furious) (empathetic) (embarrassed) (reluctant) (disgusted) (keen) (moved) (proud) (relaxed) (grateful) (confident) (interested) (curious) (confused) (joyful) (disapproving) (negative) (denying) (astonished) (serious) (sarcastic) (conciliative) (comforting) (sincere) (sneering) (hesitating) (yielding) (painful) (awkward) (amused)

   またトーンマーカーもサポートしています：

   (急いだトーン) (叫び) (絶叫) (ささやき) (柔らかいトーン)

    サポートされているいくつかの特別なマーカーがあります：

    (笑い) (くすくす笑い) (すすり泣き) (大声で泣く) (ため息) (あえぎ) (うめき) (群衆の笑い) (背景の笑い) (観客の笑い)

    また、**ハ、ハ、ハ**を使って制御することもでき、あなた自身が探索を待っている他の多くのケースがあります。

3. OpenAudio S1には以下のサイズが含まれています：
-   **S1 (4B, プロプライエタリ):** フルサイズのモデル。
-   **S1-mini (0.5B, オープンソース):** S1の蒸留版。

    S1とS1-miniの両方がオンライン人間フィードバック強化学習（RLHF）を組み込んでいます。

4. 評価

    **Seed TTS評価メトリクス（英語、自動評価、OpenAI gpt-4o-transcribeベース、Revai/pyannote-wespeaker-voxceleb-resnet34-LMを使用したスピーカー距離）：**

    -   **S1:**
        -   WER（単語誤り率）：**0.008**
        -   CER（文字誤り率）：**0.004**
        -   距離：**0.332**
    -   **S1-mini:**
        -   WER（単語誤り率）：**0.011**
        -   CER（文字誤り率）：**0.005**
        -   距離：**0.380**
    

## 免責事項

コードベースの違法な使用について、いかなる責任も負いません。DMCAおよびその他の関連法律に関する現地の法律をご参照ください。

## 動画

#### 続く予定。

## ドキュメント

- [環境構築](en/install.md)
- [推論](en/inference.md)

現在のモデルは**ファインチューニングをサポートしていない**ことに注意してください。

## クレジット

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

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
