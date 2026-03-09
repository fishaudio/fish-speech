<div align="center">
<h1>Fish Speech</h1>

[English](../en/) | [简体中文](../zh/) | [Portuguese](../pt/) | **日本語** | [한국어](../ko/) | [العربية](../ar/) <br>

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

!!! info "ライセンス通知"
    このコードベースは **Apache License** の下でリリースされており、すべてのモデルの重みは **CC-BY-NC-SA-4.0 License** の下でリリースされています。詳細は [LICENSE](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) を参照してください。

!!! warning "法的免責事項"
    私たちは、コードベースのいかなる違法な使用に対しても責任を負いません。DMCA およびその他の関連法に関する現地の規制を参照してください。

## ここから始める

これは Fish Speech の公式ドキュメントです。説明に従って簡単に使い始めることができます。

- [インストール](install.md)
- [推論](inference.md)

## Fish Audio S2
**オープンソースおよびクローズドソースの中で最高峰のテキスト読み上げシステム**

Fish Audio S2 は [Fish Audio](https://fish.audio/) によって開発された最新のモデルで、自然でリアル、かつ感情豊かな音声を生成するように設計されています。機械的でも平坦でもなく、スタジオスタイルの朗読に限定されません。

Fish Audio S2 は日常会話に焦点を当てており、ネイティブなマルチ話者およびマルチターン生成をサポートしています。また、指示制御もサポートしています。

S2 シリーズには複数のモデルが含まれており、オープンソースモデルは S2-Pro で、シリーズの中で最も強力なモデルです。

リアルタイム体験については、[Fish Audio Webサイト](https://fish.audio/) をご覧ください。

### モデルバリアント

| モデル | サイズ | 利用可能性 | 説明 |
|------|------|-------------|-------------|
| S2-Pro | 4B パラメータ | [huggingface](https://huggingface.co/fishaudio/s2-pro) | 最高品質と安定性を備えたフル機能のフラッグシップモデル |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | より高速で低遅延のクローズドソースモデル |

モデルの詳細については、技術レポートを参照してください。

## ハイライト

<img src="../assets/totalability.png" width=200%>

### 自然言語制御

Fish Audio S2 では、ユーザーが自然言語を使用して各文のパフォーマンス、副言語情報、感情、その他の音声特性を制御できます。短いタグを使用してモデルのパフォーマンスを曖昧に制御するだけでなく、生成されるコンテンツ全体の品質を大幅に向上させます。

### 多言語サポート

Fish Audio S2 は、音素や特定の言語のプリプロセスを必要とせず、高品質な多言語テキスト読み上げをサポートしています。以下を含みます：

**英語、中国語、日本語、韓国語、アラビア語、ドイツ語、フランス語...**

**さらに追加予定！**

リストは常に拡大しています。最新のリリースについては [Fish Audio](https://fish.audio/) を確認してください。

### ネイティブマルチ話者生成

<img src="../assets/chattemplate.png" width=200%>

Fish Audio S2 では、ユーザーが複数の話者を含むリファレンスオーディオをアップロードでき、モデルは `<|speaker:i|>` トークンを通じて各話者の特徴を処理します。その後、話者 ID トークンを介してモデルのパフォーマンスを制御し、1 回の生成で複数の話者を実現できます。話者ごとに個別にリファレンスオーディオをアップロードして音声を生成する必要はもうありません。

### マルチターン対話生成

モデルのコンテキストの拡張により、以前のコンテキストの情報を使用して、その後に生成されるコンテンツの表現力を向上させ、コンテンツの自然度を高めることができるようになりました。

### 高速音声クローン

Fish Audio S2 は、短いリファレンスサンプル（通常 10〜30 秒）を使用した正確な音声クローンをサポートしています。モデルは音色、話し方、感情的な傾向を捉えることができ、追加の微調整なしでリアルで一貫したクローン音声を生成できます。

---

## 謝辞

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## 技術報告

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
