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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
</div>

> [!IMPORTANT]
> **ライセンス注意事項**
> このコードベースは **Apache License** の下でリリースされ、すべてのモデルウェイトは **CC-BY-NC-SA-4.0 License** の下でリリースされています。詳細については [LICENSE](../LICENSE) をご参照ください。

> [!WARNING]
> **法的免責事項**
> 私たちはコードベースの不法な使用について一切の責任を負いません。DMCA 及びその他の関連法律について、現地の法律をご参照ください。

## ここから始める

こちらは Fish Speech の公式ドキュメントです。手順に従って簡単に始めることができます。

- [インストール](https://speech.fish.audio/ja/install/)
- [推論](https://speech.fish.audio/ja/inference/)

## Fish Audio S2
**オープンソースおよびクローズドソースの中で最も優れたテキスト読み上げシステム**

Fish Audio S2 は、[Fish Audio](https://fish.audio/) によって開発された最新のモデルで、自然でリアル、かつ感情豊かな音声を生成するように設計されています。ロボット的ではなく、平坦でもなく、スタジオスタイルのナレーションに限定されません。

Fish Audio S2 は日常の会話に焦点を当てており、ネイティブなマルチスピーカーおよびマルチターンの生成をサポートしています。また、命令制御もサポートしています。

S2 シリーズには複数のモデルが含まれており、オープンソースモデルは S2-Pro で、このシリーズの中で最もパフォーマンスの高いモデルです。

リアルタイムのエクスペリエンスについては、[Fish Audio Webサイト](https://fish.audio/) にアクセスしてください。

### モデルバリアント

| モデル | サイズ | 利用可能性 | 説明 |
|------|------|-------------|-------------|
| S2-Pro | 4B パラメータ | [huggingface](https://huggingface.co/fishaudio/s2-pro) | 最高の品質と安定性を備えた、フル機能のフラッグシップモデル |
| S2-Flash | - - - - | [fish.audio](https://fish.audio/) | より高速で低遅延なクローズドソースモデル |

モデルの詳細については、技術レポートを参照してください。

## ハイライト

<img src="./assets/totalability.png" width=200%>

### 自然言語制御

Fish Audio S2 では、ユーザーが自然言語を使用して、各文の表現、副言語情報、感情、その他の音声特性を制御できます。単に短いタグを使用してモデルのパフォーマンスを曖昧に制御するだけでなく、生成されたコンテンツ全体の品質が大幅に向上します。

### 多言語サポート

Fish Audio S2 は、音素や言語固有の前処理を必要とせずに、高品質な多言語テキスト読み上げをサポートします。以下を含みます：

**英語、中国語、日本語、韓国語、アラビア語、ドイツ語、フランス語...**

**さらに多く！**

リストは常に拡大しています。最新のリリースについては [Fish Audio](https://fish.audio/) を確認してください。

### ネイティブなマルチスピーカー生成

<img src="./assets/chattemplate.png" width=200%>

Fish Audio S2 では、ユーザーが複数のスピーカーを含む参照オーディオをアップロードでき、モデルは `<|speaker:i|>` トークンを介して各スピーカーの特徴を処理します。その後、スピーカーIDトークンを使用してモデルのパフォーマンスを制御し、1回の生成で複数のスピーカーを含めることができます。以前のように各スピーカーに対して個別に参照オーディオをアップロードして音声を生成する必要はもうありません。

### マルチターン対話生成

モデルのコンテキストの拡張により、以前の情報を使用して後続の生成されたコンテンツの表現力を向上させ、コンテンツの自然さを高めることができるようになりました。

### 高速音声クローニング

Fish Audio S2 は、短い参照サンプル（通常10〜30秒）を使用した正確な音声クローニングをサポートしています。モデルは音色、話し方、感情的な傾向を捉え、追加の微調整なしでリアルで一貫したクローン音声を生成します。

---

## クレジット

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
- [Qwen3](https://github.com/QwenLM/Qwen3)

## 技術レポート
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
