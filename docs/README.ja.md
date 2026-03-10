<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | **日本語** | [한국어](README.ko.md) | [العربية](README.ar.md) <br>

<a href="https://www.producthunt.com/products/fish-speech?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-fish&#0045;audio&#0045;s1" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1023740&theme=light&period=daily&t=1761164814710" alt="Fish&#0032;Audio&#0032;S1 - Expressive&#0032;Voice&#0032;Cloning&#0032;and&#0032;Text&#0045;to&#0045;Speech | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://github.com/fishaudio/fish-speech/blob/main/FishAudioS2TecReport.pdf">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Tecnical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **ライセンス注意事項**
> このコードベースおよび関連するモデルウェイトは **[FISH AUDIO RESEARCH LICENSE](../LICENSE)** の下でリリースされています。詳細については [LICENSE](../LICENSE) をご参照ください。

> [!WARNING]
> **法的免責事項**
> 私たちはコードベースの不法な使用について一切の責任を負いません。DMCA 及びその他の関連法律について、現地の法律をご参照ください。

## クイックスタート

### まずはドキュメントから

Fish Audio S2 の公式ドキュメントです。以下からすぐに始められます。

- [インストール](https://speech.fish.audio/ja/install/)
- [コマンドライン推論](https://speech.fish.audio/ja/inference/)
- [WebUI 推論](https://speech.fish.audio/ja/inference/)
- [サーバー推論](https://speech.fish.audio/ja/server/)
- [Docker セットアップ](https://speech.fish.audio/ja/install/)

> [!IMPORTANT]
> **SGLang サーバーについては [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md) を参照してください。**

### LLM Agent 向け

```
https://speech.fish.audio/ja/install/ の手順に従って、Fish Audio S2 をインストール・設定してください。
```

## Fish Audio S2
**オープンソースおよびクローズドソースの中で最も優れたテキスト読み上げシステム**

Fish Audio S2 は [Fish Audio](https://fish.audio/) が開発した最新モデルです。約 50 言語・1,000 万時間超の音声データで学習され、強化学習アラインメントと Dual-Autoregressive アーキテクチャを組み合わせることで、自然でリアルかつ感情表現豊かな音声を生成します。

S2 は `[laugh]`、`[whispers]`、`[super happy]` といった自然言語タグで、韻律や感情を文中の任意位置で細かく制御できます。さらに、マルチスピーカー生成とマルチターン生成にもネイティブ対応しています。

ライブデモは [Fish Audio ウェブサイト](https://fish.audio/) から、詳細は [ブログ記事](https://fish.audio/blog/fish-audio-open-sources-s2/) をご覧ください。

### モデルバリアント

| モデル | サイズ | 利用可能性 | 説明 |
|------|------|-------------|-------------|
| S2-Pro | 4B パラメータ | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | 品質と安定性を最大化したフル機能のフラッグシップモデル |

モデルの詳細は[技術レポート](https://arxiv.org/abs/2411.01156)をご参照ください。

## ベンチマーク結果

| ベンチマーク | Fish Audio S2 |
|------|------|
| Seed-TTS Eval — WER（中国語） | **0.54%**（全体最良） |
| Seed-TTS Eval — WER（英語） | **0.99%**（全体最良） |
| Audio Turing Test（指示あり） | **0.515** 事後平均値 |
| EmergentTTS-Eval — 勝率 | **81.88%**（全体最高） |
| Fish Instruction Benchmark — TAR | **93.3%** |
| Fish Instruction Benchmark — 品質 | **4.51 / 5.0** |
| 多言語（MiniMax Testset）— 最良 WER | **24 言語中 11 言語** |
| 多言語（MiniMax Testset）— 最良 SIM | **24 言語中 17 言語** |

Seed-TTS Eval では、S2 はクローズドソースを含む全評価モデルの中で最小 WER を達成しました：Qwen3-TTS（0.77/1.24）、MiniMax Speech-02（0.99/1.90）、Seed-TTS（1.12/2.25）。Audio Turing Test では 0.515 を記録し、Seed-TTS（0.417）比で 24%、MiniMax-Speech（0.387）比で 33% 上回りました。EmergentTTS-Eval では、副言語情報（91.61%）、疑問文（84.41%）、統語的複雑性（83.39%）で特に高い成績を示しています。

## ハイライト

<img src="./assets/totalability.png" width=200%>

### 自然言語による細粒度インライン制御

Fish Audio S2 では、テキスト内の特定の単語やフレーズ位置に自然言語の指示を直接埋め込むことで、音声生成を局所的に制御できます。固定の事前定義タグに依存するのではなく、S2 は [whisper in small voice]、[professional broadcast tone]、[pitch up] のような自由形式のテキスト記述を受け付け、単語レベルで表現をオープンエンドに制御できます。

### 二重自己回帰（Dual-Autoregressive）アーキテクチャ

S2 はデコーダー専用 Transformer と RVQ ベースの音声コーデック（10 codebooks、約 21 Hz）を組み合わせています。Dual-AR は生成を 2 段階に分割します。

- **Slow AR** は時間軸方向に動作し、主となる semantic codebook を予測。
- **Fast AR** は各時刻で残り 9 個の residual codebook を生成し、細かな音響ディテールを復元。

この非対称設計（時間軸 4B パラメータ、深さ軸 400M パラメータ）により、音質を保ちながら推論効率を高めています。

### 強化学習アラインメント

S2 は後学習アラインメントに Group Relative Policy Optimization（GRPO）を採用しています。学習データのフィルタリングとアノテーションに使った同一モデル群を、そのまま RL の報酬モデルとして再利用することで、事前学習データ分布と事後学習目的のミスマッチを抑制しています。報酬信号には、意味的正確性、指示追従性、音響的選好スコア、音色類似度が含まれます。

### SGLang による本番向けストリーミング

Dual-AR は構造的に標準的な自己回帰 LLM と同型のため、S2 は SGLang の LLM 向け最適化をそのまま活用できます。たとえば continuous batching、paged KV cache、CUDA graph replay、RadixAttention ベースの prefix caching です。

単一の NVIDIA H200 GPU での実測:

- **RTF（Real-Time Factor）:** 0.195
- **初回音声出力までの時間:** 約 100 ms
- **スループット:** RTF 0.5 未満を維持しつつ 3,000+ acoustic tokens/s

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
SGLang サーバーの利用については https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md を参照してください。

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
