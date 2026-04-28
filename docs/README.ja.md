<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | **日本語** | [한국어](README.ko.md) | [العربية](README.ar.md) | [Español](docs/README.es.md)  <br>

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
    <a target="_blank" href="https://huggingface.co/fishaudio/s2-pro">
        <img alt="HuggingFace Model" src="https://img.shields.io/badge/🤗%20-models-orange"/>
    </a>
    <a target="_blank" href="https://fish.audio/blog/fish-audio-open-sources-s2/">
        <img alt="Fish Audio Blog" src="https://img.shields.io/badge/Blog-Fish_Audio_S2-1f7a8c?style=flat-square&logo=readme&logoColor=white"/>
    </a>
    <a target="_blank" href="https://arxiv.org/abs/2603.08823">
        <img alt="Paper | Technical Report" src="https://img.shields.io/badge/Paper-Technical_Report-b31b1b?style=flat-square"/>
    </a>
</div>

> [!IMPORTANT]
> **ライセンス注意事項**
> このコードベースおよび関連するモデルウェイトは **[FISH AUDIO RESEARCH LICENSE](../LICENSE)** の下でリリースされています。詳細については [LICENSE](../LICENSE) をご参照ください。


> [!WARNING]
> **法的免責事項**
> 私たちはコードベースの不法な使用について一切の責任を負いません。DMCA 及びその他の関連法律について、現地の法律をご参照ください。

## クイックスタート

### ドキュメント入口

Fish Audio S2 の公式ドキュメントです。以下からすぐに始められます。

- [インストール](https://speech.fish.audio/ja/install/)
- [コマンドライン推論](https://speech.fish.audio/ja/inference/)
- [WebUI 推論](https://speech.fish.audio/ja/inference/)
- [サーバー推論](https://speech.fish.audio/ja/server/)
- [Docker デプロイ](https://speech.fish.audio/ja/install/)

> [!IMPORTANT]
> **SGLang サーバーについては [SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md) を参照してください。**
>
> **vLLM Omni サーバーについては [vLLM-Omni Fish Speech README](https://github.com/vllm-project/vllm-omni/blob/main/examples/online_serving/fish_speech/README.md) を参照してください。**

### LLM Agent 指南

```
https://speech.fish.audio/ja/install/ の手順に従って、Fish Audio S2 をインストール・設定してください。
```

## Fish Audio S2 Pro
**業界最先端の多言語テキスト読み上げ (TTS) システム。音声生成の限界を再定義します。**

Fish Audio S2 Pro は [Fish Audio](https://fish.audio/) が開発した最高峰のマルチモーダルモデルです。世界 **80 言語以上**、**1,000 万時間** を超える膨大な音声データで学習されています。革新的な **二重自己回帰 (Dual-AR)** アーキテクチャと強化学習 (RL) アライメント技術を組み合わせることで、極めて自然でリアル、かつ感情豊かな音声を生成し、オープンソースおよびクローズドソースの双方でリーダーシップを発揮しています。

S2 Pro の最大の特徴は、自然言語タグ（例：`[whisper]`、`[excited]`、`[angry]`）による韻律や感情の **サブワードレベル (Sub-word Level)** での極めて細やかなインライン制御が可能である点です。また、マルチスピーカー生成や長文コンテキストのマルチターン対話生成にもネイティブ対応しています。

今すぐ [Fish Audio 公式サイト](https://fish.audio/) でプレイグラウンドを体験するか、[技術レポート](https://arxiv.org/abs/2603.08823) や [ブログ記事](https://fish.audio/blog/fish-audio-open-sources-s2/) を読んで詳細を確認してください。

### モデルバリアント

| モデル | サイズ | 利用可能性 | 説明 |
|------|------|-------------|-------------|
| S2-Pro | 4B パラメータ | [HuggingFace](https://huggingface.co/fishaudio/s2-pro) | 品質と安定性を最大化した、フル機能のフラッグシップモデル |

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

S2 Pro は音声にこれまでにない「魂」を宿らせます。シンプルな `[tag]` 構文を使用して、テキスト内の任意の場所に感情の指示を正確に埋め込むことができます。
- **1万5,000以上のユニークタグに対応**：固定のプリセットに限定されず、**自由形式のテキスト記述** をサポートします。`[whisper in small voice]` (ささやき声で), `[professional broadcast tone]` (プロのナレーション風), `[pitch up]` (ピッチを上げる) などを試してみてください。
- **豊富な感情ライブラリ**:
  `[pause]` `[emphasis]` `[laughing]` `[inhale]` `[chuckle]` `[tsk]` `[singing]` `[excited]` `[laughing tone]` `[interrupting]` `[chuckling]` `[excited tone]` `[volume up]` `[echo]` `[angry]` `[low volume]` `[sigh]` `[low voice]` `[whisper]` `[screaming]` `[shouting]` `[loud]` `[surprised]` `[short pause]` `[exhale]` `[delight]` `[panting]` `[audience laughter]` `[with strong accent]` `[volume down]` `[clearing throat]` `[sad]` `[moaning]` `[shocked]`

### 革新的な二重自己回帰 (Dual-Autoregressive) アーキテクチャ

S2 Pro は、Decoder-only Transformer と RVQ オーディオコーデック（10 コードブック、約 21 Hz）で構成されるマスター・スレーブ型の Dual-AR アーキテクチャを採用しています：

- **Slow AR (4B パラメータ)**: 時間軸方向に動作し、核となるセマンティックコードブックを予測。
- **Fast AR (400M パラメータ)**: 各時間ステップで残り 9 個の残差コードブックを生成し、極めて繊細な音響ディテールを復元。

この非対称設計により、究極のオーディオ忠実度を維持しながら、推論速度を大幅に向上させています。

### 強化学習 (RL) アライメント

S2 Pro は、事後学習アライメントに **Group Relative Policy Optimization (GRPO)** 技術を採用しています。データのクリーニングとアノテーションに使用したモデルセットをそのまま報酬モデル (Reward Model) として使用することで、事前学習データの分布と事後学習の目標との間のミスマッチを完璧に解決しました。
- **多次元の報酬信号**: 意味の正確性、指示追従性、音響的な好み、音色の類似性を総合的に評価し、生成される一秒一秒の音声が人間の直感に沿うようにしています。

### SGLang による究極のストリーミング推論性能

Dual-AR アーキテクチャは標準的な LLM 構造と同型であるため、S2 Pro は SGLang のすべての推論加速機能をネイティブにサポートしています。これには、Continuous Batching、Paged KV Cache、CUDA Graph、RadixAttention ベースの Prefix Caching が含まれます。

**NVIDIA H200 GPU 1枚でのパフォーマンス表現:**
- **リアルタイム係数 (RTF)**: 0.195
- **初回音声出力までの時間 (TTFA)**: 約 100 ms
- **極速スループット**: RTF < 0.5 を維持しつつ 3,000+ acoustic tokens/s

### 強力な多言語サポート

S2 Pro は 80 以上の言語をサポートしており、音素や特定の言語に対する前処理なしで高品質な合成を実現します：

- **第1層 (Tier 1)**: 日本語 (ja), 英語 (en), 中国語 (zh)
- **第2層 (Tier 2)**: 韓国語 (ko), スペイン語 (es), ポルトガル語 (pt), アラビア語 (ar), ロシア語 (ru), フランス語 (fr), ドイツ語 (de)
- **グローバルカバレッジ**: sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, e!t, hi, la, ur, th, vi, jw, bn, yo, xsl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo など。

### ネイティブなマルチスピーカー生成

<img src="./assets/chattemplate.png" width=200%>

Fish Audio S2 では、複数のスピーカーを含む参照オーディオをアップロードでき、モデルは `<|speaker:i|>` トークンを介して各スピーカーの特徴を処理します。スピーカー ID トークンを使用してモデルの出力を制御することで、1回の生成に複数のスピーカーを混在させることが可能です。個別のスピーカーごとに参照オーディオをアップロードし直す手間はもう不要です。

### マルチターン対話生成

コンテキストの拡張により、以前のターンの情報を利用して後続の生成内容の表現力を高めることができ、対話としての自然さが大幅に向上しました。

### 高速音声クローニング

Fish Audio S2 は、短い参照サンプル（通常 10〜30 秒）を使用した正確な音声クローニングをサポートしています。モデルは音色、話し方、感情を捉え、追加の微調整なしでリアルで一貫したクローン音声を生成します。
SGLang サーバーの利用については、[SGLang-Omni README](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/models/fishaudio_s2_pro/README.md) を参照してください。

---

## 謝辞

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

@misc{liao2026fishaudios2technical,
      title={Fish Audio S2 Technical Report}, 
      author={Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
      year={2026},
      eprint={2603.08823},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2603.08823}, 
}
```
