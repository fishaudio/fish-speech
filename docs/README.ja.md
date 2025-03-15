<div align="center">
<h1>Fish Speech</h1>

[English](../README.md) | [简体中文](README.zh.md) | [Portuguese](README.pt-BR.md) | **日本語** | [한국어](README.ko.md)<br>

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
</div>

このコードリポジトリはApache 2.0ライセンスの下で公開されており、モデルはCC-BY-NC-SA-4.0ライセンスの下で公開されています。詳細については[LICENSE](../LICENSE)をご参照ください。

---

## 機能

1. **ゼロショット & フューショット TTS**：10〜30 秒の音声サンプルを入力して、高品質の TTS 出力を生成します。**詳細は [音声クローンのベストプラクティス](https://docs.fish.audio/text-to-speech/voice-clone-best-practices) を参照してください。**
2. **多言語 & クロスリンガル対応**：多言語テキストを入力ボックスにコピーペーストするだけで、言語を気にする必要はありません。現在、英語、日本語、韓国語、中国語、フランス語、ドイツ語、アラビア語、スペイン語に対応しています。
3. **音素依存なし**：このモデルは強力な汎化能力を持ち、TTS に音素を必要としません。あらゆる言語スクリプトに対応可能です。
4. **高精度**：5 分間の英語テキストに対し、CER（文字誤り率）と WER（単語誤り率）は約 2%の精度を達成します。
5. **高速**：fish-tech アクセラレーションにより、Nvidia RTX 4060 ラップトップではリアルタイムファクターが約 1:5、Nvidia RTX 4090 では約 1:15 です。
6. **WebUI 推論**：使いやすい Gradio ベースの Web ユーザーインターフェースを搭載し、Chrome、Firefox、Edge などのブラウザに対応しています。
7. **GUI 推論**：PyQt6 のグラフィカルインターフェースを提供し、API サーバーとシームレスに連携します。Linux、Windows、macOS に対応しています。[GUI を見る](https://github.com/AnyaCoder/fish-speech-gui)。
8. **デプロイしやすい**：Linux、Windows、macOS にネイティブ対応した推論サーバーを簡単にセットアップでき、速度の低下を最小限に抑えます。

## 免責事項

コードベースの違法な使用については一切責任を負いません。DMCA（デジタルミレニアム著作権法）およびその他の関連法については、地域の法律を参照してください。

## オンラインデモ

[Fish Audio](https://fish.audio)

## ローカル推論のクイックスタート

[inference.ipynb](/inference.ipynb)

## ビデオ

#### V1.5 デモビデオ: [Watch the video on X (Twitter).](https://x.com/FishAudio/status/1864370933496205728)

## ドキュメント

- [英語](https://speech.fish.audio/)
- [中文](https://speech.fish.audio/zh/)
- [日本語](https://speech.fish.audio/ja/)
- [ポルトガル語 (ブラジル)](https://speech.fish.audio/pt/)

## サンプル (2024/10/02 V1.4)

- [英語](https://speech.fish.audio/samples/)
- [中文](https://speech.fish.audio/zh/samples/)
- [日本語](https://speech.fish.audio/ja/samples/)
- [ポルトガル語 (ブラジル)](https://speech.fish.audio/pt/samples/)

## クレジット

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

## スポンサー

<div>
  <a href="https://6block.com/">
    <img src="https://avatars.githubusercontent.com/u/60573493" width="100" height="100" alt="6Block Avatar"/>
  </a>
  <br>
  <a href="https://6block.com/">データ処理スポンサー：6Block</a>
</div>
