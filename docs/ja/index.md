# OpenAudio (旧 Fish-Speech)

<div align="center">

<div align="center">

<img src="../assets/openaudio.jpg" alt="OpenAudio" style="display: block; margin: 0 auto; width: 35%;"/>

</div>

<strong>先進的なText-to-Speechモデルシリーズ</strong>

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/fishaudio/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/fishaudio/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>

<strong>今すぐ試す：</strong> <a href="https://fish.audio">Fish Audio Playground</a> | <strong>詳細情報：</strong> <a href="https://openaudio.com">OpenAudio ウェブサイト</a>

</div>

---

!!! note "ライセンスに関するお知らせ"
    このコードベースは **Apache ライセンス** の下でリリースされ、すべてのモデルウェイトは **CC-BY-NC-SA-4.0 ライセンス** の下でリリースされています。詳細については、[コードライセンス](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) と [モデルライセンス](https://spdx.org/licenses/CC-BY-NC-SA-4.0) を参照してください。

!!! warning "法的免責事項"
    コードベースの違法な使用について、当方は一切の責任を負いません。お住まいの地域のDMCAおよびその他の関連法規をご参照ください。

## **紹介**

私たちは **OpenAudio** への改名を発表できることを嬉しく思います。Fish-Speechを基盤とし、大幅な改善と新機能を加えた、新しい先進的なText-to-Speechモデルシリーズを紹介します。

**Openaudio-S1-mini**: [ブログ](https://openaudio.com/blogs/s1); [動画](https://www.youtube.com/watch?v=SYuPvd7m06A); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [動画](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **ハイライト**

### **優秀なTTS品質**

Seed TTS評価指標を使用してモデルのパフォーマンスを評価した結果、OpenAudio S1は英語テキストで**0.008 WER**と**0.004 CER**を達成し、以前のモデルより大幅に改善されました。（英語、自動評価、OpenAI gpt-4o-転写に基づく、話者距離はRevai/pyannote-wespeaker-voxceleb-resnet34-LM使用）

| モデル | 単語誤り率 (WER) | 文字誤り率 (CER) | 話者距離 |
|:-----:|:--------------------:|:-------------------------:|:----------------:|
| **S1** | **0.008** | **0.004** | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **TTS-Arena2最高モデル**

OpenAudio S1は[TTS-Arena2](https://arena.speechcolab.org/)で**#1ランキング**を達成しました。これはtext-to-speech評価のベンチマークです：

<div align="center">
    <img src="../assets/Elo.jpg" alt="TTS-Arena2 Ranking" style="width: 75%;" />
</div>

### **音声制御**
OpenAudio S1は**多様な感情、トーン、特殊マーカーをサポート**して音声合成を強化します：

- **基本感情**：
```
(怒った) (悲しい) (興奮した) (驚いた) (満足した) (喜んだ) 
(怖がった) (心配した) (動揺した) (緊張した) (欲求不満な) (落ち込んだ)
(共感した) (恥ずかしい) (嫌悪した) (感動した) (誇らしい) (リラックスした)
(感謝した) (自信のある) (興味のある) (好奇心のある) (困惑した) (楽しい)
```

- **高度な感情**：
```
(軽蔑的な) (不幸な) (不安な) (ヒステリックな) (無関心な) 
(いらいらした) (罪悪感のある) (軽蔑的な) (パニックした) (激怒した) (不本意な)
(熱心な) (不賛成の) (否定的な) (否定する) (驚いた) (真剣な)
(皮肉な) (和解的な) (慰める) (誠実な) (冷笑的な)
(躊躇する) (譲歩する) (痛々しい) (気まずい) (面白がった)
```

（現在英語、中国語、日本語をサポート、より多くの言語が近日公開予定！）

- **トーンマーカー**：
```
(急いだ調子で) (叫んで) (悲鳴をあげて) (ささやいて) (柔らかい調子で)
```

- **特殊音響効果**：
```
(笑って) (くすくす笑って) (すすり泣いて) (大声で泣いて) (ため息をついて) (息を切らして)
(うめいて) (群衆の笑い声) (背景の笑い声) (観客の笑い声)
```

Ha,ha,haを使用してコントロールすることもでき、他にも多くの使用法があなた自身の探索を待っています。

### **2つのモデルタイプ**

異なるニーズに対応する2つのモデルバリエーションを提供しています：

- **OpenAudio S1 (40億パラメータ)**：[fish.audio](https://fish.audio) で利用可能な全機能搭載のフラッグシップモデルで、すべての高度な機能を備えた最高品質の音声合成を提供します。

- **OpenAudio S1-mini (5億パラメータ)**：コア機能を備えた蒸留版で、[Hugging Face Space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) で利用可能です。優秀な品質を維持しながら、より高速な推論のために最適化されています。

S1とS1-miniの両方にオンライン人間フィードバック強化学習（RLHF）が組み込まれています。

## **機能**

1. **ゼロショット・フューショットTTS：** 10〜30秒の音声サンプルを入力するだけで高品質なTTS出力を生成します。**詳細なガイドラインについては、[音声クローニングのベストプラクティス](https://docs.fish.audio/text-to-speech/voice-clone-best-practices)をご覧ください。**

2. **多言語・言語横断サポート：** 多言語テキストを入力ボックスにコピー＆ペーストするだけで、言語を気にする必要はありません。現在、英語、日本語、韓国語、中国語、フランス語、ドイツ語、アラビア語、スペイン語をサポートしています。

3. **音素依存なし：** このモデルは強力な汎化能力を持ち、TTSに音素に依存しません。あらゆる言語スクリプトのテキストを処理できます。

4. **高精度：** Seed-TTS Evalで低い文字誤り率（CER）約0.4%と単語誤り率（WER）約0.8%を達成します。

5. **高速：** torch compile加速により、Nvidia RTX 4090でリアルタイム係数約1:7。

6. **WebUI推論：** Chrome、Firefox、Edge、その他のブラウザと互換性のあるGradioベースの使いやすいWebUIを備えています。

7. **GUI推論：** APIサーバーとシームレスに連携するPyQt6グラフィカルインターフェースを提供します。Linux、Windows、macOSをサポートします。[GUIを見る](https://github.com/AnyaCoder/fish-speech-gui)。

8. **デプロイフレンドリー：** Linux、Windows（MacOS近日公開）のネイティブサポートで推論サーバーを簡単にセットアップし、速度低下を最小化します。

## **メディア・デモ**

<!-- <div align="center"> -->

<h3><strong>ソーシャルメディア</strong></h3>
<a href="https://x.com/FishAudio/status/1929915992299450398" target="_blank">
    <img src="https://img.shields.io/badge/𝕏-最新デモ-black?style=for-the-badge&logo=x&logoColor=white" alt="Latest Demo on X" />
</a>

<h3><strong>インタラクティブデモ</strong></h3>

<a href="https://fish.audio" target="_blank">
    <img src="https://img.shields.io/badge/Fish_Audio-OpenAudio_S1を試す-blue?style=for-the-badge" alt="Try OpenAudio S1" />
</a>
<a href="https://huggingface.co/spaces/fishaudio/openaudio-s1-mini" target="_blank">
    <img src="https://img.shields.io/badge/Hugging_Face-S1_Miniを試す-yellow?style=for-the-badge" alt="Try S1 Mini" />
</a>

<h3><strong>動画ショーケース</strong></h3>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/SYuPvd7m06A" title="OpenAudio S1 Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## **ドキュメント**

### クイックスタート
- [環境構築](install.md) - 開発環境をセットアップ
- [推論ガイド](inference.md) - モデルを実行して音声を生成

## **コミュニティ・サポート**

- **Discord：** [Discordコミュニティ](https://discord.gg/Es5qTB9BcN)に参加
- **ウェブサイト：** 最新アップデートは[OpenAudio.com](https://openaudio.com)をご覧ください
- **オンライン試用：** [Fish Audio Playground](https://fish.audio)

このコードベースは **Apache ライセンス** の下でリリースされ、すべてのモデルウェイトは **CC-BY-NC-SA-4.0 ライセンス** の下でリリースされています。詳細については、[コードライセンス](https://github.com/fishaudio/fish-speech/blob/main/LICENSE) と [モデルライセンス](https://spdx.org/licenses/CC-BY-NC-SA-4.0) を参照してください。

## モデル

OpenAudio S1 は OpenAudio シリーズの最初のモデルです。これは、VQ コードからオーディオを再構築できるデュアルデコーダ VQ-GAN ボコーダです。
