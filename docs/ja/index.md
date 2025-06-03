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

!!! warning "法的通知"
    このコードベースの違法な使用について、当方は一切の責任を負いません。お住まいの地域のDMCA（デジタルミレニアム著作権法）およびその他の関連法規をご参照ください。
    
    **ライセンス：** このコードベースはApache 2.0ライセンスの下でリリースされ、すべてのモデルはCC-BY-NC-SA-4.0ライセンスの下でリリースされています。

## **紹介**

私たちは **OpenAudio** への改名を発表できることを嬉しく思います。Fish-Speechを基盤とし、大幅な改善と新機能を加えた、新しい先進的なText-to-Speechモデルシリーズを紹介します。

**Openaudio-S1-mini**: [動画](アップロード予定); [Hugging Face](https://huggingface.co/fishaudio/openaudio-s1-mini);

**Fish-Speech v1.5**: [動画](https://www.bilibili.com/video/BV1EKiDYBE4o/); [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5);

## **ハイライト** ✨

### **感情制御**
OpenAudio S1は**多様な感情、トーン、特殊マーカーをサポート**して音声合成を強化します：

- **基本感情**：
```
(angry) (sad) (excited) (surprised) (satisfied) (delighted)
(scared) (worried) (upset) (nervous) (frustrated) (depressed)
(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)
(grateful) (confident) (interested) (curious) (confused) (joyful)
```

- **高度な感情**：
```
(disdainful) (unhappy) (anxious) (hysterical) (indifferent) 
(impatient) (guilty) (scornful) (panicked) (furious) (reluctant)
(keen) (disapproving) (negative) (denying) (astonished) (serious)
(sarcastic) (conciliative) (comforting) (sincere) (sneering)
(hesitating) (yielding) (painful) (awkward) (amused)
```

- **トーンマーカー**：
```
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)
```

- **特殊音響効果**：
```
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting)
(groaning) (crowd laughing) (background laughter) (audience laughing)
```

Ha,ha,haを使用してコントロールすることもでき、他にも多くの使用法があなた自身の探索を待っています。

### **優秀なTTS品質**

Seed TTS評価指標を使用してモデルのパフォーマンスを評価した結果、OpenAudio S1は英語テキストで**0.008 WER**と**0.004 CER**を達成し、以前のモデルより大幅に改善されました。（英語、自動評価、OpenAI gpt-4o-転写に基づく、話者距離はRevai/pyannote-wespeaker-voxceleb-resnet34-LM使用）

| モデル | 単語誤り率 (WER) | 文字誤り率 (CER) | 話者距離 |
|-------|----------------------|---------------------------|------------------|
| **S1** | **0.008**  | **0.004**  | **0.332** |
| **S1-mini** | **0.011** | **0.005** | **0.380** |

### **2つのモデルタイプ**

| モデル | サイズ | 利用可能性 | 特徴 |
|-------|------|--------------|----------|
| **S1** | 40億パラメータ | [fish.audio](fish.audio) で利用可能 | 全機能搭載のフラッグシップモデル |
| **S1-mini** | 5億パラメータ | huggingface [hf space](https://huggingface.co/spaces/fishaudio/openaudio-s1-mini) で利用可能 | コア機能を備えた蒸留版 |

S1とS1-miniの両方にオンライン人間フィードバック強化学習（RLHF）が組み込まれています。

## **機能**

1. **ゼロショット・フューショットTTS：** 10〜30秒の音声サンプルを入力するだけで高品質なTTS出力を生成します。**詳細なガイドラインについては、[音声クローニングのベストプラクティス](https://docs.fish.audio/text-to-speech/voice-clone-best-practices)をご覧ください。**

2. **多言語・言語横断サポート：** 多言語テキストを入力ボックスにコピー＆ペーストするだけで、言語を気にする必要はありません。現在、英語、日本語、韓国語、中国語、フランス語、ドイツ語、アラビア語、スペイン語をサポートしています。

3. **音素依存なし：** このモデルは強力な汎化能力を持ち、TTSに音素に依存しません。あらゆる言語スクリプトのテキストを処理できます。

4. **高精度：** Seed-TTS Evalで低い文字誤り率（CER）約0.4%と単語誤り率（WER）約0.8%を達成します。

5. **高速：** fish-tech加速により、Nvidia RTX 4060ラップトップでリアルタイム係数約1:5、Nvidia RTX 4090で約1:15を実現します。

6. **WebUI推論：** Chrome、Firefox、Edge、その他のブラウザと互換性のあるGradioベースの使いやすいWebUIを備えています。

7. **GUI推論：** APIサーバーとシームレスに連携するPyQt6グラフィカルインターフェースを提供します。Linux、Windows、macOSをサポートします。[GUIを見る](https://github.com/AnyaCoder/fish-speech-gui)。

8. **デプロイフレンドリー：** Linux、Windows、MacOSの native サポートで推論サーバーを簡単にセットアップし、速度低下を最小化します。

## **免責事項**

コードベースの違法な使用について、当方は一切の責任を負いません。お住まいの地域のDMCAやその他の関連法律をご参照ください。

## **メディア・デモ**

#### 🚧 近日公開
動画デモとチュートリアルは現在開発中です。

## **ドキュメント**

### クイックスタート
- [環境構築](install.md) - 開発環境をセットアップ
- [推論ガイド](inference.md) - モデルを実行して音声を生成

## **コミュニティ・サポート**

- **Discord：** [Discordコミュニティ](https://discord.gg/Es5qTB9BcN)に参加
- **ウェブサイト：** 最新アップデートは[OpenAudio.com](https://openaudio.com)をご覧ください
- **オンライン試用：** [Fish Audio Playground](https://fish.audio)
