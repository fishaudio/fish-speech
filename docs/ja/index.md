# イントロダクション

<div>
<a target="_blank" href="https://discord.gg/Es5qTB9BcN">
<img alt="Discord" src="https://img.shields.io/discord/1214047546020728892?color=%23738ADB&label=Discord&logo=discord&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=jCKlUP7QgSm9kh95UlBoYv6s1I-Apl1M&authKey=xI5ttVAp3do68IpEYEalwXSYZFdfxZSkah%2BctF5FIMyN2NqAa003vFtLqJyAVRfF&noverify=0&group_code=593946093">
<img alt="QQ" src="https://img.shields.io/badge/QQ Group-%2312B7F5?logo=tencent-qq&logoColor=white&style=flat-square"/>
</a>
<a target="_blank" href="https://hub.docker.com/r/lengyue233/fish-speech">
<img alt="Docker" src="https://img.shields.io/docker/pulls/lengyue233/fish-speech?style=flat-square&logo=docker"/>
</a>
</div>

!!! warning
私たちは、コードベースの違法な使用について一切の責任を負いません。お住まいの地域のDMCA（デジタルミレニアム著作権法）およびその他の関連法については、現地の法律を参照してください。

このコードベースは `BSD-3-Clause` ライセンスの下でリリースされており、すべてのモデルは CC-BY-NC-SA-4.0 ライセンスの下でリリースされています。

<p align="center">
<img src="/docs/assets/figs/diagram.png" width="75%">
</p>

## 要件

- GPUメモリ: 4GB（推論用）、16GB（微調整用）
- システム: Linux、Windows

## Windowsセットアップ

Windowsのプロユーザーは、コードベースを実行するためにWSL2またはDockerを検討することができます。

非プロのWindowsユーザーは、Linux環境なしでコードベースを実行するために以下の方法を検討することができます（モデルコンパイル機能付き、つまり `torch.compile`）：

<ol>
   <li>プロジェクトパッケージを解凍します。</li>
   <li><code>install_env.bat</code>をクリックして環境をインストールします。
      <ul>
            <li><code>install_env.bat</code>の<code>USE_MIRROR</code>項目を編集して、ミラーサイトを使用するかどうかを決定できます。</li>
            <li><code>USE_MIRROR=false</code>は、最新の安定版<code>torch</code>をオリジナルサイトからダウンロードします。<code>USE_MIRROR=true</code>は、最新の<code>torch</code>をミラーサイトからダウンロードします。デフォルトは<code>true</code>です。</li>
            <li><code>install_env.bat</code>の<code>INSTALL_TYPE</code>項目を編集して、コンパイル環境のダウンロードを有効にするかどうかを決定できます。</li>
            <li><code>INSTALL_TYPE=preview</code>は、コンパイル環境付きのプレビュー版をダウンロードします。<code>INSTALL_TYPE=stable</code>は、コンパイル環境なしの安定版をダウンロードします。</li>
      </ul>
   </li>
   <li>ステップ2で<code>USE_MIRROR=preview</code>の場合、このステップを実行します（オプション、コンパイルモデル環境を有効にするため）：
      <ol>
            <li>以下のリンクを使用してLLVMコンパイラをダウンロードします：
               <ul>
                  <li><a href="https://huggingface.co/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6（オリジナルサイトダウンロード）</a></li>
                  <li><a href="https://hf-mirror.com/fishaudio/fish-speech-1/resolve/main/LLVM-17.0.6-win64.exe?download=true">LLVM-17.0.6（ミラーサイトダウンロード）</a></li>
                  <li><code>LLVM-17.0.6-win64.exe</code>をダウンロードした後、ダブルクリックしてインストールし、適切なインストール場所を選択し、最も重要なのは<code>Add Path to Current User</code>をチェックして環境変数に追加することです。</li>
                  <li>インストールが完了したことを確認します。</li>
               </ul>
            </li>
            <li>Microsoft Visual C++ 再頒布可能パッケージをダウンロードしてインストールし、潜在的な.dllの欠落問題を解決します。
               <ul>
                  <li><a href="https://aka.ms/vs/17/release/vc_redist.x64.exe">MSVC++ 14.40.33810.0 ダウンロード</a></li>
               </ul>
            </li>
            <li>Visual Studio Community Editionをダウンロードしてインストールし、MSVC++ビルドツールを取得し、LLVMのヘッダーファイル依存関係を解決します。
               <ul>
                  <li><a href="https://visualstudio.microsoft.com/zh-hans/downloads/">Visual Studio ダウンロード</a></li>
                  <li>Visual Studio Installerをインストールした後、Visual Studio Community 2022をダウンロードします。</li>
                  <li>以下の図のように<code>Modify</code>ボタンをクリックし、<code>Desktop development with C++</code>オプションを見つけてチェックしてダウンロードします。</li>
                  <p align="center">
                     <img src="/docs/assets/figs/VS_1.jpg" width="75%">
                  </p>
               </ul>
            </li>
      </ol>
   </li>
   <li><code>start.bat</code>をダブルクリックして、Fish-Speechトレーニング推論設定WebUIページに入ります。
      <ul>
            <li>（オプション）直接推論ページに行きたい場合は、プロジェクトルートディレクトリの<code>API_FLAGS.txt</code>を編集し、最初の3行を次のように変更します：
               <pre><code>--infer
# --api
# --listen ...
...</code></pre>
            </li>
            <li>（オプション）APIサーバーを起動したい場合は、プロジェクトルートディレクトリの<code>API_FLAGS.txt</code>を編集し、最初の3行を次のように変更します：
               <pre><code># --infer
--api
--listen ...
...</code></pre>
            </li>
      </ul>
   </li>
   <li>（オプション）<code>run_cmd.bat</code>をダブルクリックして、このプロジェクトのconda/pythonコマンドライン環境に入ります。</li>
</ol>

## Linuxセットアップ

```bash
# python 3.10仮想環境を作成します。virtualenvも使用できます。
conda create -n fish-speech python=3.10
conda activate fish-speech

# pytorchをインストールします。
pip3 install torch torchvision torchaudio

# fish-speechをインストールします。
pip3 install -e .

# (Ubuntu / Debianユーザー) soxをインストールします。
apt install libsox-dev
```

## 変更履歴

- 2024/07/02: Fish-Speechを1.2バージョンに更新し、VITSデコーダーを削除し、ゼロショット能力を大幅に強化しました。
- 2024/05/10: Fish-Speechを1.1バージョンに更新し、VITSデコーダーを実装してWERを減少させ、音色の類似性を向上させました。
- 2024/04/22: Fish-Speech 1.0バージョンを完成させ、VQGANおよびLLAMAモデルを大幅に修正しました。
- 2023/12/28: `lora`微調整サポートを追加しました。
- 2023/12/27: `gradient checkpointing`、`causual sampling`、および`flash-attn`サポートを追加しました。
- 2023/12/19: webuiおよびHTTP APIを更新しました。
- 2023/12/18: 微調整ドキュメントおよび関連例を更新しました。
- 2023/12/17: `text2semantic`モデルを更新し、音素フリーモードをサポートしました。
- 2023/12/13: ベータ版をリリースし、VQGANモデルおよびLLAMAに基づく言語モデル（音素のみサポート）を含みます。

## 謝辞

- [VITS2 (daniilrobnikov)](https://github.com/daniilrobnikov/vits2)
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [GPT VITS](https://github.com/innnky/gpt-vits)
- [MQTTS](https://github.com/b04901014/MQTTS)
- [GPT Fast](https://github.com/pytorch-labs/gpt-fast)
- [Transformers](https://github.com/huggingface/transformers)
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
