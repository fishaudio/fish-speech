"""
東京・注文住宅をテーマにした多話者・多ターン対話の音声合成サンプル
Multi-speaker, multi-turn TTS dialogue about custom homes (注文住宅) in Tokyo

使い方 / Usage:
  1. APIサーバーを起動する:
       python tools/api_server.py \
           --llama-checkpoint-path checkpoints/s2-pro \
           --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
           --listen 0.0.0.0:8080

  2. (任意) 話者ごとの参照音声を用意する:
       references/staff/sample.wav + references/staff/sample.lab
       references/customer/sample.wav + references/customer/sample.lab
       references/consultant/sample.wav + references/consultant/sample.lab

  3. このスクリプトを実行:
       python examples/multi_speaker_housing_dialogue.py \
           --output output/housing_dialogue
"""

import argparse
import os
import time

import ormsgpack
import requests

from fish_speech.utils.file import audio_to_bytes, read_ref_text
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

# ---------------------------------------------------------------------------
# 対話スクリプト（約2分 / ~700文字）
#
# 話者マッピング:
#   <|speaker:0|> … 営業スタッフ（鈴木さん）
#   <|speaker:1|> … お客様（田中さん）
# ---------------------------------------------------------------------------
DIALOGUE_SCRIPT = """\
<|speaker:0|>こんにちは、東京ホームプランニングへようこそ。\
本日はどのようなご相談でしょうか？
<|speaker:1|>はじめまして、田中と申します。\
東京で注文住宅を建てたいと思っているのですが、\
何から始めればいいのかわからなくて。
<|speaker:0|>ご来店ありがとうございます、田中様。\
注文住宅は完全に自由設計できる分、選択肢が多くて迷いますよね。\
まず、どのエリアをお考えでしょうか？
<|speaker:1|>世田谷区か杉並区あたりを希望しています。\
子どもの学校のことも考えて、住環境を大切にしたいんです。
<|speaker:0|>世田谷・杉並は人気エリアですね。\
ただ、都内の土地相場は坪単価が高めで、\
世田谷区では平均で一坪200万円前後になることもあります。\
予算はどのくらいお考えですか？
<|speaker:1|>土地込みで7000万円くらいを目安にしています。\
建物にもこだわりたくて、吹き抜けと書斎は絶対に欲しいと思っていて。
<|speaker:0|>7000万円であれば、土地に4500万円、建物に2500万円ほどの\
配分が現実的かと思います。\
吹き抜けと書斎はどちらも設計次第で十分実現できますよ。
<|speaker:1|>それは心強いですね。\
でも、具体的にどう進めればいいのかまだ不安で。
<|speaker:0|>そのお気持ち、よく分かります。\
実は弊社には、注文住宅専門の建築顧問・山田建築士がおります。\
東京の狭小地や変形地での設計実績が豊富で、\
資金計画から法規制の確認まで、トータルでサポートしてくれます。
<|speaker:1|>ぜひ相談してみたいです。どうすれば会えますか？
<|speaker:0|>山田建築士との初回相談は無料です。\
平日は毎日10時から18時まで対応しております。\
田中様のご都合のよい日時をお聞かせいただければ、\
すぐにご予約をお取りできます。\
今週末か来週あたり、いかがでしょうか？
<|speaker:1|>来週の水曜日、午後2時はいかがでしょうか。
<|speaker:0|>ありがとうございます。\
来週水曜日の午後2時、山田建築士との初回相談、\
ただいまご予約を承りました。\
当日は土地の候補エリアや間取りのイメージなど、\
お気軽にお持ちください。\
田中様の理想の住まいが実現できるよう、\
スタッフ一同、精一杯サポートいたします。\
本日はご来店いただき、ありがとうございました。"""

# ---------------------------------------------------------------------------
# 参照音声の設定（ファイルがなければ空リストにフォールバック）
# ---------------------------------------------------------------------------
REFERENCE_CONFIGS = [
    # speaker:0 - 営業スタッフ
    {
        "audio": "references/staff/sample.wav",
        "text": "references/staff/sample.lab",
    },
    # speaker:1 - お客様
    {
        "audio": "references/customer/sample.wav",
        "text": "references/customer/sample.lab",
    },
]


def build_references(configs: list[dict]) -> list[ServeReferenceAudio]:
    """参照音声ファイルが存在する場合のみ読み込む"""
    refs = []
    for cfg in configs:
        audio_path = cfg["audio"]
        text_path = cfg["text"]
        if os.path.exists(audio_path) and os.path.exists(text_path):
            refs.append(
                ServeReferenceAudio(
                    audio=audio_to_bytes(audio_path),
                    text=read_ref_text(text_path),
                )
            )
            print(f"[参照音声] 読み込み完了: {audio_path}")
        else:
            print(f"[参照音声] ファイルなし (スキップ): {audio_path}")
    return refs


def generate_dialogue(
    url: str,
    output_path: str,
    references: list[ServeReferenceAudio],
    format: str = "wav",
    temperature: float = 0.8,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    max_new_tokens: int = 0,
    chunk_length: int = 300,
    api_key: str = "",
):
    """対話音声をストリーミングで生成してファイルに保存する"""
    # streaming=True でチャンクを受信しながら書き込む（タイムアウト回避）
    request = ServeTTSRequest(
        text=DIALOGUE_SCRIPT,
        references=references,
        format="wav",  # ストリーミング時は WAV のみ対応
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        streaming=True,
        use_memory_cache="on",
    )

    headers = {"content-type": "application/msgpack"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    print(f"\n[生成開始] APIサーバー: {url}")
    print(f"[テキスト文字数] {len(DIALOGUE_SCRIPT)} 文字")
    print("[モード] ストリーミング（PCM → WAV変換）")
    start = time.time()

    response = requests.post(
        url,
        params={"format": "msgpack"},
        data=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers=headers,
        stream=True,
        timeout=(10, 600),  # (接続タイムアウト, 読み取りタイムアウト)
    )

    if response.status_code != 200:
        print(f"[エラー] ステータスコード: {response.status_code}")
        print(response.text)
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # PCM チャンクを収集
    pcm_chunks = []
    received = 0
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            pcm_chunks.append(chunk)
            received += len(chunk)
            print(f"\r[受信中] {received // 1024} KB ...", end="", flush=True)

    elapsed = time.time() - start
    print(f"\n[完了] 生成時間: {elapsed:.1f}秒")

    # WAV ヘッダーのサイズフィールドを正しい値に修正して保存
    # ストリーミング時はヘッダーのサイズが 0 のため QuickTime が読めない
    import struct
    data = b"".join(pcm_chunks)
    data_size = len(data) - 44  # 44バイトが標準 WAV ヘッダー
    data = (
        data[:4]
        + struct.pack("<I", 36 + data_size)  # ChunkSize（RIFF チャンク全体）
        + data[8:40]
        + struct.pack("<I", data_size)       # Subchunk2Size（PCM データ長）
        + data[44:]
    )
    out_file = f"{output_path}.wav"
    with open(out_file, "wb") as f:
        f.write(data)
    print(f"[出力] ファイル保存: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="東京・注文住宅テーマの多話者対話音声を生成する",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080/v1/tts", help="APIサーバーのURL"
    )
    parser.add_argument(
        "--output", default="output/housing_dialogue", help="出力ファイルのパス（拡張子なし）"
    )
    parser.add_argument(
        "--format", choices=["wav", "mp3", "flac"], default="wav", help="音声フォーマット"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="サンプリング温度 (0.1–1.0)"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8, help="Top-pサンプリング (0.1–1.0)"
    )
    parser.add_argument(
        "--api_key", type=str, default="", help="APIキー（ローカルサーバーでは不要）"
    )
    args = parser.parse_args()

    references = build_references(REFERENCE_CONFIGS)
    generate_dialogue(
        url=args.url,
        output_path=args.output,
        references=references,
        format=args.format,
        temperature=args.temperature,
        top_p=args.top_p,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
