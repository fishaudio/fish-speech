"""
話者の声質（性別・音色）を探すためのヘルパースクリプト
Helper to find a suitable seed for a desired speaker voice

使い方 / Usage:
  1. APIサーバーを起動しておく
  2. 複数のシードを試して好みの声を探す:

     # 営業スタッフ（男性）の声を探す:
     python examples/find_speaker_voice.py \\
         --url http://127.0.0.1:8081/v1/tts \\
         --speaker 0 \\
         --seeds 0,42,100,200,300,500,1000,2025

     # お客様（女性）の声を探す:
     python examples/find_speaker_voice.py \\
         --url http://127.0.0.1:8081/v1/tts \\
         --speaker 1 \\
         --seeds 0,42,100,200,300,500,1000,2025

  3. 気に入った声のシードを確認して、対話生成時に使う:
     python examples/multi_speaker_housing_dialogue.py \\
         --url http://127.0.0.1:8081/v1/tts \\
         --seed 42 \\
         --output output/housing_dialogue

NOTE: 1つのseedで両話者の声が決まる（話者ごとに独立したseedは設定不可）。
      最良の結果は参照音声（reference audio）を使うこと。
"""

import argparse
import os
import wave

import ormsgpack
import requests

from fish_speech.utils.schema import ServeTTSRequest

# テスト用テキスト（話者ごと）
TEST_TEXTS = {
    0: "<|speaker:0|>[calm professional male voice]こんにちは、東京ホームプランニングへようこそ。本日はどのようなご相談でしょうか？7000万円の予算であれば、十分に実現できると思います。",
    1: "<|speaker:1|>[gentle female voice]はじめまして、田中と申します。東京で注文住宅を建てたいと思っているのですが、何から始めればいいのかわからなくて。",
}


def test_seed(
    url: str, speaker: int, seed: int, output_dir: str, api_key: str = ""
) -> str:
    """指定シードで音声を生成し、WAVファイルを保存する"""
    text = TEST_TEXTS[speaker]
    http_headers = {"content-type": "application/msgpack"}
    if api_key:
        http_headers["authorization"] = f"Bearer {api_key}"

    req = ServeTTSRequest(
        text=text,
        references=[],
        format="wav",
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=1.1,
        streaming=True,
        use_memory_cache="off",
        seed=seed,
    )

    response = requests.post(
        url,
        params={"format": "msgpack"},
        data=ormsgpack.packb(req, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers=http_headers,
        stream=True,
        timeout=(10, 120),
    )

    if response.status_code != 200:
        print(f"  [エラー] seed={seed}: HTTP {response.status_code}")
        return ""

    pcm_chunks = []
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            pcm_chunks.append(chunk)

    if not pcm_chunks:
        print(f"  [エラー] seed={seed}: 音声なし")
        return ""

    import numpy as np

    all_pcm = b"".join(pcm_chunks)
    samples = len(all_pcm) // 2  # int16 = 2 bytes
    out_path = os.path.join(output_dir, f"speaker{speaker}_seed{seed}.wav")
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(all_pcm)

    duration = samples / 44100
    print(f"  seed={seed:6d}  → {out_path}  ({duration:.1f}秒)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="異なるシードで声質をテストする",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080/v1/tts", help="APIサーバーのURL"
    )
    parser.add_argument(
        "--speaker",
        type=int,
        choices=[0, 1],
        default=0,
        help="テストする話者ID (0=営業スタッフ, 1=お客様)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,42,100,200,300,500,1000,2025",
        help="カンマ区切りのシード値リスト",
    )
    parser.add_argument(
        "--output-dir",
        default="output/voice_test",
        help="出力ディレクトリ",
    )
    parser.add_argument("--api_key", type=str, default="", help="APIキー")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    speaker_label = "営業スタッフ（male）" if args.speaker == 0 else "お客様（female）"
    print(f"\n[声質テスト] speaker:{args.speaker} ({speaker_label})")
    print(f"[シード数] {len(seeds)} 個: {seeds}")
    print(f"[出力先] {args.output_dir}/\n")

    for seed in seeds:
        test_seed(args.url, args.speaker, seed, args.output_dir, args.api_key)

    print(
        f"\n完了。{args.output_dir}/ の WAV ファイルを再生して好みの声を確認してください。"
    )
    print(
        "気に入ったシードを --seed オプションで multi_speaker_housing_dialogue.py に渡せます。"
    )


if __name__ == "__main__":
    main()
