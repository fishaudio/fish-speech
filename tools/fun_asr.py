import functools
from pathlib import Path

import click
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tqdm import tqdm


@functools.lru_cache(maxsize=1)
def asr_pipeline():
    return pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )


def transcribe(audio_file):
    _pipeline = asr_pipeline()
    rec_result = _pipeline(audio_in=audio_file)
    return rec_result["text"]


@click.command()
@click.option("--audio_dir", type=str)
def main(audio_dir):
    audio_dir = Path(audio_dir)

    wav_files = list(audio_dir.glob("*.wav"))
    flac_files = list(audio_dir.glob("*.flac"))
    mp3_files = list(audio_dir.glob("*.mp3"))

    all_audio_files = wav_files + flac_files + mp3_files

    for filepath in tqdm(all_audio_files, desc="Processing files"):
        text = transcribe(str(filepath))

        with open(
            (audio_dir / filepath.stem).with_suffix(".lab"), "w", encoding="utf-8"
        ) as f:
            f.write(text)


if __name__ == "__main__":
    main()
