"""
Used to transcribe all audio files in one folder into another folder.
e.g.
Directory structure:
--pre_data_root
----SP_1
------01.wav
------02.wav
------......
----SP_2
------01.wav
------02.wav
------......
Use 
python tools/whisper_asr.py --audio-dir pre_data_root/SP_1 --save-dir data/SP_1 
to transcribe the first speaker.

Use 
python tools/whisper_asr.py --audio-dir pre_data_root/SP_2 --save-dir data/SP_2 
to transcribe the second speaker.

Note: Be aware of your audio sample rate, which defaults to 44.1kHz.
"""

import re
from pathlib import Path

import click
import soundfile as sf
from faster_whisper import WhisperModel
from loguru import logger
from pydub import AudioSegment
from tqdm import tqdm

from tools.file import AUDIO_EXTENSIONS, list_files


@click.command()
@click.option("--model-size", default="large-v3", help="Size of the Whisper model")
@click.option(
    "--compute-type",
    default="float16",
    help="Computation Precision of the Whisper model [float16 / int8_float16 / int8]",
)
@click.option("--audio-dir", required=True, help="Directory containing audio files")
@click.option(
    "--save-dir", required=True, help="Directory to save processed audio files"
)
@click.option(
    "--sample-rate",
    default=44100,
    type=int,
    help="Output sample rate, default to input sample rate",
)
@click.option("--device", default="cuda", help="Device to use [cuda / cpu]")
@click.option("--language", default="auto", help="Language of the transcription")
@click.option("--initial-prompt", default=None, help="Initial prompt for transcribing")
def main(
    model_size,
    compute_type,
    audio_dir,
    save_dir,
    sample_rate,
    device,
    language,
    initial_prompt,
):
    logger.info("Loading / Downloading Faster Whisper model...")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root="faster_whisper",
    )

    logger.info("Model loaded.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    audio_files = list_files(
        path=audio_dir, extensions=AUDIO_EXTENSIONS, recursive=True
    )

    for file_path in tqdm(audio_files, desc="Processing audio file"):
        file_stem = file_path.stem
        file_suffix = file_path.suffix

        rel_path = Path(file_path).relative_to(audio_dir)
        (save_path / rel_path.parent).mkdir(parents=True, exist_ok=True)

        audio = AudioSegment.from_file(file_path)

        segments, info = model.transcribe(
            file_path,
            beam_size=5,
            language=None if language == "auto" else language,
            initial_prompt=initial_prompt,
        )

        print(
            "Detected language '%s' with probability %f"
            % (info.language, info.language_probability)
        )
        print("Total len(ms): ", len(audio))

        whole_text = None
        for segment in segments:
            id, start, end, text = (
                segment.id,
                segment.start,
                segment.end,
                segment.text,
            )
            print("Segment %03d [%.2fs -> %.2fs] %s" % (id, start, end, text))
            if not whole_text:
                whole_text = text
            else:
                whole_text += ", " + text

        whole_text += "."

        audio_save_path = save_path / rel_path.parent / f"{file_stem}{file_suffix}"
        audio.export(audio_save_path, format=file_suffix[1:])
        print(f"Exported {audio_save_path}")

        transcript_save_path = save_path / rel_path.parent / f"{file_stem}.lab"
        with open(
            transcript_save_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(whole_text)


if __name__ == "__main__":
    main()
    exit(0)

    audio = AudioSegment.from_wav(
        r"D:\PythonProject\原神语音中文\胡桃\vo_hutao_draw_appear.wav"
    )

    model_size = "large-v3"

    model = WhisperModel(
        model_size,
        device="cuda",
        compute_type="float16",
        download_root="faster_whisper",
    )

    segments, info = model.transcribe(
        r"D:\PythonProject\原神语音中文\胡桃\vo_hutao_draw_appear.wav",
        beam_size=5,
    )

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )
    print("Total len(ms): ", len(audio))

    for i, segment in enumerate(segments):
        print(
            "Segment %03d [%.2fs -> %.2fs] %s"
            % (i, segment.start, segment.end, segment.text)
        )
        start_ms = int(segment.start * 1000)
        end_ms = int(segment.end * 1000)
        segment_audio = audio[start_ms:end_ms]
        segment_audio.export(f"segment_{i:03d}.wav", format="wav")
        print(f"Exported segment_{i:03d}.wav")

    print("All segments have been exported.")
