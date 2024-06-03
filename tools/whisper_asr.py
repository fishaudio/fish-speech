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

from pathlib import Path

import click
import librosa
import soundfile as sf
import whisper
from loguru import logger
from merge_asr_files import merge_and_delete_files
from tqdm import tqdm

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files


@click.command()
@click.option("--model-size", default="large", help="Size of the Whisper model")
@click.option("--audio-dir", required=True, help="Directory containing audio files")
@click.option(
    "--save-dir", required=True, help="Directory to save processed audio files"
)
@click.option(
    "--sample-rate",
    default=None,
    type=int,
    help="Output sample rate, default to input sample rate",
)
@click.option("--device", default="cuda", help="Device to use")
@click.option("--language", default="ZH", help="Language of the transcription")
def main(model_size, audio_dir, save_dir, sample_rate, device, language):
    logger.info("Loading / Downloading OpenAI Whisper model...")
    model = whisper.load_model(
        name=model_size,
        device=device,
        download_root=str(Path(".cache/whisper").resolve()),
    )
    logger.info("Model loaded.")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    original_files = []
    audio_files = list_files(
        path=audio_dir, extensions=AUDIO_EXTENSIONS, recursive=True
    )
    for file_path in tqdm(audio_files, desc="Processing audio file"):
        file_stem = file_path.stem
        file_suffix = file_path.suffix

        rel_path = Path(file_path).relative_to(audio_dir)
        (save_path / rel_path.parent).mkdir(parents=True, exist_ok=True)

        if (save_path / rel_path.parent / f"{rel_path.stem}.wav").exists() and (
            save_path / rel_path.parent / f"{rel_path.stem}.lab"
        ).exists():
            continue

        audio, sr = librosa.load(file_path, sr=sample_rate, mono=False)
        transcription = model.transcribe(str(file_path), language=language)

        for segment in transcription.get("segments", []):
            id, text, start, end = (
                segment["id"],
                segment["text"],
                segment["start"],
                segment["end"],
            )

            extract = audio[..., int(start * sr) : int(end * sr)]
            audio_save_path = (
                save_path / rel_path.parent / f"{file_stem}-{id}{file_suffix}"
            )
            sf.write(
                audio_save_path,
                extract,
                samplerate=sr,
            )
            original_files.append(audio_save_path)

            transcript_save_path = save_path / rel_path.parent / f"{file_stem}-{id}.lab"
            with open(
                transcript_save_path,
                "w",
                encoding="utf-8",
            ) as f:
                f.write(text)
            original_files.append(transcript_save_path)

    merge_and_delete_files(save_dir, original_files)


if __name__ == "__main__":
    main()
