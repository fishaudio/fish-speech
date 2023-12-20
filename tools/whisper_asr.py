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
python tools/whisper_asr.py --audio_dir pre_data_root/SP_1 --save_dir data/SP_1 
to transcribe the first speaker.

Use 
python tools/whisper_asr.py --audio_dir pre_data_root/SP_2 --save_dir data/SP_2 
to transcribe the second speaker.

Note: Be aware of your audio sample rate, which defaults to 44.1kHz.
"""

import argparse
import os
from glob import glob

import numpy as np
import whisper
from pydub import AudioSegment
from tqdm import tqdm


def transcribe_audio(model, filepath, language):
    return model.transcribe(filepath, language=language)


def transcribe_segment(model, filepath):
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)
    return result.text, lang


def main(model_size, audio_dir, save_dir, out_sr, language):
    print("Loading/Downloading OpenAI Whisper model...")
    model = whisper.load_model(model_size)
    os.makedirs(save_dir, exist_ok=True)
    audio_files = []
    for ext in ["*.wav", "*.flac", "*.mp3"]:  # 支持多种文件格式
        audio_files.extend(glob(os.path.join(audio_dir, ext)))
    for file_path in tqdm(audio_files, desc="Processing audio file"):
        file_name, extension = os.path.splitext(os.path.basename(file_path))

        if extension.lower() == ".wav":
            audio = AudioSegment.from_wav(file_path)
        elif extension.lower() == ".mp3":
            audio = AudioSegment.from_mp3(file_path)
        elif extension.lower() == ".flac":
            audio = AudioSegment.from_file(file_path, format="flac")

        transcription = transcribe_audio(model, file_path, language)
        for segment in transcription.get("segments", []):
            id, text, start, end = (
                segment["id"],
                segment["text"],
                segment["start"],
                segment["end"],
            )
            extract = audio[start:end]
            extract.export(
                os.path.join(save_dir, f"{file_name}_{id}{extension}"),
                format=extension.lower().strip("."),
            )
            with open(
                os.path.join(save_dir, f"{file_name}_{id}.lab"), "w", encoding="utf-8"
            ) as f:
                f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Transcription with Whisper")
    parser.add_argument(
        "--model_size", type=str, default="large", help="Size of the Whisper model"
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save processed audio files",
    )
    parser.add_argument(
        "--language", type=str, default="ZH", help="Language of the transcription"
    )
    parser.add_argument("--out_sr", type=int, default=44100, help="Output sample rate")
    args = parser.parse_args()

    main(args.model_size, args.audio_dir, args.save_dir, args.out_sr, args.language)
