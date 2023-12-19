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

import os
from pathlib import Path
import librosa
from scipy.io import wavfile
import numpy as np
import whisper
import argparse
from tqdm import tqdm

def load_and_normalize_audio(filepath, target_sr):
    wav, sr = librosa.load(filepath, sr=None, mono=True)
    wav, _ = librosa.effects.trim(wav, top_db=20)
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak / 0.98
    return librosa.resample(wav, orig_sr=sr, target_sr=target_sr), target_sr

def transcribe_audio(model, filepath):
    return model.transcribe(filepath, word_timestamps=True, task="transcribe", beam_size=5, best_of=5)

def save_audio_segments(segments, wav, sr, save_path):
    for i, seg in enumerate(segments):
        start_time, end_time = seg['start'], seg['end']
        wav_seg = wav[int(start_time * sr):int(end_time * sr)]
        wav_seg_name = f"{save_path.stem}_{i}.wav"
        out_fpath = save_path / wav_seg_name
        wavfile.write(out_fpath, rate=sr, data=(wav_seg * np.iinfo(np.int16).max).astype(np.int16))

def transcribe_segment(model, filepath):
    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)
    options = whisper.DecodingOptions(beam_size=5)
    result = whisper.decode(model, mel, options)
    return result.text, lang

def process_output(save_dir, language, out_file):
    with open(out_file, 'w', encoding="utf-8") as wf:
        ch_name = save_dir.stem
        for file in save_dir.glob("*.lab"):
            with open(file, 'r', encoding="utf-8") as perFile:
                line = perFile.readline().strip()
                result = f"{save_dir}/{ch_name}/{file.stem}.wav|{ch_name}|{language}|{line}"
                wf.write(f"{result}\n")

def main(model_size, audio_dir, save_dir, out_sr, language):
    model = whisper.load_model(model_size)
    audio_dir, save_dir = Path(audio_dir), Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    for filepath in tqdm(list(audio_dir.glob("*.wav")), desc="Processing files"):
        wav, sr = load_and_normalize_audio(filepath, out_sr)
        transcription = transcribe_audio(model, filepath)
        save_path = save_dir / filepath.stem
        save_audio_segments(transcription['segments'], wav, sr, save_path)

        for segment_file in tqdm(list(save_path.glob("*.wav")), desc="Transcribing segments"):
            text, _ = transcribe_segment(model, segment_file)
            with open(segment_file.with_suffix(".lab"), 'w', encoding="utf-8") as f:
                f.write(text)

    # process_output(save_dir, language, save_dir / "output.txt") # Dont need summarize to one file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Audio Transcription with Whisper")
    parser.add_argument("--model_size", type=str, default="large", help="Size of the Whisper model")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save processed audio files")
    parser.add_argument("--language", type=str, default="ZH", help="Language of the transcription")
    parser.add_argument("--out_sr", type=int, default=44100, help="Output sample rate")
    args = parser.parse_args()

    main(args.model_size, args.audio_dir, args.save_dir, args.out_sr, args.language)
