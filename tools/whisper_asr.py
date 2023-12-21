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
from pathlib import Path

import click
import whisper
from pydub import AudioSegment

from fish_speech.utils.file import list_files 

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

def load_audio(file_path, file_suffix):
    try:
        if file_suffix == '.wav':
            audio = AudioSegment.from_wav(file_path)
        elif file_suffix == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif file_suffix == '.flac':
            audio = AudioSegment.from_file(file_path, format='flac')
        return audio
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

@click.command()
@click.option('--model_size', default='large', help='Size of the Whisper model')
@click.option('--audio_dir', required=True, help='Directory containing audio files')
@click.option('--save_dir', required=True, help='Directory to save processed audio files')
@click.option('--language', default='ZH', help='Language of the transcription')
@click.option('--out_sr', default=44100, type=int, help='Output sample rate')
def main(model_size, audio_dir, save_dir, out_sr, language):
    print("Loading/Downloading OpenAI Whisper model...")
    model = whisper.load_model(model_size)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    audio_files = list_files(path=audio_dir, extensions=[".wav", ".mp3", ".flac"], recursive=True)
    for file_path in tqdm(audio_files, desc="Processing audio file"):
        file_stem = file_path.stem
        file_suffix = file_path.suffix
        file_path = str(file_path)
        audio = load_audio(file_path, file_suffix)
        if not audio: continue
        transcription = transcribe_audio(model, file_path, language)
        for segment in transcription.get('segments', []):
            print(segment)
            id, text, start, end = segment['id'], segment['text'], segment['start'], segment['end']
            extract = audio[int(start * 1000):int(end * 1000)].set_frame_rate(out_sr)
            extract.export(save_path / f"{file_stem}_{id}{file_suffix}", format=file_suffix.lower().strip('.'))
            with open(save_path / f"{file_stem}_{id}.lab", "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    main()
