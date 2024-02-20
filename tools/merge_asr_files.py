import os
from pathlib import Path

from pydub import AudioSegment
from tqdm import tqdm

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files


def merge_and_delete_files(save_dir, original_files):
    save_path = Path(save_dir)
    audio_slice_files = list_files(
        path=save_dir, extensions=AUDIO_EXTENSIONS.union([".lab"]), recursive=True
    )
    audio_files = {}
    label_files = {}
    for file_path in tqdm(audio_slice_files, desc="Merging audio files"):
        rel_path = Path(file_path).relative_to(save_path)
        (save_path / rel_path.parent).mkdir(parents=True, exist_ok=True)
        if file_path.suffix == ".wav":
            prefix = rel_path.parent / file_path.stem.rsplit("-", 1)[0]
            if prefix == rel_path.parent / file_path.stem:
                continue
            audio = AudioSegment.from_wav(file_path)
            if prefix in audio_files.keys():
                audio_files[prefix] = audio_files[prefix] + audio
            else:
                audio_files[prefix] = audio

        elif file_path.suffix == ".lab":
            prefix = rel_path.parent / file_path.stem.rsplit("-", 1)[0]
            if prefix == rel_path.parent / file_path.stem:
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                label = f.read()
            if prefix in label_files.keys():
                label_files[prefix] = label_files[prefix] + ", " + label
            else:
                label_files[prefix] = label

    for prefix, audio in audio_files.items():
        output_audio_path = save_path / f"{prefix}.wav"
        audio.export(output_audio_path, format="wav")

    for prefix, label in label_files.items():
        output_label_path = save_path / f"{prefix}.lab"
        with open(output_label_path, "w", encoding="utf-8") as f:
            f.write(label)

    for file_path in original_files:
        os.remove(file_path)


if __name__ == "__main__":
    merge_and_delete_files("/made/by/spicysama/laziman", [__file__])
