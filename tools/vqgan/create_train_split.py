from pathlib import Path
from random import Random

import click
from tqdm import tqdm

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files


@click.command()
@click.argument("root", type=click.Path(exists=True, path_type=Path))
def main(root):
    files = list_files(root, AUDIO_EXTENSIONS, recursive=True, show_progress=True)
    print(f"Found {len(files)} files")

    files = [str(file) for file in tqdm(files) if file.with_suffix(".npy").exists()]
    print(f"Found {len(files)} files with features")

    Random(42).shuffle(files)

    with open(root / "vq_train_filelist.txt", "w") as f:
        f.write("\n".join(files[:-100]))

    with open(root / "vq_val_filelist.txt", "w") as f:
        f.write("\n".join(files[-100:]))

    print("Done")


if __name__ == "__main__":
    main()
