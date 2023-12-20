import math
import os
from pathlib import Path
from random import Random

import click
from tqdm import tqdm

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files, load_filelist


@click.command()
@click.argument("root", type=click.Path(exists=True, path_type=Path))
@click.option("--val-ratio", type=float, default=0.2)
@click.option("--val-count", type=int, default=None)
@click.option("--filelist", default=None, type=Path)
def main(root, val_ratio, val_count, filelist):
    if filelist:
        files = [i[0] for i in load_filelist(filelist)]
    else:
        files = list_files(root, AUDIO_EXTENSIONS, recursive=True, sort=True)

    print(f"Found {len(files)} files")
    files = [str(file.relative_to(root)) for file in tqdm(files)]

    Random(42).shuffle(files)

    if val_count is not None:
        if val_count < 1 or val_count > len(files):
            raise ValueError("val_count must be between 1 and number of files")
        val_size = val_count
    else:
        val_size = math.ceil(len(files) * val_ratio)

    with open(root / "vq_train_filelist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(files[val_size:]))

    with open(root / "vq_val_filelist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(files[:val_size]))

    print("Done")


if __name__ == "__main__":
    main()
