import math
from pathlib import Path
from random import Random

import click
from tqdm import tqdm

from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files


@click.command()
@click.argument("root", type=click.Path(exists=True, path_type=Path))
@click.option("val_ratio", type=float, default=0.2)
@click.option("val_count", type=int, default=None)
def main(root, val_ratio, val_count):
    files = list_files(root, AUDIO_EXTENSIONS, recursive=True)
    print(f"Found {len(files)} files")

    files = [str(file.relative_to(root)) for file in tqdm(files)]

    Random(42).shuffle(files)

    # use 3:1 ratio on train vs val
    val_size = math.ceil(len(files) / 4)  # at least one val file

    with open(root / "vq_train_filelist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(files[val_size:]))

    with open(root / "vq_val_filelist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(files[:val_size]))

    print("Done")


if __name__ == "__main__":
    main()
