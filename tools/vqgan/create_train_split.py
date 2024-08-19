import math
from pathlib import Path
from random import Random

import click
from loguru import logger
from pydub import AudioSegment
from tqdm import tqdm

from tools.file import AUDIO_EXTENSIONS, list_files, load_filelist


@click.command()
@click.argument("root", type=click.Path(exists=True, path_type=Path))
@click.option("--val-ratio", type=float, default=None)
@click.option("--val-count", type=int, default=None)
@click.option("--filelist", default=None, type=Path)
@click.option("--min-duration", default=None, type=float)
@click.option("--max-duration", default=None, type=float)
def main(root, val_ratio, val_count, filelist, min_duration, max_duration):
    if filelist:
        files = [i[0] for i in load_filelist(filelist)]
    else:
        files = list_files(root, AUDIO_EXTENSIONS, recursive=True, sort=True)

    if min_duration is None and max_duration is None:
        filtered_files = list(map(str, [file.relative_to(root) for file in files]))
    else:
        filtered_files = []
        for file in tqdm(files):
            try:
                audio = AudioSegment.from_file(str(file))
                duration = len(audio) / 1000.0

                if min_duration is not None and duration < min_duration:
                    logger.info(
                        f"Skipping {file} due to duration {duration:.2f} < {min_duration:.2f}"
                    )
                    continue

                if max_duration is not None and duration > max_duration:
                    logger.info(
                        f"Skipping {file} due to duration {duration:.2f} > {max_duration:.2f}"
                    )
                    continue

                filtered_files.append(str(file.relative_to(root)))
            except Exception as e:
                logger.info(f"Error processing {file}: {e}")

    logger.info(
        f"Found {len(files)} files, remaining {len(filtered_files)} files after filtering"
    )

    Random(42).shuffle(filtered_files)

    if val_count is None and val_ratio is None:
        logger.info("Validation ratio and count not specified, using min(20%, 100)")
        val_size = min(100, math.ceil(len(filtered_files) * 0.2))
    elif val_count is not None and val_ratio is not None:
        logger.error("Cannot specify both val_count and val_ratio")
        return
    elif val_count is not None:
        if val_count < 1 or val_count > len(filtered_files):
            logger.error("val_count must be between 1 and number of files")
            return
        val_size = val_count
    else:
        val_size = math.ceil(len(filtered_files) * val_ratio)

    logger.info(f"Using {val_size} files for validation")

    with open(root / "vq_train_filelist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_files[val_size:]))

    with open(root / "vq_val_filelist.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_files[:val_size]))

    logger.info("Done")


if __name__ == "__main__":
    main()
