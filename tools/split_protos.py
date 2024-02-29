from pathlib import Path

import click
from loguru import logger

from fish_speech.datasets.protos.text_data_stream import split_pb_stream


@click.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--chunk-size", type=int, default=1024**3)  # 1GB
def main(input, output, chunk_size):
    chunk_idx = 0
    current_size = 0
    current_file = None

    if output.exists() is False:
        output.mkdir(parents=True)

    with open(input, "rb") as f:
        for chunk in split_pb_stream(f):
            if current_file is None or current_size + len(chunk) > chunk_size:
                if current_file is not None:
                    current_file.close()

                current_file = open(
                    output / f"{input.stem}.{chunk_idx:04d}.protos", "wb"
                )
                chunk_idx += 1
                current_size = 0
                logger.info(f"Writing to {current_file.name}")

            current_file.write(chunk)
            current_size += len(chunk)

    if current_file is not None:
        current_file.close()

    logger.info(f"Split {input} into {chunk_idx} files")


if __name__ == "__main__":
    main()
