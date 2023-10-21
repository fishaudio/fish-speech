import random
from pathlib import Path

import click
from loguru import logger


@click.command()
@click.argument(
    "list-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option("--train-proportion", "-p", type=float, default=0.95)
def main(list_file, train_proportion):
    lines = list_file.read_text().splitlines()
    logger.info(f"Found {len(lines)} lines in {list_file}")

    random.shuffle(lines)

    train_size = int(len(lines) * train_proportion)

    train_file = list_file.with_suffix(f".train{list_file.suffix}")
    train_file.write_text("\n".join(lines[:train_size]))

    test_file = list_file.with_suffix(f".test{list_file.suffix}")
    test_file.write_text("\n".join(lines[train_size:]))

    logger.info(f"Wrote {len(lines[:train_size])} lines to {train_file}")
    logger.info(f"Wrote {len(lines[train_size:])} lines to {test_file}")


if __name__ == "__main__":
    main()
