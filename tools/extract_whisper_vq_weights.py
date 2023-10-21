from pathlib import Path

import click
import torch
from loguru import logger


@click.command()
@click.argument(
    "input-file",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
)
@click.argument(
    "output-file",
    type=click.Path(exists=False, dir_okay=False, file_okay=True, path_type=Path),
)
def extract(input_file: Path, output_file: Path):
    model = torch.load(input_file, map_location="cpu")["model"]
    state_dict = {k: v for k, v in model.items() if k.startswith("whisper") is False}

    torch.save(state_dict, output_file)
    logger.info(f"Saved {len(state_dict)} keys to {output_file}")


if __name__ == "__main__":
    extract()
