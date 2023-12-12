import click
import torch
from loguru import logger


@click.command()
@click.argument("model_path")
@click.argument("output_path")
def main(model_path, output_path):
    if model_path == output_path:
        logger.error("Model path and output path are the same")
        return

    logger.info(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    torch.save(state_dict, output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
