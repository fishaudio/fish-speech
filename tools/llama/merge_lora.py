import shutil
from copy import deepcopy
from pathlib import Path

import click
import hydra
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger

from fish_speech.models.text2semantic.llama import BaseTransformer
from fish_speech.models.text2semantic.lora import get_merged_state_dict


@click.command()
@click.option("--lora-config", type=str, default="r_8_alpha_16")
@click.option("--base-weight", type=str, default="checkpoints/fish-speech-1.4")
@click.option("--lora-weight", type=str, required=True)
@click.option("--output", type=str, required=True)
def merge(lora_config, base_weight, lora_weight, output):
    output = Path(output)
    logger.info(
        f"Merging {base_weight} and {lora_weight} into {output} with {lora_config}"
    )

    with initialize(version_base="1.3", config_path="../../fish_speech/configs/lora"):
        cfg = compose(config_name=lora_config)

    lora_config = instantiate(cfg)
    logger.info(f"Loaded lora model with config {lora_config}")

    llama_model = BaseTransformer.from_pretrained(
        path=base_weight,
        load_weights=True,
        lora_config=lora_config,
    )
    logger.info(f"Loaded llama model")

    llama_state_dict = llama_model.state_dict()
    llama_state_dict = {k: v for k, v in llama_state_dict.items() if "lora" not in k}
    llama_state_dict_copy = deepcopy(llama_state_dict)
    lora_state_dict = torch.load(lora_weight, map_location="cpu")

    if "state_dict" in llama_state_dict:
        llama_state_dict = llama_state_dict["state_dict"]

    if "state_dict" in lora_state_dict:
        lora_state_dict = lora_state_dict["state_dict"]

    # remove prefix model.
    if any(k.startswith("model.") for k in llama_state_dict.keys()):
        llama_state_dict = {
            k.replace("model.", ""): v
            for k, v in llama_state_dict.items()
            if k.startswith("model.")
        }
    if any(k.startswith("model.") for k in lora_state_dict.keys()):
        lora_state_dict = {
            k.replace("model.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("model.")
        }

    logger.info(f"Found {len(llama_state_dict)} keys in llama model")
    logger.info(f"Found {len(lora_state_dict)} keys in lora model")

    merged_state_dict = llama_state_dict | lora_state_dict
    llama_model.load_state_dict(merged_state_dict, strict=True)
    logger.info(f"Merged model loaded")

    # Trigger eval mode to merge lora
    llama_model.eval()
    llama_model.save_pretrained(output, drop_lora=True)
    logger.info(f"Saved merged model to {output}, validating")

    new_state_dict = torch.load(output / "model.pth", map_location="cpu")
    original_keys = set(llama_state_dict_copy.keys())

    tolerance = 1e-5
    for key in original_keys:
        diff_l1 = (new_state_dict[key] - llama_state_dict_copy[key]).abs().sum().item()
        if diff_l1 > tolerance:
            logger.info(f"Significant difference found in key: {key}")
            break

    if diff_l1 <= tolerance:
        logger.warning(
            "Merged model seems identical to the original model. Further validation might be needed."
        )
    else:
        logger.info("Merged model is different from the original model, check passed")


if __name__ == "__main__":
    merge()
