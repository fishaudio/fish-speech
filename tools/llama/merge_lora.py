import click
import hydra
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger

from fish_speech.models.text2semantic.lora_utils import (
    get_merged_state_dict,
    setup_lora,
)


@click.command()
@click.option("--llama-config", type=str, default="dual_ar_2_codebook_medium")
@click.option("--lora-config", type=str, default="r_8_alpha_16")
@click.option(
    "--llama-weight", type=str, default="checkpoints/text2semantic-sft-medium-v1-4k.pth"
)
@click.option("--lora-weight", type=str, required=True)
@click.option("--output", type=str, required=True)
def merge(llama_config, lora_config, llama_weight, lora_weight, output):
    logger.info(
        f"Merging {llama_weight} and {lora_weight} into {output} with configs {llama_config} and {lora_config}"
    )

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../fish_speech/configs/model"):
        # The max_seq_len here doesn't matter.
        cfg = compose(config_name=llama_config, overrides=[f"config.max_seq_len=2048"])

    llama_model = instantiate(cfg)
    logger.info(f"Loaded llama model with config {llama_config}")

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../../fish_speech/configs/lora"):
        cfg = compose(config_name=lora_config)

    lora_config = instantiate(cfg)
    logger.info(f"Loaded lora model with config {lora_config}")

    setup_lora(llama_model, lora_config)
    logger.info(f"Merged model setup complete")

    llama_state_dict = torch.load(llama_weight, map_location="cpu")
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

    state_dict = get_merged_state_dict(llama_model)
    torch.save(state_dict, output)
    logger.info(f"Merged model saved to {output}")


if __name__ == "__main__":
    merge()
