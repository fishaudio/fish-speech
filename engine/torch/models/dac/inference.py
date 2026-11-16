"""Loading the DAC codec for the PyTorch backend.

Lifted from ``fish_speech/models/dac/inference.py``. The old Hydra ``compose``
and project-root lookup are replaced by the bundled config loader, so this no
longer depends on the working directory or the old repo layout.
"""

import torch
from hydra.utils import instantiate
from loguru import logger

from engine.torch.configs import load_dac_config


def load_model(config_name, checkpoint_path, device="cuda"):
    """Instantiate the DAC model from a bundled config and load its weights."""
    model = instantiate(load_dac_config(config_name))

    state_dict = torch.load(
        checkpoint_path, map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model: {result}")
    return model
