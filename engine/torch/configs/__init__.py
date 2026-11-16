"""Model configs bundled with the PyTorch backend.

Paths here are resolved from ``__file__`` instead of the working directory, so
the backend can be loaded from anywhere in the app.
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent


def dac_config_path(config_name: str = "modded_dac_vq") -> Path:
    """Absolute path of a bundled DAC config."""
    path = CONFIG_DIR / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown DAC config {config_name!r}: {path}")
    return path


def load_dac_config(config_name: str = "modded_dac_vq"):
    """Load a bundled DAC config as an OmegaConf DictConfig."""
    from omegaconf import OmegaConf

    # Kept from fish_speech/models/dac/inference.py: configs may use ${eval:...}.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    return OmegaConf.load(str(dac_config_path(config_name)))
