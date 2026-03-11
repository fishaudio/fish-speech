import os

VALID_FISH_MODEL_TYPES = ("s1", "s2")


def get_fish_model_type() -> str:
    """Return validated model type from FISH_MODEL_TYPE env var.

    Defaults to "s2" when unset. Any explicitly provided invalid value raises
    ValueError to fail fast at startup.
    """

    raw_value = os.getenv("FISH_MODEL_TYPE")
    if raw_value is None:
        return "s2"

    model_type = raw_value.strip().lower()
    if model_type not in VALID_FISH_MODEL_TYPES:
        raise ValueError(
            f"FISH_MODEL_TYPE='{model_type}' is not valid. Must be one of: {{'s1', 's2'}}"
        )

    return model_type
