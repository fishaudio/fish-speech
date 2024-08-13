from contextlib import nullcontext

import torch


def autocast_exclude_mps(
    device_type: str, dtype: torch.dtype
) -> nullcontext | torch.autocast:
    return (
        nullcontext()
        if torch.backends.mps.is_available()
        else torch.autocast(device_type, dtype)
    )
