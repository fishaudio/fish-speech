import os
from pathlib import Path


def get_latest_checkpoint(path: Path | str) -> Path | None:
    # Find the latest checkpoint
    ckpt_dir = Path(path)

    if ckpt_dir.exists() is False:
        return None

    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime)
    if len(ckpts) == 0:
        return None

    return ckpts[-1]
