"""Resolve Hub models through the shared cache without importing the ML stack."""
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_MODEL = 'fishaudio/s2-pro'


def resolve_model(model=None, checkpoint_path=None, revision=None, offline=False) -> Path:
    if model and checkpoint_path:
        raise ValueError('Use either --model or a local checkpoint path, not both')
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f'Local model directory does not exist: {path}')
        return path
    return Path(snapshot_download(
        repo_id=model or DEFAULT_MODEL, revision=revision, local_files_only=offline,
    ))


def resolve_codec(checkpoint_path, codec_path=None, repo_id=DEFAULT_MODEL,
                  revision=None, offline=False) -> Path:
    if codec_path is not None:
        path = Path(codec_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Local codec does not exist: {path}')
        return path
    root = Path(checkpoint_path)
    # Preserve legacy local layouts and custom model bundles.
    for candidate in (root / 'codec.pth', root.parent / 's2-pro' / 'codec.pth'):
        if candidate.is_file():
            return candidate
    return Path(hf_hub_download(
        repo_id=repo_id, filename='codec.pth', revision=revision,
        local_files_only=offline,
    ))
