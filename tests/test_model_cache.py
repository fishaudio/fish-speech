import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from fish_speech.model_cache import resolve_codec, resolve_model


def test_local_model_never_contacts_hub():
    with tempfile.TemporaryDirectory() as directory:
        with patch("fish_speech.model_cache.snapshot_download") as download:
            assert resolve_model(checkpoint_path=directory) == Path(directory)
            download.assert_not_called()


def test_missing_explicit_local_path_does_not_download():
    with pytest.raises(FileNotFoundError):
        resolve_model(checkpoint_path="/missing/model")


def test_model_and_path_are_exclusive():
    with pytest.raises(ValueError):
        resolve_model(model="org/model", checkpoint_path="local")


def test_hub_revision_and_offline_are_forwarded():
    with patch(
        "fish_speech.model_cache.snapshot_download", return_value="/cache/snapshot"
    ) as download:
        assert resolve_model(model="org/model", revision="abc", offline=True) == Path(
            "/cache/snapshot"
        )
        download.assert_called_once_with(
            repo_id="org/model", revision="abc", local_files_only=True
        )


def test_codec_local_then_shared_cache():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch(
            "fish_speech.model_cache.hf_hub_download", return_value="/cache/codec.pth"
        ) as download:
            assert resolve_codec(root, revision="codec-rev", offline=True) == Path(
                "/cache/codec.pth"
            )
            download.assert_called_once_with(
                repo_id="fishaudio/s2-pro",
                filename="codec.pth",
                revision="codec-rev",
                local_files_only=True,
            )
            (root / "codec.pth").write_bytes(b"codec")
            download.reset_mock()
            assert resolve_codec(root) == root / "codec.pth"
            download.assert_not_called()


def test_api_options_preserve_local_paths_and_hub_controls():
    from tools.server.cli import parse_args

    with patch(
        "sys.argv",
        ["server", "--model", "org/model", "--revision", "commit", "--offline"],
    ):
        args = parse_args()
    assert args.model == "org/model" and args.revision == "commit" and args.offline
    assert args.llama_checkpoint_path is None
    with patch("sys.argv", ["server", "--llama-checkpoint-path", "/local/model"]):
        assert parse_args().llama_checkpoint_path == "/local/model"
    with patch(
        "sys.argv",
        ["server", "--model", "org/model", "--llama-checkpoint-path", "/local/model"],
    ):
        with pytest.raises(SystemExit):
            parse_args()


def test_explicit_missing_codec_does_not_download():
    with patch("fish_speech.model_cache.hf_hub_download") as download:
        with pytest.raises(FileNotFoundError):
            resolve_codec("/model", "/missing/codec.pth")
        download.assert_not_called()
