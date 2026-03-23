import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tools.server.model_manager import ModelManager


def test_model_manager_builds_reference_compatibility_snapshot():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        llama_dir = root / "llama"
        llama_dir.mkdir()
        decoder_path = root / "codec.pth"
        decoder_path.write_bytes(b"codec-weights")

        (llama_dir / "config.json").write_text(
            json.dumps({"semantic_start_token_id": 1024}), encoding="utf-8"
        )
        (llama_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

        manager = ModelManager.__new__(ModelManager)
        manager.llama_checkpoint_path = llama_dir
        manager.decoder_checkpoint_path = decoder_path
        manager.decoder_config_name = "modded_dac_vq"
        manager.decoder_model = SimpleNamespace(
            sample_rate=24000,
            quantizer=SimpleNamespace(n_codebooks=2),
        )

        snapshot = manager.get_reference_compatibility_snapshot()

        assert snapshot.artifact_schema_version == 1
        assert snapshot.decoder_config_name == "modded_dac_vq"
        assert snapshot.num_codebooks == 3
        assert snapshot.semantic_begin_id == 1024
        assert snapshot.sample_rate_hz == 24000
        assert snapshot.codec_checkpoint_sha256.startswith("sha256:")
        assert snapshot.text2semantic_checkpoint_sha256.startswith("sha256:")
        assert snapshot.tokenizer_sha256.startswith("sha256:")
