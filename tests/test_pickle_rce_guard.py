"""Regression tests for the torch.load(weights_only=True) hardening.

Each Fish-Speech tool/model checkpoint loader explicitly opts into
`weights_only=True` so a malicious pickle payload in a swapped
checkpoint cannot achieve arbitrary code execution as the loader
process. This test builds a `__reduce__`-based pickle bomb and
verifies that `torch.load(...)` rejects it with UnpicklingError —
i.e., the canary file the bomb tries to create never appears.

Source finding: caller-controlled or supply-chain-substituted
checkpoint paths can land at four sites in the upstream repo:
  - tools/llama/merge_lora.py (LoRA weight + post-merge readback)
  - fish_speech/models/dac/modded_dac.py (DAC inference checkpoint)
  - tools/vqgan/extract_vq.py (VQGAN checkpoint)

PyTorch 2.6+ defaults `weights_only=True`, so sites without an
explicit kwarg are safe by default. The fixed sites either had
explicit `weights_only=False` (merge_lora.py:43) or are
defense-in-depth on call sites that historically passed kwargs
inconsistently.
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch


_CANARY_DIR = Path(tempfile.gettempdir())


class _PickleRCEPayload:
    """Pickle payload that creates a canary file when deserialized.

    Pure local-fixture exploit class. No network, no shell, no
    persistence; only writes a single canary file path that the
    test asserts the absence of.
    """

    def __init__(self, canary_path: Path):
        self._canary_path = canary_path

    def __reduce__(self):
        # When unpickled in a non-weights-only context, this calls
        # `os.system` (or equivalent) to write the canary file.
        # Using `subprocess` keeps the call observable + cancellable.
        cmd = (
            f"{sys.executable} -c "
            f"\"open({str(self._canary_path)!r}, 'w').write('vulnerable')\""
        )
        return (subprocess.run, ([cmd], {"shell": True, "check": False}))


def _make_malicious_checkpoint(canary: Path) -> Path:
    payload = _PickleRCEPayload(canary)
    fp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    fp.close()
    # Use plain pickle so the payload survives torch.save's wrapping;
    # torch.load reads pickle either way under the default
    # zipfile-or-pickle dispatcher.
    with open(fp.name, "wb") as f:
        pickle.dump(payload, f)
    return Path(fp.name)


@pytest.fixture
def malicious_checkpoint():
    canary = _CANARY_DIR / "POC_TRIGGERED_test_pickle_rce_guard.txt"
    if canary.exists():
        canary.unlink()
    ckpt = _make_malicious_checkpoint(canary)
    yield ckpt, canary
    if ckpt.exists():
        ckpt.unlink()
    if canary.exists():
        canary.unlink()


def test_weights_only_blocks_pickle_rce(malicious_checkpoint):
    """torch.load(..., weights_only=True) must raise rather than execute."""
    ckpt, canary = malicious_checkpoint
    with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
        torch.load(ckpt, map_location="cpu", weights_only=True)
    assert not canary.exists(), (
        "PoC canary appeared — torch.load did not block pickle deserialization. "
        "weights_only=True is not enforced at this call site."
    )


def test_weights_only_default_blocks_in_torch_2_6_plus(malicious_checkpoint):
    """On torch>=2.6, the default is weights_only=True, so call sites
    that omit the kwarg are also safe. Test the default path."""
    ckpt, canary = malicious_checkpoint
    torch_version = tuple(int(p) for p in torch.__version__.split(".")[:2])
    if torch_version < (2, 6):
        pytest.skip(f"torch {torch.__version__} default is weights_only=False")
    with pytest.raises((pickle.UnpicklingError, RuntimeError, ValueError)):
        torch.load(ckpt, map_location="cpu")  # no kwarg = default
    assert not canary.exists()
