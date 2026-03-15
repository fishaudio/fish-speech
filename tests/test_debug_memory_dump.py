"""
Tests for /v1/debug/memory dump (CUDA memory snapshot).
Run in env with torch + tools deps: pytest tests/test_debug_memory_dump.py -v
"""

import tempfile
import unittest.mock as mock

import pytest

# Need full app deps (torch, ormsgpack, kui, …) to import views
pytest.importorskip("torch")
pytest.importorskip("ormsgpack")


def test_dump_memory_snapshot_no_cuda():
    """When CUDA is not available, _dump_memory_snapshot returns (None, error message)."""
    from tools.server.views import _dump_memory_snapshot

    with mock.patch("tools.server.views.torch") as mtorch:
        mtorch.cuda.is_available.return_value = False
        path, err = _dump_memory_snapshot(out_dir=tempfile.gettempdir())
    assert path is None
    assert err == "CUDA not available"


def test_dump_memory_snapshot_no_dump_fn():
    """When _dump_snapshot is missing, returns (None, error message)."""
    import torch

    from tools.server.views import _dump_memory_snapshot

    with mock.patch("tools.server.views.torch") as mtorch:
        mtorch.cuda.is_available.return_value = True
        mtorch.cuda.memory._dump_snapshot = None
        path, err = _dump_memory_snapshot(out_dir=tempfile.gettempdir())
    assert path is None
    assert "_dump_snapshot" in (err or "")


def test_dump_memory_snapshot_dump_fails():
    """When _dump_snapshot raises, returns (None, error message)."""
    import torch

    from tools.server.views import _dump_memory_snapshot

    with mock.patch("tools.server.views.torch") as mtorch:
        mtorch.cuda.is_available.return_value = True
        mtorch.cuda.memory._dump_snapshot = mock.Mock(
            side_effect=RuntimeError("_cuda_memorySnapshot not found")
        )
        path, err = _dump_memory_snapshot(out_dir=tempfile.gettempdir())
    assert path is None
    assert err is not None
    assert "Dump failed" in err or "RuntimeError" in err
