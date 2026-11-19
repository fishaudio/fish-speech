import runpy
import sys
from types import ModuleType

import ormsgpack
import pytest
import requests


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], {}),
        (
            ["--top_k", "0", "--epsilon_cutoff", "0.002"],
            {"top_k": 0, "epsilon_cutoff": 0.002},
        ),
    ],
)
def test_drama3_sampling_options_are_optional(monkeypatch, options, expected):
    def post(*args, **kwargs):
        payload = ormsgpack.unpackb(kwargs["data"])
        assert {
            key: payload[key] for key in ("top_k", "epsilon_cutoff") if key in payload
        } == expected
        raise StopIteration

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setitem(sys.modules, "pyaudio", ModuleType("pyaudio"))
    monkeypatch.setattr(sys, "argv", ["api_client.py", "-t", "test", *options])
    with pytest.raises(StopIteration):
        runpy.run_module("tools.api_client", run_name="__main__")
