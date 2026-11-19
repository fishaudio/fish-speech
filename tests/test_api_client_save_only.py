import runpy
import sys
import wave
from types import SimpleNamespace

import pytest
import requests


@pytest.mark.parametrize("streaming", [False, True])
def test_no_play_can_save_without_audio_playback_dependencies(
    monkeypatch, tmp_path, streaming
):
    samples = bytes((0, 0, 255, 127))
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        wav.writeframes(samples)
    payload = source.read_bytes()
    output = tmp_path / "out"
    response = SimpleNamespace(
        status_code=200, content=payload, iter_content=lambda **kw: [payload]
    )
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    monkeypatch.setitem(sys.modules, "pyaudio", None)
    monkeypatch.setitem(sys.modules, "pydub", None)
    args = ["api_client.py", "-t", "test", "-o", str(output), "--no-play"]
    if streaming:
        args.append("--streaming")
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_module("tools.api_client", run_name="__main__")
    with wave.open(str(output.with_suffix(".wav")), "rb") as wav:
        assert wav.readframes(2) == samples
