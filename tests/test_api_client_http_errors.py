import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest
import requests


def run_client(monkeypatch, tmp_path, response, *args):
    monkeypatch.setitem(sys.modules, "pyaudio", ModuleType("pyaudio"))
    monkeypatch.setattr(requests, "post", lambda *a, **kw: response)
    monkeypatch.setattr(
        sys, "argv", ["api_client.py", "-t", "test", "-o", str(tmp_path / "out"), *args]
    )
    runpy.run_module("tools.api_client", run_name="__main__")


@pytest.mark.parametrize("body", [b'{"error":"unavailable"}', b"service unavailable"])
def test_http_error_exits_with_diagnostic(monkeypatch, tmp_path, capsys, body):
    response = requests.Response()
    response.status_code = 503
    response._content = body

    with pytest.raises(SystemExit) as error:
        run_client(monkeypatch, tmp_path, response)

    assert error.value.code == 1
    diagnostic = capsys.readouterr().err
    assert "503" in diagnostic
    assert "unavailable" in diagnostic
    assert not (tmp_path / "out.wav").exists()
