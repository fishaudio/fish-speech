from types import SimpleNamespace

import pytest

from tools import tts_sample


def test_partial_batch_failure_exits_after_saving_later_successes(
    monkeypatch, tmp_path, capsys
):
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"reference")
    monkeypatch.setattr(
        tts_sample,
        "parse_args",
        lambda: SimpleNamespace(
            ref_audio=str(reference),
            ref_text="reference",
            out_dir=tmp_path,
            prefix="sample",
            temperature=None,
            top_p=None,
            max_new_tokens=None,
            text=["first", "second"],
            url="http://localhost/v1/tts",
            format="wav",
            timeout=10,
        ),
    )
    responses = iter(
        [
            SimpleNamespace(status_code=503, text="unavailable"),
            SimpleNamespace(status_code=200, content=b"audio"),
        ]
    )
    monkeypatch.setattr(
        tts_sample.requests, "post", lambda *args, **kw: next(responses)
    )
    with pytest.raises(SystemExit) as error:
        tts_sample.main()
    assert error.value.code == 1
    assert "503" in capsys.readouterr().err
    assert not (tmp_path / "sample_00.wav").exists()
    assert (tmp_path / "sample_01.wav").read_bytes() == b"audio"
