import pytest

from fish_speech.models.text2semantic.fish_omni.chunking import (
    split_drama3_text,
    utf8_len,
)


@pytest.mark.parametrize(
    "text,budget",
    [
        ("<|speaker:15|>" + "声" * 14, 30),
        ("<|speaker:0|> a<|speaker:1|> b", 30),
    ],
)
def test_tagged_chunks_count_their_full_prompt_bytes(text, budget):
    chunks = split_drama3_text(text, max_bytes=budget)
    assert len(chunks) > 1
    assert all(utf8_len(chunk) <= budget for chunk in chunks)
