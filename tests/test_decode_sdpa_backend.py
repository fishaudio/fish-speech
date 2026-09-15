from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from fish_speech.models.text2semantic import inference


@pytest.mark.parametrize(
    ("compile", "expected_backends"),
    [
        (False, []),
        (True, [inference.SDPBackend.MATH, inference.SDPBackend.MATH]),
    ],
)
def test_decode_uses_math_sdpa_only_when_compiled(
    monkeypatch, compile, expected_backends
):
    backends = []

    @contextmanager
    def record_sdpa(backend):
        backends.append(backend)
        yield

    monkeypatch.setattr(inference, "sdpa_kernel", record_sdpa)
    monkeypatch.setattr(inference, "tqdm", lambda iterable: iterable)

    model = SimpleNamespace(
        config=SimpleNamespace(num_codebooks=1),
        tokenizer=SimpleNamespace(get_token_id=lambda _: -1),
    )

    inference.decode_n_tokens(
        model=model,
        cur_token=torch.zeros((1, 2, 1), dtype=torch.int),
        input_pos=torch.tensor([0]),
        num_new_tokens=2,
        temperature=torch.tensor(1.0),
        top_p=torch.tensor(0.9),
        top_k=30,
        semantic_logit_bias=torch.empty(0),
        audio_masks=torch.empty(0),
        audio_parts=torch.empty(0),
        decode_one_token=lambda **_: torch.tensor([[1], [2]]),
        compile=compile,
    )

    assert backends == expected_backends
