"""
Regression test for DEFEND-20260424T1515-fish-max-new-tokens-unbounded.

An unauthenticated caller could POST max_new_tokens=2147483647 to /v1/tts,
causing the server to run the LLaMA generation loop for up to max_seq_len
steps and monopolize GPU compute (DoS). Fix: bound max_new_tokens to [1, 8192]
at the schema layer so the value is rejected before reaching inference.

8192 upper bound rationale: at typical TTS token rates (~40-50 tokens/sec of
audio output), 8192 tokens corresponds to roughly 160-200 seconds (~2.7-3.3
minutes) of audio. This is generous for any legitimate TTS request and prevents
the GPU-monopolization attack vector. A more conservative cap (e.g. 4096) could
be applied in production deployments.
"""

import pytest
from pydantic import ValidationError

from fish_speech.utils.schema import ServeTTSRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**kwargs):
    """Construct a minimal valid ServeTTSRequest with overrides."""
    defaults = dict(text="Hello world.")
    defaults.update(kwargs)
    return ServeTTSRequest(**defaults)


# ---------------------------------------------------------------------------
# DoS-vector test
# ---------------------------------------------------------------------------

def test_max_new_tokens_dos_value_rejected():
    """The INT32_MAX value used in the PoC must be rejected at validation time."""
    with pytest.raises(ValidationError) as exc_info:
        _make_request(max_new_tokens=2_147_483_647)
    errors = exc_info.value.errors()
    assert any(
        "max_new_tokens" in str(e.get("loc", "")) for e in errors
    ), f"Expected error on max_new_tokens, got: {errors}"


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------

def test_max_new_tokens_upper_boundary_accepted():
    """8192 is the maximum allowed value and must be accepted."""
    req = _make_request(max_new_tokens=8192)
    assert req.max_new_tokens == 8192


def test_max_new_tokens_one_above_upper_boundary_rejected():
    """8193 is one above the cap and must be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        _make_request(max_new_tokens=8193)
    errors = exc_info.value.errors()
    assert any(
        "max_new_tokens" in str(e.get("loc", "")) for e in errors
    ), f"Expected error on max_new_tokens, got: {errors}"


def test_max_new_tokens_zero_rejected():
    """0 is below the minimum (ge=1) and must be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        _make_request(max_new_tokens=0)
    errors = exc_info.value.errors()
    assert any(
        "max_new_tokens" in str(e.get("loc", "")) for e in errors
    ), f"Expected error on max_new_tokens, got: {errors}"


def test_max_new_tokens_negative_rejected():
    """Negative values must be rejected."""
    with pytest.raises(ValidationError):
        _make_request(max_new_tokens=-1)


def test_max_new_tokens_default_is_valid():
    """Default value (1024) must pass validation without being specified."""
    req = _make_request()
    assert req.max_new_tokens == 1024
    assert 1 <= req.max_new_tokens <= 8192


def test_max_new_tokens_lower_boundary_accepted():
    """1 is the minimum allowed value and must be accepted."""
    req = _make_request(max_new_tokens=1)
    assert req.max_new_tokens == 1
