"""
Regression tests for DEFEND-20260424T1530-fish-vqgan-encode-audio-memory-bomb.

Verifies that crafted oversized audio payloads are rejected at the Pydantic
validation layer before any memory allocation can occur.
"""

import pytest
from pydantic import ValidationError

from fish_speech.utils.schema import (
    ServeReferenceAudio,
    ServeVQGANEncodeRequest,
    _MAX_AUDIO_BYTES,
    _MAX_AUDIO_LIST_ITEMS,
)


# ---------------------------------------------------------------------------
# ServeVQGANEncodeRequest — per-item size
# ---------------------------------------------------------------------------


class TestServeVQGANEncodeRequestAudioSize:
    """Oversized individual audio items must be rejected."""

    def test_single_audio_over_limit_rejected(self):
        """26 MB audio → ValidationError (exceeds 25 MB cap)."""
        oversized = b"\x00" * (26 * 1024 * 1024)
        with pytest.raises(ValidationError, match="exceeds size limit"):
            ServeVQGANEncodeRequest(audios=[oversized])

    def test_single_audio_at_limit_accepted(self):
        """25 MB audio → accepted (exactly at cap)."""
        at_limit = b"\x00" * (25 * 1024 * 1024)
        req = ServeVQGANEncodeRequest(audios=[at_limit])
        assert len(req.audios) == 1
        assert len(req.audios[0]) == _MAX_AUDIO_BYTES

    def test_single_audio_under_limit_accepted(self):
        """Small audio payload → accepted."""
        small = b"\x52\x49\x46\x46" + b"\x00" * 100  # RIFF header + padding
        req = ServeVQGANEncodeRequest(audios=[small])
        assert len(req.audios) == 1


# ---------------------------------------------------------------------------
# ServeVQGANEncodeRequest — list item count
# ---------------------------------------------------------------------------


class TestServeVQGANEncodeRequestListCount:
    """List with too many items must be rejected."""

    def _small_audio(self) -> bytes:
        return b"\x00" * 1024

    def test_seventeen_audios_rejected(self):
        """17 audio items → ValidationError (exceeds cap of 16)."""
        audios = [self._small_audio() for _ in range(17)]
        with pytest.raises(ValidationError, match="Too many audio items"):
            ServeVQGANEncodeRequest(audios=audios)

    def test_sixteen_audios_accepted(self):
        """16 audio items → accepted (exactly at cap)."""
        audios = [self._small_audio() for _ in range(16)]
        req = ServeVQGANEncodeRequest(audios=audios)
        assert len(req.audios) == _MAX_AUDIO_LIST_ITEMS

    def test_one_audio_accepted(self):
        """Single audio item → accepted."""
        req = ServeVQGANEncodeRequest(audios=[self._small_audio()])
        assert len(req.audios) == 1

    def test_empty_list_accepted(self):
        """Empty list → accepted (no audio, no risk)."""
        req = ServeVQGANEncodeRequest(audios=[])
        assert req.audios == []


# ---------------------------------------------------------------------------
# ServeReferenceAudio — per-field size
# ---------------------------------------------------------------------------


class TestServeReferenceAudioSize:
    """ServeReferenceAudio.audio must be bounded."""

    def test_oversized_audio_rejected(self):
        """26 MB bytes reference audio → ValidationError."""
        oversized = b"\x00" * (26 * 1024 * 1024)
        with pytest.raises(ValidationError, match="exceeds size limit"):
            ServeReferenceAudio(audio=oversized, text="hello")

    def test_at_limit_audio_accepted(self):
        """25 MB bytes reference audio → accepted."""
        at_limit = b"\x00" * (25 * 1024 * 1024)
        ref = ServeReferenceAudio(audio=at_limit, text="hello")
        assert len(ref.audio) == _MAX_AUDIO_BYTES

    def test_small_audio_accepted(self):
        """Small bytes reference audio → accepted."""
        small = b"\x00" * 1024
        ref = ServeReferenceAudio(audio=small, text="hello")
        assert len(ref.audio) == 1024

    def test_base64_oversized_audio_rejected(self):
        """Base64-encoded 26 MB audio string → ValidationError after decode."""
        import base64

        oversized = b"\x00" * (26 * 1024 * 1024)
        b64 = base64.b64encode(oversized).decode()
        with pytest.raises(ValidationError, match="exceeds size limit"):
            ServeReferenceAudio(audio=b64, text="hello")

    def test_base64_at_limit_audio_accepted(self):
        """Base64-encoded 25 MB audio string → accepted after decode."""
        import base64

        at_limit = b"\x00" * (25 * 1024 * 1024)
        b64 = base64.b64encode(at_limit).decode()
        ref = ServeReferenceAudio(audio=b64, text="hello")
        assert len(ref.audio) == _MAX_AUDIO_BYTES
