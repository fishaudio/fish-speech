"""
Unit tests for text2semantic inference streaming helpers.
No model/checkpoint required.

Run from repo root with project venv:
  uv sync --extra cpu --extra dev && .venv/bin/python -m pytest tests/test_text2semantic_streaming.py -v
Or: uv run --extra dev pytest tests/test_text2semantic_streaming.py -v  (if uv uses project venv)
"""
import unittest.mock as mock

import pytest
import torch

from fish_speech.models.text2semantic.inference import (
    _to_normal_tensor,
    decode_n_tokens,
    split_text_by_bytes,
    split_text_by_speaker,
)


class TestSplitTextByBytes:
    """Tests for split_text_by_bytes (used when no speaker turns, chunk_length controls batching)."""

    def test_empty_or_whitespace_returns_empty(self):
        assert split_text_by_bytes("", 100) == []
        assert split_text_by_bytes("   \n", 100) == []

    def test_short_text_single_chunk(self):
        assert split_text_by_bytes("Hello", 100) == ["Hello"]
        assert split_text_by_bytes("Hello world.", 50) == ["Hello world."]

    def test_respects_max_bytes(self):
        text = "a" * 200
        chunks = split_text_by_bytes(text, 50)
        assert len(chunks) == 4
        assert sum(len(c.encode("utf-8")) for c in chunks) == 200
        assert all(len(c.encode("utf-8")) <= 50 for c in chunks)

    def test_utf8_boundary_no_split_mid_codepoint(self):
        # Cyrillic "Привет" is 12 bytes in UTF-8; split at 6 could split inside a character
        text = "Привет"
        chunks = split_text_by_bytes(text, 6)
        # Should not raise; chunks should be valid (no replacement chars from bad split)
        assert "�" not in "".join(chunks)
        assert "".join(chunks) == text

    def test_ascii_splits_at_boundary(self):
        text = "one two three four"
        chunks = split_text_by_bytes(text, 8)
        assert all(len(c.encode("utf-8")) <= 8 for c in chunks)
        assert "".join(chunks) == text


class TestSplitTextBySpeaker:
    """Smoke tests for split_text_by_speaker (used for turn-based batching)."""

    def test_no_speaker_tags_returns_empty_list(self):
        # No tags -> empty list (caller uses split_text_by_bytes)
        result = split_text_by_speaker("Just some text.")
        assert result == []

    def test_with_speaker_tag(self):
        text = "<|speaker:0|>Hello world."
        result = split_text_by_speaker(text)
        assert len(result) >= 1
        assert any("Hello" in r for r in result)


def _is_inference(t: torch.Tensor) -> bool:
    """True if tensor is an inference tensor (PyTorch 2.5+)."""
    fn = getattr(t, "is_inference", None)
    return fn() if callable(fn) else False


class TestToNormalTensor:
    """Tests for _to_normal_tensor (streaming + torch.compile workaround)."""

    def test_returns_copy_on_same_device(self):
        t = torch.tensor([1.0, 2.0])
        out = _to_normal_tensor(t)
        assert out.device == t.device
        assert out.shape == t.shape
        assert torch.equal(out, t)
        assert out is not t

    def test_cuda_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.tensor([1.0, 2.0], device="cuda")
        out = _to_normal_tensor(t)
        assert out.device.type == "cuda"
        assert torch.equal(out.cpu(), t.cpu())

    def test_clears_inference_flag_when_called_from_inference_mode(self):
        """
        When called from inside inference_mode(), result must not be an inference tensor.
        This is required so AOT-compiled decode_one_token does not get inference tensors
        as inputs (runtime does original_inpt.copy_(updated_inpt) and fails on inference tensors).
        """
        with torch.inference_mode():
            t = torch.tensor([1.0, 2.0])
            out = _to_normal_tensor(t)
        assert out is not None
        assert torch.equal(out, t)
        if hasattr(out, "is_inference") and callable(out.is_inference):
            assert not out.is_inference(), "_to_normal_tensor must return a normal tensor when used under inference_mode"


class TestDecodeNTokens:
    """Tests for decode_n_tokens loop: chunking, EOS, and streaming vs non-streaming."""

    @pytest.fixture
    def mock_model(self):
        """Minimal mock model for decode_n_tokens (config + tokenizer)."""
        model = mock.Mock()
        model.config.num_codebooks = 10
        model.tokenizer.get_token_id = mock.Mock(return_value=99999)  # EOS id
        return model

    def _make_inputs(self, device="cpu", seq_len=1):
        codebook_dim = 11  # 1 + num_codebooks
        cur_token = torch.randint(0, 100, (1, codebook_dim, seq_len), device=device, dtype=torch.long)
        input_pos = torch.tensor([0], device=device, dtype=torch.long)
        temperature = torch.tensor(0.7, device=device)
        top_p = torch.tensor(0.8, device=device)
        semantic_logit_bias = torch.zeros(1, 1, 100, device=device)
        return cur_token, input_pos, temperature, top_p, semantic_logit_bias

    @mock.patch("fish_speech.models.text2semantic.inference.tqdm", lambda x: x)
    def test_streaming_yields_chunks_of_requested_size(self, mock_model):
        """With stream_chunk_size=5 and mock returning 12 tokens then EOS, we get chunks [5, 5, 2]."""
        cur_token, input_pos, temperature, top_p, semantic_logit_bias = self._make_inputs()
        EOS_ID = 99999
        num_codebooks = 10
        chunk_size = 5
        tokens_to_generate = 12  # then EOS

        call_count = 0

        def fake_decode_one_token(*, model, x, **kwargs):
            nonlocal call_count
            call_count += 1
            # Return (1, 11, 1) token; last one has EOS at [0,0,0] to stop loop
            out = torch.randint(1, 100, (1, num_codebooks + 1, 1), dtype=torch.long)
            if call_count >= tokens_to_generate:
                out[0, 0, 0] = EOS_ID
            return out

        with mock.patch("fish_speech.models.text2semantic.inference.logger"):
            chunks = list(
                decode_n_tokens(
                    mock_model,
                    cur_token,
                    input_pos,
                    num_new_tokens=100,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=30,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=None,
                    audio_parts=None,
                    decode_one_token=fake_decode_one_token,
                    stream_chunk_size=chunk_size,
                )
            )

        # decode_n_tokens cat's on dim=1: each token (1, 11, 1) -> chunk (1, 11*K, 1)
        assert len(chunks) == 3, "Expected 3 chunks: 5 + 5 + 2"
        codebook_dim = 11
        assert chunks[0].shape == (1, codebook_dim * chunk_size, 1)
        assert chunks[1].shape == (1, codebook_dim * chunk_size, 1)
        assert chunks[2].shape == (1, codebook_dim * 2, 1)
        assert call_count == tokens_to_generate

    @mock.patch("fish_speech.models.text2semantic.inference.tqdm", lambda x: x)
    def test_non_streaming_yields_single_chunk(self, mock_model):
        """With stream_chunk_size=None we get one chunk with all generated tokens."""
        cur_token, input_pos, temperature, top_p, semantic_logit_bias = self._make_inputs()
        EOS_ID = 99999
        num_codebooks = 10
        total_tokens = 7

        call_count = 0

        def fake_decode_one_token(*, model, x, **kwargs):
            nonlocal call_count
            call_count += 1
            out = torch.randint(1, 100, (1, num_codebooks + 1, 1), dtype=torch.long)
            if call_count >= total_tokens:
                out[0, 0, 0] = EOS_ID
            return out

        with mock.patch("fish_speech.models.text2semantic.inference.logger"):
            chunks = list(
                decode_n_tokens(
                    mock_model,
                    cur_token,
                    input_pos,
                    num_new_tokens=100,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=30,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=None,
                    audio_parts=None,
                    decode_one_token=fake_decode_one_token,
                    stream_chunk_size=None,
                )
            )

        # Non-streaming: one chunk (1, 11*K, 1)
        assert len(chunks) == 1
        assert chunks[0].shape == (1, 11 * total_tokens, 1)
        assert call_count == total_tokens

    @mock.patch("fish_speech.models.text2semantic.inference.tqdm", lambda x: x)
    def test_streaming_eos_before_full_chunk(self, mock_model):
        """EOS after 2 tokens with stream_chunk_size=5 yields one remainder chunk of 2."""
        cur_token, input_pos, temperature, top_p, semantic_logit_bias = self._make_inputs()
        EOS_ID = 99999
        num_codebooks = 10

        call_count = 0

        def fake_decode_one_token(*, model, x, **kwargs):
            nonlocal call_count
            call_count += 1
            out = torch.randint(1, 100, (1, num_codebooks + 1, 1), dtype=torch.long)
            if call_count >= 2:
                out[0, 0, 0] = EOS_ID
            return out

        with mock.patch("fish_speech.models.text2semantic.inference.logger"):
            chunks = list(
                decode_n_tokens(
                    mock_model,
                    cur_token,
                    input_pos,
                    num_new_tokens=100,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=30,
                    semantic_logit_bias=semantic_logit_bias,
                    audio_masks=None,
                    audio_parts=None,
                    decode_one_token=fake_decode_one_token,
                    stream_chunk_size=5,
                )
            )

        assert len(chunks) == 1
        assert chunks[0].shape == (1, 11 * 2, 1)
        assert call_count == 2
