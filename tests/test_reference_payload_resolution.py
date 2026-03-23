from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import (
    ServeReferenceCompatibility,
    ServeReferencePayload,
)


def _compatibility(
    codec_checkpoint_sha256: str = "sha256:codec",
) -> ServeReferenceCompatibility:
    return ServeReferenceCompatibility(
        artifact_schema_version=1,
        codec_checkpoint_sha256=codec_checkpoint_sha256,
        decoder_config_name="modded_dac_vq",
        text2semantic_checkpoint_sha256="sha256:llama",
        tokenizer_sha256="sha256:tokenizer",
        num_codebooks=3,
        semantic_begin_id=1024,
        sample_rate_hz=24000,
    )


def _payload(
    *,
    compatibility: ServeReferenceCompatibility,
    reference_fingerprint: str | None = None,
) -> ServeReferencePayload:
    return ServeReferencePayload(
        reference_id="debug-sky",
        reference_text="Hello there.",
        prompt_tokens=[[1, 2], [3, 4], [5, 6]],
        reference_fingerprint=reference_fingerprint,
        compatibility=compatibility,
    )


def _engine() -> TTSInferenceEngine:
    engine = TTSInferenceEngine.__new__(TTSInferenceEngine)
    engine.reference_compatibility = _compatibility()
    return engine


def test_resolve_reference_payload_accepts_valid_payload():
    engine = _engine()
    payload = _payload(compatibility=_compatibility())

    prompt_tokens, prompt_texts, reference_fingerprint = engine.resolve_reference_payload(
        payload
    )

    assert prompt_texts == ["Hello there."]
    assert len(prompt_tokens) == 1
    assert prompt_tokens[0].shape == (3, 2)
    assert reference_fingerprint.startswith("sha256:")


def test_resolve_reference_payload_rejects_compatibility_mismatch():
    engine = _engine()
    payload = _payload(
        compatibility=_compatibility(codec_checkpoint_sha256="sha256:other")
    )

    try:
        engine.resolve_reference_payload(payload)
    except ValueError as exc:
        assert "compatibility mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched compatibility")


def test_resolve_reference_payload_rejects_fingerprint_mismatch():
    engine = _engine()
    payload = _payload(
        compatibility=_compatibility(),
        reference_fingerprint="sha256:not-the-real-fingerprint",
    )

    try:
        engine.resolve_reference_payload(payload)
    except ValueError as exc:
        assert "fingerprint mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched fingerprint")
