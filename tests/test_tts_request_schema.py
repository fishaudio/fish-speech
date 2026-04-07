from fish_speech.utils.schema import (
    ServeReferenceAudio,
    ServeReferenceCompatibility,
    ServeReferencePayload,
    ServeTTSRequest,
)


def _payload() -> ServeReferencePayload:
    return ServeReferencePayload(
        reference_id="debug-sky",
        reference_text="Hello there.",
        prompt_tokens=[[1, 2], [3, 4], [5, 6]],
        reference_fingerprint="sha256:test",
        compatibility=ServeReferenceCompatibility(
            artifact_schema_version=1,
            codec_checkpoint_sha256="sha256:codec",
            decoder_config_name="modded_dac_vq",
            text2semantic_checkpoint_sha256="sha256:llama",
            tokenizer_sha256="sha256:tokenizer",
            num_codebooks=3,
            semantic_begin_id=1024,
            sample_rate_hz=24000,
        ),
    )


def test_effective_reference_source_prefers_reference_payload():
    request = ServeTTSRequest(
        text="Say hello",
        references=[ServeReferenceAudio(audio=b"audio", text="hello")],
        reference_id="legacy-id",
        reference_payload=_payload(),
    )

    assert request.effective_reference_source() == "reference_payload"
