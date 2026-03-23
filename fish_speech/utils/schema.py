import base64
import os
import queue
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, conint, model_validator
from pydantic.functional_validators import SkipValidation
from typing_extensions import Annotated

from fish_speech.content_sequence import TextPart, VQPart


class ServeVQPart(BaseModel):
    type: Literal["vq"] = "vq"
    codes: SkipValidation[list[list[int]]]


class ServeTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ServeAudioPart(BaseModel):
    type: Literal["audio"] = "audio"
    audio: bytes


class ServeRequest(BaseModel):
    # Raw content sequence dict that we can use with ContentSequence(**content)
    content: dict
    max_new_tokens: int = 600
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    streaming: bool = False
    num_samples: int = 1
    early_stop_threshold: float = 1.0


class ServeVQGANEncodeRequest(BaseModel):
    # The audio here should be in wav, mp3, etc
    audios: list[bytes]


class ServeVQGANEncodeResponse(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeRequest(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeResponse(BaseModel):
    # The audio here should be in PCM float16 format
    audios: list[bytes]


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if (
            isinstance(audio, str) and len(audio) > 255
        ):  # Check if audio is a string (Base64)
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception:
                # If the audio is not a valid base64 string, we will just ignore it and let the server handle it
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeReferenceCompatibility(BaseModel):
    artifact_schema_version: Literal[1] = 1
    codec_checkpoint_sha256: str
    decoder_config_name: str
    text2semantic_checkpoint_sha256: str
    tokenizer_sha256: str
    num_codebooks: Annotated[int, Field(gt=0, strict=True)]
    semantic_begin_id: Annotated[int, Field(ge=0, strict=True)]
    sample_rate_hz: Annotated[int, Field(gt=0, strict=True)]


class ServeReferencePayload(BaseModel):
    reference_id: str | None = None
    reference_text: str
    prompt_tokens: SkipValidation[list[list[int]]]
    reference_fingerprint: str | None = None
    compatibility: ServeReferenceCompatibility

    @model_validator(mode="after")
    def validate_payload(self):
        if not self.reference_text.strip():
            raise ValueError("reference_text cannot be empty")

        if not self.prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")

        if len(self.prompt_tokens) != self.compatibility.num_codebooks:
            raise ValueError(
                "prompt_tokens row count must equal compatibility.num_codebooks"
            )

        row_lengths = {len(row) for row in self.prompt_tokens}
        if 0 in row_lengths:
            raise ValueError("prompt_tokens rows cannot be empty")
        if len(row_lengths) != 1:
            raise ValueError("all prompt_tokens rows must have the same length")

        for row in self.prompt_tokens:
            for token in row:
                if not isinstance(token, int) or token < 0:
                    raise ValueError(
                        "prompt_tokens must contain only non-negative integers"
                    )

        return self


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=1000, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3", "opus"] = "wav"
    # Latency mode (used by api.fish.audio; "normal" or "balanced")
    latency: Literal["normal", "balanced"] = "normal"
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    reference_payload: ServeReferencePayload | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # not usually used below
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8

    def effective_reference_source(
        self,
    ) -> Literal["reference_payload", "references", "reference_id", "none"]:
        if self.reference_payload is not None:
            return "reference_payload"
        if self.references:
            return "references"
        if self.reference_id is not None:
            return "reference_id"
        return "none"

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True


class AddReferenceRequest(BaseModel):
    id: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9\-_ ]+$")
    audio: bytes
    text: str = Field(..., min_length=1)


class AddReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class ListReferencesResponse(BaseModel):
    success: bool
    reference_ids: list[str]
    message: str = "Success"


class DeleteReferenceResponse(BaseModel):
    success: bool
    message: str
    reference_id: str


class UpdateReferenceResponse(BaseModel):
    success: bool
    message: str
    old_reference_id: str
    new_reference_id: str


class ReferenceCompatibilityResponse(BaseModel):
    success: bool
    compatibility: ServeReferenceCompatibility
    message: str = "Success"
