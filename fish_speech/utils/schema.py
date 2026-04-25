import base64
import os
import queue
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, Field, conint, field_validator, model_validator
from pydantic.functional_validators import SkipValidation
from typing_extensions import Annotated

from fish_speech.content_sequence import TextPart, VQPart

# Security: per-audio size cap (25 MB ~ 2.5 min uncompressed 24 kHz mono).
# Prevents OOM via crafted WAV with a 2^31-byte sample-count header
# (DEFEND-20260424T1530-fish-vqgan-encode-audio-memory-bomb).
_MAX_AUDIO_BYTES: int = 25 * 1024 * 1024  # 25 MB

# Security: maximum number of reference-audio items in a single encode request.
_MAX_AUDIO_LIST_ITEMS: int = 16


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
    # The audio here should be in wav, mp3, etc.
    # Security: bounded to _MAX_AUDIO_LIST_ITEMS items, each at most _MAX_AUDIO_BYTES.
    audios: list[bytes]

    @field_validator("audios")
    @classmethod
    def validate_audios(cls, v: list[bytes]) -> list[bytes]:
        if len(v) > _MAX_AUDIO_LIST_ITEMS:
            raise ValueError(
                f"Too many audio items: {len(v)} > {_MAX_AUDIO_LIST_ITEMS}. "
                "Send at most 16 reference audios per request."
            )
        for i, audio in enumerate(v):
            if len(audio) > _MAX_AUDIO_BYTES:
                raise ValueError(
                    f"Audio item {i} exceeds size limit: "
                    f"{len(audio)} bytes > {_MAX_AUDIO_BYTES} bytes (25 MB). "
                    "Reduce the audio length or bitrate."
                )
        return v


class ServeVQGANEncodeResponse(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeRequest(BaseModel):
    tokens: SkipValidation[list[list[list[int]]]]


class ServeVQGANDecodeResponse(BaseModel):
    # The audio here should be in PCM float16 format
    audios: list[bytes]


class ServeReferenceAudio(BaseModel):
    # Security: bounded to _MAX_AUDIO_BYTES. Prevents OOM via crafted audio payloads
    # (DEFEND-20260424T1530-fish-vqgan-encode-audio-memory-bomb).
    audio: bytes
    text: str

    @model_validator(mode="before")
    @classmethod
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

    @field_validator("audio")
    @classmethod
    def validate_audio_size(cls, v: bytes) -> bytes:
        if len(v) > _MAX_AUDIO_BYTES:
            raise ValueError(
                f"Reference audio exceeds size limit: "
                f"{len(v)} bytes > {_MAX_AUDIO_BYTES} bytes (25 MB). "
                "Reduce the audio length or bitrate."
            )
        return v

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


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
