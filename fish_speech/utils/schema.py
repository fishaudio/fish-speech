import base64
import os
import queue
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import BaseModel, Field, conint, conlist, model_validator
from pydantic.functional_validators import SkipValidation
from typing_extensions import Annotated

from fish_speech.conversation import Message, TextPart, VQPart


class ServeVQPart(BaseModel):
    type: Literal["vq"] = "vq"
    codes: SkipValidation[list[list[int]]]


class ServeTextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ServeAudioPart(BaseModel):
    type: Literal["audio"] = "audio"
    audio: bytes


@dataclass
class ASRPackRequest:
    audio: torch.Tensor
    result_queue: queue.Queue
    language: str


class ServeASRRequest(BaseModel):
    # The audio should be an uncompressed PCM float16 audio
    audios: list[bytes]
    sample_rate: int = 44100
    language: Literal["zh", "en", "ja", "auto"] = "auto"


class ServeASRTranscription(BaseModel):
    text: str
    duration: float
    huge_gap: bool


class ServeASRSegment(BaseModel):
    text: str
    start: float
    end: float


class ServeTimedASRResponse(BaseModel):
    text: str
    segments: list[ServeASRSegment]
    duration: float


class ServeASRResponse(BaseModel):
    transcriptions: list[ServeASRTranscription]


class ServeMessage(BaseModel):
    role: Literal["system", "assistant", "user"]
    parts: list[ServeVQPart | ServeTextPart]

    def to_conversation_message(self):
        new_message = Message(role=self.role, parts=[])
        if self.role == "assistant":
            new_message.modality = "voice"

        for part in self.parts:
            if isinstance(part, ServeTextPart):
                new_message.parts.append(TextPart(text=part.text))
            elif isinstance(part, ServeVQPart):
                new_message.parts.append(
                    VQPart(codes=torch.tensor(part.codes, dtype=torch.int))
                )
            else:
                raise ValueError(f"Unsupported part type: {part}")

        return new_message


class ServeChatRequest(BaseModel):
    messages: Annotated[list[ServeMessage], conlist(ServeMessage, min_length=1)]
    max_new_tokens: int = 1024
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


class ServeForwardMessage(BaseModel):
    role: str
    content: str


class ServeResponse(BaseModel):
    messages: list[ServeMessage]
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] = {}


class ServeStreamDelta(BaseModel):
    role: Literal["system", "assistant", "user"] | None = None
    part: ServeVQPart | ServeTextPart | None = None


class ServeStreamResponse(BaseModel):
    sample_id: int = 0
    delta: ServeStreamDelta | None = None
    finish_reason: Literal["stop", "error"] | None = None
    stats: dict[str, int | float | str] | None = None


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
            except Exception as e:
                # If the audio is not a valid base64 string, we will just ignore it and let the server handle it
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
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
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True
