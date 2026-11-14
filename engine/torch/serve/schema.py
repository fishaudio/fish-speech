"""Request contract for POST /v1/tts, matching the hosted API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, Field

# Validated but inert locally: platform services or inference support missing.
IGNORED_FIELDS = (
    "normalize",
    "language",
    "prosody",
    "sample_rate",
    "mp3_bitrate",
    "opus_bitrate",
    "latency",
    "repetition_penalty",
    "min_chunk_length",
    "condition_on_previous_chunks",
    "early_stop_threshold",
    "features",
    "pronunciation_dictionary",
)

LOCAL_EXTENSIONS = ("streaming", "single_chunk", "top_k", "epsilon_cutoff", "seed")

AudioFormat = Literal["wav", "pcm", "mp3", "opus"]
LatencyMode = Literal["low", "normal", "balanced"]

SUPPORTED_FORMATS: tuple[AudioFormat, ...] = ("wav", "pcm")
MAX_SPEAKER_ID = 15


class ReferenceAudio(BaseModel):
    audio: str = Field(description="Base64-encoded audio bytes when sent as JSON")
    text: str = ""


class ProsodyControl(BaseModel):
    speed: float = Field(default=1, ge=0.5, le=2.0)
    volume: float = 0
    normalize_loudness: bool = True


class PronunciationDictionaryRef(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    version: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")


class PronunciationItem(BaseModel):
    key: str = Field(min_length=1, max_length=256)
    value: str = Field(min_length=1, max_length=1024)
    type: Literal["phoneme", "alias"] = "phoneme"
    case_sensitive: bool = False


class PronunciationDictionaryInline(BaseModel):
    items: list[PronunciationItem] = Field(default_factory=list, max_length=5000)


@dataclass(frozen=True, kw_only=True)
class ResolvedReference:
    """One inline clip with its speaker index."""

    audio: str
    text: str
    speaker: int


class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    references: list[ReferenceAudio] | list[list[ReferenceAudio]] | None = None
    reference_id: str | list[str] | None = None
    temperature: float = Field(default=0.7, ge=0, le=1)
    top_p: float = Field(default=0.7, ge=0, le=1)
    prosody: ProsodyControl | None = None
    chunk_length: int = Field(default=300, ge=100, le=300)
    normalize: bool = True
    language: str | None = Field(default=None, max_length=35, pattern=r"^[A-Za-z0-9_-]*$")
    format: AudioFormat = "wav"  # No local MP3/Opus encoder.
    sample_rate: int | None = None
    mp3_bitrate: Literal[64, 128, 192] = 128
    opus_bitrate: Literal[-1000, 24000, 32000, 48000, 64000] = -1000
    latency: LatencyMode = "normal"
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.2
    min_chunk_length: int = Field(default=50, ge=0, le=100)
    condition_on_previous_chunks: bool = True
    early_stop_threshold: float = Field(default=1, ge=0, le=1)
    features: list[str] = Field(default_factory=list)
    pronunciation_dictionary: (
        list[PronunciationDictionaryRef | PronunciationDictionaryInline] | None
    ) = Field(default=None, max_length=3)

    streaming: bool = False
    single_chunk: bool = False
    top_k: int | None = None
    epsilon_cutoff: float | None = None
    seed: int | None = None

    def resolved_references(self) -> list[ResolvedReference]:
        """Flatten 1D/2D references; the Nth group is speaker N."""
        if not self.references:
            return []
        if isinstance(self.references[0], ReferenceAudio):
            flat = cast(list[ReferenceAudio], self.references)
            return [
                ResolvedReference(audio=clip.audio, text=clip.text, speaker=0)
                for clip in flat
            ]
        nested = cast(list[list[ReferenceAudio]], self.references)
        return [
            ResolvedReference(audio=clip.audio, text=clip.text, speaker=speaker)
            for speaker, group in enumerate(nested)
            for clip in group
        ]

    def reference_ids(self) -> list[str]:
        if self.reference_id is None:
            return []
        if isinstance(self.reference_id, str):
            return [self.reference_id]
        return list(self.reference_id)
