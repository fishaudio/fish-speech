import io
import wave
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from fish_speech.text.chn_text_norm.text import Text as ChnNormedText


@dataclass
class InferenceResult:
    code: Literal["header", "segment", "error", "final"]
    audio: Optional[Tuple[int, np.ndarray]]
    error: Optional[Exception]


def normalize_text(user_input: str, use_normalization: bool) -> str:
    """Normalize user input text if needed."""
    if use_normalization:
        return ChnNormedText(raw_text=user_input).normalize()
    else:
        return user_input


def wav_chunk_header(
    sample_rate: int = 44100, bit_depth: int = 16, channels: int = 1
) -> bytes:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()

    return wav_header_bytes
