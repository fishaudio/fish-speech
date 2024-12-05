import io
import wave
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from fish_speech.text.chn_text_norm.text import Text as ChnNormedText


@dataclass
class InferenceResult:
    code: str
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
) -> np.ndarray:
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()

    # Convert to numpy array
    wav_header = np.frombuffer(wav_header_bytes, dtype=np.uint8)

    return wav_header