import io
import struct
import wave
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np


@dataclass
class InferenceResult:
    code: Literal["header", "segment", "error", "final"]
    audio: Optional[Tuple[int, np.ndarray]]
    error: Optional[Exception]


def wav_chunk_header(
    sample_rate: int = 44100, bit_depth: int = 16, channels: int = 1
) -> bytes:
    """The 44-byte header of a WAV stream whose length is not known yet.

    The RIFF and data sizes are 0xFFFFFFFF, which readers take as "until the end
    of the stream"; with 0, some (libsndfile) read no audio at all.
    """
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    header = buffer.getvalue()
    buffer.close()

    unknown = struct.pack("<I", 0xFFFFFFFF)
    return header[:4] + unknown + header[8:40] + unknown
