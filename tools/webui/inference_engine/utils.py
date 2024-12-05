import io
import html
import wave
import numpy as np
from functools import partial
from typing import Any, Callable
from dataclasses import dataclass
from typing import Optional, Tuple

from fish_speech.i18n import i18n
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
from tools.schema import ServeReferenceAudio, ServeTTSRequest


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


def inference_wrapper(
    text,
    normalize,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    """

    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    req = ServeTTSRequest(
        text=text,
        normalize=normalize,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
    )

    for result in engine.inference(req):
        match result.code:
            case "final":
                return result.audio
            case "error":
                return build_html_error_message(i18n(result.error))
            case _:
                pass

    return None, i18n("No audio generated")


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def wav_chunk_header(sample_rate: int=44100, bit_depth: int=16, channels: int=1) -> np.ndarray:
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