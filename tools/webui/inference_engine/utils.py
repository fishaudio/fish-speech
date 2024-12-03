from typing import Callable
from functools import partial

from fish_speech.i18n import i18n
from tools.webui.inference_engine import inference
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText

from tools.schema import (
    ServeReferenceAudio,
    ServeTTSRequest,
)


def normalize_text(user_input : str, use_normalization : bool) -> str:
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
    decoder_model,
    llama_queue,
    compile,
    precision,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    """

    references = []
    if reference_audio:
        with open(reference_audio, "rb") as audio_file:
            audio_bytes = audio_file.read()
        references = [
            ServeReferenceAudio(audio=audio_bytes, text=reference_text)
        ]

    req = ServeTTSRequest(
        text = text,
        normalize = normalize,
        reference_id = reference_id if reference_id else None,
        references = references,
        max_new_tokens = max_new_tokens,
        chunk_length = chunk_length,
        top_p = top_p,
        repetition_penalty = repetition_penalty,
        temperature = temperature,
        seed = int(seed) if seed else None,
        use_memory_cache = use_memory_cache,
        decoder_model = decoder_model,
        llama_queue = llama_queue,
        compile = compile,
        precision = precision,
    )

    for result in inference(req):
        if result[2]:  # Error message
            return None, result[2]
        elif result[1]:  # Audio data
            return result[1], None

    return None, i18n("No audio generated")


def get_inference_wrapper(
        llama_queue,
        decoder_model,
        compile,
        precision,
) -> Callable:
    
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        llama_queue = llama_queue,
        decoder_model = decoder_model,
        compile = compile,
        precision = precision,
    )