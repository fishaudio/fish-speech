from http import HTTPStatus

import numpy as np
from kui.asgi import HTTPException

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.utils.schema import ServeTTSRequest

AMPLITUDE = 32768  # Needs an explaination


def inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    count = 0
    for result in engine.inference(req):
        match result.code:
            case "header":
                # inference_async passes on bytes only.
                if isinstance(result.audio, tuple):
                    yield result.audio[1].tobytes()

            case "error":
                # Only input the model rejects is the client's fault; anything else,
                # including other ValueErrors, is a server error.
                raise HTTPException(
                    (
                        HTTPStatus.BAD_REQUEST
                        if isinstance(result.error, InvalidInputError)
                        else HTTPStatus.INTERNAL_SERVER_ERROR
                    ),
                    content=str(result.error),
                )

            case "segment":
                count += 1
                if isinstance(result.audio, tuple):
                    # Clip so that 1.0 does not wrap around to -32768.
                    samples = np.clip(
                        result.audio[1] * AMPLITUDE, -AMPLITUDE, AMPLITUDE - 1
                    )
                    yield samples.astype(np.int16).tobytes()

            case "final":
                count += 1
                if isinstance(result.audio, tuple):
                    yield result.audio[1]
                return None  # Stop the generator

    if count == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )
