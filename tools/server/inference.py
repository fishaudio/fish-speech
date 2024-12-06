import numpy as np
from http import HTTPStatus
from kui.asgi import HTTPException

from tools.schema import ServeTTSRequest
from tools.inference_engine import TTSInferenceEngine


def inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    for result in engine.inference(req):
        match result.code:
            case "header":
                if isinstance(result.audio, tuple):
                    yield result.audio[1]

            case "error":
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(result.error),
                )

            case "segment":
                # Needs an explaination
                if isinstance(result.audio, tuple):
                    yield (result.audio[1] * 32768).astype(np.int16).tobytes()

            case "final":
                if isinstance(result.audio, tuple):
                    yield result.audio[1]
                return None # Stop the generator
    
    raise HTTPException(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        content="No audio generated, please check the input text.",
    )