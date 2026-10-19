import io
from http import HTTPStatus
from typing import Annotated, Any

import numpy as np
import ormsgpack
import soundfile as sf
import torch
import torchaudio
from baize.datastructures import ContentType
from kui.asgi import (
    HTTPException,
    HttpRequest,
    JSONResponse,
    request,
)
from loguru import logger
from pydantic import BaseModel

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference


from tools.server.cli import parse_args


class MsgPackRequest(HttpRequest):
    async def data(
        self,
    ) -> Annotated[
        Any,
        ContentType("application/msgpack"),
        ContentType("application/json"),
        ContentType("multipart/form-data"),
    ]:
        if self.content_type == "application/msgpack":
            return ormsgpack.unpackb(await self.body)

        elif self.content_type == "application/json":
            return await self.json

        elif self.content_type == "multipart/form-data":
            return await self.form

        raise HTTPException(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            headers={
                "Accept": "application/msgpack, application/json, multipart/form-data"
            },
        )


async def inference_async(req: ServeTTSRequest, engine: TTSInferenceEngine):
    for chunk in inference(req, engine):
        print("Got chunk")
        if isinstance(chunk, bytes):
            yield chunk


async def buffer_to_async_generator(buffer):
    yield buffer


def encode_audio(audio: np.ndarray, sample_rate: int, audio_format: str) -> bytes:
    """A mono float waveform as a wav, mp3, opus or headerless 16-bit pcm file."""
    buffer = io.BytesIO()
    if audio_format == "pcm":
        sf.write(buffer, audio, sample_rate, format="RAW", subtype="PCM_16")
    elif audio_format == "opus":
        # Opus takes only 8, 12, 16, 24 or 48 kHz.
        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        waveform = torchaudio.functional.resample(waveform, sample_rate, 48000)
        sf.write(buffer, waveform.numpy(), 48000, format="OGG", subtype="OPUS")
    else:
        sf.write(buffer, audio, sample_rate, format=audio_format)
    return buffer.getvalue()


def get_content_type(audio_format):
    if audio_format == "wav":
        return "audio/wav"
    elif audio_format == "flac":
        return "audio/flac"
    elif audio_format == "mp3":
        return "audio/mpeg"
    elif audio_format == "opus":
        return "audio/ogg"
    else:
        return "application/octet-stream"


def wants_json(req):
    """Helper method to determine if the client wants a JSON response

    Parameters
    ----------
    req : Request
        The request object

    Returns
    -------
    bool
        True if the client wants a JSON response, False otherwise
    """
    q = req.query_params.get("format", "").strip().lower()
    if q in {"json", "application/json", "msgpack", "application/msgpack"}:
        return q in ("json", "application/json")
    accept = req.headers.get("Accept", "").strip().lower()
    return "application/json" in accept and "application/msgpack" not in accept


def format_response(response: BaseModel, status_code=200):
    """
    Helper function to format responses consistently based on client preference.

    Parameters
    ----------
    response : BaseModel
        The response object to format
    status_code : int
        HTTP status code (default: 200)

    Returns
    -------
    Response
        Formatted response in the client's preferred format
    """
    try:
        if wants_json(request):
            return JSONResponse(
                response.model_dump(mode="json"), status_code=status_code
            )

        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            status_code,
            {"Content-Type": "application/msgpack"},
        )
    except Exception as e:
        logger.error(f"Error formatting response: {e}", exc_info=True)
        # Fallback to JSON response if formatting fails
        return JSONResponse(
            {"error": "Response formatting failed", "details": str(e)}, status_code=500
        )
