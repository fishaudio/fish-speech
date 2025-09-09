from argparse import ArgumentParser
from http import HTTPStatus
from typing import Annotated, Any

import ormsgpack
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["tts"], default="tts")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/openaudio-s1-mini",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/openaudio-s1-mini/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key", type=str, default=None)

    return parser.parse_args()


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


def get_content_type(audio_format):
    if audio_format == "wav":
        return "audio/wav"
    elif audio_format == "flac":
        return "audio/flac"
    elif audio_format == "mp3":
        return "audio/mpeg"
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
        return q == "json"
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
