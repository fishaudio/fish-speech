import io
import os
import tempfile
import time
from http import HTTPStatus

import numpy as np
import ormsgpack
import soundfile as sf
import torch
from fish_speech.utils.schema import (
    AddReferenceRequest,
    AddReferenceResponse,
    ListReferencesResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
)
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Routes,
    StreamResponse,
    UploadFile,
    request,
)
from loguru import logger
from tools.server.api_utils import (
    buffer_to_async_generator,
    get_content_type,
    inference_async,
    wants_json,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)
from typing_extensions import Annotated

MAX_NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", 1))

routes = Routes()


@routes.http("/v1/health")
class Health(HttpView):
    @classmethod
    async def get(cls):
        return JSONResponse({"status": "ok"})

    @classmethod
    async def post(cls):
        return JSONResponse({"status": "ok"})


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    # Get the model from the app
    model_manager: ModelManager = request.app.state.model_manager
    decoder_model = model_manager.decoder_model

    # Encode the audio
    start_time = time.time()
    tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
    logger.info(f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms")

    # Return the response
    return ormsgpack.packb(
        ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    # Get the model from the app
    model_manager: ModelManager = request.app.state.model_manager
    decoder_model = model_manager.decoder_model

    # Decode the audio
    tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
    start_time = time.time()
    audios = batch_vqgan_decode(decoder_model, tokens)
    logger.info(f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms")
    audios = [audio.astype(np.float16).tobytes() for audio in audios]

    # Return the response
    return ormsgpack.packb(
        ServeVQGANDecodeResponse(audios=audios),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    # Get the model from the app
    app_state = request.app.state
    model_manager: ModelManager = app_state.model_manager
    engine = model_manager.tts_inference_engine
    sample_rate = engine.decoder_model.sample_rate

    # Check if the text is too long
    if app_state.max_text_length > 0 and len(req.text) > app_state.max_text_length:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content=f"Text is too long, max length is {app_state.max_text_length}",
        )

    # Check if streaming is enabled
    if req.streaming and req.format != "wav":
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content="Streaming only supports WAV format",
        )

    # Perform TTS
    if req.streaming:
        return StreamResponse(
            iterable=inference_async(req, engine),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )
    else:
        fake_audios = next(inference(req, engine))
        buffer = io.BytesIO()
        sf.write(
            buffer,
            fake_audios,
            sample_rate,
            format=req.format,
        )

        return StreamResponse(
            iterable=buffer_to_async_generator(buffer.getvalue()),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )


@routes.http.post("/v1/references/add")
async def add_reference(req: Annotated[AddReferenceRequest, Body(exclusive=True)]):
    """
    Add a new reference voice with audio file and text.
    """
    try:
        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(req.audio)
            temp_file_path = temp_file.name

        try:
            # Add the reference using the engine's reference loader
            engine.add_reference(req.id, temp_file_path, req.text)

            response = AddReferenceResponse(success=True, message=f"Reference voice '{req.id}' added successfully", reference_id=req.id)

            return (
                ormsgpack.packb(
                    response,
                    option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
                ),
                200,
                {"Content-Type": "application/msgpack"},
            )

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except FileExistsError as e:
        response = AddReferenceResponse(success=False, message=str(e), reference_id=req.id)
        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            400,
            {"Content-Type": "application/msgpack"},
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        response = AddReferenceResponse(success=False, message=str(e), reference_id=req.id)
        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            400,
            {"Content-Type": "application/msgpack"},
        )
    except Exception as e:
        logger.error(f"Unexpected error adding reference: {e}")
        response = AddReferenceResponse(success=False, message=f"Internal server error: {str(e)}", reference_id=req.id)
        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            500,
            {"Content-Type": "application/msgpack"},
        )


@routes.http.get("/v1/references/list")
async def list_references():
    """
    Get a list of all available reference voice IDs.
    """
    try:
        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Get the list of reference IDs
        reference_ids = engine.list_reference_ids()

        response = ListReferencesResponse(success=True, reference_ids=reference_ids, message=f"Found {len(reference_ids)} reference voices")

        if wants_json(request):
            return JSONResponse(response.model_dump(mode="json"))

        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            200,
            {"Content-Type": "application/msgpack"},
        )

    except Exception as e:
        logger.error(f"Unexpected error listing references: {e}")
        response = ListReferencesResponse(success=False, reference_ids=[], message=f"Internal server error: {str(e)}")

        if wants_json(request):
            return JSONResponse(response.model_dump(mode="json"))

        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            500,
            {"Content-Type": "application/msgpack"},
        )
