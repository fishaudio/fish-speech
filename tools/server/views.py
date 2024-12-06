import io
import time
from http import HTTPStatus

import numpy as np
import ormsgpack
import soundfile as sf
import torch
from kui.asgi import HTTPException, HttpView, JSONResponse, StreamResponse
from loguru import logger

from tools.schema import (
    ServeASRRequest,
    ServeASRResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
)
from tools.server.api_utils import (
    buffer_to_async_generator,
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import batch_asr, cached_vqgan_batch_encode, vqgan_decode


class HealthView(HttpView):
    """
    Return the health status of the server.
    """

    async def post(self, request):
        return JSONResponse({"status": "ok"})


class VQGANEncodeView(HttpView):
    """
    Encode the audio into symbolic tokens.
    """

    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeVQGANEncodeRequest(**payload)

        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Encode the audio
        start_time = time.time()
        tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
        logger.info(
            f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms"
        )

        # Return the response
        return ormsgpack.packb(
            ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )


class VQGANDecodeView(HttpView):
    """
    Decode the symbolic tokens into audio.
    """

    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeVQGANDecodeRequest(**payload)

        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Decode the audio
        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = vqgan_decode(decoder_model, tokens)
        logger.info(
            f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms"
        )
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        # Return the response
        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )


class ASRView(HttpView):
    """
    Perform automatic speech recognition on the audio.
    """

    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeASRRequest(**payload)

        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        asr_model = model_manager.asr_model
        lock = request.app.state.lock

        # Perform ASR
        start_time = time.time()
        audios = [np.frombuffer(audio, dtype=np.float16) for audio in req.audios]
        audios = [torch.from_numpy(audio).float() for audio in audios]

        if any(audios.shape[-1] >= 30 * req.sample_rate for audios in audios):
            raise HTTPException(status_code=400, content="Audio length is too long")

        transcriptions = batch_asr(
            asr_model, lock, audios=audios, sr=req.sample_rate, language=req.language
        )
        logger.info(f"[EXEC] ASR time: {(time.time() - start_time) * 1000:.2f}ms")

        # Return the response
        return ormsgpack.packb(
            ServeASRResponse(transcriptions=transcriptions),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )


class TTSView(HttpView):
    """
    Perform text-to-speech on the input text.
    """

    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeTTSRequest(**payload)

        # Get the model from the app
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine
        sample_rate = engine.decoder_model.spec_transform.sample_rate

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
