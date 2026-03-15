import io
import os
import re
import shutil
import tempfile
import time
from http import HTTPStatus
from pathlib import Path

import numpy as np
import ormsgpack
import soundfile as sf
import torch
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
from starlette.responses import Response
from typing_extensions import Annotated

from fish_speech.utils.schema import (
    AddEncodedReferenceResponse,
    AddReferenceRequest,
    AddReferenceResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    UpdateReferenceResponse,
)
from tools.server.api_utils import (
    buffer_to_async_generator,
    format_response,
    get_content_type,
    inference_async,
)
from tools.server.inference import inference_wrapper as inference
from tools.server.model_manager import ModelManager
from tools.server.model_utils import (
    batch_vqgan_decode,
    cached_vqgan_batch_encode,
)

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


def _model_param_memory_gb(module: torch.nn.Module) -> tuple[float, int]:
    """Weights-only memory (GB) and param count for a module."""
    total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    count = sum(p.numel() for p in module.parameters())
    return (round(total_bytes / (1024**3), 3), count)


def _gpu_memory_info(model_manager: ModelManager | None = None):
    """Current GPU memory usage (read-only, safe to call anytime)."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    gb = 1024**3
    out = {
        "cuda_available": True,
        "allocated_gb": round(torch.cuda.memory_allocated() / gb, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / gb, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 3),
        "device_count": torch.cuda.device_count(),
    }
    if model_manager is not None:
        models = {}
        if (
            hasattr(model_manager, "decoder_model")
            and model_manager.decoder_model is not None
        ):
            dac_gb, dac_count = _model_param_memory_gb(model_manager.decoder_model)
            models["dac"] = {"param_gb": dac_gb, "param_count": dac_count}
        if getattr(model_manager, "_worker_memory_info", None):
            wi = model_manager._worker_memory_info
            if "llama_param_gb" in wi:
                models["llama"] = {
                    "param_gb": wi["llama_param_gb"],
                    "param_count": wi.get("llama_param_count"),
                }
        if models:
            out["models"] = models
    if torch.cuda.device_count() > 1:
        out["devices"] = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                out["devices"].append(
                    {
                        "device": i,
                        "name": torch.cuda.get_device_name(i),
                        "allocated_gb": round(torch.cuda.memory_allocated() / gb, 3),
                        "reserved_gb": round(torch.cuda.memory_reserved() / gb, 3),
                        "max_allocated_gb": round(
                            torch.cuda.max_memory_allocated() / gb, 3
                        ),
                    }
                )
    try:
        out["memory_summary"] = torch.cuda.memory_summary(abbreviated=True)
    except Exception:
        pass
    return out


def _gpu_memory_text(info: dict) -> str:
    """Format GPU memory info as plain text (readable table)."""
    if not info.get("cuda_available"):
        return "CUDA not available.\n"
    lines = [
        "GPU memory (current)",
        f"  allocated_gb:    {info['allocated_gb']}",
        f"  reserved_gb:     {info['reserved_gb']}",
        f"  max_allocated_gb: {info['max_allocated_gb']}",
        f"  device_count:    {info['device_count']}",
        "",
    ]
    if "models" in info:
        lines.append("Model weights (params only)")
        for name, m in info["models"].items():
            lines.append(
                f"  {name}: param_gb={m.get('param_gb')} param_count={m.get('param_count', 'N/A')}"
            )
        lines.append("")
    if "devices" in info:
        for d in info["devices"]:
            lines.append(f"  device {d['device']} ({d['name']})")
            lines.append(
                f"    allocated_gb: {d['allocated_gb']}  reserved_gb: {d['reserved_gb']}  max_allocated_gb: {d['max_allocated_gb']}"
            )
        lines.append("")
    if "memory_summary" in info:
        lines.append(info["memory_summary"])
    return "\n".join(lines)


def _dump_memory_snapshot(out_dir: str = "/workspace") -> tuple[str | None, str | None]:
    """Dump current CUDA memory snapshot to a pickle file. Returns (path, None) or (None, error_message)."""
    if not torch.cuda.is_available():
        return None, "CUDA not available"
    dump_fn = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if dump_fn is None:
        return None, "_dump_snapshot not found (PyTorch too old?)"
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(out_dir, f"memory_snapshot_{int(time.time())}.pickle")
        dump_fn(path)
        return path, None
    except Exception as e:
        return (
            None,
            f"Dump failed: {e!r} (need PyTorch built with CUDA memory snapshot support?)",
        )


@routes.http.get("/v1/debug/memory")
async def debug_memory():
    """
    Return current GPU memory usage (allocated, reserved, max).
    If model_manager is available, includes models.dac and models.llama param memory (weights only).
    Query: ?format=text for plain text (table aligned), default JSON.
    Query: ?dump=1 to also write torch.cuda.memory._dump_snapshot() to /workspace (requires FISH_RECORD_MEMORY_HISTORY=1 at startup).
    """
    model_manager = getattr(request.app.state, "model_manager", None)
    info = _gpu_memory_info(model_manager)
    if request.query_params.get("dump", "").strip() in ("1", "true", "True"):
        out_dir = request.query_params.get("dump_dir", "").strip() or "/workspace"
        snapshot_path, snapshot_err = _dump_memory_snapshot(out_dir=out_dir)
        if snapshot_path:
            info["snapshot_path"] = snapshot_path
            info["snapshot_note"] = (
                "Open at https://pytorch.org/memory_viz or: python -m torch.cuda._memory_viz trace_plot <path> -o out.html"
            )
        else:
            info["snapshot_error"] = (
                snapshot_err
                or "Set FISH_RECORD_MEMORY_HISTORY=1 at startup for alloc history in snapshot."
            )
    if request.query_params.get("format", "").strip().lower() == "text":
        return Response(_gpu_memory_text(info), media_type="text/plain; charset=utf-8")
    return JSONResponse(info)


@routes.http.post("/v1/vqgan/encode")
async def vqgan_encode(req: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):
    """
    Encode audio using VQGAN model.
    """
    try:
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
    except Exception as e:
        logger.error(f"Error in VQGAN encode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to encode audio"
        )


@routes.http.post("/v1/vqgan/decode")
async def vqgan_decode(req: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    """
    Decode tokens to audio using VQGAN model.
    """
    try:
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Decode the audio
        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = batch_vqgan_decode(decoder_model, tokens)
        logger.info(
            f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms"
        )
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        # Return the response
        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )
    except Exception as e:
        logger.error(f"Error in VQGAN decode: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to decode tokens to audio"
        )


@routes.http.post("/v1/tts")
async def tts(req: Annotated[ServeTTSRequest, Body(exclusive=True)]):
    """
    Generate speech from text using TTS model.
    """
    try:
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
            chunks = []
            for result in engine.inference(req):
                if result.code == "error" and result.error:
                    raise RuntimeError(str(result.error))
                if result.code in ("segment", "final") and isinstance(
                    result.audio, tuple
                ):
                    chunks.append(result.audio[1])
            if not chunks:
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content="No audio generated, please check the input text.",
                )
            fake_audios = np.concatenate(chunks, axis=0)
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
    except HTTPException:
        # Re-raise HTTP exceptions as they are already properly formatted
        raise
    except Exception as e:
        logger.error(f"Error in TTS generation: {e}", exc_info=True)
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR, content="Failed to generate speech"
        )


@routes.http.post("/v1/references/add")
async def add_reference(
    id: str = Body(...), audio: UploadFile = Body(...), text: str = Body(...)
):
    """
    Add a new reference voice with audio file and text.
    """
    temp_file_path = None

    try:
        # Validate input parameters
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")

        if not text or not text.strip():
            raise ValueError("Reference text cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Read the uploaded audio file
        audio_content = audio.read()
        if not audio_content:
            raise ValueError("Audio file is empty or could not be read")

        # Create a temporary file for the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        # Add the reference using the engine's reference loader
        engine.add_reference(id, temp_file_path, text)

        response = AddReferenceResponse(
            success=True,
            message=f"Reference voice '{id}' added successfully",
            reference_id=id,
        )
        return format_response(response)

    except FileExistsError as e:
        logger.warning(f"Reference ID '{id}' already exists: {e}")
        response = AddReferenceResponse(
            success=False,
            message=f"Reference ID '{id}' already exists",
            reference_id=id,
        )
        return format_response(response, status_code=409)  # Conflict

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{id}': {e}")
        response = AddReferenceResponse(success=False, message=str(e), reference_id=id)
        return format_response(response, status_code=400)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"File system error for reference '{id}': {e}")
        response = AddReferenceResponse(
            success=False, message="File system error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error adding reference '{id}': {e}", exc_info=True)
        response = AddReferenceResponse(
            success=False, message="Internal server error occurred", reference_id=id
        )
        return format_response(response, status_code=500)

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(
                    "Failed to clean up temporary file %s: %s", temp_file_path, e
                )


@routes.http.post("/v1/references/add_encoded")
async def add_reference_encoded(
    id: str = Body(...),
    codes: UploadFile = Body(...),
    lab: UploadFile = Body(...),
    stem: str | None = Body(None),
):
    """
    Add or update a pre-encoded reference (.codes.pt + .lab). Skips write if content hash matches.
    Optional stem: file names under references/<id>/ (default id). Use for multi-stem references.
    """
    try:
        if not id or not id.strip():
            raise ValueError("Reference ID cannot be empty")
        codes_bytes = await codes.read()
        lab_bytes = await lab.read()
        if not codes_bytes:
            raise ValueError("Codes file is empty")
        lab_text = lab_bytes.decode("utf-8", errors="replace").strip()
        if not lab_text:
            raise ValueError("Lab content is empty")
        stem_val = stem.strip() if stem and stem.strip() else None

        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        status = engine.add_reference_encoded(id, codes_bytes, lab_text, stem=stem_val)

        response = AddEncodedReferenceResponse(
            success=True,
            status=status,
            message=f"Reference '{id}' {status}",
            reference_id=id,
        )
        return format_response(response)

    except ValueError as e:
        logger.warning("add_encoded invalid input: %s", e)
        return format_response(
            AddEncodedReferenceResponse(
                success=False, status="error", message=str(e), reference_id=id or ""
            ),
            status_code=400,
        )
    except Exception as e:
        logger.error("add_encoded error: %s", e, exc_info=True)
        return format_response(
            AddEncodedReferenceResponse(
                success=False,
                status="error",
                message="Internal server error",
                reference_id=id,
            ),
            status_code=500,
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

        response = ListReferencesResponse(
            success=True,
            reference_ids=reference_ids,
            message=f"Found {len(reference_ids)} reference voices",
        )
        return format_response(response)

    except Exception as e:
        logger.error(f"Unexpected error listing references: {e}", exc_info=True)
        response = ListReferencesResponse(
            success=False, reference_ids=[], message="Internal server error occurred"
        )
        return format_response(response, status_code=500)


@routes.http.delete("/v1/references/delete")
async def delete_reference(reference_id: str = Body(...)):
    """
    Delete a reference voice by ID.
    """
    try:
        # Validate input parameters
        if not reference_id or not reference_id.strip():
            raise ValueError("Reference ID cannot be empty")

        # Get the model manager to access the reference loader
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        # Delete the reference using the engine's reference loader
        engine.delete_reference(reference_id)

        response = DeleteReferenceResponse(
            success=True,
            message=f"Reference voice '{reference_id}' deleted successfully",
            reference_id=reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(f"Reference ID '{reference_id}' not found: {e}")
        response = DeleteReferenceResponse(
            success=False,
            message=f"Reference ID '{reference_id}' not found",
            reference_id=reference_id,
        )
        return format_response(response, status_code=404)  # Not Found

    except ValueError as e:
        logger.warning(f"Invalid input for reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False, message=str(e), reference_id=reference_id
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error deleting reference '{reference_id}': {e}")
        response = DeleteReferenceResponse(
            success=False,
            message="File system error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(
            f"Unexpected error deleting reference '{reference_id}': {e}", exc_info=True
        )
        response = DeleteReferenceResponse(
            success=False,
            message="Internal server error occurred",
            reference_id=reference_id,
        )
        return format_response(response, status_code=500)


@routes.http.post("/v1/references/update")
async def update_reference(
    old_reference_id: str = Body(...), new_reference_id: str = Body(...)
):
    """
    Rename a reference voice directory from old_reference_id to new_reference_id.
    """
    try:
        # Validate input parameters
        if not old_reference_id or not old_reference_id.strip():
            raise ValueError("Old reference ID cannot be empty")
        if not new_reference_id or not new_reference_id.strip():
            raise ValueError("New reference ID cannot be empty")
        if old_reference_id == new_reference_id:
            raise ValueError("New reference ID must be different from old reference ID")

        # Validate ID format per ReferenceLoader rules
        id_pattern = r"^[a-zA-Z0-9\-_ ]+$"
        if not re.match(id_pattern, new_reference_id) or len(new_reference_id) > 255:
            raise ValueError(
                "New reference ID contains invalid characters or is too long"
            )

        # Access engine to update caches after renaming
        app_state = request.app.state
        model_manager: ModelManager = app_state.model_manager
        engine = model_manager.tts_inference_engine

        refs_base = Path("references")
        old_dir = refs_base / old_reference_id
        new_dir = refs_base / new_reference_id

        # Existence checks
        if not old_dir.exists() or not old_dir.is_dir():
            raise FileNotFoundError(f"Reference ID '{old_reference_id}' not found")
        if new_dir.exists():
            # Conflict: destination already exists
            response = UpdateReferenceResponse(
                success=False,
                message=f"Reference ID '{new_reference_id}' already exists",
                old_reference_id=old_reference_id,
                new_reference_id=new_reference_id,
            )
            return format_response(response, status_code=409)

        # Perform rename
        old_dir.rename(new_dir)

        # Update in-memory cache key if present
        if old_reference_id in engine.ref_by_id:
            engine.ref_by_id[new_reference_id] = engine.ref_by_id.pop(old_reference_id)

        response = UpdateReferenceResponse(
            success=True,
            message=(
                f"Reference voice renamed from '{old_reference_id}' to '{new_reference_id}' successfully"
            ),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response)

    except FileNotFoundError as e:
        logger.warning(str(e))
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=404)

    except ValueError as e:
        logger.warning(f"Invalid input for update reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message=str(e),
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=400)

    except OSError as e:
        logger.error(f"File system error renaming reference: {e}")
        response = UpdateReferenceResponse(
            success=False,
            message="File system error occurred",
            old_reference_id=old_reference_id,
            new_reference_id=new_reference_id,
        )
        return format_response(response, status_code=500)

    except Exception as e:
        logger.error(f"Unexpected error updating reference: {e}", exc_info=True)
        response = UpdateReferenceResponse(
            success=False,
            message="Internal server error occurred",
            old_reference_id=old_reference_id if "old_reference_id" in locals() else "",
            new_reference_id=new_reference_id if "new_reference_id" in locals() else "",
        )
        return format_response(response, status_code=500)
