"""Model Context Protocol server for a self-hosted Fish Speech API.

Run the API with ``./start.sh``, then point an MCP client at ``./start-mcp.sh``.
The server is a thin client: it does not import Torch and can run on a laptop
that reaches the Spark over the network.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fishapi import FishAPI, FishAPIError  # noqa: E402

DEFAULT_API_URL = "http://127.0.0.1:8080"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mcp"
FORMAT_SUFFIX = {"wav": ".wav", "pcm": ".pcm", "mp3": ".mp3", "opus": ".opus"}
MIME_TYPE = {
    "wav": "audio/wav",
    "pcm": "audio/L16",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
}


def _api() -> FishAPI:
    return FishAPI(
        os.environ.get("FISH_API_URL", DEFAULT_API_URL),
        api_key=os.environ.get("FISH_API_KEY") or None,
        timeout=float(os.environ.get("FISH_API_TIMEOUT", "600")),
    )


def _output_dir() -> Path:
    directory = Path(
        os.environ.get("FISH_MCP_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
    ).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def _safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())[:48].strip("-._")
    return stem or "speech"


def _output_path(text: str, fmt: str, output_name: str | None) -> Path:
    suffix = FORMAT_SUFFIX[fmt]
    if output_name:
        name = Path(output_name).name
        if name in ("", ".", ".."):
            raise ValueError("output_name must be a file name")
        stem = Path(name).stem
        if Path(name).suffix.lower() != suffix:
            stem = name
        stem = _safe_stem(stem)
    else:
        stem = _safe_stem(text)
    directory = _output_dir()
    path = (
        directory
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}-{uuid.uuid4().hex[:8]}{suffix}"
    )
    if path.exists():  # practically impossible, but never overwrite silently
        path = directory / f"{path.stem}-{uuid.uuid4().hex[:8]}{suffix}"
    return path


def _wav_metadata(audio: bytes) -> dict[str, int | float]:
    try:
        with wave.open(io.BytesIO(audio)) as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
            return {
                "sample_rate": rate,
                "channels": wav.getnchannels(),
                "duration_s": round(frames / rate, 3) if rate else 0.0,
            }
    except (wave.Error, EOFError):
        return {}


mcp = FastMCP(
    "fish-speech",
    instructions=(
        "Text-to-speech and voice management for a Fish Speech server. "
        "Call fish_health first if you are unsure the server is running. "
        "Generated audio is written on the server host; the returned path can be "
        "read by an MCP client running on that host."
    ),
)


@mcp.tool()
def fish_health() -> dict:
    """Check that the Fish Speech API is reachable and healthy."""
    started = time.perf_counter()
    try:
        status = _api().health()
    except FishAPIError as exc:
        raise RuntimeError(f"Fish Speech API is not reachable: {exc}") from exc
    return {
        "ok": True,
        "api_url": _api().base_url,
        "status": status,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


@mcp.tool()
def fish_defaults() -> dict:
    """Return the loaded model family and its sampling defaults."""
    try:
        return _api().tts_defaults()
    except FishAPIError as exc:
        raise RuntimeError(f"Could not read TTS defaults: {exc}") from exc


@mcp.tool()
def fish_voices_list() -> dict:
    """List saved reference voice IDs on the server."""
    try:
        voices = _api().list_voices()
    except FishAPIError as exc:
        raise RuntimeError(f"Could not list voices: {exc}") from exc
    return {"count": len(voices), "voices": voices}


@mcp.tool()
def fish_voice_add(voice_id: str, audio_path: str, text: str) -> dict:
    """Register a reference voice from a WAV file on the server host.

    Args:
        voice_id: Stable ID (letters, digits, spaces, hyphen, underscore).
        audio_path: Path to the reference audio on the server host.
        text: Exact transcript of the reference audio.
    """
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Reference audio does not exist: {path}")
    if not text.strip():
        raise ValueError("Reference transcript must not be empty")
    try:
        result = _api().add_voice(voice_id, path, text)
    except FishAPIError as exc:
        raise RuntimeError(f"Could not add voice: {exc}") from exc
    return {"voice_id": voice_id, "audio_path": str(path.resolve()), **result}


@mcp.tool()
def fish_voice_delete(voice_id: str) -> dict:
    """Delete a saved reference voice by ID."""
    try:
        return _api().delete_voice(voice_id)
    except FishAPIError as exc:
        raise RuntimeError(f"Could not delete voice: {exc}") from exc


@mcp.tool()
def fish_tts(
    text: str,
    voice: str | None = None,
    reference_audio: str | None = None,
    reference_text: str | None = None,
    speaker: int | None = None,
    format: Literal["wav", "pcm", "mp3", "opus"] = "wav",
    seed: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_new_tokens: int | None = None,
    chunk_length: int | None = None,
    single_chunk: bool | None = None,
    normalize: bool | None = None,
    latency: Literal["normal", "balanced"] | None = None,
    use_memory_cache: Literal["on", "off"] | None = None,
    output_name: str | None = None,
    include_audio_base64: bool = False,
) -> dict:
    """Generate speech and save it to a WAV (or PCM/MP3/Opus) file.

    Args:
        text: Text to speak, including Fish inline tags such as ``[whisper]``.
        voice: A voice ID previously registered with fish_voice_add.
        reference_audio: Optional one-shot reference WAV on the server host.
            Use instead of ``voice`` for an ad-hoc clone.
        reference_text: Exact transcript of ``reference_audio``.
        speaker: Speaker number 0-15 for the reference or text (Drama 3).
        format: Audio container. WAV is the safest choice.
        seed: Sampling seed for reproducible output.
        temperature: Sampling temperature, typically 0.5-0.9.
        top_p: Nucleus sampling probability.
        max_new_tokens: Per-chunk cap; 0 uses the model's own cap.
        chunk_length: S2-Pro text chunk size in bytes, 100-1000.
        single_chunk: Drama 3 only: generate the whole text in one pass.
        normalize: Normalize numbers and punctuation.
        latency: Server latency mode.
        use_memory_cache: Reuse in-memory reference audio when "on".
        output_name: Optional output file name; server-side path is still used.
        include_audio_base64: Also return the audio as base64 (large).
    """
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")
    if format not in FORMAT_SUFFIX:
        raise ValueError(f"unsupported format: {format}")
    if voice and reference_audio:
        raise ValueError("voice and reference_audio are mutually exclusive")
    if reference_audio and not reference_text:
        raise ValueError("reference_text is required with reference_audio")
    if speaker is not None and not 0 <= speaker <= 15:
        raise ValueError("speaker must be between 0 and 15")
    if chunk_length is not None and not 100 <= chunk_length <= 1000:
        raise ValueError("chunk_length must be between 100 and 1000")
    if temperature is not None and not 0.1 <= temperature <= 1.0:
        raise ValueError("temperature must be between 0.1 and 1.0")
    if top_p is not None and not 0.1 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.1 and 1.0")

    options: dict = {}
    for key, value in (
        ("seed", seed),
        ("temperature", temperature),
        ("top_p", top_p),
        ("max_new_tokens", max_new_tokens),
        ("chunk_length", chunk_length),
        ("single_chunk", single_chunk),
        ("normalize", normalize),
        ("latency", latency),
        ("use_memory_cache", use_memory_cache),
    ):
        if value is not None:
            options[key] = value

    if reference_audio:
        path = Path(reference_audio).expanduser()
        if not path.is_file():
            raise ValueError(f"Reference audio does not exist: {path}")
        reference = {
            "audio": base64.b64encode(path.read_bytes()).decode("ascii"),
            "text": reference_text,
        }
        if speaker is not None:
            reference["speaker"] = speaker
        options["references"] = [reference]

    destination = _output_path(text, format, output_name)
    started = time.perf_counter()
    try:
        audio = _api().generate(text, voice=voice, format=format, **options)
    except FishAPIError as exc:
        raise RuntimeError(f"Speech generation failed: {exc}") from exc
    elapsed = time.perf_counter() - started
    destination.write_bytes(audio)
    metadata = _wav_metadata(audio) if format == "wav" else {}
    duration = metadata.get("duration_s")
    result = {
        "path": str(destination),
        "format": format,
        "mime_type": MIME_TYPE[format],
        "bytes": len(audio),
        "elapsed_s": round(elapsed, 3),
        "rtf": round(elapsed / duration, 3) if duration else None,
        "text": text,
        "voice": voice,
        **metadata,
    }
    if include_audio_base64:
        result["audio_base64"] = base64.b64encode(audio).decode("ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default=os.environ.get("FISH_MCP_TRANSPORT", "stdio"),
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("FISH_MCP_HOST", "127.0.0.1"),
        help="Bind host for SSE/streamable-http (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FISH_MCP_PORT", "8765")),
        help="Bind port for SSE/streamable-http (default: 8765)",
    )
    args = parser.parse_args()
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
