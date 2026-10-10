"""STDIO MCP bridge to the already-running local Fish Speech server."""

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:7860"
mcp = FastMCP(
    "local-voice",
    instructions=(
        "Generate English video narration with the local S1 Mini model. Check "
        "voice_server_status first. The user starts/stops the model with the "
        "project's .command shortcuts. Generate short sections separately. "
        "When cloning a voice, supply its local audio path and exact transcript. "
        "Pass the returned WAV path to a separately connected video editor tool."
    ),
)


@mcp.tool()
def voice_server_status() -> dict[str, Any]:
    """Check whether the local S1 Mini web server is ready for voice generation."""
    try:
        with urllib.request.urlopen(f"{URL}/config", timeout=5) as response:
            config = json.load(response)
        title = config.get("title", "")
        ready = "S1 Mini" in title
        return {"ready": ready, "url": URL, "title": title,
                "message": "Ready" if ready else "Port 7860 is serving another app."}
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return {"ready": False, "url": URL,
                "message": f"Start Fish Speech.command and wait for the web UI. ({exc})"}


@mcp.tool()
def generate_voiceover(
    text: str,
    output_path: str,
    reference_audio_path: str = "",
    reference_text: str = "",
    seed: int = 42,
) -> dict[str, Any]:
    """Generate English narration and return a local WAV path and duration.

    output_path must be an absolute, new .wav filename in the video project.
    For voice cloning, provide reference_audio_path and its exact reference_text.
    Use short sections (roughly 2-3 sentences), then import each returned file
    into the separately connected video editor. Generation can take minutes.
    Existing files are never overwritten.
    """
    if not Path(output_path).is_absolute():
        raise ValueError("output_path must be an absolute path ending in .wav.")
    if len(text) > 2000:
        raise ValueError("Split narration into sections of at most 2000 characters.")
    command = [str(ROOT / "venv/bin/python"), str(ROOT / "generate_voiceover.py"),
               "--text", text, "--output", output_path, "--seed", str(seed)]
    if reference_audio_path:
        command.extend(["--reference-audio", reference_audio_path])
    if reference_text:
        command.extend(["--reference-text", reference_text])
    result = subprocess.run(command, capture_output=True, text=True, timeout=900, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-2000:] or "Voice generation failed.")
    return json.loads(result.stdout)


if __name__ == "__main__":
    mcp.run(transport="stdio")
