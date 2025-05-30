import json
import urllib.request
import sys
import os
import types
import importlib.util

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Provide minimal stubs for dependencies that require heavy packages
# ---------------------------------------------------------------------------
fish_speech = types.ModuleType("fish_speech")
fish_speech.__path__ = []
utils = types.ModuleType("fish_speech.utils")
utils.__path__ = []
fish_speech.utils = utils
sys.modules.setdefault("fish_speech", fish_speech)
sys.modules.setdefault("fish_speech.utils", utils)

spec = importlib.util.spec_from_file_location(
    "voicereel.server", os.path.join(PROJECT_ROOT, "voicereel", "server.py")
)
server_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_mod)
VoiceReelServer = server_mod.VoiceReelServer


def _base_url(server: VoiceReelServer) -> str:
    host, port = server.address
    return f"http://{host}:{port}"


def test_health_endpoint():
    server = VoiceReelServer()
    server.start()
    try:
        with urllib.request.urlopen(f"{_base_url(server)}/health") as resp:
            data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
    finally:
        server.stop()


def test_register_and_list_speakers():
    server = VoiceReelServer()
    server.start()
    try:
        req = urllib.request.Request(
            f"{_base_url(server)}/v1/speakers", data=b"{}", method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
        assert "job_id" in body
        assert "speaker_id" in body

        with urllib.request.urlopen(f"{_base_url(server)}/v1/speakers") as resp:
            data = json.loads(resp.read().decode())
        assert any(s["id"] == body["speaker_id"] for s in data["speakers"])
    finally:
        server.stop()
