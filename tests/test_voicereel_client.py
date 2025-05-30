import json
import os
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# Ensure project root is on sys.path for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Provide minimal stubs for dependencies that require heavy packages
# ---------------------------------------------------------------------------
class ServeReferenceAudio:
    def __init__(self, audio: bytes, text: str):
        self.audio = audio
        self.text = text

    def model_dump(self):
        return {
            "audio": (
                self.audio.decode() if isinstance(self.audio, bytes) else self.audio
            ),
            "text": self.text,
        }


class ServeTTSRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return self.__dict__


def audio_to_bytes(path: str):
    with open(path, "rb") as f:
        return f.read()


def read_ref_text(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Build fake fish_speech.utils package
fish_speech = types.ModuleType("fish_speech")
fish_speech.__path__ = []  # mark as package

utils = types.ModuleType("fish_speech.utils")
utils.__path__ = []  # mark as package
file_mod = types.ModuleType("fish_speech.utils.file")
file_mod.audio_to_bytes = audio_to_bytes
file_mod.read_ref_text = read_ref_text
schema_mod = types.ModuleType("fish_speech.utils.schema")
schema_mod.ServeReferenceAudio = ServeReferenceAudio
schema_mod.ServeTTSRequest = ServeTTSRequest
utils.file = file_mod
utils.schema = schema_mod
fish_speech.utils = utils
sys.modules.setdefault("fish_speech", fish_speech)
sys.modules.setdefault("fish_speech.utils", utils)
sys.modules.setdefault("fish_speech.utils.file", file_mod)
sys.modules.setdefault("fish_speech.utils.schema", schema_mod)

from voicereel import VoiceReelClient


class MockHandler(BaseHTTPRequestHandler):
    last_payload: dict | None = None

    def do_POST(self):
        if self.path == "/v1/tts":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"audio-bytes")
        elif self.path == "/v1/speakers":
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {"job_id": "job123", "speaker_temp_id": "spk_tmp"}
            self.wfile.write(json.dumps(body).encode())
        elif self.path == "/v1/synthesize":
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                MockHandler.last_payload = json.loads(raw.decode()) if raw else {}
            except json.JSONDecodeError:
                MockHandler.last_payload = {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {"job_id": "job456"}
            self.wfile.write(json.dumps(body).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/v1/speakers":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {"speakers": [{"id": "spk1", "name": "minho"}]}
            self.wfile.write(json.dumps(body).encode())
        elif self.path.startswith("/v1/speakers/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {"id": "spk1", "name": "minho"}
            self.wfile.write(json.dumps(body).encode())
        elif self.path.startswith("/v1/jobs/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = {
                "status": "succeeded",
                "audio_url": "http://example.com/out.wav",
                "captions": [],
            }
            self.wfile.write(json.dumps(body).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        if self.path.startswith("/v1/jobs/"):
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        return


class MockServer:
    def __init__(self):
        self.server = HTTPServer(("127.0.0.1", 0), MockHandler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.thread.join()

    @property
    def url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}"


@pytest.fixture
def client():
    server = MockServer()
    server.start()
    client = VoiceReelClient(api_url=server.url)
    try:
        yield client
    finally:
        server.stop()


def test_tts(client):
    result = client.tts("hello")
    assert result == b"audio-bytes"


def test_register_and_list(client):
    with (
        tempfile.NamedTemporaryFile("wb", suffix=".wav") as fa,
        tempfile.NamedTemporaryFile("w", suffix=".txt") as ft,
    ):
        fa.write(b"data")
        fa.flush()
        ft.write("hello")
        ft.flush()
        resp = client.register_speaker("name", "en", fa.name, ft.name)
    assert resp["job_id"] == "job123"
    speakers = client.list_speakers()
    assert speakers["speakers"][0]["id"] == "spk1"


def test_synthesize_and_job(client):
    job = client.synthesize([{"speaker_id": "spk1", "text": "hi"}])
    assert job["job_id"] == "job456"
    status = client.get_job(job["job_id"])
    assert status["status"] == "succeeded"
    client.delete_job(job["job_id"])


def test_synthesize_caption_option(client):
    _ = client.synthesize([{"speaker_id": "spk1", "text": "hi"}], caption_format="vtt")
    assert MockHandler.last_payload["caption_format"] == "vtt"
