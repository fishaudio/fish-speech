import importlib.util
import json
import os
import sys
import types
import urllib.error
import urllib.request
from datetime import datetime
import hmac
import hashlib

import pytest

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


def test_usage_report():
    server = VoiceReelServer()
    server.start()
    try:
        payload = json.dumps({"script": [{"speaker_id": 1, "text": "hi"}]}).encode()
        for _ in range(3):
            req = urllib.request.Request(
                f"{_base_url(server)}/v1/synthesize", data=payload, method="POST"
            )
            urllib.request.urlopen(req).read()
        server.wait_all_jobs()
        now = datetime.now()
        report = server.usage_report(now.year, now.month)
        assert report["count"] >= 3
        assert report["total_length"] >= 1.5
    finally:
        server.stop()


def test_register_and_list_speakers():
    server = VoiceReelServer()
    server.start()
    try:
        payload = json.dumps({"duration": 30, "script": "hello"}).encode()
        req = urllib.request.Request(
            f"{_base_url(server)}/v1/speakers", data=payload, method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
        assert "job_id" in body
        assert "speaker_id" in body

        with urllib.request.urlopen(
            f"{_base_url(server)}/v1/speakers?page=1&page_size=1"
        ) as resp:
            data = json.loads(resp.read().decode())
        assert any(s["id"] == body["speaker_id"] for s in data["speakers"])

        server.wait_all_jobs()
        with urllib.request.urlopen(
            f"{_base_url(server)}/v1/jobs/{body['job_id']}"
        ) as resp:
            info = json.loads(resp.read().decode())
        assert info["status"] == "succeeded"
    finally:
        server.stop()


def test_get_single_speaker():
    server = VoiceReelServer()
    server.start()
    try:
        payload = json.dumps({"duration": 30, "script": "hello"}).encode()
        req = urllib.request.Request(
            f"{_base_url(server)}/v1/speakers", data=payload, method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode())
        speaker_id = body["speaker_id"]

        server.wait_all_jobs()
        with urllib.request.urlopen(
            f"{_base_url(server)}/v1/speakers/{speaker_id}"
        ) as resp:
            info = json.loads(resp.read().decode())
        assert info["id"] == speaker_id
    finally:
        server.stop()


def test_register_invalid_duration():
    server = VoiceReelServer()
    server.start()
    try:
        payload = json.dumps({"duration": 10, "script": "hi"}).encode()
        req = urllib.request.Request(
            f"{_base_url(server)}/v1/speakers", data=payload, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 422
    finally:
        server.stop()


def test_synthesize_endpoint():
    server = VoiceReelServer()
    server.start()
    try:
        payload = json.dumps({"script": [{"speaker_id": 1, "text": "hi"}]}).encode()
        req = urllib.request.Request(
            f"{_base_url(server)}/v1/synthesize", data=payload, method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert "job_id" in data
        server.wait_all_jobs()
        with urllib.request.urlopen(
            f"{_base_url(server)}/v1/jobs/{data['job_id']}"
        ) as resp:
            info = json.loads(resp.read().decode())
        assert info["status"] == "succeeded"
    finally:
        server.stop()


def test_job_get_and_delete():
    server = VoiceReelServer()
    server.start()
    try:
        payload = json.dumps({"script": [{"speaker_id": 1, "text": "hi"}]}).encode()
        req = urllib.request.Request(
            f"{_base_url(server)}/v1/synthesize", data=payload, method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        job_id = data["job_id"]

        server.wait_all_jobs()
        with urllib.request.urlopen(f"{_base_url(server)}/v1/jobs/{job_id}") as resp:
            info = json.loads(resp.read().decode())
        assert info["status"] == "succeeded"
        assert "audio_url" in info

        req = urllib.request.Request(
            f"{_base_url(server)}/v1/jobs/{job_id}", method="DELETE"
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 204

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"{_base_url(server)}/v1/jobs/{job_id}")
        assert exc.value.code == 404
    finally:
        server.stop()


def test_synthesize_caption_formats():
    server = VoiceReelServer()
    server.start()
    try:
        for fmt in ["json", "vtt", "srt"]:
            payload = json.dumps(
                {
                    "script": [{"speaker_id": 1, "text": "hi"}],
                    "caption_format": fmt,
                }
            ).encode()
            req = urllib.request.Request(
                f"{_base_url(server)}/v1/synthesize", data=payload, method="POST"
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
            job_id = data["job_id"]
            server.wait_all_jobs()
            with urllib.request.urlopen(
                f"{_base_url(server)}/v1/jobs/{job_id}"
            ) as resp:
                info = json.loads(resp.read().decode())
            assert info["caption_format"] == fmt
            assert os.path.exists(info["caption_url"])
    finally:
        server.stop()

def test_api_key_required(monkeypatch):
    monkeypatch.setenv("VR_API_KEY", "secret")
    server = VoiceReelServer()
    server.start()
    try:
        req = urllib.request.Request(f"{_base_url(server)}/health")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 401

        req = urllib.request.Request(
            f"{_base_url(server)}/health", headers={"X-VR-APIKEY": "secret"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
    finally:
        server.stop()


def test_hmac_signature(monkeypatch):
    monkeypatch.setenv("VR_API_KEY", "secret")
    monkeypatch.setenv("VR_HMAC_SECRET", "hmac")
    server = VoiceReelServer()
    server.start()
    try:
        req = urllib.request.Request(
            f"{_base_url(server)}/health", headers={"X-VR-APIKEY": "secret"}
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 401

        sign = hmac.new(b"hmac", b"", hashlib.sha256).hexdigest()
        req = urllib.request.Request(
            f"{_base_url(server)}/health",
            headers={"X-VR-APIKEY": "secret", "X-VR-SIGN": sign},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        assert data["status"] == "ok"
    finally:
        server.stop()

