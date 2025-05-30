"""End-to-end tests for a real VoiceReel deployment."""

import os
import time

import pytest

VoiceReelClient = pytest.importorskip("voicereel").VoiceReelClient

# Optional dependency for database validation
psycopg2 = pytest.importorskip("psycopg2")

SERVER_URL = os.getenv("VOICE_REEL_E2E_URL")
DSN = os.getenv("VOICE_REEL_E2E_DSN")
AUDIO = os.getenv("VOICE_REEL_E2E_AUDIO")
SCRIPT = os.getenv("VOICE_REEL_E2E_SCRIPT")

skip_reason = "E2E environment variables not set"

pytestmark = pytest.mark.skipif(
    not (SERVER_URL and DSN and AUDIO and SCRIPT), reason=skip_reason
)


def wait_job(client: VoiceReelClient, job_id: str, timeout: float = 60.0) -> dict:
    """Poll the job endpoint until completion."""
    end = time.time() + timeout
    while time.time() < end:
        info = client.get_job(job_id)
        if info.get("status") == "succeeded":
            return info
        time.sleep(1)
    raise RuntimeError(f"Job {job_id} did not finish in time")


def test_register_synthesize_flow():
    client = VoiceReelClient(api_url=SERVER_URL)

    # Register a new speaker
    resp = client.register_speaker("e2e_spk", "en", AUDIO, SCRIPT)
    job_id = resp["job_id"]
    speaker_id = resp["speaker_id"]

    info = wait_job(client, job_id)
    assert info["status"] == "succeeded"

    # Verify record exists in DB
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    cur.execute("SELECT id FROM speakers WHERE id = %s", (speaker_id,))
    assert cur.fetchone() is not None
    conn.close()

    # Synthesis using the new speaker
    script = [{"speaker_id": speaker_id, "text": "hello from e2e"}]
    synth = client.synthesize(script)
    synth_job = synth["job_id"]

    result = wait_job(client, synth_job)
    assert result["status"] == "succeeded"
    assert result.get("audio_url")
    assert result.get("caption_url")
