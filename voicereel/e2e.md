# VoiceReel End-to-End Testing Guide

This document describes how to run the optional end-to-end (E2E) tests against a real VoiceReel deployment and database. These tests exercise the HTTP API using `VoiceReelClient` and verify records in the backing database.

## Prerequisites

1. A running VoiceReel API server that exposes the endpoints described in `PRD.md`.
2. A PostgreSQL database used by the server.
3. Sample reference audio (`wav`) and matching script (`txt`) for registering a speaker.
4. `psycopg2` installed in the Python environment for database access.

## Environment Variables

Set the following variables before executing the tests:

- `VOICE_REEL_E2E_URL` – Base URL of the VoiceReel API (e.g. `http://localhost:8080`).
- `VOICE_REEL_E2E_DSN` – PostgreSQL DSN for verifying database state (e.g. `postgresql://user:pass@localhost/db`).
- `VOICE_REEL_E2E_AUDIO` – Path to a reference audio file (≥30 sec recommended).
- `VOICE_REEL_E2E_SCRIPT` – Path to the matching transcript file.

## Running the Tests

```bash
# Install additional dependencies
pip install psycopg2-binary pytest

# Execute only the E2E suite
pytest tests/test_voicereel_e2e.py -v
```

The tests are skipped automatically if any of the required environment variables are missing or if `psycopg2` is not installed.

### Test Skip Conditions

The E2E tests will be automatically skipped in these scenarios:
1. **Missing environment variables**: Any of the four required environment variables is not set
2. **Missing psycopg2**: The PostgreSQL Python adapter is not installed
3. **Missing voicereel module**: The VoiceReel Python package dependencies are not available

### Verifying Test Prerequisites

You can check if the E2E test environment is properly set up:

```bash
python -c "
import os
print('Environment variables:')
for var in ['VOICE_REEL_E2E_URL', 'VOICE_REEL_E2E_DSN', 'VOICE_REEL_E2E_AUDIO', 'VOICE_REEL_E2E_SCRIPT']:
    print(f'  {var}: {\"✓\" if os.getenv(var) else \"✗\"}')

try:
    import psycopg2
    print('psycopg2: ✓')
except ImportError:
    print('psycopg2: ✗')

try:
    from voicereel import VoiceReelClient
    print('voicereel: ✓')
except ImportError:
    print('voicereel: ✗')
"
```

## What the Test Covers

1. **Speaker registration** – uploads the reference audio and waits for the job to succeed.
2. **Database check** – connects to PostgreSQL and confirms the new speaker row exists.
3. **Synthesis** – requests synthesis using the new speaker and waits for completion.
4. **Artifact validation** – ensures the API returns URLs for the generated audio and captions.

These steps verify that the HTTP layer, worker queue, and database are all wired correctly.

