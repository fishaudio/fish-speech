# VoiceReel Studio

Multi-speaker text-to-speech API service built on fish-speech 1.5 engine.

## Quick Start

### Development Setup

1. **Install dependencies:**
```bash
pip install -e .
pip install 'celery[redis]' redis psycopg2-binary
```

2. **Start services with Docker Compose:**
```bash
cd voicereel
docker-compose up -d
```

3. **Run tests:**
```bash
pytest tests/test_voicereel*.py
```

### Using the API

#### Register a speaker:
```bash
python -m voicereel.client register \
  --name "John Doe" \
  --lang en \
  --audio reference.wav \
  --script reference.txt
```

#### Synthesize multi-speaker audio:
```bash
python -m voicereel.client synthesize script.json
```

Where `script.json` contains:
```json
[
  {"speaker_id": "spk_1", "text": "Hello, this is speaker one."},
  {"speaker_id": "spk_2", "text": "And this is speaker two!"}
]
```

## Architecture

### With Celery (Production)
```
Client → API Server → Redis Queue → Celery Workers → PostgreSQL
                                 ↘ S3 Storage ↙
```

### Without Celery (Development)
```
Client → API Server → In-Memory Queue → Worker Thread → SQLite
                   ↘ Local Filesystem ↙
```

## Environment Variables

- `VR_DSN`: Database connection string (PostgreSQL or SQLite)
- `VR_REDIS_URL`: Redis URL for Celery broker/backend
- `VR_API_KEY`: API authentication key
- `VR_HMAC_SECRET`: HMAC signature secret (optional)

## API Endpoints

- `POST /v1/speakers` - Register new speaker
- `GET /v1/speakers` - List speakers
- `GET /v1/speakers/{id}` - Get speaker details
- `POST /v1/synthesize` - Multi-speaker synthesis
- `GET /v1/jobs/{id}` - Check job status
- `DELETE /v1/jobs/{id}` - Delete job and files

## Development

### Running without Docker:

1. Start Redis:
```bash
redis-server
```

2. Start Celery workers:
```bash
celery -A voicereel.celery_app worker -Q speakers,synthesis -l info
```

3. Start API server:
```bash
python -m voicereel.server
```

### Running tests:
```bash
# Unit tests
pytest tests/test_voicereel_infra.py

# Client tests
pytest tests/test_voicereel_client.py

# E2E tests (requires running server)
VOICE_REEL_E2E_URL=http://localhost:8080 \
VOICE_REEL_E2E_DSN=postgresql://... \
VOICE_REEL_E2E_AUDIO=test.wav \
VOICE_REEL_E2E_SCRIPT=test.txt \
pytest tests/test_voicereel_e2e.py
```

## Monitoring

Access Flower dashboard at http://localhost:5555 to monitor Celery tasks.

## Production Deployment

1. Set secure environment variables
2. Use proper PostgreSQL database
3. Configure S3 bucket for audio storage
4. Enable GPU support for synthesis workers
5. Set up monitoring and alerting
6. Configure TLS/SSL certificates

See PRD.md for detailed requirements and specifications.
