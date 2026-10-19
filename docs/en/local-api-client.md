# Self-hosted Python API client

`fishapi.py` wraps every explicit application route in `tools/server/views.py`.
It is a standalone Python 3.10+ module, not a published PyPI package. Copy it
beside your dataset script, or import it while working from the repository root.
No Torch, CUDA, or model installation is required on the client computer.

The official [Fish Audio SDK](https://github.com/fishaudio/fish-audio-python)
targets the hosted API. Its voice management uses `/model`, whereas this local
server uses `/v1/references`. This small wrapper follows the local server.

## Launch

On the Spark, after setup (model weights download into the shared HF cache on first launch):

```bash
./start-api.sh
# Or, after building Awesome WebUI:
./start-webui.sh
```

Both scripts select the isolated Spark CUDA 13 environment and bind to
`0.0.0.0:8080` for LAN access. The built WebUI is served at `/ui`; build it with
`(cd awesome_webui && npm ci && npm run build)`. Run only one launcher per port.
Extra arguments are forwarded. Compilation is opt-in (`--compile`, S2-Pro only)
and reuses `$HOME/.cache/fish-speech/inductor` (or `TORCHINDUCTOR_CACHE_DIR`).
For Drama-3 use `--model fishaudio/drama-3 --device cuda:0`
without `--compile`. Each running server loads its own model.

API authentication is optional: start with `--api-key YOUR_KEY` and pass the
same value as `api_key=` to the client. Without it, LAN clients can generate
speech and manage references. For loopback only, pass `--listen 127.0.0.1:8080`.

## Generate speech

```python
from fishapi import FishAPI

api = FishAPI('192.168.1.50')  # bare host/IP defaults to port 8080
# Also accepts '192.168.1.50:8081' or 'http://192.168.1.50:8080'.
print(api.health())

# Register once. Use a reference WAV and its actual transcript.
api.add_voice('speaker_a', 'speaker_a.wav', 'Words spoken in the reference.')
# addvoice(...) is an alias.
print(api.list_voices())

audio = api.generate(
    'I agree to deliver the furniture on Friday.',
    voice='speaker_a',
    output='turn_001.wav',
    seed=42,
    temperature=0.8,
)
```

`generate()` always returns audio bytes; `output` additionally saves them.
WAV is the default. Output directories must already exist. Existing output
files are overwritten on success. HTTP failure does not overwrite the output.
`voice` is an alias for the server's `reference_id` field. Other keyword options
are passed as TTS request fields, including `references`, `normalize`, `top_p`,
`repetition_penalty`, `max_new_tokens`, `chunk_length`, and `use_memory_cache`.
Inline references use base64-encoded audio strings and their transcripts.
Unknown fields may be ignored by the server; consult `ServeTTSRequest` in
`fish_speech/utils/schema.py`. Use WAV for offline datasets; other formats depend
on the server's encoder support. Cloud-only controls such as speed are not added.

## Endpoint coverage

| Method | HTTP endpoint | Return value |
| --- | --- | --- |
| `health()` | GET `/v1/health` | Status dictionary |
| `health(method='POST')` | POST `/v1/health` | Status dictionary |
| `webui()` | GET `/ui` | HTML string (404 if UI not built) |
| `generate(text, ...)` | POST `/v1/tts` | Complete audio bytes |
| `generate_stream(text, ...)` | POST `/v1/tts` | Iterator of WAV stream bytes |
| `add_voice(id, audio_path, text)` | POST `/v1/references/add` | Result dictionary |
| `list_voices()` | GET `/v1/references/list` | List of IDs |
| `rename_voice(old_id, new_id)` | POST `/v1/references/update` | Result dictionary |
| `delete_voice(id)` | DELETE `/v1/references/delete` | Result dictionary |
| `encode(audio_files)` | POST `/v1/vqgan/encode` | Nested token arrays |
| `decode(tokens)` | POST `/v1/vqgan/decode` | List of raw float16 PCM buffers |

`rename_voice` only renames a reference; it does not replace its recording.
Adding an existing ID raises an error. Metadata calls request JSON explicitly.
Framework-generated OpenAPI documentation routes are not application operations.

## Streaming and codec operations

```python
from contextlib import closing

with closing(api.generate_stream('Hello!', voice='speaker_a')) as chunks:
    with open('stream.wav', 'wb') as out:
        for chunk in chunks:
            out.write(chunk)
```

Streaming supports WAV only. Chunks are parts of one stream, not independent
WAV files. The WAV stream header may not contain a finalized duration; use
non-streaming generation for dataset files. Closing the generator releases its
connection if the caller stops early.

Only codec methods need an extra dependency:

```bash
uv pip install ormsgpack
# In a separate uv project, use: uv add ormsgpack
```

```python
from pathlib import Path

tokens = api.encode([Path('reference.wav').read_bytes()])
raw_audio_buffers = api.decode(tokens)
```

Codec requests and responses use MessagePack. Decode returns raw float16 PCM,
**not WAV**, and does not return the sample rate. Obtain the rate from the
server's decoder configuration before converting to a playable file.

## Errors and dataset jobs

HTTP and connection errors raise `FishAPIError` (HTTP errors include status and
server response). Local file and serialization errors propagate normally.
Read-time errors after response headers can also propagate as standard Python
I/O exceptions. The timeout defaults to 600 seconds; override with
`FishAPI(host, timeout=1200)` if the first compilation needs longer.
There are no automatic retries: retrying a generation may create duplicate work.

Use one request at a time initially. Store the turn text, speaker ID, seed,
settings and returned audio in your dataset pipeline. This client does not
provide a job queue, alignment, conversation generation, or resume tracking.

## Client validation

```bash
uv run --no-project --with ormsgpack python -m unittest discover -s tests -p test_fishapi.py
```

Tests use a local HTTP fixture to check request bodies, multipart uploads,
MessagePack, authentication, streaming, and output preservation on HTTP errors.
They do not load the speech model or validate GPU inference.
