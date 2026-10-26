"""Small client for the self-hosted Fish Speech API (Python 3.10+).

Copy this file into your project. Only encode()/decode() require ormsgpack.
This is independent of the hosted Fish Audio SDK and imports no ML packages.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4


class FishAPIError(RuntimeError):
    """Server or transport failure. HTTP failures include status and response body."""


class FishAPI:
    def __init__(self, host: str = 'http://127.0.0.1:8080', *,
                 api_key: str | None = None, timeout: float = 600):
        bare_host = '://' not in host
        host = host if not bare_host else 'http://' + host
        parsed = urlsplit(host)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            raise ValueError('Use an HTTP(S) server URL or hostname/IP')
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError('URL must not contain credentials, query, or fragment')
        if bare_host and parsed.port is None:
            host = parsed._replace(netloc=parsed.netloc + ':8080').geturl()
        self.base_url = host.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def _open(self, method, path, *, body=None, content_type=None, accept='application/json'):
        headers = {'Accept': accept}
        if content_type:
            headers['Content-Type'] = content_type
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        req = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            return urlopen(req, timeout=self.timeout)
        except HTTPError as exc:
            with exc:
                detail = exc.read().decode('utf-8', errors='replace')
            raise FishAPIError(f'HTTP {exc.code} {method} {path}: {detail}') from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise FishAPIError(f'{method} {path}: {exc}') from exc

    def _json(self, method, path, payload=None, *, body=None, content_type=None):
        if payload is not None:
            body, content_type = json.dumps(payload).encode(), 'application/json'
        with self._open(method, path, body=body, content_type=content_type) as response:
            result = json.load(response)
        if isinstance(result, dict) and result.get('success') is False:
            raise FishAPIError(result.get('message', 'Server reported failure'))
        return result

    def health(self, *, method='GET') -> dict:
        """GET or POST /v1/health."""
        if method not in ('GET', 'POST'):
            raise ValueError('Health method must be GET or POST')
        return self._json(method, '/v1/health')

    def webui(self) -> str:
        """GET /ui: return built WebUI HTML; raises if the server UI is not built."""
        with self._open('GET', '/ui', accept='text/html') as response:
            return response.read().decode('utf-8')

    def tts_defaults(self) -> dict:
        """GET /v1/tts/defaults; model family and sampling defaults."""
        return self._json('GET', '/v1/tts/defaults')

    def generate(self, text: str, *, voice: str | None = None,
                 output: str | Path | None = None, **options) -> bytes:
        """POST /v1/tts; return complete audio bytes and optionally save them.

        Defaults to WAV. Options match ServeTTSRequest, e.g. seed, temperature,
        reference_id, references (base64 audio + transcript), max_new_tokens.
        Use generate_stream() for streaming. No automatic retries.
        """
        if options.get('streaming'):
            raise ValueError('Use generate_stream() for streaming')
        payload = self._tts_payload(text, voice, options, streaming=False)
        with self._open('POST', '/v1/tts', body=json.dumps(payload).encode(),
                        content_type='application/json', accept='audio/*') as response:
            audio = response.read()
        if output is not None:
            Path(output).write_bytes(audio)
        return audio

    @staticmethod
    def _tts_payload(text, voice, options, *, streaming):
        payload = {'format': 'wav', **options, 'text': text, 'streaming': streaming}
        if voice is not None:
            if options.get('reference_id') not in (None, voice):
                raise ValueError('voice and reference_id must match when both supplied')
            payload['reference_id'] = voice
        return payload

    def generate_stream(self, text: str, *, voice: str | None = None,
                        chunk_size: int = 65536, **options):
        """Yield raw pieces of one WAV stream; concatenate in order.

        Chunks are transport bytes, not separate playable WAV files. Close the
        generator if stopping early. The server only supports WAV streaming.
        """
        if chunk_size <= 0 or options.get('format', 'wav') != 'wav':
            raise ValueError('Streaming requires WAV and a positive chunk_size')
        payload = self._tts_payload(text, voice, options, streaming=True)
        with self._open('POST', '/v1/tts', body=json.dumps(payload).encode(),
                        content_type='application/json', accept='audio/wav') as response:
            while chunk := response.read(chunk_size):
                yield chunk

    def add_voice(self, voice_id: str, audio: str | Path, text: str) -> dict:
        """POST /v1/references/add; upload a reference WAV and exact transcript."""
        boundary = uuid4().hex
        parts = []
        for name, value in [('id', voice_id), ('text', text)]:
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                          f'\r\n\r\n{value}\r\n').encode())
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; '
                      'filename="reference.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode())
        parts.extend([Path(audio).read_bytes(), f'\r\n--{boundary}--\r\n'.encode()])
        return self._json('POST', '/v1/references/add', body=b''.join(parts),
                          content_type=f'multipart/form-data; boundary={boundary}')

    addvoice = add_voice

    def list_voices(self) -> list[str]:
        """GET /v1/references/list; return saved reference IDs."""
        return self._json('GET', '/v1/references/list')['reference_ids']

    def rename_voice(self, old_id: str, new_id: str) -> dict:
        """POST /v1/references/update (rename only, not audio replacement)."""
        return self._json('POST', '/v1/references/update',
                          {'old_reference_id': old_id, 'new_reference_id': new_id})

    def delete_voice(self, voice_id: str) -> dict:
        """DELETE /v1/references/delete."""
        return self._json('DELETE', '/v1/references/delete', {'reference_id': voice_id})

    def _codec(self, operation, payload):
        try:
            import ormsgpack
        except ImportError as exc:
            raise ImportError('Codec methods require ormsgpack: uv pip install ormsgpack') from exc
        with self._open('POST', '/v1/vqgan/' + operation,
                        body=ormsgpack.packb(payload), content_type='application/msgpack',
                        accept='application/msgpack') as response:
            return ormsgpack.unpackb(response.read())

    def encode(self, audios: list[bytes]) -> list:
        """POST /v1/vqgan/encode; list of encoded audio file bytes -> token arrays."""
        return self._codec('encode', {'audios': audios})['tokens']

    def decode(self, tokens: list) -> list[bytes]:
        """POST /v1/vqgan/decode; token arrays -> raw float16 PCM buffers.

        These are NOT WAV files. The response does not include sample rate;
        obtain it from the server's decoder configuration before saving/playing.
        """
        return self._codec('decode', {'tokens': tokens})['audios']
