from __future__ import annotations

import argparse
import json
import mimetypes
import uuid
from dataclasses import dataclass
from typing import Iterable
from urllib import request as urlrequest
from urllib.error import HTTPError

import ormsgpack
from fish_speech.utils.file import audio_to_bytes, read_ref_text
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

DEFAULT_API_URL = "http://127.0.0.1:8080"


@dataclass
class VoiceReelClient:
    """Client for interacting with the VoiceReel API."""

    api_url: str = DEFAULT_API_URL
    api_key: str | None = None

    @property
    def tts_endpoint(self) -> str:
        return f"{self.api_url.rstrip('/')}/v1/tts"

    @property
    def speakers_endpoint(self) -> str:
        return f"{self.api_url.rstrip('/')}/v1/speakers"

    @property
    def synth_endpoint(self) -> str:
        return f"{self.api_url.rstrip('/')}/v1/synthesize"

    @property
    def jobs_endpoint(self) -> str:
        return f"{self.api_url.rstrip('/')}/v1/jobs"

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["X-VR-APIKEY"] = self.api_key
        return headers

    def _post_json(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        headers.update(self._headers())
        req = urlrequest.Request(url, data=data, headers=headers, method="POST")
        try:
            with urlrequest.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            raise RuntimeError(f"Request failed: {exc.read().decode()}") from exc

    def _get_json(self, url: str) -> dict:
        headers = self._headers()
        req = urlrequest.Request(url, headers=headers)
        try:
            with urlrequest.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            raise RuntimeError(f"Request failed: {exc.read().decode()}") from exc

    def _delete(self, url: str) -> None:
        headers = self._headers()
        req = urlrequest.Request(url, headers=headers, method="DELETE")
        try:
            urlrequest.urlopen(req).read()
        except HTTPError as exc:
            raise RuntimeError(f"Request failed: {exc.read().decode()}") from exc

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def register_speaker(
        self,
        name: str,
        lang: str,
        reference_audio: str,
        reference_script: str,
    ) -> dict:
        """Register a speaker with reference audio and script."""
        boundary = uuid.uuid4().hex
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        headers.update(self._headers())

        def _field(name: str, value: bytes, filename: str | None = None) -> bytes:
            disposition = f'form-data; name="{name}"'
            if filename:
                disposition += f'; filename="{filename}"'
            part = [f"--{boundary}", f"Content-Disposition: {disposition}", ""]
            return "\r\n".join(part).encode() + value + b"\r\n"

        body = b""
        body += _field("name", name.encode())
        body += _field("lang", lang.encode())
        with open(reference_audio, "rb") as f:
            data = f.read()
        mime = mimetypes.guess_type(reference_audio)[0] or "application/octet-stream"
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="reference_audio"; filename="{reference_audio}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode() + data + b"\r\n"
        )
        body += _field("reference_script", reference_script.encode())
        body += f"--{boundary}--\r\n".encode()

        req = urlrequest.Request(
            self.speakers_endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urlrequest.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            raise RuntimeError(f"Request failed: {exc.read().decode()}") from exc

    def list_speakers(
        self, page: int | None = None, page_size: int | None = None
    ) -> dict:
        url = self.speakers_endpoint
        params = []
        if page is not None:
            params.append(f"page={page}")
        if page_size is not None:
            params.append(f"page_size={page_size}")
        if params:
            url = f"{url}?{'&'.join(params)}"
        return self._get_json(url)

    def get_speaker(self, speaker_id: str) -> dict:
        return self._get_json(f"{self.speakers_endpoint}/{speaker_id}")

    def synthesize(
        self,
        script: list[dict],
        output_format: str = "wav",
        sample_rate: int = 48000,
        caption_format: str = "json",
    ) -> dict:
        payload = {
            "script": script,
            "output_format": output_format,
            "sample_rate": sample_rate,
            "caption_format": caption_format,
        }
        return self._post_json(self.synth_endpoint, payload)

    def get_job(self, job_id: str) -> dict:
        return self._get_json(f"{self.jobs_endpoint}/{job_id}")

    def delete_job(self, job_id: str) -> None:
        self._delete(f"{self.jobs_endpoint}/{job_id}")

    def tts(
        self,
        text: str,
        reference_audios: Iterable[str] | None = None,
        reference_texts: Iterable[str] | None = None,
        reference_id: str | None = None,
        fmt: str = "wav",
        streaming: bool = False,
        **kwargs,
    ) -> bytes:
        """Synthesize speech using the API.

        Parameters
        ----------
        text : str
            Input text to synthesize.
        reference_audios : Iterable[str] | None
            Paths to reference audios.
        reference_texts : Iterable[str] | None
            Text corresponding to reference audios.
        reference_id : str | None
            Prebuilt reference id stored on server.
        fmt : str
            Audio format returned by the server.
        streaming : bool
            Whether to request streaming response.
        **kwargs
            Additional parameters passed to :class:`ServeTTSRequest`.
        """
        if reference_audios is None:
            reference_audios = []
        if reference_texts is None:
            reference_texts = []

        references = [
            ServeReferenceAudio(audio=audio_to_bytes(a), text=read_ref_text(t))
            for a, t in zip(reference_audios, reference_texts)
        ]

        req = ServeTTSRequest(
            text=text,
            references=references,
            reference_id=reference_id,
            format=fmt,
            streaming=streaming,
            **kwargs,
        )
        data = ormsgpack.packb(req, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
        headers = {"Content-Type": "application/msgpack"}
        headers.update(self._headers())
        http_req = urlrequest.Request(
            self.tts_endpoint, data=data, headers=headers, method="POST"
        )
        if streaming:
            with urlrequest.urlopen(http_req) as resp:
                return resp.read()
        else:
            with urlrequest.urlopen(http_req) as resp:
                return resp.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceReel API client")
    parser.add_argument("--server-url", default=DEFAULT_API_URL, help="API base url")
    sub = parser.add_subparsers(dest="command", required=True)

    tts_p = sub.add_parser("tts", help="Synthesize text using /v1/tts")
    tts_p.add_argument("text")
    tts_p.add_argument("--output", default="output.wav")

    reg_p = sub.add_parser("register", help="Register a speaker")
    reg_p.add_argument("--name", required=True)
    reg_p.add_argument("--lang", required=True)
    reg_p.add_argument("--audio", required=True)
    reg_p.add_argument("--script", required=True)

    synth_p = sub.add_parser("synthesize", help="Multi-speaker synthesis")
    synth_p.add_argument("script_json", help="Path to script JSON file")

    job_p = sub.add_parser("job", help="Get job status")
    job_p.add_argument("job_id")

    list_p = sub.add_parser("list-speakers", help="List registered speakers")
    list_p.add_argument("--page", type=int)
    list_p.add_argument("--page-size", type=int)

    del_job_p = sub.add_parser("delete-job", help="Delete a job")
    del_job_p.add_argument("job_id")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = VoiceReelClient(api_url=args.server_url)

    if args.command == "tts":
        audio = client.tts(args.text)
        with open(args.output, "wb") as f:
            f.write(audio)
        print(f"Audio saved to {args.output}")
    elif args.command == "register":
        result = client.register_speaker(args.name, args.lang, args.audio, args.script)
        print(json.dumps(result, indent=2))
    elif args.command == "synthesize":
        with open(args.script_json) as f:
            script = json.load(f)
        result = client.synthesize(script)
        print(json.dumps(result, indent=2))
    elif args.command == "job":
        result = client.get_job(args.job_id)
        print(json.dumps(result, indent=2))
    elif args.command == "list-speakers":
        result = client.list_speakers(page=args.page, page_size=args.page_size)
        print(json.dumps(result, indent=2))
    elif args.command == "delete-job":
        client.delete_job(args.job_id)
        print(f"Job {args.job_id} deleted")


if __name__ == "__main__":
    main()
