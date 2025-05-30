from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import ormsgpack
import requests

from fish_speech.utils.file import audio_to_bytes, read_ref_text
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

DEFAULT_API_URL = "http://127.0.0.1:8080"


@dataclass
class VoiceReelClient:
    """Client for interacting with the Fish Speech HTTP API."""

    api_url: str = DEFAULT_API_URL

    @property
    def tts_endpoint(self) -> str:
        return f"{self.api_url.rstrip('/')}/v1/tts"

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
        response = requests.post(
            self.tts_endpoint,
            data=ormsgpack.packb(req, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
            stream=streaming,
            headers={"content-type": "application/msgpack"},
        )
        response.raise_for_status()
        return response.content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoiceReel TTS client")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--server-url", default=DEFAULT_API_URL, help="Fish Speech API base url"
    )
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = VoiceReelClient(api_url=args.server_url)
    audio = client.tts(args.text)
    with open(args.output, "wb") as f:
        f.write(audio)
    print(f"Audio saved to {args.output}")


if __name__ == "__main__":
    main()
