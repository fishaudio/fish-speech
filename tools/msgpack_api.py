from typing import Annotated, AsyncGenerator, Literal, Optional

import httpx
import ormsgpack
from pydantic import AfterValidator, BaseModel, Field, conint


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    mp3_bitrate: Literal[64, 128, 192] = 128
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    mp3_bitrate: Optional[int] = 64
    opus_bitrate: Optional[int] = -1000
    # Balance mode will reduce latency to 300ms, but may decrease stability
    latency: Literal["normal", "balanced"] = "normal"
    # not usually used below
    streaming: bool = False
    emotion: Optional[str] = None
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7


# priority: ref_id > references
request = ServeTTSRequest(
    text="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
    # reference_id="114514",
    references=[
        ServeReferenceAudio(
            audio=open("lengyue.wav", "rb").read(),
            text=open("lengyue.lab", "r", encoding="utf-8").read(),
        )
    ],
    streaming=True,
)

with (
    httpx.Client() as client,
    open("hello.wav", "wb") as f,
):
    with client.stream(
        "POST",
        "http://127.0.0.1:8080/v1/tts",
        content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        headers={
            "authorization": "Bearer YOUR_API_KEY",
            "content-type": "application/msgpack",
        },
        timeout=None,
    ) as response:
        for chunk in response.iter_bytes():
            f.write(chunk)
