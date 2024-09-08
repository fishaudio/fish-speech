import httpx
import ormsgpack

from tools.commons import ServeReferenceAudio, ServeTTSRequest

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
