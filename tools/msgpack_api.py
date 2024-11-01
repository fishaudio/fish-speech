import os
from argparse import ArgumentParser
from pathlib import Path

import httpx
import ormsgpack

from tools.schema import ServeReferenceAudio, ServeTTSRequest

api_key = os.environ.get("FISH_API_KEY", "YOUR_API_KEY")


def audio_request():
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

    api_key = os.environ.get("FISH_API_KEY", "YOUR_API_KEY")

    with (
        httpx.Client() as client,
        open("hello.wav", "wb") as f,
    ):
        with client.stream(
            "POST",
            "http://127.0.0.1:8080/v1/tts",
            content=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/msgpack",
            },
            timeout=None,
        ) as response:
            for chunk in response.iter_bytes():
                f.write(chunk)


def asr_request(audio_path: Path):

    # Read the audio file
    with open(
        str(audio_path),
        "rb",
    ) as audio_file:
        audio_data = audio_file.read()

    # Prepare the request data
    request_data = {
        "audio": audio_data,
        "language": "en",  # Optional: specify the language
        "ignore_timestamps": False,  # Optional: set to True to ignore precise timestamps
    }

    # Send the request
    with httpx.Client() as client:
        response = client.post(
            "https://api.fish.audio/v1/asr",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/msgpack",
            },
            content=ormsgpack.packb(request_data),
        )

    # Parse the response
    result = response.json()

    print(f"Transcribed text: {result['text']}")
    print(f"Audio duration: {result['duration']} seconds")

    for segment in result["segments"]:
        print(f"Segment: {segment['text']}")
        print(f"Start time: {segment['start']}, End time: {segment['end']}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--audio_path", type=Path, default="audio/ref/trump.mp3")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    asr_request(args.audio_path)
