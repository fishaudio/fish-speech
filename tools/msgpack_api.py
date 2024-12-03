import os
from argparse import ArgumentParser
from pathlib import Path

import httpx
import ormsgpack

api_key = os.environ.get("FISH_API_KEY", "YOUR_API_KEY")


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
