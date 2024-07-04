import argparse
import base64
import json

import pyaudio
import requests


def play_audio(audio_content, format, channels, rate):
    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, output=True)
    stream.write(audio_content)
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Send a WAV file and text to a server and receive synthesized audio."
    )

    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default="http://127.0.0.1:8000/v1/invoke",
        help="URL of the server",
    )
    parser.add_argument(
        "--text", "-t", type=str, required=True, help="Text to be synthesized"
    )
    parser.add_argument(
        "--reference_audio",
        "-ra",
        type=str,
        required=False,
        help="Path to the WAV file",
    )
    parser.add_argument(
        "--reference_text",
        "-rt",
        type=str,
        required=False,
        help="Reference text for voice synthesis",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--chunk_length", type=int, default=100, help="Chunk length for synthesis"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.7, help="Top-p sampling for synthesis"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty for synthesis",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--speaker", type=str, default=None, help="Speaker ID for voice synthesis"
    )
    parser.add_argument("--format", type=str, default="wav", help="Audio format")
    parser.add_argument(
        "--streaming", type=bool, default=False, help="Enable streaming response"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels"
    )
    parser.add_argument("--rate", type=int, default=44100, help="Sample rate for audio")

    args = parser.parse_args()

    data = {
        "text": args.text,
        "reference_text": args.reference_text,
        "reference_audio": args.reference_audio,
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "speaker": args.speaker,
        "format": args.format,
        "streaming": args.streaming,
    }

    response = requests.post(args.url, json=data, stream=args.streaming)

    audio_format = pyaudio.paInt16  # Assuming 16-bit PCM format

    if response.status_code == 200:
        if args.streaming:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=audio_format, channels=args.channels, rate=args.rate, output=True
            )
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    stream.write(chunk)
            stream.stop_stream()
            stream.close()
            p.terminate()
        else:
            audio_content = response.content

            with open("generated_audio.wav", "wb") as audio_file:
                audio_file.write(audio_content)

            play_audio(audio_content, audio_format, args.channels, args.rate)
            print("Audio has been saved to 'generated_audio.wav'.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())
