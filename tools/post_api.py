import argparse
import base64
import json
import wave
from pathlib import Path

import pyaudio
import requests


def wav_to_base64(file_path):
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as wav_file:
        wav_content = wav_file.read()
        base64_encoded = base64.b64encode(wav_content)
        return base64_encoded.decode("utf-8")


def read_ref_text(ref_text):
    path = Path(ref_text)
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as file:
            return file.read()
    return ref_text


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
        default="http://127.0.0.1:8080/v1/invoke",
        help="URL of the server",
    )
    parser.add_argument(
        "--text", "-t", type=str, required=True, help="Text to be synthesized"
    )
    parser.add_argument(
        "--reference_audio",
        "-ra",
        type=str,
        default=None,
        help="Path to the WAV file",
    )
    parser.add_argument(
        "--reference_text",
        "-rt",
        type=str,
        default=None,
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
    parser.add_argument("--emotion", type=str, default=None, help="Speaker's Emotion")
    parser.add_argument("--format", type=str, default="wav", help="Audio format")
    parser.add_argument(
        "--streaming", type=bool, default=False, help="Enable streaming response"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels"
    )
    parser.add_argument("--rate", type=int, default=44100, help="Sample rate for audio")

    args = parser.parse_args()

    base64_audio = wav_to_base64(args.reference_audio)

    ref_text = args.reference_text
    if ref_text:
        ref_text = read_ref_text(ref_text)

    data = {
        "text": args.text,
        "reference_text": ref_text,
        "reference_audio": base64_audio,
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "speaker": args.speaker,
        "emotion": args.emotion,
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

            wf = wave.open("generated_audio.wav", "wb")
            wf.setnchannels(args.channels)
            wf.setsampwidth(p.get_sample_size(audio_format))
            wf.setframerate(args.rate)

            stream_stopped_flag = False

            try:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        stream.write(chunk)
                        wf.writeframesraw(chunk)
                    else:
                        if not stream_stopped_flag:
                            stream.stop_stream()
                            stream_stopped_flag = True
            finally:
                stream.close()
                p.terminate()
                wf.close()
        else:
            audio_content = response.content

            with open("generated_audio.wav", "wb") as audio_file:
                audio_file.write(audio_content)

            play_audio(audio_content, audio_format, args.channels, args.rate)
            print("Audio has been saved to 'generated_audio.wav'.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())
