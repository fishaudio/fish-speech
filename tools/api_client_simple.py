import argparse
import base64
import wave

import ormsgpack
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play
from pydantic import BaseModel, Field, conint, model_validator
from typing_extensions import Annotated
from typing import Literal
from pathlib import Path

def audio_to_bytes(file_path):
    if not file_path or not Path(file_path).exists():
        return None
    with open(file_path, "rb") as wav_file:
        wav = wav_file.read()
    return wav


def read_ref_text(ref_text):
    path = Path(ref_text)
    if path.exists() and path.is_file():
        with path.open("r", encoding="utf-8") as file:
            return file.read()
    return ref_text


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if (
            isinstance(audio, str) and len(audio) > 255
        ):  # Check if audio is a string (Base64)
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception as e:
                # If the audio is not a valid base64 string, we will just ignore it and let the server handle it
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # not usually used below
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.1
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.8

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True



def parse_args():

    parser = argparse.ArgumentParser(
        description="Send a WAV file and text to a server and receive synthesized audio.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default="http://127.0.0.1:8080/v1/tts",
        help="URL of the server",
    )
    parser.add_argument(
        "--text", "-t", type=str, required=True, help="Text to be synthesized"
    )
    parser.add_argument(
        "--reference_id",
        "-id",
        type=str,
        default=None,
        help="ID of the reference model to be used for the speech\n(Local: name of folder containing audios and files)",
    )
    parser.add_argument(
        "--reference_audio",
        "-ra",
        type=str,
        nargs="+",
        default=None,
        help="Path to the audio file",
    )
    parser.add_argument(
        "--reference_text",
        "-rt",
        type=str,
        nargs="+",
        default=None,
        help="Reference text for voice synthesis",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="generated_audio",
        help="Output audio file name",
    )
    parser.add_argument(
        "--play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to play audio after receiving data",
    )
    parser.add_argument(
        "--format", type=str, choices=["wav", "mp3", "flac"], default="wav"
    )
    parser.add_argument(
        "--latency",
        type=str,
        default="normal",
        choices=["normal", "balanced"],
        help="Used in api.fish.audio/v1/tts",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate. \n0 means no limit.",
    )
    parser.add_argument(
        "--chunk_length", type=int, default=300, help="Chunk length for synthesis"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8, help="Top-p sampling for synthesis"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for synthesis",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling"
    )

    parser.add_argument(
        "--streaming", type=bool, default=False, help="Enable streaming response"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels"
    )
    parser.add_argument("--rate", type=int, default=44100, help="Sample rate for audio")
    parser.add_argument(
        "--use_memory_cache",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Cache encoded references codes in memory.\n",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="`None` means randomized inference, otherwise deterministic.\n"
        "It can't be used for fixing a timbre.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="YOUR_API_KEY",
        help="API key for authentication",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    idstr: str | None = args.reference_id
    # priority: ref_id > [{text, audio},...]
    if idstr is None:
        ref_audios = args.reference_audio
        ref_texts = args.reference_text
        if ref_audios is None:
            byte_audios = []
        else:
            byte_audios = [audio_to_bytes(ref_audio) for ref_audio in ref_audios]
        if ref_texts is None:
            ref_texts = []
        else:
            ref_texts = [read_ref_text(ref_text) for ref_text in ref_texts]
    else:
        byte_audios = []
        ref_texts = []
        pass  # in api.py

    data = {
        "text": args.text,
        "references": [
            ServeReferenceAudio(
                audio=ref_audio if ref_audio is not None else b"", text=ref_text
            )
            for ref_text, ref_audio in zip(ref_texts, byte_audios)
        ],
        "reference_id": idstr,
        "format": args.format,
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "streaming": args.streaming,
        "use_memory_cache": args.use_memory_cache,
        "seed": args.seed,
    }

    pydantic_data = ServeTTSRequest(**data)

    response = requests.post(
        args.url,
        data=ormsgpack.packb(pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        stream=args.streaming,
        headers={
            "authorization": f"Bearer {args.api_key}",
            "content-type": "application/msgpack",
        },
    )

    if response.status_code == 200:
        if args.streaming:
            p = pyaudio.PyAudio()
            audio_format = pyaudio.paInt16  # Assuming 16-bit PCM format
            stream = p.open(
                format=audio_format, channels=args.channels, rate=args.rate, output=True
            )

            wf = wave.open(f"{args.output}.wav", "wb")
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
            audio_path = f"{args.output}.{args.format}"
            with open(audio_path, "wb") as audio_file:
                audio_file.write(audio_content)

            audio = AudioSegment.from_file(audio_path, format=args.format)
            if args.play:
                play(audio)
            print(f"Audio has been saved to '{audio_path}'.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.json())
