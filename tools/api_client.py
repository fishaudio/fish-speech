import argparse
import base64
import wave

import ormsgpack
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play

from fish_speech.utils.file import audio_to_bytes, read_ref_text
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


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
