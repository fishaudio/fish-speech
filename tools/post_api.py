import argparse
import base64
import wave

import ormsgpack
import pyaudio
import requests
from pydub import AudioSegment
from pydub.playback import play

from tools.commons import ServeReferenceAudio, ServeTTSRequest
from tools.file import audio_to_bytes, read_ref_text


def parse_args():

    parser = argparse.ArgumentParser(
        description="Send a WAV file and text to a server and receive synthesized audio."
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
        help="ID of the reference model o be used for the speech",
    )
    parser.add_argument(
        "--reference_audio",
        "-ra",
        type=str,
        nargs="+",
        default=None,
        help="Path to the WAV file",
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
        type=bool,
        default=True,
        help="Whether to play audio after receiving data",
    )
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument(
        "--format", type=str, choices=["wav", "mp3", "flac"], default="wav"
    )
    parser.add_argument("--mp3_bitrate", type=int, default=64)
    parser.add_argument("--opus_bitrate", type=int, default=-1000)
    parser.add_argument("--latency", type=str, default="normal", help="延迟选项")
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
    parser.add_argument(
        "--streaming", type=bool, default=False, help="Enable streaming response"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels"
    )
    parser.add_argument("--rate", type=int, default=44100, help="Sample rate for audio")

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
            ServeReferenceAudio(audio=ref_audio, text=ref_text)
            for ref_text, ref_audio in zip(ref_texts, byte_audios)
        ],
        "reference_id": idstr,
        "normalize": args.normalize,
        "format": args.format,
        "mp3_bitrate": args.mp3_bitrate,
        "opus_bitrate": args.opus_bitrate,
        "max_new_tokens": args.max_new_tokens,
        "chunk_length": args.chunk_length,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
        "speaker": args.speaker,
        "emotion": args.emotion,
        "streaming": args.streaming,
    }

    pydantic_data = ServeTTSRequest(**data)

    response = requests.post(
        args.url,
        data=ormsgpack.packb(pydantic_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
        stream=args.streaming,
        headers={
            "authorization": "Bearer YOUR_API_KEY",
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
