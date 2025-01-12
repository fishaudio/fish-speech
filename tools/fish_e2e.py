import base64
import ctypes
import io
import json
import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Union

import httpx
import numpy as np
import ormsgpack
import soundfile as sf

from fish_speech.utils.schema import (
    ServeChatRequest,
    ServeMessage,
    ServeTextPart,
    ServeVQGANDecodeRequest,
    ServeVQGANEncodeRequest,
    ServeVQPart,
)


class CustomAudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        if len(data) < num_channels * samples_per_channel * ctypes.sizeof(
            ctypes.c_int16
        ):
            raise ValueError(
                "data length must be >= num_channels * samples_per_channel * sizeof(int16)"
            )

        self._data = bytearray(data)
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._samples_per_channel = samples_per_channel

    @property
    def data(self):
        return memoryview(self._data).cast("h")

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def num_channels(self):
        return self._num_channels

    @property
    def samples_per_channel(self):
        return self._samples_per_channel

    @property
    def duration(self):
        return self.samples_per_channel / self.sample_rate

    def __repr__(self):
        return (
            f"CustomAudioFrame(sample_rate={self.sample_rate}, "
            f"num_channels={self.num_channels}, "
            f"samples_per_channel={self.samples_per_channel}, "
            f"duration={self.duration:.3f})"
        )


class FishE2EEventType(Enum):
    SPEECH_SEGMENT = 1
    TEXT_SEGMENT = 2
    END_OF_TEXT = 3
    END_OF_SPEECH = 4
    ASR_RESULT = 5
    USER_CODES = 6


@dataclass
class FishE2EEvent:
    type: FishE2EEventType
    frame: np.ndarray = None
    text: str = None
    vq_codes: list[list[int]] = None


client = httpx.AsyncClient(
    timeout=None,
    limits=httpx.Limits(
        max_connections=None,
        max_keepalive_connections=None,
        keepalive_expiry=None,
    ),
)


class FishE2EAgent:
    def __init__(self):
        self.llm_url = "http://localhost:8080/v1/chat"
        self.vqgan_url = "http://localhost:8080"
        self.client = httpx.AsyncClient(timeout=None)

    async def get_codes(self, audio_data, sample_rate):
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_data, sample_rate, format="WAV")
        audio_buffer.seek(0)
        # Step 1: Encode audio using VQGAN
        encode_request = ServeVQGANEncodeRequest(audios=[audio_buffer.read()])
        encode_request_bytes = ormsgpack.packb(
            encode_request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC
        )
        encode_response = await self.client.post(
            f"{self.vqgan_url}/v1/vqgan/encode",
            data=encode_request_bytes,
            headers={"Content-Type": "application/msgpack"},
        )
        encode_response_data = ormsgpack.unpackb(encode_response.content)
        codes = encode_response_data["tokens"][0]
        return codes

    async def stream(
        self,
        system_audio_data: np.ndarray | None,
        user_audio_data: np.ndarray | None,
        sample_rate: int,
        num_channels: int,
        chat_ctx: dict | None = None,
    ) -> AsyncGenerator[bytes, None]:

        if system_audio_data is not None:
            sys_codes = await self.get_codes(system_audio_data, sample_rate)
        else:
            sys_codes = None
        if user_audio_data is not None:
            user_codes = await self.get_codes(user_audio_data, sample_rate)
        # Step 2: Prepare LLM request
        if chat_ctx is None:
            sys_parts = [
                ServeTextPart(
                    text='您是由 Fish Audio 设计的语音助手，提供端到端的语音交互，实现无缝用户体验。首先转录用户的语音，然后使用以下格式回答："Question: [用户语音]\n\nAnswer: [你的回答]\n"。'
                ),
            ]
            if system_audio_data is not None:
                sys_parts.append(ServeVQPart(codes=sys_codes))
            chat_ctx = {
                "messages": [
                    ServeMessage(
                        role="system",
                        parts=sys_parts,
                    ),
                ],
            }
        else:
            if chat_ctx["added_sysaudio"] is False and sys_codes:
                chat_ctx["added_sysaudio"] = True
                chat_ctx["messages"][0].parts.append(ServeVQPart(codes=sys_codes))

        prev_messages = chat_ctx["messages"].copy()
        if user_audio_data is not None:
            yield FishE2EEvent(
                type=FishE2EEventType.USER_CODES,
                vq_codes=user_codes,
            )
        else:
            user_codes = None

        request = ServeChatRequest(
            messages=prev_messages
            + (
                [
                    ServeMessage(
                        role="user",
                        parts=[ServeVQPart(codes=user_codes)],
                    )
                ]
                if user_codes
                else []
            ),
            streaming=True,
            num_samples=1,
        )

        # Step 3: Stream LLM response and decode audio
        buffer = b""
        vq_codes = []
        current_vq = False

        async def decode_send():
            nonlocal current_vq
            nonlocal vq_codes

            data = np.concatenate(vq_codes, axis=1).tolist()
            # Decode VQ codes to audio
            decode_request = ServeVQGANDecodeRequest(tokens=[data])
            decode_response = await self.client.post(
                f"{self.vqgan_url}/v1/vqgan/decode",
                data=ormsgpack.packb(
                    decode_request,
                    option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
                ),
                headers={"Content-Type": "application/msgpack"},
            )
            decode_data = ormsgpack.unpackb(decode_response.content)

            # Convert float16 audio data to int16
            audio_data = np.frombuffer(decode_data["audios"][0], dtype=np.float16)
            audio_data = (audio_data * 32768).astype(np.int16).tobytes()

            audio_frame = CustomAudioFrame(
                data=audio_data,
                samples_per_channel=len(audio_data) // 2,
                sample_rate=44100,
                num_channels=1,
            )
            yield FishE2EEvent(
                type=FishE2EEventType.SPEECH_SEGMENT,
                frame=audio_frame,
                vq_codes=data,
            )

            current_vq = False
            vq_codes = []

        async with self.client.stream(
            "POST",
            self.llm_url,
            data=ormsgpack.packb(request, option=ormsgpack.OPT_SERIALIZE_PYDANTIC),
            headers={"Content-Type": "application/msgpack"},
        ) as response:

            async for chunk in response.aiter_bytes():
                buffer += chunk

                while len(buffer) >= 4:
                    read_length = struct.unpack("I", buffer[:4])[0]
                    if len(buffer) < 4 + read_length:
                        break

                    body = buffer[4 : 4 + read_length]
                    buffer = buffer[4 + read_length :]
                    data = ormsgpack.unpackb(body)

                    if data["delta"] and data["delta"]["part"]:
                        if current_vq and data["delta"]["part"]["type"] == "text":
                            async for event in decode_send():
                                yield event
                        if data["delta"]["part"]["type"] == "text":
                            yield FishE2EEvent(
                                type=FishE2EEventType.TEXT_SEGMENT,
                                text=data["delta"]["part"]["text"],
                            )
                        elif data["delta"]["part"]["type"] == "vq":
                            vq_codes.append(np.array(data["delta"]["part"]["codes"]))
                            current_vq = True

        if current_vq and vq_codes:
            async for event in decode_send():
                yield event

        yield FishE2EEvent(type=FishE2EEventType.END_OF_TEXT)
        yield FishE2EEvent(type=FishE2EEventType.END_OF_SPEECH)


# Example usage:
async def main():
    import torchaudio

    agent = FishE2EAgent()

    # Replace this with actual audio data loading
    with open("uz_story_en.m4a", "rb") as f:
        audio_data = f.read()

    audio_data, sample_rate = torchaudio.load("uz_story_en.m4a")
    audio_data = (audio_data.numpy() * 32768).astype(np.int16)

    stream = agent.stream(audio_data, sample_rate, 1)
    if os.path.exists("audio_segment.wav"):
        os.remove("audio_segment.wav")

    async for event in stream:
        if event.type == FishE2EEventType.SPEECH_SEGMENT:
            # Handle speech segment (e.g., play audio or save to file)
            with open("audio_segment.wav", "ab+") as f:
                f.write(event.frame.data)
        elif event.type == FishE2EEventType.ASR_RESULT:
            print(event.text, flush=True)
        elif event.type == FishE2EEventType.TEXT_SEGMENT:
            print(event.text, flush=True, end="")
        elif event.type == FishE2EEventType.END_OF_TEXT:
            print("\nEnd of text reached.")
        elif event.type == FishE2EEventType.END_OF_SPEECH:
            print("End of speech reached.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
