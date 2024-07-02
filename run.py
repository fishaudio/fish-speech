import audioop
import base64

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from loguru import logger
from stream_service import FishAgentPipeline

app = FastAPI()


@app.post("/incoming")
async def handle_incoming():
    xml = """<Response>
    <Connect>
    <Stream url="wss://2427-24-4-31-213.ngrok-free.app/connection" />
    </Connect>
</Response>"""

    logger.info("Incoming call received")
    return Response(media_type="text/xml", content=xml)


async def send_audio(ws, audio, stream_sid=""):
    await ws.send_json(
        {
            "streamSid": stream_sid,
            "event": "media",
            "media": {
                "payload": audio,
            },
        }
    )


def decode_mu_law(data):
    samples = audioop.ulaw2lin(data, 2)
    samples = np.frombuffer(samples, dtype=np.int16)
    samples = samples.astype(np.float32) / 32768.0

    return samples


def encode_mu_law(data):
    samples = np.clip(data, -1.0, 1.0)
    samples = (samples * 32768).astype(np.int16)
    samples = audioop.lin2ulaw(samples.tobytes(), 2)

    return samples


is_working = False


@app.websocket("/connection")
async def handle_connection(websocket: WebSocket):
    global is_working

    await websocket.accept()
    logger.info("Connection established")
    stream_sid = None
    call_sid = None

    if is_working:
        logger.info("Already working, closing connection")
        await websocket.close()
        return

    is_working = True
    pipe.reset()

    while True:
        data = await websocket.receive_json()
        if data["event"] == "connected":
            logger.info("Connected message received")
        elif data["event"] == "start":
            stream_sid = data["start"]["streamSid"]
            call_sid = data["start"]["callSid"]
            logger.info(f"Start media streaming: {stream_sid} - {call_sid}")
        elif data["event"] == "media":
            payload = data["media"]["payload"]
            chunk = base64.b64decode(payload)
            samples = decode_mu_law(chunk)
            for i in pipe.add_chunk(samples, sr=8000):
                await send_audio(
                    websocket, base64.b64encode(encode_mu_law(i)).decode(), stream_sid
                )
        elif data["event"] == "closed":
            logger.info("Connection closed")
            await websocket.close()
            break
        elif data["event"] == "stop":
            logger.info("Stop media streaming")
            await websocket.close()
            break
        else:
            logger.info(f"Unknown event: {data}")

    is_working = False


if __name__ == "__main__":
    import uvicorn

    pipe = FishAgentPipeline()
    pipe.warmup()

    logger.info("Starting server")
    uvicorn.run(app, host="localhost", port=5000)
