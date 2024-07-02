import io
import wave
from typing import List

import av
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-time Chat Room</title>
</head>
<body>
    <h1>Real-time Chat Room</h1>
    <button id="start">Start Streaming</button>
    <button id="stop">Stop Streaming</button>
    <script type="module">
        import { MediaRecorder, register } from 'https://dev.jspm.io/npm:extendable-media-recorder';
        import { connect } from 'https://dev.jspm.io/npm:extendable-media-recorder-wav-encoder';
    
        await register(await connect());

        let socket;
        let mediaRecorder;
        let audioContext;

        function startStreaming() {
            initWebSocket();

            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            navigator.mediaDevices.getUserMedia({ audio: {
                channelCount: 1,  
                sampleRate: 44100,
                sampleSize: 16,
                echoCancellation: true,
                noiseSuppression: true
            } })
                .then(function (stream) {
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                    mediaRecorder.start(100);
                    mediaRecorder.addEventListener("dataavailable", function (event) {
                        socket.send(event.data);
                    });
                })
                .catch(function (err) {
                    console.error("Error accessing microphone:", err);
                });

                // Create a MediaSource
                const mediaSource = new MediaSource();
                const mediaStream = new MediaStream();

                // Create an HTMLVideoElement and attach the MediaSource to it
                const audioElement = document.createElement('audio');
                audioElement.src = URL.createObjectURL(mediaSource);
                audioElement.autoplay = true;
                document.body.appendChild(audioElement);

                mediaSource.addEventListener('sourceopen', function() {
                    const sourceBuffer = mediaSource.addSourceBuffer('audio/webm; codecs=opus');

                    socket.onmessage = function(event) {
                        const arrayBuffer = event.data;

                        sourceBuffer.appendBuffer(arrayBuffer);
                    };
                });
        }

        function stopStreaming() {
            mediaRecorder.stop();
        }

        function initWebSocket() {
            const is_wss = window.location.protocol === "https:";
            socket = new WebSocket(`${is_wss ? "wss" : "ws"}://${window.location.host}/ws`);
            socket.binaryType = 'arraybuffer';
        }

        document.getElementById("start").onclick = startStreaming;
        document.getElementById("stop").onclick = stopStreaming;
    </script>
</body>
</html>
"""


def encode_wav(data):
    sample_rate = 44100
    samples = np.frombuffer(data, dtype=np.int16)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())

    return buffer.getvalue()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: bytes, sender: WebSocket):
        for connection in self.active_connections:
            if connection == sender:
                #     print("Sending message to client", connection)
                await connection.send_bytes(message)


manager = ConnectionManager()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        buffer = io.BytesIO()
        container = None
        cur_pos = 0
        total_size = 0

        while True:
            data = await websocket.receive_bytes()
            # data = encode_wav(data)
            # if len(data) == 1:
            #     print(f"len(data): {len(data)}, data: {data}")
            # if len(data) > 1:
            #     data = b'\x1a' + data
            #     with open("output.webm", "wb") as f:
            #         f.write(data)
            #     exit()
            # print(f"len(data): {len(data)}")

            # print("Received data:", data)
            # Save as webm file and exit
            # with open("output.wav", "wb") as f:
            #     f.write(encode_wav(data))

            buffer.write(data)
            buffer.seek(cur_pos)
            total_size += len(data)

            if not container and total_size > 1000:
                container = av.open(buffer, "r", format="webm")
                print(container)
            elif container:
                for packet in container.decode(video=0):
                    if packet.size == 0:
                        continue

                    cur_pos += packet.size
                    for frame in packet.decode():
                        print(frame.to_ndarray().shape)

            await manager.broadcast(data, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
