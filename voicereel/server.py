from __future__ import annotations

import json
import os
import queue
import sqlite3
import tempfile
import threading
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

from .caption import export_captions


class VoiceReelServer:
    """Minimal HTTP server skeleton for VoiceReel."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.job_queue: queue.Queue = queue.Queue()
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()
        handler = self._make_handler()
        self.httpd = HTTPServer((self.host, self.port), handler)
        self.thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Internal setup methods
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        cur = self.db.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                lang TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                type TEXT,
                status TEXT,
                audio_url TEXT,
                caption_path TEXT,
                caption_format TEXT
            )
            """
        )
        self.db.commit()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                elif self.path.startswith("/v1/jobs/"):
                    job_id = self.path.rsplit("/", 1)[-1]
                    cur = server.db.cursor()
                    cur.execute(
                        "SELECT id, type, status, audio_url, caption_path, caption_format FROM jobs WHERE id=?",
                        (job_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        body = json.dumps(
                            {
                                "id": row[0],
                                "type": row[1],
                                "status": row[2],
                                "audio_url": row[3],
                                "caption_url": row[4],
                                "caption_format": row[5],
                            }
                        ).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self.send_response(404)
                        self.end_headers()
                elif self.path.startswith("/v1/speakers"):
                    query = urllib.parse.urlparse(self.path).query
                    params = urllib.parse.parse_qs(query)
                    page = int(params.get("page", ["1"])[0])
                    page_size = int(params.get("page_size", ["10"])[0])
                    offset = (page - 1) * page_size
                    cur = server.db.cursor()
                    cur.execute(
                        "SELECT id, name, lang FROM speakers LIMIT ? OFFSET ?",
                        (page_size, offset),
                    )
                    speakers = [
                        {"id": row[0], "name": row[1], "lang": row[2]}
                        for row in cur.fetchall()
                    ]
                    body = json.dumps({"speakers": speakers}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_DELETE(self):
                if self.path.startswith("/v1/jobs/"):
                    job_id = self.path.rsplit("/", 1)[-1]
                    cur = server.db.cursor()
                    cur.execute(
                        "SELECT audio_url, caption_path FROM jobs WHERE id=?", (job_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        self.send_response(404)
                        self.end_headers()
                        return
                    audio_url, caption_path = row
                    for path in (audio_url, caption_path):
                        if path:
                            try:
                                os.remove(path)
                            except FileNotFoundError:
                                pass
                    cur.execute("DELETE FROM jobs WHERE id=?", (job_id,))
                    server.db.commit()
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == "/v1/speakers":
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length)
                    try:
                        payload = json.loads(raw.decode()) if raw else {}
                    except json.JSONDecodeError:
                        self.send_response(400)
                        self.end_headers()
                        return

                    duration = float(payload.get("duration", 0))
                    script = payload.get("script", "")
                    if duration < 30:
                        self.send_response(422)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error":"INSUFFICIENT_REF"}')
                        return
                    if not script:
                        self.send_response(400)
                        self.end_headers()
                        return

                    cur = server.db.cursor()
                    cur.execute(
                        "INSERT INTO speakers (name, lang) VALUES (?, ?)",
                        ("unknown", "en"),
                    )
                    speaker_id = cur.lastrowid
                    job_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO jobs (id, type, status, audio_url, caption_path, caption_format) VALUES (?, ?, ?, ?, ?, ?)",
                        (job_id, "register_speaker", "succeeded", None, None, None),
                    )
                    server.db.commit()
                    server.job_queue.put(("register_speaker", speaker_id))
                    body = json.dumps(
                        {
                            "job_id": job_id,
                            "speaker_id": speaker_id,
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/v1/synthesize":
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length)
                    try:
                        payload = json.loads(raw.decode()) if raw else {}
                    except json.JSONDecodeError:
                        self.send_response(400)
                        self.end_headers()
                        return

                    script = payload.get("script")
                    caption_format = payload.get("caption_format", "json")
                    if not isinstance(script, list) or not script:
                        self.send_response(400)
                        self.end_headers()
                        return

                    job_id = str(uuid.uuid4())
                    cur = server.db.cursor()
                    audio_path = os.path.join(tempfile.gettempdir(), f"{job_id}.wav")
                    with open(audio_path, "wb") as f:
                        f.write(b"FAKE")

                    caption_units = [
                        {
                            "start": i * 0.5,
                            "end": i * 0.5 + 0.5,
                            "speaker": seg.get("speaker_id"),
                            "text": seg.get("text", ""),
                        }
                        for i, seg in enumerate(script)
                    ]
                    caption_text = export_captions(caption_units, caption_format)
                    caption_path = os.path.join(
                        tempfile.gettempdir(), f"{job_id}.{caption_format}"
                    )
                    with open(caption_path, "w", encoding="utf-8") as f:
                        f.write(caption_text)

                    cur.execute(
                        "INSERT INTO jobs (id, type, status, audio_url, caption_path, caption_format) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            job_id,
                            "synthesize",
                            "succeeded",
                            audio_path,
                            caption_path,
                            caption_format,
                        ),
                    )
                    server.db.commit()
                    server.job_queue.put(("synthesize", job_id))
                    body = json.dumps({"job_id": job_id}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args) -> None:
                # Suppress default logging
                return

        return Handler

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    @property
    def address(self) -> tuple[str, int]:
        return self.httpd.server_address

    def start(self) -> None:
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> None:
        if self.thread:
            self.httpd.shutdown()
            self.thread.join()
            self.thread = None
