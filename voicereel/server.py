from __future__ import annotations

import json
import queue
import sqlite3
import threading
import uuid
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer


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
                    server.db.commit()
                    job_id = str(uuid.uuid4())
                    server.job_queue.put(("register_speaker", speaker_id))
                    body = json.dumps({
                        "job_id": job_id,
                        "speaker_id": speaker_id,
                    }).encode()
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
