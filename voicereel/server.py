from __future__ import annotations

import json
import os
import queue
import sqlite3
import tempfile
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from .caption import export_captions

# Try to import Celery tasks
try:
    from .tasks import register_speaker as celery_register_speaker
    from .tasks import synthesize as celery_synthesize

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False


class VoiceReelServer:
    """Minimal HTTP server skeleton for VoiceReel."""

    PRESIGNED_TTL = 15 * 60  # 15 minutes

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        dsn: str | None = None,
        api_key: str | None = None,
        hmac_secret: str | None = None,
        redis_url: str | None = None,
        use_celery: bool = None,
    ):
        self.host = host
        self.port = port
        self.job_queue: queue.Queue = queue.Queue()
        self.api_key = api_key or os.getenv("VR_API_KEY")
        self.hmac_secret = hmac_secret or os.getenv("VR_HMAC_SECRET")
        self.redis_url = redis_url or os.getenv("VR_REDIS_URL")

        # Determine whether to use Celery
        if use_celery is None:
            # Auto-detect: use Celery if available and Redis URL is configured
            self.use_celery = CELERY_AVAILABLE and bool(self.redis_url)
        else:
            self.use_celery = use_celery and CELERY_AVAILABLE

        dsn = dsn or os.getenv("VR_DSN", ":memory:")
        self.db = sqlite3.connect(dsn, check_same_thread=False)
        self._init_db()
        handler = self._make_handler()
        self.httpd = HTTPServer((self.host, self.port), handler)
        self.thread: threading.Thread | None = None
        self.worker: threading.Thread | None = None
        self._stop_event = threading.Event()

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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                ts TEXT,
                length REAL
            )
            """
        )
        self.db.commit()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code: int, body: dict) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def _error(self, code: int, name: str) -> None:
                self._json(code, {"error": name})

            def _require_key(self, body: bytes = b"") -> bool:
                if server.api_key:
                    if self.headers.get("X-VR-APIKEY") != server.api_key:
                        self._error(401, "UNAUTHORIZED")
                        return False
                    if server.hmac_secret:
                        import hashlib
                        import hmac

                        expected = hmac.new(
                            server.hmac_secret.encode(), body, hashlib.sha256
                        ).hexdigest()
                        if self.headers.get("X-VR-SIGN") != expected:
                            self._error(401, "UNAUTHORIZED")
                            return False
                return True

            def do_GET(self):
                if not self._require_key():
                    return
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
                        audio = server._presign_path(row[3]) if row[3] else None
                        caption = server._presign_path(row[4]) if row[4] else None
                        body = json.dumps(
                            {
                                "id": row[0],
                                "type": row[1],
                                "status": row[2],
                                "audio_url": audio,
                                "caption_url": caption,
                                "caption_format": row[5],
                            }
                        ).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self._error(404, "NOT_FOUND")
                elif self.path.startswith("/v1/speakers/"):
                    speaker_id = self.path.rsplit("/", 1)[-1]
                    cur = server.db.cursor()
                    cur.execute(
                        "SELECT id, name, lang FROM speakers WHERE id=?",
                        (speaker_id,),
                    )
                    row = cur.fetchone()
                    if row:
                        body = json.dumps(
                            {"id": row[0], "name": row[1], "lang": row[2]}
                        ).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(body)
                    else:
                        self._error(404, "NOT_FOUND")
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
                    self._error(404, "NOT_FOUND")

            def do_DELETE(self):
                if not self._require_key():
                    return
                if self.path.startswith("/v1/jobs/"):
                    job_id = self.path.rsplit("/", 1)[-1]
                    cur = server.db.cursor()
                    cur.execute(
                        "SELECT audio_url, caption_path FROM jobs WHERE id=?", (job_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        self._error(404, "NOT_FOUND")
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
                    self._error(404, "NOT_FOUND")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                if length > 30 * 1024 * 1024:
                    self._error(413, "PAYLOAD_TOO_LARGE")
                    _ = self.rfile.read(length)
                    return
                raw = self.rfile.read(length)
                if not self._require_key(raw):
                    return
                if self.path == "/v1/speakers":
                    try:
                        payload = json.loads(raw.decode()) if raw else {}
                    except json.JSONDecodeError:
                        self._error(400, "INVALID_INPUT")
                        return

                    duration = float(payload.get("duration", 0))
                    script = payload.get("script", "")
                    name = payload.get("name", "unknown")
                    lang = payload.get("lang", "en")
                    allowed_langs = {"en", "ko", "ja"}
                    if lang not in allowed_langs:
                        self._error(400, "INVALID_INPUT")
                        return
                    if duration < 30:
                        self._error(422, "INSUFFICIENT_REF")
                        return
                    if not script:
                        self._error(400, "INVALID_INPUT")
                        return

                    cur = server.db.cursor()
                    cur.execute(
                        "INSERT INTO speakers (name, lang) VALUES (?, ?)",
                        (name, lang),
                    )
                    speaker_id = cur.lastrowid
                    job_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO jobs (id, type, status, audio_url, caption_path, caption_format) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            job_id,
                            "register_speaker",
                            "pending",
                            None,
                            None,
                            None,
                        ),
                    )
                    server.db.commit()

                    # Queue the task
                    if server.use_celery:
                        # Use Celery for async processing
                        celery_register_speaker.delay(
                            job_id,
                            speaker_id,
                            audio_path="dummy.wav",
                            script=script,
                            lang=lang,
                        )
                    else:
                        # Use in-memory queue
                        server.job_queue.put(("register_speaker", job_id, speaker_id))

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
                    try:
                        payload = json.loads(raw.decode()) if raw else {}
                    except json.JSONDecodeError:
                        self._error(400, "INVALID_INPUT")
                        return

                    script = payload.get("script")
                    caption_format = payload.get("caption_format", "json")
                    if not isinstance(script, list) or not script:
                        self._error(400, "INVALID_INPUT")
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
                            "pending",
                            audio_path,
                            caption_path,
                            caption_format,
                        ),
                    )
                    cur.execute(
                        "INSERT INTO usage (ts, length) VALUES (?, ?)",
                        (datetime.now().isoformat(), len(script) * 0.5),
                    )
                    server.db.commit()

                    # Queue the synthesis task
                    if server.use_celery:
                        # Use Celery for async processing
                        celery_synthesize.delay(
                            job_id, script, output_format, sample_rate, caption_format
                        )
                    else:
                        # Use in-memory queue
                        server.job_queue.put(("synthesize", job_id))

                    body = json.dumps({"job_id": job_id}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._error(404, "NOT_FOUND")

        def log_message(self, format: str, *args) -> None:
            # Suppress default logging
            return

        return Handler

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self.job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                continue
            if len(item) == 2:
                job_type, job_id = item
            else:
                job_type, job_id, *_ = item
            cur = self.db.cursor()
            cur.execute(
                "UPDATE jobs SET status=? WHERE id=?",
                ("succeeded", job_id),
            )
            self.db.commit()
            self.job_queue.task_done()

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

        # Only start worker thread if not using Celery
        if not self.use_celery:
            self.worker = threading.Thread(target=self._worker_loop)
            self.worker.daemon = True
            self.worker.start()

    def stop(self) -> None:
        if self.thread:
            self.httpd.shutdown()
            self.thread.join()
            self.thread = None
        if self.worker:
            self._stop_event.set()
            self.job_queue.put(None)
            self.worker.join()
            self.worker = None
            self._stop_event.clear()

    def wait_all_jobs(self, timeout: float = 1.0) -> None:
        end = datetime.now().timestamp() + timeout
        while datetime.now().timestamp() < end:
            if self.job_queue.empty():
                return
            time.sleep(0.01)

    def usage_report(self, year: int, month: int) -> dict:
        start = datetime(year, month, 1)
        if month == 12:
            end_dt = datetime(year + 1, 1, 1)
        else:
            end_dt = datetime(year, month + 1, 1)
        cur = self.db.cursor()
        cur.execute(
            "SELECT COUNT(*), COALESCE(SUM(length), 0) FROM usage WHERE ts >= ? AND ts < ?",
            (start.isoformat(), end_dt.isoformat()),
        )
        count, total = cur.fetchone()
        return {"count": count, "total_length": total}

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _presign_path(self, path: str | None) -> str | None:
        if not path:
            return None
        expiry = int(time.time()) + self.PRESIGNED_TTL
        return f"{path}?expires={expiry}"

    def cleanup_old_files(self, max_age_hours: float = 48) -> None:
        cutoff = time.time() - max_age_hours * 3600
        cur = self.db.cursor()
        cur.execute(
            "SELECT id, audio_url, caption_path FROM jobs WHERE status='succeeded'"
        )
        rows = cur.fetchall()
        for job_id, audio, caption in rows:
            keep = False
            for p in (audio, caption):
                if p and os.path.exists(p):
                    if os.path.getmtime(p) < cutoff:
                        try:
                            os.remove(p)
                        except FileNotFoundError:
                            pass
                    else:
                        keep = True
            if not keep:
                cur.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        self.db.commit()
