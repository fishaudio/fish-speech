"""Persist reference audio/text pairs from /v1/tts into the repo's .app_cache.

The server already decodes ``TTSRequest.references`` to build the prompt, so the
audio bytes and transcripts are available without any extra client work. This
module only uses the standard library, so it can be exercised without loading a
model. Layout::

    .app_cache/
        speakers.json      # index + transcripts
        audio/<id>.<ext>   # one file per distinct clip

Entries are deduplicated by audio content and capped, so repeatedly generating
with the same voice does not fill the directory.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

def _repo_root() -> Path:
    # Anchor on pyproject.toml rather than a fixed number of parent levels, so
    # moving this module around does not silently relocate the cache.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).resolve().parents[3]


CACHE_DIR = _repo_root() / ".app_cache"
INDEX_FILE = CACHE_DIR / "speakers.json"
AUDIO_DIR = CACHE_DIR / "audio"
INDEX_VERSION = 1
MAX_ENTRIES = 64
_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sniff(data: bytes) -> tuple[str, str]:
    """Return (mime, extension) from the file header."""
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav", "wav"
    if data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0):
        return "audio/mpeg", "mp3"
    if data[:4] == b"OggS":
        return "audio/ogg", "ogg"
    if data[:4] == b"fLaC":
        return "audio/flac", "flac"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "audio/webm", "webm"
    return "application/octet-stream", "bin"


def _read_index() -> dict:
    if not INDEX_FILE.exists():
        return {"version": INDEX_VERSION, "entries": []}
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": INDEX_VERSION, "entries": []}
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        return {"version": INDEX_VERSION, "entries": []}
    return data


def _write_index(index: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_name(INDEX_FILE.name + ".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(INDEX_FILE)


def _label(text: str, speaker: int | None) -> str:
    cleaned = " ".join(text.split())[:36]
    return cleaned or f"Speaker {speaker if speaker is not None else 0}"


def cache_references(references: Iterable) -> int:
    """Store each reference clip once. Returns the number of new entries.

    ``references`` may be pydantic models; only ``audio``, ``text`` and
    ``speaker`` attributes are read. Decoding errors are skipped rather than
    raised, so a bad clip never breaks synthesis.
    """
    decoded: list[tuple[bytes, str, int | None]] = []
    for reference in references or []:
        raw = getattr(reference, "audio", "") or ""
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            continue
        if not data:
            continue
        text = getattr(reference, "text", "") or ""
        speaker = getattr(reference, "speaker", None)
        decoded.append((data, text, speaker))

    if not decoded:
        return 0

    with _LOCK:
        index = _read_index()
        entries = index["entries"]
        by_id = {entry.get("id"): entry for entry in entries}
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        added = 0

        for data, text, speaker in decoded:
            entry_id = hashlib.sha256(data).hexdigest()[:16]
            mime, extension = _sniff(data)
            existing = by_id.get(entry_id)
            if existing is None:
                (AUDIO_DIR / f"{entry_id}.{extension}").write_bytes(data)
                entry = {
                    "id": entry_id,
                    "name": _label(text, speaker),
                    "text": text,
                    "speaker": speaker,
                    "mime": mime,
                    "file": f"audio/{entry_id}.{extension}",
                    "size": len(data),
                    "created_at": _now(),
                    "last_used": _now(),
                }
                entries.append(entry)
                by_id[entry_id] = entry
                added += 1
            else:
                # Keep the most informative transcript, and refresh recency.
                if text and (not existing.get("text") or len(text) > len(existing["text"])):
                    existing["text"] = text
                    existing["name"] = _label(text, speaker)
                if speaker is not None:
                    existing["speaker"] = speaker
                existing["last_used"] = _now()

        if len(entries) > MAX_ENTRIES:
            entries.sort(key=lambda entry: entry.get("last_used", ""), reverse=True)
            for stale in entries[MAX_ENTRIES:]:
                path = CACHE_DIR / stale.get("file", "")
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            entries[:] = entries[:MAX_ENTRIES]

        _write_index(index)
    return added


def list_entries() -> list[dict]:
    """Return cache entries with audio inlined as base64 for the web UI."""
    with _LOCK:
        index = _read_index()

    result: list[dict] = []
    for entry in index["entries"]:
        path = CACHE_DIR / entry.get("file", "")
        try:
            data = path.read_bytes()
        except OSError:
            continue
        result.append(
            {
                "id": entry.get("id", ""),
                "name": entry.get("name", "Reference clip"),
                "text": entry.get("text", ""),
                "speaker": entry.get("speaker"),
                "mime": entry.get("mime", "application/octet-stream"),
                "size": len(data),
                "created_at": entry.get("created_at", ""),
                "last_used": entry.get("last_used", ""),
                "audio": base64.b64encode(data).decode("ascii"),
            }
        )
    return result


def delete_entry(entry_id: str) -> bool:
    """Remove one entry. Returns False when the id is unknown."""
    if not _ID_PATTERN.fullmatch(entry_id):
        return False

    with _LOCK:
        index = _read_index()
        entries = index["entries"]
        target = next((entry for entry in entries if entry.get("id") == entry_id), None)
        if target is None:
            return False
        try:
            (CACHE_DIR / target.get("file", "")).unlink(missing_ok=True)
        except OSError:
            pass
        index["entries"] = [entry for entry in entries if entry.get("id") != entry_id]
        _write_index(index)
    return True
