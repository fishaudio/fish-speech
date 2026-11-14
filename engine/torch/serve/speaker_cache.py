"""Reference clip persistence and request resolution for the local /v1/tts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

from engine.torch.serve.schema import MAX_SPEAKER_ID, ResolvedReference, TTSRequest

CACHE_DIR = next(
    parent / ".app_cache"
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
INDEX_FILE = CACHE_DIR / "speakers.json"
AUDIO_DIR = CACHE_DIR / "audio"
INDEX_VERSION = 1
MAX_ENTRIES = 64
MAX_LABEL_CHARS = 36
ENTRY_ID_LENGTH = 16

_LOCK = threading.Lock()


class CacheEntry(TypedDict):
    id: str
    name: str
    text: str
    speaker: int
    mime: str
    file: str
    size: int
    created_at: str
    last_used: str


class CacheIndex(TypedDict):
    version: int
    entries: list[CacheEntry]


def read_index() -> CacheIndex:
    if INDEX_FILE.is_file():
        try:
            data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return cast(CacheIndex, data)
    return {"version": INDEX_VERSION, "entries": []}


def write_index(index: CacheIndex) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_FILE.with_name(INDEX_FILE.name + ".tmp")
    tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(INDEX_FILE)


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def clip_label(text: str, speaker: int) -> str:
    return " ".join(text.split())[:MAX_LABEL_CHARS] or f"Speaker {speaker}"


def sniff_audio(data: bytes) -> tuple[str, str]:
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


def cache_references(references: list[ResolvedReference]) -> int:
    """Store each clip once; return the number of new entries."""
    decoded: list[tuple[bytes, str, int]] = []
    for reference in references:
        try:
            data = base64.b64decode(reference.audio, validate=True)
        except (binascii.Error, ValueError):
            continue
        if data:
            decoded.append((data, reference.text, reference.speaker))
    if not decoded:
        return 0

    with _LOCK:
        index = read_index()
        entries = index["entries"]
        by_id = {entry["id"]: entry for entry in entries}
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        added = 0

        for data, text, speaker in decoded:
            entry_id = hashlib.sha256(data).hexdigest()[:ENTRY_ID_LENGTH]
            existing = by_id.get(entry_id)
            if existing is None:
                mime, extension = sniff_audio(data)
                (AUDIO_DIR / f"{entry_id}.{extension}").write_bytes(data)
                now = timestamp()
                entry: CacheEntry = {
                    "id": entry_id,
                    "name": clip_label(text, speaker),
                    "text": text,
                    "speaker": speaker,
                    "mime": mime,
                    "file": f"audio/{entry_id}.{extension}",
                    "size": len(data),
                    "created_at": now,
                    "last_used": now,
                }
                entries.append(entry)
                by_id[entry_id] = entry
                added += 1
            else:
                if len(text) > len(existing["text"]):
                    existing["text"] = text
                    existing["name"] = clip_label(text, speaker)
                existing["speaker"] = speaker
                existing["last_used"] = timestamp()

        if len(entries) > MAX_ENTRIES:
            entries.sort(key=lambda item: item["last_used"], reverse=True)
            for stale in entries[MAX_ENTRIES:]:
                (CACHE_DIR / stale["file"]).unlink(missing_ok=True)
            entries[:] = entries[:MAX_ENTRIES]
        write_index(index)
    return added


def list_entries() -> list[dict[str, object]]:
    """Return cache entries with audio inlined as base64."""
    with _LOCK:
        index = read_index()

    result: list[dict[str, object]] = []
    for entry in index["entries"]:
        path = CACHE_DIR / entry["file"]
        if not path.is_file():
            continue
        data = path.read_bytes()
        result.append(
            {
                "id": entry["id"],
                "name": entry["name"],
                "text": entry["text"],
                "speaker": entry["speaker"],
                "mime": entry["mime"],
                "size": len(data),
                "created_at": entry["created_at"],
                "last_used": entry["last_used"],
                "audio": base64.b64encode(data).decode("ascii"),
            }
        )
    return result


def resolve_references(req: TTSRequest) -> list[ResolvedReference]:
    """Resolve inline clips, or voices addressed by reference_id."""
    resolved = req.resolved_references()
    if not resolved:
        # With inline audio, reference_id is only a label.
        with _LOCK:
            entries = read_index()["entries"]
        by_id = {entry["id"]: entry for entry in entries}
        for speaker, entry_id in enumerate(req.reference_ids()):
            entry = by_id.get(entry_id)
            if entry is None:
                raise ValueError(f"reference_id not found in .app_cache: {entry_id}")
            data = (CACHE_DIR / entry["file"]).read_bytes()
            resolved.append(
                ResolvedReference(
                    audio=base64.b64encode(data).decode("ascii"),
                    text=entry["text"],
                    speaker=speaker,
                )
            )

    for reference in resolved:
        if reference.speaker > MAX_SPEAKER_ID:
            raise ValueError(f"speaker index must be 0-{MAX_SPEAKER_ID}")
    return resolved


def delete_entry(entry_id: str) -> bool:
    """Remove one entry; return False when the id is unknown."""
    with _LOCK:
        index = read_index()
        target = next((item for item in index["entries"] if item["id"] == entry_id), None)
        if target is None:
            return False
        (CACHE_DIR / target["file"]).unlink(missing_ok=True)
        index["entries"] = [item for item in index["entries"] if item["id"] != entry_id]
        write_index(index)
    return True
