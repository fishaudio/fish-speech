from __future__ import annotations

import json
from typing import Any, Dict, Iterable


def _sec_to_timestamp(sec: float, sep: str = ".") -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02}:{m:02}:{s:06.3f}".replace(".", sep)


def export_captions(units: Iterable[Dict[str, Any]], fmt: str = "json") -> str:
    """Return caption text in the requested format."""
    units = list(units)
    if fmt == "json":
        return json.dumps(units, ensure_ascii=False)
    lines: list[str] = []
    if fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
        for i, u in enumerate(units, 1):
            start = _sec_to_timestamp(u["start"])
            end = _sec_to_timestamp(u["end"])
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            prefix = f"{u.get('speaker', '')}: " if u.get("speaker") else ""
            lines.append(f"{prefix}{u['text']}")
            lines.append("")
        return "\n".join(lines)
    if fmt == "srt":
        for i, u in enumerate(units, 1):
            start = _sec_to_timestamp(u["start"], sep=",")
            end = _sec_to_timestamp(u["end"], sep=",")
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(u["text"])
            lines.append("")
        return "\n".join(lines)
    raise ValueError(f"Unsupported caption format: {fmt}")


__all__ = ["export_captions"]
