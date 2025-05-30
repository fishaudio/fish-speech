from __future__ import annotations

import sqlite3
from typing import Optional


def init_db(dsn: Optional[str] = None) -> sqlite3.Connection:
    """Initialize and return a database connection.

    This uses SQLite for testing purposes but mirrors the intended
    PostgreSQL schema described in the PRD.
    """

    conn = sqlite3.connect(dsn or ":memory:", check_same_thread=False)
    cur = conn.cursor()
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
    conn.commit()
    return conn


__all__ = ["init_db"]
