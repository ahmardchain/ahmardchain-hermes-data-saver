"""SQLite ledger of the data Hermes used: one row per tool call, model request or preflight."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    day TEXT NOT NULL,
    month TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    source TEXT NOT NULL,
    rx INTEGER NOT NULL DEFAULT 0,
    tx INTEGER NOT NULL DEFAULT 0,
    measured INTEGER NOT NULL DEFAULT 1,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS usage_day ON usage(day);
CREATE INDEX IF NOT EXISTS usage_month ON usage(month);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

CATEGORIES = ("tool", "model", "preflight")


def day_key(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts).strftime("%Y-%m-%d")


def month_key(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts).strftime("%Y-%m")


class Ledger:
    """Thread-safe wrapper around one SQLite connection.

    ``connect`` is called lazily, so nothing touches the disk while Hermes is
    only registering the plugin.
    """

    def __init__(self, connect: Callable[[], sqlite3.Connection], clock: Callable[[], float] = time.time):
        self._connect = connect
        self._clock = clock
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = self._connect()
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    # ── writes ────────────────────────────────────────────────────────────
    def record(self, category: str, source: str, rx: int = 0, tx: int = 0, *, session_id: str = "",
               measured: bool = True, detail: str = "") -> None:
        rx, tx = max(0, int(rx)), max(0, int(tx))
        if rx == 0 and tx == 0:
            return
        now = self._clock()
        with self._lock:
            db = self._db()
            db.execute(
                "INSERT INTO usage (ts, day, month, session_id, category, source, rx, tx, measured, detail)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, day_key(now), month_key(now), session_id or "", category, source, rx, tx,
                 1 if measured else 0, (detail or "")[:300]),
            )
            db.commit()

    def set_kv(self, key: str, value: Any) -> None:
        with self._lock:
            db = self._db()
            db.execute("INSERT INTO kv (key, value) VALUES (?, ?)"
                       " ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, str(value)))
            db.commit()

    def delete_kv(self, key: str) -> None:
        with self._lock:
            db = self._db()
            db.execute("DELETE FROM kv WHERE key = ?", (key,))
            db.commit()

    # ── reads ─────────────────────────────────────────────────────────────
    def get_kv(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            row = self._db().execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def total(self, *, day: Optional[str] = None, month: Optional[str] = None,
              session_id: Optional[str] = None) -> int:
        """Bytes (received + sent) for a day, a month or a session."""
        where, params = self._where(day, month, session_id)
        with self._lock:
            row = self._db().execute(f"SELECT COALESCE(SUM(rx + tx), 0) FROM usage{where}", params).fetchone()
        return int(row[0] or 0)

    def by_category(self, *, day: Optional[str] = None, month: Optional[str] = None) -> dict[str, int]:
        where, params = self._where(day, month, None)
        with self._lock:
            rows = self._db().execute(
                f"SELECT category, COALESCE(SUM(rx + tx), 0) FROM usage{where} GROUP BY category", params).fetchall()
        out = {name: 0 for name in CATEGORIES}
        out.update({str(cat): int(total) for cat, total in rows})
        return out

    def top_sources(self, *, day: Optional[str] = None, month: Optional[str] = None,
                    limit: int = 5) -> list[dict[str, Any]]:
        where, params = self._where(day, month, None)
        with self._lock:
            rows = self._db().execute(
                f"SELECT category, source, COALESCE(SUM(rx + tx), 0) AS total, COUNT(*) FROM usage{where}"
                " GROUP BY category, source ORDER BY total DESC LIMIT ?", (*params, int(limit))).fetchall()
        return [{"category": c, "source": s, "bytes": int(t), "calls": int(n)} for c, s, t, n in rows]

    def biggest_calls(self, *, day: Optional[str] = None, limit: int = 5) -> list[dict[str, Any]]:
        where, params = self._where(day, None, None)
        with self._lock:
            rows = self._db().execute(
                f"SELECT source, rx + tx AS total, detail FROM usage{where}"
                " ORDER BY total DESC LIMIT ?", (*params, int(limit))).fetchall()
        return [{"source": s, "bytes": int(t), "detail": d} for s, t, d in rows]

    # ── lookup store (see forecast.lookup_store) ──────────────────────────
    def get(self, key: str, max_age: float) -> Optional[Any]:
        raw = self.get_kv("cache:" + key)
        if not raw:
            return None
        try:
            stamp, value = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return value if self._clock() - float(stamp) < max_age else None

    def put(self, key: str, value: Any) -> None:
        self.set_kv("cache:" + key, json.dumps([self._clock(), value]))

    @staticmethod
    def _where(day: Optional[str], month: Optional[str], session_id: Optional[str]) -> tuple[str, tuple]:
        clauses, params = [], []
        if day:
            clauses.append("day = ?")
            params.append(day)
        if month:
            clauses.append("month = ?")
            params.append(month)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", tuple(params)
