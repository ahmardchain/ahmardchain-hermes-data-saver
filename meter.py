"""Real network metering from the operating system's interface counters.

The counters are system-wide, so bytes other apps move while a tool call runs
are counted too. To avoid double counting when several tool calls overlap, the
traffic between two events is split evenly between the calls active at the time.

A background download (a terminal command Hermes moved to the background) is
tracked as an *absorbing* key: while one is running, the traffic belongs to it,
not to the model requests and polls that happen while it downloads.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

_LOOPBACK_PREFIXES = ("lo", "loopback")


def read_counters() -> Optional[tuple[int, int]]:
    """(bytes received, bytes sent) across non-loopback interfaces, or None when unavailable."""
    try:
        import psutil

        per_nic = psutil.net_io_counters(pernic=True)
    except Exception:
        return None
    if not per_nic:
        return None
    rx = tx = 0
    for name, stats in per_nic.items():
        if name.lower().startswith(_LOOPBACK_PREFIXES):
            continue
        rx += int(stats.bytes_recv)
        tx += int(stats.bytes_sent)
    return rx, tx


class Meter:
    """Attributes counter deltas to the tool calls that were running."""

    def __init__(self, reader: Callable[[], Optional[tuple[int, int]]] = read_counters):
        self._reader = reader
        self._lock = threading.Lock()
        self._active: dict[object, list[float]] = {}
        self._absorbing: set[object] = set()
        self._last: Optional[tuple[int, int]] = None

    def _advance(self) -> bool:
        """Split the traffic since the last event between the active calls. Lock held."""
        now = self._reader()
        if now is None:
            return False
        if self._last is not None and self._active:
            # A counter that went backwards (interface reset, wraparound) contributes nothing.
            d_rx = max(0, now[0] - self._last[0])
            d_tx = max(0, now[1] - self._last[1])
            targets = [key for key in self._active if key in self._absorbing] or list(self._active)
            for key in targets:
                self._active[key][0] += d_rx / len(targets)
                self._active[key][1] += d_tx / len(targets)
        self._last = now
        return True

    def begin(self, key: object, *, absorb: bool = False) -> bool:
        """Start metering a call. False when counters are unavailable on this machine."""
        with self._lock:
            if not self._advance():
                return False
            self._active[key] = [0.0, 0.0]
            if absorb:
                self._absorbing.add(key)
            return True

    def take(self, key: object) -> Optional[tuple[int, int]]:
        """Bytes attributed to a still-running call since the last take; the call keeps running."""
        with self._lock:
            if key not in self._active:
                return None
            self._advance()
            rx, tx = self._active[key]
            self._active[key] = [rx - int(rx), tx - int(tx)]
            return int(rx), int(tx)

    def end(self, key: object) -> Optional[tuple[int, int]]:
        """Stop metering a call; (received, sent) bytes attributed to it."""
        with self._lock:
            if key not in self._active:
                return None
            self._advance()
            self._absorbing.discard(key)
            rx, tx = self._active.pop(key)
            return int(rx), int(tx)

    @property
    def available(self) -> bool:
        return self._reader() is not None
