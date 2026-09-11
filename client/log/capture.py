"""Turn a managed child's stdout into log entries worth shipping.

The supervisor already tees llama-server / rpc-server output to a file and to
container stdout. Neither is reachable from the orchestrator: `kubectl logs`
proxies through the node's kubelet on :10250, which the k3d server container
cannot route to for any LAN-joined node, and a shard worker serves no HTTP at
all, so without this a crash on a worker is invisible from the orchestrator
unless someone SSHes in.

This module is the filter that makes shipping those lines affordable. llama.cpp
is extremely chatty at the default verbosity — it emits a `print_timing` line
every few tokens — so forwarding everything would swamp the orchestrator's log
store and cost real bandwidth on a LAN that inference already shares.

Two things are worth having centrally, and this keeps only those:
  * warnings and errors, as they happen;
  * the last N lines before a child dies, which is the context that actually
    explains the death.
"""

from __future__ import annotations

import logging
import re
from collections import deque

# llama.cpp log lines look like:
#     561.18.787.826 I slot print_timing: id  0 | task 4936 | eval time = ...
#     0.00.034.776 W srv  llama_server: CORS is set to allow all origins
#     15.29.768.587 E recv failed (bytes_recv=0, size_to_recv=8)
# i.e. <mm.ss.mmm.uuu> <LEVEL> <component> <text>, where LEVEL is a single
# character. Anchored so a stray "E" inside a message can never be read as a
# level marker.
_LLAMA_LINE = re.compile(r"^[\d.]+\s+([IWED])\s+")

_LEVEL_BY_CHAR = {"I": "info", "W": "warn", "E": "error", "D": "debug"}

# Paired with ``_MAX_TEXT`` in orchestrator/logs/receiver.py, which truncates on
# arrival anyway. Bounded HERE as well because _pump_child_output splits only on
# b"\n": llama.cpp's \r-based load progress and a GGML abort backtrace each
# arrive as one very large line, and an unbounded one is held whole in the ring
# and shipped whole for the orchestrator to throw the excess away.
_MAX_TEXT = 4000

# Lines llama.cpp emits without a level marker but which are unambiguously a
# failure — GGML aborts print a bare file:line assertion followed by a raw
# backtrace, and that is precisely the crash text worth keeping.
_FAILURE_HINTS = (
    "GGML_ASSERT",
    "ggml_abort",
    "error loading model",
    "failed to",
    "Remote RPC server crashed",
    "CUDA error",
    "out of memory",
    "Segmentation fault",
)


def classify(line: str) -> str:
    """Best-effort severity for one child output line.

    Returns one of ``debug|info|warn|error``. Unmarked lines default to
    ``info`` unless they match a known failure signature, because an
    unrecognised line during a crash is far more useful mislabelled as an
    error than dropped as noise.
    """
    stripped = line.strip()
    if not stripped:
        return "debug"

    m = _LLAMA_LINE.match(stripped)
    if m:
        return _LEVEL_BY_CHAR.get(m.group(1), "info")

    lowered = stripped.lower()
    for hint in _FAILURE_HINTS:
        if hint.lower() in lowered:
            return "error"
    return "info"


class ChildLogCapture:
    """Decides which child lines reach the log store.

    ``ship_levels`` are recorded as they arrive. Everything else goes into a
    fixed-size ring buffer that is flushed only if the child dies, so the run-up
    to a crash survives without paying for it during normal operation.
    """

    def __init__(
        self,
        store,
        source: str,
        ship_levels: tuple[str, ...] = ("warn", "error"),
        context_lines: int = 80,
    ) -> None:
        self.store = store
        self.source = source
        self.ship_levels = set(ship_levels)
        self._ring: deque[tuple[str, str]] = deque(maxlen=max(0, context_lines))
        self._store_failed = False

    def observe(self, line: str) -> None:
        """Record one line of child output."""
        text = line.rstrip("\n")[:_MAX_TEXT]
        if not text.strip():
            return
        level = classify(text)
        self._ring.append((level, text))
        if level in self.ship_levels:
            self._append(level, text)

    def flush_crash_context(self, exit_code: int) -> None:
        """Persist the buffered run-up after the child has exited.

        Called on ANY exit, including a clean one: a zero exit that was not
        expected is itself worth explaining, and the volume is bounded by the
        ring size.
        """
        # One transaction for the whole flush. Row-at-a-time it is up to 81
        # separate append() calls — a connection and an fsync each, ~0.4 ms on
        # fast storage and more on flash — spent inside _exit_reporting before
        # drain() even starts, already racing the grace period. The trade is
        # that one failure loses the whole flush rather than one row. Rows
        # share one ts, but recent() orders by id DESC, so the exit marker
        # still leads and the run-up keeps its order.
        records = [("error", f"child exited with code {exit_code} — last lines follow", self.source)]
        records.extend(
            # Everything at ship_levels was already stored as it happened;
            # re-storing it here would duplicate the interesting lines.
            (level, text, self.source)
            for level, text in self._ring
            if level not in self.ship_levels
        )
        self._ring.clear()
        self._append_many(records)

    def _append(self, level: str, text: str) -> None:
        # Never let logging break the pump thread that feeds container stdout —
        # losing the tee would make the crash less diagnosable, not more.
        try:
            self.store.append(level, text, source=self.source)
        except Exception as e:
            self._note_store_failure(e)

    def _append_many(self, records: list[tuple[str, str, str]]) -> None:
        try:
            self.store.append_many(records)
        except Exception as e:
            self._note_store_failure(e)

    def _note_store_failure(self, exc: Exception) -> None:
        """Report the first store failure of this capture's life, then stay quiet.

        Swallowed outright, a full disk, a corrupt system.db or a failed
        extension load is a permanent silent no-op: the operator's only symptom
        is a node that stops appearing in the shipped-log console, which looks
        exactly like a healthy quiet node. This is the one path that carries a
        shard worker's crash off the box.

        The latch is per-instance, not per-process, so a fresh capture after a
        restart reports again. Logged under "aureax." so it reaches container
        stdout and the child log even while the store itself is dead.
        """
        if self._store_failed:
            return
        self._store_failed = True
        logging.getLogger("aureax.telemetry").warning("log store append failing: %s", exc)
