"""Ship edge log entries to the orchestrator as JSONL.

Deliberately mirrors ``client/telemetry/reporter.py``: same transport, same
retry shape, same failure posture — literally so, since the POST loop is that
module's :func:`post_jsonl`. Logs ride the road telemetry already proved works,
which is what makes this independent of `kubectl logs` — that path proxies
through each node's kubelet on :10250 and the k3d server container cannot reach
it for any LAN-joined node.

The one structural difference is progress tracking. ``TelemetryStorage`` marks
rows pushed in-place; ``LogStore`` is shared verbatim with the orchestrator
dashboard, where a per-shipper column means nothing and would still have to be
migrated onto every database already on disk on both sides. It already exposes
``since_id``, so this keeps a cursor instead — persisted to a file beside the
database so a pod restart resumes rather than re-shipping the whole local
history, and reconciled against the store on load because the two can be
separated (see ``_reconciled``).
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
from pathlib import Path

from client.telemetry.reporter import post_jsonl

logger = logging.getLogger("aureax.logship")


class LogReporter:
    """Pushes unshipped log entries to the orchestrator endpoint as JSONL."""

    def __init__(
        self,
        store,
        endpoint: str,
        node_id: str,
        cursor_path: str | Path,
        batch_size: int = 200,
        max_attempts: int = 3,
        backoff_base: int = 30,
        backoff_max: int = 300,
        verify: ssl.SSLContext | None = None,
    ) -> None:
        self.store = store
        self.endpoint = endpoint
        self.node_id = node_id
        self.cursor_path = Path(str(cursor_path)).expanduser()
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        # The fleet CA context, threaded straight into post_jsonl. Same
        # trailing position as TelemetryReporter's, for the same reason the
        # rest of this signature mirrors that one: two shippers over one
        # transport, and one signature is what keeps them from drifting.
        self.verify = verify
        self._cursor = self._read_cursor()
        # Serializes since_id -> POST -> cursor-advance. The background
        # logship thread and the exit path's drain() otherwise race the
        # cursor and double-ship (or skip) a batch.
        self._lock = threading.Lock()

    # --- cursor persistence --------------------------------------------------

    def _read_cursor(self) -> int:
        try:
            # Clamped at 0: since_id(-1) returns the whole table, so a negative
            # value on disk would ship the node's local history all over again.
            cursor = max(0, int(self.cursor_path.read_text().strip()))
        except (OSError, ValueError):
            # No cursor yet, or a truncated write from a hard kill. Starting at
            # 0 re-ships local history once, which is noisy but never loses the
            # crash that is the reason this exists.
            return 0
        return self._reconciled(cursor)

    def _reconciled(self, cursor: int) -> int:
        """Refuse a cursor that points past the end of the table it indexes.

        The cursor file and the database are siblings, but they are not written
        together and nothing ties one to the other. A system.db removed by hand
        to recover from corruption comes back with ids restarting at 1 while
        the surviving cursor still names a far higher row, and from then on
        ``since_id`` returns nothing every cycle and ``push`` reports success:
        the node looks healthy and ships not one line for the life of the pod.
        Restarting from 0 re-ships whatever local history is left, once.
        """
        if cursor <= 0:
            return cursor
        try:
            newest = self.store.recent(limit=1)
        except Exception as e:
            # An unreadable store is reported by its own writers. Trusting the
            # file here costs at most duplicates, which this module already
            # prefers over a gap.
            logger.warning("Could not check the log cursor against the store: %s", e)
            return cursor
        highest = newest[0].id if newest else 0
        if cursor <= highest:
            return cursor
        logger.warning(
            "Log cursor %d is ahead of the store's highest id %d — the database "
            "was recreated under it; shipping again from the start",
            cursor,
            highest,
        )
        return 0

    def _write_cursor(self, value: int) -> None:
        try:
            self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
            # Written to a sibling and renamed. ``write_text`` truncates in
            # place, so a kill between truncate and write leaves an empty file
            # that _read_cursor reads as 0 — re-shipping every row still inside
            # logging.retention_days, and putting that backlog in front of the
            # fresh crash context drain() exists to reach. A SIGKILL here is
            # expected, not exotic: the pod is killed at the end of
            # daemonset-edge.yaml's terminationGracePeriodSeconds (45s).
            # os.replace is atomic within one filesystem, and the cursor lives
            # beside the database it tracks.
            tmp = self.cursor_path.with_name(self.cursor_path.name + ".tmp")
            tmp.write_text(str(value))
            os.replace(tmp, self.cursor_path)
        except OSError as e:
            # A cursor we cannot persist means duplicates after a restart, which
            # is strictly better than dropping entries or crashing the thread.
            logger.warning("Could not persist log cursor: %s", e)

    # --- shipping ------------------------------------------------------------

    def push(self, max_batches: int = 1, attempts: int | None = None) -> bool:
        """Send up to ``max_batches`` batches newer than the cursor.

        Batches are FIFO, so a single batch only reaches the newest entries when
        the backlog is smaller than ``batch_size``. That matters on the way out
        of a dying pod: ``flush_crash_context`` appends the post-mortem at the
        END of the table, so with a backlog the one batch that gets sent is the
        oldest history and the crash context — the entire point — never leaves.

        ``max_batches`` is what lets a backlog drain: one batch per call
        clears (batch_size - production rate) rows per push interval, so an
        outage long enough to matter needs hours of catch-up during which the
        newest entries — the crash context above among them — stay on the node.
        The loop stops early on an empty store or a failed send, so a healthy
        node still sends exactly one batch.

        ``attempts`` overrides the configured retry count for one call. The
        entries are already durable on disk, so in-loop backoff (3 attempts at
        30s and 60s, each over a 30s POST timeout) buys no durability and
        instead blocks this thread for ~3 minutes per cycle, holding up the
        hourly prune that shares it. The next cycle is the retry.

        The signature matches ``TelemetryReporter.push`` deliberately: the two
        reporters ship over the same transport with the same policy, and one
        signature is what keeps them from drifting.
        """
        with self._lock:
            return self._push_locked(max_batches, attempts or self.max_attempts)

    def _push_locked(self, max_batches: int, attempts: int, deadline: float | None = None) -> bool:
        for _ in range(max(1, max_batches)):
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning("drain(): budget spent, entries still pending")
                return True
            entries = self.store.since_id(self._cursor, limit=self.batch_size)
            if not entries:
                return True
            if not self._send(entries, attempts):
                return False
        return True

    def drain(self, max_batches: int = 25, budget: float = 10.0) -> bool:
        """Ship everything pending, fast, for the shutdown path.

        One attempt per batch, no backoff, and a wall-clock budget. This runs on
        the main thread immediately before ``sys.exit``, where the background
        retry policy (3 attempts with 30s and 60s backoff) would keep a
        container Running for up to three minutes after its child is already
        dead — delaying the restart K8s is waiting to perform.

        ``budget`` covers the lock wait AND the batch loop behind it, as one
        deadline taken at the top: against a black-holing orchestrator, 25
        batches at one 30s POST apiece would overshoot
        daemonset-edge.yaml's terminationGracePeriodSeconds (45s) many times
        over and be SIGKILLed anyway, having shipped no more than a bounded
        drain would. The deadline is only checked BETWEEN batches, so a request
        already in flight can still overrun it by its own 30s timeout: 10 + 30
        is the real ceiling on this path, inside the grace.

        The lock acquire is bounded for the same reason — the logship thread
        can sit inside a 30s POST, and an exit path that waits minutes for it
        defeats drain's whole purpose — but at HALF the budget, so winning the
        lock late still leaves time to ship rather than returning on a deadline
        already spent. On timeout, ship anyway: the race costs at worst a
        duplicated batch, which this module already prefers over a dropped one
        (see ``_write_cursor``). That path gets its own half-budget rather than
        the shared deadline, which by then is gone; its ceiling is 5 + 5 + 30,
        also inside the grace.
        """
        deadline = time.monotonic() + budget
        if self._lock.acquire(timeout=budget / 2):
            try:
                return self._push_locked(max_batches, attempts=1, deadline=deadline)
            finally:
                self._lock.release()
        logger.warning("drain(): logship thread still busy — shipping without the lock")
        return self._push_locked(
            max_batches, attempts=1, deadline=time.monotonic() + budget / 2
        )

    def _send(self, entries, attempts: int) -> bool:
        """POST one batch and advance the cursor on success."""
        payload = "\n".join(
            json.dumps(
                {
                    "node_id": self.node_id,
                    "ts": e.ts,
                    "level": e.level,
                    "source": e.source,
                    "text": e.text,
                }
            )
            for e in entries
        )
        highest = entries[-1].id

        if not post_jsonl(
            self.endpoint,
            payload,
            attempts,
            backoff_base=self.backoff_base,
            backoff_max=self.backoff_max,
            log=logger,
            what="log entries",
            verify=self.verify,
        ):
            return False

        self._cursor = highest
        self._write_cursor(highest)
        logger.debug("Shipped %d log entries to %s", len(entries), self.endpoint)
        return True
