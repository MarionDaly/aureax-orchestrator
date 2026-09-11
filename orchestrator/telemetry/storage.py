"""Orchestrator-side SQLite metrics storage."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from client.sqlite import connection

logger = logging.getLogger("aureax.telemetry.storage")


class MetricsStorage:
    """SQLite storage for aggregated telemetry from all edge nodes."""

    def __init__(self, db_path: str = "~/.aureax/telemetry/metrics.db") -> None:
        """Open or create the orchestrator metrics database."""
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self):
        return connection(self.db_path, rows=False, foreign_keys=False)

    def _init_db(self) -> None:
        """Create the metrics table and indexes if they don't exist."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    received_at TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_ts ON metrics(node_id, timestamp)"
            )
            # The covering index for get_latest_all's fleet-wide
            # `GROUP BY node_id, metric_name` — one grouped scan per dashboard
            # refresh — and for the single-node form beside it. Without it each
            # refresh reads every row the fleet has ever reported.
            #
            # No index leads with metric_name: nothing filters, groups or
            # orders by it, so one would be write cost on every insert for a
            # reader that does not exist.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_metric_ts "
                "ON metrics(node_id, metric_name, timestamp DESC)"
            )
            # One leading-column index per retention DELETE below. SQLite
            # cannot use a composite index without its leading column, and no
            # index above leads with either of these — without them the hourly
            # prune is a bare SCAN of the whole table, inside a write
            # transaction that blocks readers, to delete the ~1/720 of it that
            # has aged out.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts ON metrics(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_received_at ON metrics(received_at)"
            )

    def insert(self, records: list[dict]) -> int:
        """Insert metric records from edge nodes. Returns count inserted."""
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO metrics (timestamp, node_id, metric_name, metric_value, received_at) VALUES (?, ?, ?, ?, ?)",
                [
                    (r["timestamp"], r["node_id"], r["metric_name"], r["metric_value"], now)
                    for r in records
                ],
            )
        self._maybe_rotate()
        return len(records)

    # Nothing else prunes this table, and a 9-node fleet reporting ~10 metrics
    # a minute writes ~3.9M rows a month. This is what gives
    # `server.telemetry.storage.retention_days` its effect.
    #
    # Driven off insert rather than a background task: inserts are the only
    # thing that grows the table, they arrive continuously while the dashboard
    # runs, and this way there is no extra thread to own or shut down.
    _ROTATE_INTERVAL_SECONDS = 3600.0
    # None, not 0.0: time.monotonic() is boot-relative, so "never rotated"
    # written as 0.0 reads as "rotated at boot" — on a machine up for less than
    # the interval, `now - 0.0` is under an hour and pruning never starts at
    # all. A sentinel that is not a valid clock reading cannot be mistaken for
    # one.
    _last_rotate: float | None = None
    retention_days: int = 30

    def _maybe_rotate(self) -> None:
        """Prune at most hourly; never let pruning break an insert."""
        now = time.monotonic()
        if self.retention_days <= 0:
            return
        if self._last_rotate is not None and now - self._last_rotate < self._ROTATE_INTERVAL_SECONDS:
            return
        self._last_rotate = now
        try:
            deleted = self.rotate(self.retention_days)
            if deleted:
                logger.info("Telemetry: pruned %d row(s) older than %dd",
                            deleted, self.retention_days)
        except Exception as e:
            logger.warning("Telemetry rotate failed: %s", e)

    def rotate(self, retention_days: int) -> int:
        """Delete metrics older than retention_days. Returns count deleted.

        No default: the only production caller passes ``self.retention_days``,
        which app.py sets from the config, so a second default here would only
        ever surface as a bare call silently ignoring the configured retention.

        Cutoff built in Python: rows store ISO timestamps with a "T" while
        SQLite's datetime() returns a space-separated form, and comparing them
        is a string comparison that never reaches the time of day.

        Two DELETEs against the same cutoff, one per column. ``timestamp`` is
        copied verbatim from the pushed JSONL, so a node with no RTC that boots
        with a future clock — or anything on the LAN posting to the
        unauthenticated /api/v1/telemetry — writes rows a timestamp-only prune
        can never reach, and retention stops bounding the table at all.
        ``received_at`` is server-stamped and no poster can move it. Two
        statements rather than one OR: an OR across two columns can use neither
        single-column index, which is the full scan this shape exists to avoid.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._conn() as conn:
            deleted = conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?", (cutoff,)
            ).rowcount
            deleted += conn.execute(
                "DELETE FROM metrics WHERE received_at < ?", (cutoff,)
            ).rowcount
            return deleted
