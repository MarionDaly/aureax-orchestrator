"""Local SQLite buffer for telemetry metrics on edge nodes."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from client.sqlite import connection


class TelemetryStorage:
    """SQLite-backed metric buffer. Survives reboots and connectivity loss."""

    def __init__(self, db_path: str = "/opt/aureax/metrics/telemetry.db") -> None:
        """Open or create the SQLite telemetry database."""
        self.db_path = Path(db_path)
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
                    pushed INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pushed ON metrics(pushed)"
            )
            # Every predicate here is timestamp-only — this DB holds one
            # node's rows, so node_id never appears in a WHERE. A composite
            # index led by node_id serves none of them (SQLite cannot use an
            # index without its leading column) and costs a b-tree write per
            # insert, so it is dropped rather than merely not created:
            # CREATE ... IF NOT EXISTS alone leaves it in place on every
            # already-provisioned node.
            conn.execute("DROP INDEX IF EXISTS idx_node_ts")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts ON metrics(timestamp)"
            )

    def insert(self, metrics: list[dict]) -> None:
        """Insert a batch of metric records."""
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO metrics (timestamp, node_id, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                [
                    (m["timestamp"], m["node_id"], m["metric_name"], m["metric_value"])
                    for m in metrics
                ],
            )

    def get_unpushed(self, limit: int = 500) -> list[dict]:
        """Get metrics that haven't been pushed to the orchestrator yet."""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM metrics WHERE pushed=0 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def latest_per_metric(self, since_minutes: int = 60) -> tuple[dict[str, float], int]:
        """(latest value per metric name, total samples) in the window.

        Aggregated in SQL so the caller never materializes the window —
        get_metrics discards everything but the newest value per name anyway.
        ``since_minutes`` is caller-controlled (it arrives from the MCP
        get_metrics tool argument), and aggregating is what keeps one request
        from loading the whole buffer into memory.
        MAX(id) stands in for "latest": ids are autoincrement and rows are
        inserted in sample order.

        The cutoff is built in Python rather than with SQLite's
        ``datetime('now', ...)``. Rows store ``datetime.now(UTC).isoformat()``
        ("2026-08-13T23:50:04+00:00", with a "T") while SQLite returns a
        space-separated form. Comparing the two is a string comparison that
        diverges at index 10, so the time of day is never reached and the
        filter silently degrades to "today or later".
        """
        cutoff = (datetime.now(UTC) - timedelta(minutes=since_minutes)).isoformat()
        with self._conn() as conn:
            latest = dict(conn.execute(
                """
                SELECT m.metric_name, m.metric_value
                FROM metrics m
                JOIN (
                    SELECT MAX(id) AS max_id FROM metrics
                    WHERE timestamp > ? GROUP BY metric_name
                ) x ON m.id = x.max_id
                """,
                (cutoff,),
            ).fetchall())
            sample_count = conn.execute(
                "SELECT COUNT(*) FROM metrics WHERE timestamp > ?", (cutoff,)
            ).fetchone()[0]
            return latest, sample_count

    def mark_pushed(self, ids: list[int]) -> None:
        """Mark metrics as successfully pushed."""
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        with self._conn() as conn:
            conn.execute(f"UPDATE metrics SET pushed=1 WHERE id IN ({placeholders})", ids)

    def rotate(self, retention_days: int = 7) -> int:
        """Prune the buffer: pushed rows older than retention_days, and ANY row
        older than 3x that. Returns count deleted.

        Same ISO-vs-space cutoff reasoning as :meth:`latest_per_metric`; here
        the mismatch would make pruning a whole day less aggressive than
        configured.

        Unpushed rows get a second cutoff at 3x the retention: a node that
        cannot reach the orchestrator (endpoint unset, dashboard down) never
        marks anything pushed, and without a hard cutoff its buffer grows
        without bound. Samples three retention periods stale are not worth a
        node's disk.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        hard_cutoff = (datetime.now(UTC) - timedelta(days=3 * retention_days)).isoformat()
        with self._conn() as conn:
            deleted = conn.execute(
                "DELETE FROM metrics WHERE pushed=1 AND timestamp < ?",
                (cutoff,),
            ).rowcount
            deleted += conn.execute(
                "DELETE FROM metrics WHERE timestamp < ?",
                (hard_cutoff,),
            ).rowcount
            return deleted

    def count(self) -> dict:
        """Return counts of total, pushed, and unpushed metrics.

        Test scaffolding, kept on purpose: production code never calls this,
        but the telemetry tests in tests/test_client_smoke.py and
        tests/test_log_shipping.py assert pushed/unpushed through it rather
        than opening their own connection against the schema.
        """
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
            pushed = conn.execute("SELECT COUNT(*) FROM metrics WHERE pushed=1").fetchone()[0]
            return {"total": total, "pushed": pushed, "unpushed": total - pushed}
