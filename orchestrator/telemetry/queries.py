"""Metric query helpers for the dashboard and CLI."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from client.sqlite import connection


class MetricsQueries:
    """Query interface for the orchestrator metrics database."""

    def __init__(self, db_path: str = "~/.aureax/telemetry/metrics.db") -> None:
        """Point at the metrics database for querying.

        Read-only: this class never creates the file or its parent, so a
        mistyped ``metrics_db_path`` reads as an empty fleet rather than
        silently building a directory tree. MetricsStorage.__init__ owns
        creating both.
        """
        self.db_path = Path(db_path).expanduser()

    def _conn(self):
        return connection(self.db_path, rows=True, foreign_keys=False)

    def _missing(self) -> bool:
        """True when no database file exists yet — an empty fleet, not an error.

        ``aureax status`` builds a MetricsQueries with no MetricsStorage beside
        it, so on an orchestrator where the dashboard has never run there is
        nothing to open and sqlite3.connect would raise "unable to open
        database file". _table_exists covers only the file-without-table case,
        which needs a connection first.
        """
        return not self.db_path.exists()

    def _table_exists(self, conn: sqlite3.Connection) -> bool:
        """Check if the metrics table exists."""
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='metrics'"
        ).fetchone()
        return row is not None

    def get_latest_metrics(self, node_id: str) -> list[dict]:
        """Get the most recent value for each metric for ONE node.

        The single-node form, for a caller that already knows which node it
        wants. Anything that wants the whole fleet calls ``get_latest_all``
        instead — this in a loop is a connection and a statement per node,
        which is why the dashboard's polled route does not use it.

        Grouped to MAX(id) and joined back, rather than a correlated
        MAX(timestamp) subquery re-run per candidate row: the join is linear in
        a table that is only ever appended to, where the correlated form is
        quadratic in rows-per-node (measured on a 10-day soak, 115k rows: 84 ms
        per call against 8.5 ms here). MAX(id) also breaks ties
        deterministically, which MAX(timestamp) does not — several metrics
        share one push's timestamp.
        """
        if self._missing():
            return []
        with self._conn() as conn:
            if not self._table_exists(conn):
                return []
            rows = conn.execute("""
                SELECT m.metric_name, m.metric_value, m.timestamp
                FROM metrics m
                JOIN (
                    SELECT metric_name, MAX(id) AS mid
                    FROM metrics
                    WHERE node_id = ?
                    GROUP BY metric_name
                ) latest ON m.id = latest.mid
                ORDER BY m.metric_name
            """, (node_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_latest_all(self) -> list[dict]:
        """Newest row per (node_id, metric_name), for the WHOLE fleet, in one query.

        What the dashboard's /api/nodes route needs, in one connection and one
        statement: the route is polled by every open dashboard tab, over an
        append-only table a live fleet grows by thousands of rows an hour, so a
        per-node query loop would cost a connection and a statement each.

        Same shape as ``get_latest_metrics``: group to MAX(id) and join back,
        rather than a correlated MAX(timestamp) subquery re-run per candidate
        row. MAX(id) also breaks ties deterministically, which MAX(timestamp)
        does not — several metrics share one push's timestamp.

        ``received_at`` comes back beside ``timestamp`` because it is the
        server-stamped one: /api/nodes reports it as last_seen, so a node's own
        clock cannot decide whether the dashboard calls it online.

        Ordered by node_id so callers grouping on it see each node's rows
        together, and so the fleet listing has a stable order.
        """
        if self._missing():
            return []
        with self._conn() as conn:
            if not self._table_exists(conn):
                return []
            rows = conn.execute("""
                SELECT m.node_id, m.metric_name, m.metric_value, m.timestamp,
                       m.received_at
                FROM metrics m
                JOIN (
                    SELECT node_id, metric_name, MAX(id) AS mid
                    FROM metrics
                    GROUP BY node_id, metric_name
                ) latest ON m.id = latest.mid
                ORDER BY m.node_id, m.metric_name
            """).fetchall()
            return [dict(r) for r in rows]

    def get_metric_history(self, node_id: str, minutes: int = 60) -> dict[str, list[list]]:
        """One node's recent samples, ``{metric_name: [[timestamp, value], ...]}``.

        The series behind the node-detail view's sparklines. Filters on
        ``timestamp``, the node's OWN clock, because that is what orders the
        samples against each other — while ``last_seen`` beside it on the same
        page is the server-stamped ``received_at``. The two therefore read
        different clocks on purpose: a node whose clock runs behind shows an
        empty series next to a green dot, which is a contradiction the operator
        can act on rather than a plausible line drawn at the wrong times.

        Cutoff built in Python, exactly like ``MetricsStorage.rotate``: rows
        store full ISO timestamps ("...T23:50:04+00:00") and SQLite's
        ``datetime('now', ...)`` returns a space-separated string, so comparing
        them diverges at index 10 and the window silently means "today or
        later".

        DESC with a LIMIT and reversed here, rather than ASC: the bound has to
        keep the NEWEST samples — an ASC limit truncates the live end of every
        series, which is the end being drawn. The ceiling is a bound on the
        response, not a window: an hour of a nine-node fleet is ~900 rows, so
        only a node reporting far faster than the collector's 60 s interval
        reaches it. Served by ``idx_node_ts``.
        """
        if self._missing():
            return {}
        cutoff = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()
        with self._conn() as conn:
            if not self._table_exists(conn):
                return {}
            rows = conn.execute("""
                SELECT metric_name, timestamp, metric_value
                FROM metrics
                WHERE node_id = ? AND timestamp > ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 5000
            """, (node_id, cutoff)).fetchall()
        series: dict[str, list[list]] = {}
        for row in reversed(rows):
            series.setdefault(row["metric_name"], []).append(
                [row["timestamp"], row["metric_value"]]
            )
        return series

    def get_node_health_summary(self) -> list[dict]:
        """Per-node last_seen and sample count, for ``aureax status``.

        ``last_seen`` is a MAX over the server-stamped ``received_at``, never
        over the node-supplied ``timestamp``, for the same reason the
        dashboard's /api/nodes reports arrival time (api._collect_nodes): it is
        a freshness signal about the last push that landed here, and a node
        with a wrong clock would otherwise read as permanently fresh or
        permanently stale. Ingest is unauthenticated, so a posted timestamp is
        also a value the operator's own status output should not be steered by.
        """
        if self._missing():
            return []
        with self._conn() as conn:
            if not self._table_exists(conn):
                return []
            rows = conn.execute("""
                SELECT
                    node_id,
                    MAX(received_at) as last_seen,
                    COUNT(*) as total_metrics
                FROM metrics
                GROUP BY node_id
                ORDER BY node_id
            """).fetchall()
            return [dict(r) for r in rows]
