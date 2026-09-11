"""SQLite-backed persistent system log.

Schema: a single ``entries`` table. Each row is one emit from a
``SystemLogger`` — timestamp, level, source, text — plus ``received_at``, the
arrival time this store stamps itself so that retention does not depend on
whoever supplied the timestamp.

Lives in ``client.log`` so both the orchestrator dashboard and an edge
client can use the same shape against their own SQLite files.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from client.sqlite import connection


@dataclass
class LogEntry:
    """One row from the system log.

    ``level`` is ``debug | info | warn | error | success | hint``.
    ``success`` and ``hint`` come only from the orchestrator; ``debug | info |
    warn | error`` come from either side — an edge node ships warn/error as
    they happen (``LoggingConfig.ship_levels``) plus whatever levels its crash
    flush carries.

    ``source`` is the emitting subsystem: ``lifecycle | chat | telemetry |
    pods | nodes | drift | config | stream | access`` on the orchestrator,
    ``supervisor`` or a child's name on an edge node, and ``node:<id>`` once a
    shipped entry has been ingested by the orchestrator.
    """

    id: int
    ts: str
    level: str
    source: str
    text: str


# How long a row survives. A module constant because there is no schema field
# for it on the orchestrator side: the edge client reads `logging.retention_days`
# from its own per-node config and passes it to prune_older_than directly (see
# client/entrypoint.py::_log_loop), but the dashboard's store is opened by
# create_app with no config of its own. 30 days matches
# MetricsStorage.retention_days, the orchestrator's other long-lived table.
LOG_RETENTION_DAYS = 30


class LogStore:
    """SQLite store for system-log entries.

    Self-pruning, because on the orchestrator side nothing else is: the
    dashboard's store collects every lifecycle emit, every watcher line and
    every entry shipped by every node, and ``prune_older_than``'s only other
    caller is the edge supervisor's hourly log loop, which runs on a node.

    Pruning rides the WRITES rather than a background task, the same shape
    MetricsStorage uses: inserts are the only thing that grows the table, and
    this way there is no extra thread to own or shut down.

    The two pruners do not fight. An edge node runs both this and its own loop,
    but a DELETE of rows another DELETE already removed is a no-op, and the
    hourly guard below means at most one runs per hour per process. The node's
    own loop is the tighter of the two (7 days by default), so it keeps
    deciding retention there; this one is the backstop for the orchestrator.
    """

    _PRUNE_INTERVAL_SECONDS = 3600.0

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(str(db_path)).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # None, not 0.0: time.monotonic() is boot-relative on this platform, so
        # "never pruned" written as 0.0 reads as "pruned at boot" — on a machine
        # up for less than the interval, `now - 0.0` is under an hour and
        # pruning would never start at all. Same sentinel as
        # MetricsStorage._last_rotate, for the same reason.
        self._last_prune: float | None = None
        self._init()

    def _maybe_prune(self) -> None:
        """Prune at most hourly; never let pruning break a write.

        Called BEFORE the insert, not after. A shipped edge entry carries the
        time of the ORIGINAL event, so a crash flush that arrives from a node
        that has been offline for weeks can be older than the cutoff — and
        pruning after the write would delete the very row just inserted while
        ``append``/``append_many`` still returns a LogEntry for it, handing the
        SSE stream an id that is not in the table. Retention still bounds the
        table; such a row simply goes on the next hour's pass.

        A failed prune is a warning, not an exception: both writers of this
        store swallow exceptions silently (client/log/capture.py and
        entrypoint.py's _StoreHandler), so raising here would kill the log path
        with no symptom.
        """
        if LOG_RETENTION_DAYS <= 0:
            return
        now = time.monotonic()
        if self._last_prune is not None and now - self._last_prune < self._PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        try:
            deleted = self.prune_older_than(LOG_RETENTION_DAYS)
            if deleted:
                # "backstop", because an edge node runs this AND its own
                # configured prune against the same database. Two lines that
                # differed only in their day count left an operator unable to
                # tell which retention had actually applied — which is the
                # thing under investigation whenever this line is read.
                logging.getLogger("aureax.logs").info(
                    "Logs: backstop prune removed %d row(s) older than %dd",
                    deleted, LOG_RETENTION_DAYS,
                )
        except Exception as e:
            logging.getLogger("aureax.logs").warning("Log backstop prune failed: %s", e)

    def _conn(self):
        # 30 s, not sqlite3's 5 s: three writers reach this file on a node (the
        # hourly log loop, the crash flush, the supervisor's log handler) and
        # every one of them swallows OperationalError, so a lock wait that
        # gives up is a dropped line with no symptom. Waiting longer is the
        # cheaper failure — the writes are millisecond batches.
        return connection(self.db_path, rows=True, foreign_keys=False, timeout=30.0)

    def _init(self) -> None:
        with self._conn() as conn:
            # Table first, migration second, indexes last: the received_at
            # index cannot be created against a store that predates the column.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    level       TEXT NOT NULL,
                    source      TEXT NOT NULL DEFAULT 'unknown',
                    text        TEXT NOT NULL,
                    received_at TEXT
                )
                """
            )
            self._add_received_at(conn)
            conn.executescript(
                """
                CREATE INDEX IF NOT EXISTS entries_ts ON entries(ts DESC);
                -- prune_older_than's second DELETE. Without it that statement
                -- scans the whole table inside a write transaction to remove
                -- the fraction of it that has aged out.
                CREATE INDEX IF NOT EXISTS entries_received_at
                    ON entries(received_at);
                -- recent(source=...): the dashboard's per-node log tail, which
                -- polls for ONE node's rows out of a table carrying every
                -- lifecycle emit, every watcher line and every other node's
                -- shipped entries. A healthy node ships almost nothing, so the
                -- unindexed form walks 30 days of rowids newest-first on every
                -- poll to find the handful that match. (id DESC) rather than
                -- (source, ts): the query orders by id, so a ts index would
                -- still sort. The per-insert b-tree write is paid on edge
                -- nodes too — this store is one shape shared by both sides —
                -- but a node writes only warn/error, so it is a few rows a day
                -- there.
                CREATE INDEX IF NOT EXISTS entries_source_id
                    ON entries(source, id DESC);
                -- Superseded by the index above, which orders the way the
                -- reader does. Dropped rather than left so a long-lived store
                -- sheds a b-tree write nothing reads.
                DROP INDEX IF EXISTS entries_source_ts;
                """
            )

    @staticmethod
    def _add_received_at(conn: sqlite3.Connection) -> None:
        """Give a store created before ``received_at`` the column.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op against a file that already
        exists, so without this an already-provisioned store — on a node or on
        the orchestrator — keeps the five-column shape and every prune raises
        "no such column". Guarded on PRAGMA table_info so it runs exactly once
        per database, and existing rows are backfilled from ``ts``, the only
        arrival time ever recorded for them.
        """
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(entries)")}
        if "received_at" in columns:
            return
        conn.execute("ALTER TABLE entries ADD COLUMN received_at TEXT")
        conn.execute("UPDATE entries SET received_at = ts WHERE received_at IS NULL")

    def append(self, level: str, text: str, source: str = "unknown") -> LogEntry:
        """Insert one log entry and return it with its assigned ``id`` and ``ts``."""
        self._maybe_prune()
        ts = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            # received_at == ts on this path: a locally emitted row arrives the
            # moment it happens.
            cur = conn.execute(
                "INSERT INTO entries (ts, level, source, text, received_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (ts, level, source, text, ts),
            )
            entry_id = cur.lastrowid
        return LogEntry(id=entry_id, ts=ts, level=level, source=source, text=text)

    def append_many(self, records: list[tuple]) -> list[LogEntry]:
        """Insert many rows in one transaction.

        Each record is ``(level, text, source)`` or ``(level, text, source, ts)``.
        The 4-tuple form exists for the ingest path: shipped edge entries carry
        the timestamp of the original event, and stamping a whole batch with
        one ingest time misorders it against everything else and lies about
        when the node said any of it.

        Batches share one SQLite connection and transaction, reducing fsync work
        when ingesting a node's buffered logs.
        """
        if not records:
            return []
        # Once per batch, not once per row: this is the orchestrator's ingest
        # path and a crash flush arrives as ~80 rows at a time.
        self._maybe_prune()
        now = datetime.now(UTC).isoformat()
        out: list[LogEntry] = []
        # One connection and one commit for the batch. Looped execute rather
        # than executemany: executemany does not report per-row ids (lastrowid
        # is None on some sqlite3 builds), and the savings here come from not
        # reopening the connection and committing per row, not from the
        # statement dispatch.
        with self._conn() as conn:
            for record in records:
                level, text, source = record[:3]
                ts = record[3] if len(record) > 3 and record[3] else now
                # ``ts`` may be the node's clock; ``received_at`` is this
                # store's, which is what keeps the row prunable.
                cur = conn.execute(
                    "INSERT INTO entries (ts, level, source, text, received_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (ts, level, source, text, now),
                )
                out.append(
                    LogEntry(id=cur.lastrowid, ts=ts, level=level, source=source, text=text)
                )
        return out

    def recent(self, limit: int = 500, *, source: str | None = None) -> list[LogEntry]:
        """Most-recent-first slice; caller typically reverses for display.

        ``source`` narrows to one emitter — ``node:<id>`` for the dashboard's
        per-node log tail. Keyword-only, because the filter and the limit are
        both easy to write in either order and swapping them silently returns
        the wrong rows rather than raising.
        """
        sql = "SELECT * FROM entries"
        params: tuple = ()
        if source is not None:
            sql += " WHERE source = ?"
            params = (source,)
        with self._conn() as conn:
            rows = conn.execute(
                f"{sql} ORDER BY id DESC LIMIT ?", (*params, limit)
            ).fetchall()
        return [_entry_from_row(r) for r in rows]

    def since_id(self, min_id: int, limit: int = 1000) -> list[LogEntry]:
        """Return entries with ``id > min_id`` in ascending order — for tail/poll consumers."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM entries WHERE id > ? ORDER BY id ASC LIMIT ?",
                (min_id, limit),
            ).fetchall()
        return [_entry_from_row(r) for r in rows]

    def prune_older_than(self, days: int) -> int:
        """Drop entries older than N days. Returns number deleted.

        Cutoff built in Python, exactly like ``TelemetryStorage.rotate``: rows
        store full ISO timestamps, and comparing one against SQLite's date-only
        ``date(...)`` string keeps every row from the cutoff day itself, so
        retention runs up to a day longer than configured.

        Two DELETEs, one per column, the same shape as
        ``MetricsStorage.rotate``. ``ts`` is the ORIGINATING clock — the ingest
        path copies it verbatim out of the shipped JSONL
        (orchestrator/logs/receiver.py::ingest) after only a parseability
        check — so a node with no RTC that boots years ahead, or anything on
        the LAN posting to the unauthenticated logs endpoint, writes rows a
        ts-only prune can never reach and retention stops bounding the table.
        ``received_at`` is stamped by this store on the way in and no poster
        can move it. Two statements rather than one OR: an OR across two
        columns can use neither single-column index, which is the full scan
        this shape exists to avoid.
        """
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            deleted = conn.execute(
                "DELETE FROM entries WHERE ts < ?", (cutoff,)
            ).rowcount
            deleted += conn.execute(
                "DELETE FROM entries WHERE received_at < ?", (cutoff,)
            ).rowcount
            return deleted


def _entry_from_row(row: sqlite3.Row) -> LogEntry:
    return LogEntry(
        id=row["id"],
        ts=row["ts"],
        level=row["level"],
        source=row["source"],
        text=row["text"],
    )
