"""Time-window filters must actually compare time of day.

Rows store ``datetime.now(UTC).isoformat()`` — "2026-08-13T23:50:04+00:00",
with a "T". SQLite's ``datetime('now', ...)`` returns "2026-08-13 23:50:04",
with a space. Comparing them is a string comparison that diverges at index 10,
where "T" (0x54) sorts above " " (0x20) — so the time-of-day was never reached
and every "last N minutes" window silently meant "today or later".
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from client.telemetry.storage import TelemetryStorage
from orchestrator.telemetry.queries import MetricsQueries


def _iso(**delta) -> str:
    """A stored-format timestamp offset from now."""
    return (datetime.now(UTC) - timedelta(**delta)).isoformat()


def _rows(db_path, sql: str) -> list[tuple]:
    """Read straight from SQLite, so a test never depends on the API it checks."""
    with sqlite3.connect(str(db_path)) as conn:
        return conn.execute(sql).fetchall()


def test_client_latest_per_metric_excludes_rows_older_than_the_window(tmp_path) -> None:
    """The MCP get_metrics tool's window, and the only caller-controlled one."""
    storage = TelemetryStorage(str(tmp_path / "t.db"))
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": _iso(minutes=5)},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 2.0, "timestamp": _iso(minutes=180)},
    ])

    latest, sample_count = storage.latest_per_metric(since_minutes=60)

    assert latest == {"cpu": 1.0}, "3h-old row passed a 60-minute filter (date-only comparison)"
    assert sample_count == 1

# The T-vs-space comparison now has to be pinned on BOTH sides: the
# orchestrator grew a time-series query of its own — get_metric_history, the
# node-detail view's sparklines — whose window is tested below.


def test_client_latest_per_metric_breaks_ties_by_id(tmp_path) -> None:
    """Several metrics share one push's timestamp, so ties must break by id.

    Its orchestrator twin is pinned below; a wrong tie-break here surfaces as a
    plausible-looking stale number on the dashboard rather than as an error.
    """
    storage = TelemetryStorage(str(tmp_path / "t.db"))
    shared = _iso(minutes=1)
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "mem", "metric_value": 2.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 3.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 9.0, "timestamp": _iso(days=2)},
    ])

    assert storage.latest_per_metric(since_minutes=60) == ({"cpu": 3.0, "mem": 2.0}, 3)


def test_rotate_prunes_by_time_not_only_by_date(tmp_path) -> None:
    """Pruning was a day less aggressive than retention_days configured."""
    storage = TelemetryStorage(str(tmp_path / "t.db"))
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 9.0,
         "timestamp": _iso(days=7, minutes=30)},
    ])
    # Mark it pushed — rotate only removes shipped rows.
    ids = [r["id"] for r in storage.get_unpushed(limit=10)]
    storage.mark_pushed(ids)

    assert storage.rotate(retention_days=7) == 1


def test_rotate_hard_cutoff_bounds_an_unreachable_nodes_buffer(tmp_path) -> None:
    """A node that never reaches the orchestrator marks nothing pushed.

    Without the second, unconditional cutoff at 3x retention its buffer grew
    without bound on a hostPath disk — and the pushed-only DELETE can never
    touch a row of it.
    """
    storage = TelemetryStorage(str(tmp_path / "t.db"))
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": _iso(days=1)},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 2.0, "timestamp": _iso(days=8)},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 3.0, "timestamp": _iso(days=22)},
    ])

    assert storage.rotate(retention_days=7) == 1
    surviving = {v for v, in _rows(tmp_path / "t.db", "SELECT metric_value FROM metrics")}
    # 8 days is past retention but unpushed and inside 3x, so it stays.
    assert surviving == {1.0, 2.0}


# --- orchestrator-side retention ---------------------------------------------

def test_orchestrator_metrics_are_pruned_to_retention(tmp_path) -> None:
    """`server.telemetry.storage.retention_days` had no implementation.

    Nothing deleted from the orchestrator metrics table, so a 9-node fleet
    reporting 8 metrics a minute grew it by ~3.1M rows a month — and the
    dashboard's per-node "latest metrics" query scans it.
    """
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    storage._last_rotate = 1e18  # isolate rotate() from the insert-path trigger
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": _iso(minutes=1)},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 2.0, "timestamp": _iso(days=40)},
        # Just over the edge — caught only if the cutoff compares time of day.
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 9.0,
         "timestamp": _iso(days=30, minutes=5)},
    ])

    assert storage.rotate(30) == 2
    remaining = MetricsQueries(str(tmp_path / "m.db")).get_latest_metrics("n1")
    assert [r["metric_value"] for r in remaining] == [1.0]


def test_metrics_rotation_is_rate_limited(tmp_path) -> None:
    """Pruning rides the insert path, so it must not run on every push."""
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    row = [{"node_id": "n", "metric_name": "c", "metric_value": 1.0, "timestamp": _iso(minutes=1)}]

    storage.insert(row)
    first = storage._last_rotate
    assert first is not None, "expected a prune on the first insert"
    storage.insert(row)
    assert storage._last_rotate == first, "pruned again within the hour"


def test_latest_metrics_picks_the_newest_row_per_metric(tmp_path) -> None:
    """Several metrics share one push's timestamp, so ties must break by id."""
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    shared = _iso(minutes=1)
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "mem", "metric_value": 2.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 3.0, "timestamp": shared},
        {"node_id": "n2", "metric_name": "cpu", "metric_value": 99.0, "timestamp": shared},
    ])

    latest = {r["metric_name"]: r["metric_value"]
              for r in MetricsQueries(str(tmp_path / "m.db")).get_latest_metrics("n1")}
    assert latest == {"cpu": 3.0, "mem": 2.0}, "tie broken wrong, or another node leaked in"


def test_latest_all_answers_the_whole_fleet_in_one_query(tmp_path, monkeypatch) -> None:
    """The dashboard's /api/nodes route ran a full-table `SELECT DISTINCT
    node_id` scan and then one connection and one query PER NODE, on a route
    every open tab polls over an append-only table.

    Same MAX(id) tie-break as the single-node form: several metrics share one
    push's timestamp, so ordering by timestamp alone picks arbitrarily.
    """
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    shared = _iso(minutes=1)
    storage.insert([
        {"node_id": "n2", "metric_name": "cpu", "metric_value": 1.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 2.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "mem", "metric_value": 3.0, "timestamp": shared},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 4.0, "timestamp": shared},
    ])

    queries = MetricsQueries(str(tmp_path / "m.db"))
    opened = []
    real_conn = MetricsQueries._conn
    monkeypatch.setattr(
        MetricsQueries, "_conn",
        lambda self: (opened.append(1), real_conn(self))[1],
    )

    rows = queries.get_latest_all()

    assert len(opened) == 1, f"{len(opened)} connections for one fleet-wide read"
    assert [(r["node_id"], r["metric_name"], r["metric_value"]) for r in rows] == [
        # ORDER BY node_id, metric_name — the node ordering the DISTINCT scan
        # used to impose, which _collect_nodes still relies on to group.
        ("n1", "cpu", 4.0),
        ("n1", "mem", 3.0),
        ("n2", "cpu", 1.0),
    ]


def test_metric_history_excludes_rows_older_than_the_window(tmp_path) -> None:
    """The orchestrator's own "last N minutes" window, and the second place the
    stored-ISO-vs-SQLite-datetime comparison could silently mean "today"."""
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": _iso(minutes=5)},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 2.0, "timestamp": _iso(minutes=180)},
        {"node_id": "n2", "metric_name": "cpu", "metric_value": 9.0, "timestamp": _iso(minutes=5)},
    ])

    history = MetricsQueries(str(tmp_path / "m.db")).get_metric_history("n1", minutes=60)

    assert [v for _ts, v in history["cpu"]] == [1.0], "3h-old row passed a 60-minute filter"


def test_metric_history_is_grouped_per_metric_and_ordered_oldest_first(tmp_path) -> None:
    """Each series is drawn left to right, so it must arrive in that order —
    the query reads DESC to bound on the newest rows and reverses here."""
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    old, mid, new = _iso(minutes=30), _iso(minutes=20), _iso(minutes=10)
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": new},
        {"node_id": "n1", "metric_name": "mem", "metric_value": 7.0, "timestamp": mid},
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 2.0, "timestamp": old},
    ])

    history = MetricsQueries(str(tmp_path / "m.db")).get_metric_history("n1")

    assert history == {"cpu": [[old, 2.0], [new, 1.0]], "mem": [[mid, 7.0]]}


def test_metric_history_truncation_keeps_the_newest_samples(tmp_path) -> None:
    """The row cap bounds the response, so it has to bite at the OLD end.

    An ``ORDER BY timestamp ASC LIMIT n`` reads the same rows and drops the
    live end of every series — the only part of a sparkline anyone looks at.
    """
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    # One over the cap, each a millisecond older than the last: distinct
    # timestamps, all well inside the window, so the cap and not the window is
    # what drops exactly one row.
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": float(i),
         "timestamp": _iso(milliseconds=i)}
        for i in range(5001)
    ])

    series = MetricsQueries(str(tmp_path / "m.db")).get_metric_history("n1")["cpu"]

    assert len(series) == 5000
    assert series[-1][1] == 0.0, "the newest sample was truncated away"
    assert series[0][1] == 4999.0, "the cap has to bite at the old end"


def test_orchestrator_retention_also_prunes_on_arrival_time(tmp_path) -> None:
    """A skewed node clock could otherwise pin rows in the table forever.

    ``timestamp`` is copied verbatim from the pushed JSONL, so a node with no
    RTC that boots ahead — or anything on the LAN posting to the
    unauthenticated ingest — writes rows a timestamp-only prune never reaches,
    and server.telemetry.storage.retention_days stops bounding the table.
    ``received_at`` is server-stamped and no poster can move it.
    """
    from orchestrator.telemetry.storage import MetricsStorage

    db = tmp_path / "m.db"
    storage = MetricsStorage(str(db))
    storage._last_rotate = 1e18  # isolate rotate() from the insert-path trigger
    future = (datetime.now(UTC) + timedelta(days=365)).isoformat()
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": future},
    ])
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE metrics SET received_at = ?", (_iso(days=40),))

    assert storage.rotate(30) == 1
    assert _rows(db, "SELECT COUNT(*) FROM metrics") == [(0,)]


def test_queries_never_create_the_database_directory(tmp_path) -> None:
    """MetricsQueries reads; MetricsStorage owns creating the file and its parent.

    The mkdir here made a mistyped ``metrics_db_path`` silently build a
    directory tree instead of failing. A database that does not exist yet must
    still read as an empty fleet — `aureax status` builds one of these with no
    MetricsStorage beside it.
    """
    missing = tmp_path / "typo" / "m.db"
    queries = MetricsQueries(str(missing))

    assert queries.get_node_health_summary() == []
    assert queries.get_latest_all() == []
    assert queries.get_latest_metrics("n1") == []
    assert queries.get_metric_history("n1") == {}
    assert not missing.parent.exists(), "a read created a directory"


def test_health_summary_last_seen_is_arrival_time_not_the_nodes_clock(tmp_path) -> None:
    """`aureax status` prints this figure, and the dashboard prints its own from
    ``received_at`` — they must not disagree for a node whose clock is wrong.

    ``timestamp`` is copied verbatim out of the pushed JSONL, so an RTC-less
    node reads as permanently fresh or permanently stale, and ingest is
    unauthenticated: anything on the LAN could otherwise move the operator's
    number. ``received_at`` is stamped here on arrival.
    """
    from orchestrator.telemetry.storage import MetricsStorage

    db = tmp_path / "m.db"
    storage = MetricsStorage(str(db))
    storage._last_rotate = 1e18  # retention prunes on both columns; keep both rows
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0,
         "timestamp": (datetime.now(UTC) + timedelta(days=3650)).isoformat()},
        {"node_id": "n1", "metric_name": "mem", "metric_value": 2.0, "timestamp": _iso(days=3650)},
    ])

    [(node_id, last_seen, total)] = [
        (r["node_id"], r["last_seen"], r["total_metrics"])
        for r in MetricsQueries(str(db)).get_node_health_summary()
    ]

    assert (node_id, total) == ("n1", 2)
    assert last_seen == max(r[0] for r in _rows(db, "SELECT received_at FROM metrics"))
    assert last_seen not in [r[0] for r in _rows(db, "SELECT timestamp FROM metrics")]


# --- ingest validation: what may reach the table at all ----------------------

def test_ingest_skips_metric_values_the_table_cannot_carry() -> None:
    """Presence was checked but not type, and insert is all-or-nothing.

    A null trips ``NOT NULL constraint failed`` and a dict raises
    sqlite3.ProgrammingError, both from inside executemany — escaping ingest as
    a 500. The reporter marks pushed only on 200, so the node then re-sends the
    same poison batch every push interval, forever.
    """
    from orchestrator.telemetry.receiver import parse_jsonl

    def line(value):
        return json.dumps(
            {"node_id": "n1", "metric_name": "cpu", "metric_value": value, "timestamp": _iso()}
        )

    body = "\n".join([
        line(None),
        line({"a": 1}),
        line([1]),
        line("12"),
        line(True),          # bool is an int subclass; it would store as 1.0
        line(float("inf")),  # 500s /api/nodes for the whole fleet on render
        line(1.5),
    ])
    assert [r["metric_value"] for r in parse_jsonl(body)] == [1.5]


def test_ingest_requires_bounded_string_identifiers() -> None:
    """node_id and metric_name reach indexed columns from an unauthenticated endpoint."""
    from orchestrator.telemetry.receiver import parse_jsonl

    body = "\n".join([
        json.dumps({"node_id": 7, "metric_name": "cpu", "metric_value": 1.0, "timestamp": _iso()}),
        json.dumps({"node_id": "n1", "metric_name": "x" * 500, "metric_value": 1.0,
                    "timestamp": _iso()}),
        json.dumps({"node_id": "", "metric_name": "cpu", "metric_value": 1.0, "timestamp": _iso()}),
        json.dumps({"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": 17}),
        json.dumps({"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0,
                    "timestamp": _iso()}),
    ])
    parsed = parse_jsonl(body)
    assert len(parsed) == 1
    assert parsed[0]["node_id"] == "n1" and parsed[0]["metric_name"] == "cpu"
