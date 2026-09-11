"""Edge -> orchestrator log shipping.

This path exists because `kubectl logs` cannot work in this topology: it
proxies through each node's kubelet on :10250, and the k3d server container
cannot route there for any LAN-joined node. A shard worker is worse still — it
runs rpc-server and no HTTP server at all, so before this its crashes were only
visible by SSHing to the node.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import ssl
import time
import types
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import certifi
import httpx
import pytest

from client.log.capture import ChildLogCapture, classify
from client.log.reporter import LogReporter
from client.log.store import LogStore
from client.telemetry.reporter import post_jsonl
from orchestrator.lifecycle import SystemLogger
from orchestrator.logs.receiver import ingest, parse_jsonl

# --- classifying llama.cpp output --------------------------------------------

def test_llama_level_marker_is_parsed() -> None:
    """llama.cpp prints <timestamp> <I|W|E> <component>, which is the cheap signal."""
    assert classify("561.18.787.826 I slot print_timing: id 0 | eval time = 1") == "info"
    assert classify("0.00.034.776 W srv  llama_server: CORS is set to allow all") == "warn"
    assert classify("15.29.768.587 E recv failed (bytes_recv=0, size_to_recv=8)") == "error"


def test_a_stray_capital_is_not_read_as_a_level() -> None:
    """The marker is anchored, or ordinary prose would be classified as an error."""
    assert classify("Extremely ordinary sentence about E and W") == "info"


def test_unmarked_failure_lines_are_treated_as_errors() -> None:
    """GGML aborts print a bare assertion and a raw backtrace, with no level char.

    That text is the entire reason this feature exists, so it must not be
    dropped as an unremarkable info line.
    """
    assert classify("/build/llama.cpp/ggml/src/ggml-rpc.cpp:644: Remote RPC server crashed") == "error"
    assert classify("GGML_ASSERT(ne00 % 512 == 0) failed") == "error"
    assert classify("error loading model: failed to open GGUF") == "error"


# --- what gets stored, and when ----------------------------------------------

def test_routine_chatter_is_buffered_not_stored(tmp_path) -> None:
    """llama.cpp emits a timing line every few tokens.

    Storing those continuously would swamp the orchestrator and put real traffic
    on a 1 GbE fleet that is already latency-bound.
    """
    store = LogStore(tmp_path / "sys.db")
    cap = ChildLogCapture(store, source="llama-server")

    for i in range(50):
        cap.observe(f"10.00.000.00{i} I slot print_timing: tg = 1.6 t/s")

    assert store.recent(limit=100) == []


def test_warnings_and_errors_are_stored_immediately(tmp_path) -> None:
    store = LogStore(tmp_path / "sys.db")
    cap = ChildLogCapture(store, source="llama-server")

    cap.observe("1.00.000.000 I slot print_timing: routine")
    cap.observe("2.00.000.000 E recv failed (bytes_recv=0)")

    stored = store.recent(limit=10)
    assert len(stored) == 1
    assert stored[0].level == "error"
    assert "recv failed" in stored[0].text
    assert stored[0].source == "llama-server"


def test_crash_flush_recovers_the_run_up(tmp_path) -> None:
    """The lines BEFORE a crash are what explain it, and are what is lost today."""
    store = LogStore(tmp_path / "sys.db")
    cap = ChildLogCapture(store, source="rpc-server", context_lines=10)

    for i in range(30):
        cap.observe(f"1.00.000.{i:03d} I loading tensor {i}")
    cap.flush_crash_context(exit_code=139)

    texts = [e.text for e in store.recent(limit=100)]
    assert any("exited with code 139" in t for t in texts)
    # Only the ring's worth of context, not all 30 lines.
    assert sum("loading tensor" in t for t in texts) == 10
    assert "loading tensor 29" in " ".join(texts)
    assert "loading tensor 5" not in " ".join(texts)


def test_crash_flush_does_not_duplicate_already_stored_errors(tmp_path) -> None:
    store = LogStore(tmp_path / "sys.db")
    cap = ChildLogCapture(store, source="llama-server", context_lines=10)

    cap.observe("1.00.000.000 E the one real error")
    cap.flush_crash_context(exit_code=1)

    texts = [e.text for e in store.recent(limit=50)]
    assert sum("the one real error" in t for t in texts) == 1


def test_a_giant_child_line_is_truncated_at_the_source(tmp_path) -> None:
    """The pump splits only on b"\\n".

    llama.cpp's \\r-based load progress and a GGML abort backtrace each arrive
    as one enormous line. ``context_lines`` bounds the ring's line COUNT, not
    its bytes, and the orchestrator discards the excess on arrival anyway — so
    shipping it whole only costs bandwidth on a fleet already latency-bound.
    """
    store = LogStore(tmp_path / "sys.db")
    ChildLogCapture(store, source="llama-server").observe("1.0 E " + "x" * 20_000)
    assert len(store.recent(limit=1)[0].text) == 4000


class _BrokenLogStore(LogStore):
    """A store whose every write fails — a full disk, or a corrupt system.db."""

    def append(self, *_a, **_kw):
        raise sqlite3.OperationalError("database or disk is full")

    def append_many(self, _records):
        raise sqlite3.OperationalError("database or disk is full")


def test_a_dead_store_never_breaks_the_pump_thread(tmp_path, caplog) -> None:
    """observe() runs on the thread that tees child output to container stdout.

    An exception escaping it kills that tee, and every later crash is reduced
    to "exited with code N" — the opposite of what this module exists for.
    """
    caplog.set_level(logging.WARNING, logger="aureax.telemetry")
    cap = ChildLogCapture(_BrokenLogStore(tmp_path / "sys.db"), source="llama-server")

    cap.observe("1.0 E boom")
    cap.observe("2.0 I routine")
    cap.flush_crash_context(139)

    # Reported, because a silently dead store looks exactly like a quiet node —
    # but once, not once per line of child output.
    reports = [r for r in caplog.records if "log store append failing" in r.getMessage()]
    assert len(reports) == 1


def test_the_store_failure_latch_is_per_capture(tmp_path) -> None:
    """A fresh capture after a restart must report, and store, again."""
    store = LogStore(tmp_path / "sys.db")
    ChildLogCapture(_BrokenLogStore(tmp_path / "broken.db"), source="x").observe("1.0 E lost")

    ChildLogCapture(store, source="llama-server").observe("2.0 E kept")
    assert [e.text for e in store.recent(limit=5)] == ["2.0 E kept"]


# --- local retention ---------------------------------------------------------

def _iso(**delta) -> str:
    """A stored-format timestamp offset from now."""
    return (datetime.now(UTC) - timedelta(**delta)).isoformat()


def test_prune_older_than_compares_the_time_of_day(tmp_path) -> None:
    """Comparing against SQLite's date-only ``date(...)`` kept every row from
    the cutoff day itself, so retention ran up to a day longer than configured.
    """
    store = LogStore(tmp_path / "sys.db")
    store.append_many([
        ("info", "ancient", "supervisor", _iso(days=10)),
        ("info", "still inside", "supervisor", _iso(days=2)),
        ("info", "now", "supervisor", None),
    ])

    assert store.prune_older_than(7) == 1
    assert {e.text for e in store.recent(limit=10)} == {"still inside", "now"}


def test_a_future_dated_row_is_pruned_on_its_arrival_time(tmp_path) -> None:
    """``ts`` is the ORIGINATING clock, copied verbatim by the ingest path.

    A node with no RTC that boots years ahead — or anything on the LAN posting
    to the unauthenticated logs endpoint — writes rows a ts-only prune can
    never reach, and the store then grows without bound. ``received_at`` is
    stamped by the store on the way in and no poster can move it.
    """
    store = LogStore(tmp_path / "orch.db")
    store.append_many([("info", "from the future", "node:edge-01", _iso(days=-3650))])

    # Nothing is due: the row arrived just now, whatever its own clock says.
    assert store.prune_older_than(7) == 0

    # An arrival a fortnight old goes, however far ahead that clock reads.
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute("UPDATE entries SET received_at = ?", (_iso(days=14),))
    assert store.prune_older_than(7) == 1
    assert store.recent(limit=5) == []


def test_recent_can_isolate_one_nodes_shipped_lines(tmp_path) -> None:
    """The dashboard's per-node log tail. Every node's entries and every
    orchestrator emit share this one table, so without the filter a node's
    detail view would render the fleet's whole console."""
    store = LogStore(tmp_path / "orch.db")
    store.append_many([
        ("error", "rpc died", "node:shard-w3", None),
        ("info", "start finished", "lifecycle", None),
        ("warn", "slow load", "node:edge-01", None),
        ("error", "oom", "node:shard-w3", None),
    ])

    assert [e.text for e in store.recent(source="node:shard-w3")] == ["oom", "rpc died"]
    assert store.recent(source="node:nobody") == []
    # Unfiltered is unchanged: the Lifecycle console reads the whole table.
    assert len(store.recent(limit=10)) == 4


def test_the_source_filter_is_indexed_and_the_superseded_index_is_gone(tmp_path) -> None:
    """recent(source=...) reads ONE node's handful of rows out of 30 days of
    everything, on a 10 s poll; unindexed it walks the whole table's rowids
    every time. The (source, ts) index it replaces ordered by a column no
    statement reads, so leaving it would be a per-insert b-tree write for
    nothing — paid on edge nodes too, since they open this same store.
    """
    LogStore(tmp_path / "sys.db")
    reopened = LogStore(tmp_path / "sys.db")

    with sqlite3.connect(str(reopened.db_path)) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        plan = " ".join(
            str(cell) for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM entries WHERE source = ? ORDER BY id DESC LIMIT 1",
                ("node:edge-01",),
            ) for cell in row
        )

    assert "entries_source_id" in names
    assert "entries_source_ts" not in names
    assert "entries_source_id" in plan, f"the filter is not using its index: {plan}"


def test_a_store_created_before_received_at_gains_the_column(tmp_path) -> None:
    """CREATE TABLE IF NOT EXISTS is a no-op against a file already on disk.

    Without the migration, every prune against an already-provisioned store —
    on a node or on the orchestrator — raises "no such column".
    """
    db = tmp_path / "legacy.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE entries (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
            " level TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'unknown', text TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO entries (ts, level, source, text) VALUES (?, 'info', 'lifecycle', 'old')",
            (_iso(days=1),),
        )

    store = LogStore(db)

    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT ts, received_at FROM entries").fetchone()
    assert row["received_at"] == row["ts"], "existing rows keep the only arrival time known"
    assert store.prune_older_than(7) == 0
    assert store.prune_older_than(0) == 1


# --- the store prunes itself (the orchestrator has no loop of its own) --------


def test_the_store_prunes_itself_on_write(tmp_path) -> None:
    """Nothing on the ORCHESTRATOR side trimmed this table.

    ``prune_older_than``'s only caller was the edge supervisor's hourly log
    loop, so ~/.aureax/logs/system.db on the Mac grew for the life of the
    install — every lifecycle emit, every watcher line and every entry shipped
    by every node. Pruning rides the writes because inserts are the only thing
    that grows the table, and that way there is no thread to own or shut down.
    """
    from client.log.store import LOG_RETENTION_DAYS

    store = LogStore(tmp_path / "sys.db")
    # Written straight to SQLite: going through append_many would prune them on
    # the way in, which is the behaviour under test.
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.executemany(
            "INSERT INTO entries (ts, level, source, text) VALUES (?, ?, ?, ?)",
            [
                (_iso(days=LOG_RETENTION_DAYS + 1), "info", "lifecycle", "ancient"),
                (_iso(days=LOG_RETENTION_DAYS - 1), "info", "lifecycle", "still inside"),
            ],
        )

    store.append("info", "now", "lifecycle")

    assert {e.text for e in store.recent(limit=10)} == {"still inside", "now"}


def test_the_stores_prune_runs_at_most_hourly(tmp_path, monkeypatch) -> None:
    """It rides every write, so it must not run a DELETE per emitted line.

    This is also what keeps it from fighting the edge supervisor's own loop: an
    edge node runs both, and at most one prune per hour per process makes the
    duplicate DELETE a no-op rather than a cost on every log line.
    """
    store = LogStore(tmp_path / "sys.db")
    pruned: list[int] = []
    monkeypatch.setattr(LogStore, "prune_older_than", lambda _s, days: pruned.append(days) or 0)

    from client.log.store import LOG_RETENTION_DAYS

    store.append("info", "one", "lifecycle")
    assert pruned == [LOG_RETENTION_DAYS], "no prune on the first write"
    store.append("info", "two", "lifecycle")
    store.append_many([("info", "three", "lifecycle")])
    assert pruned == [LOG_RETENTION_DAYS], "pruned again within the hour"

    # An hour on, the next write prunes again.
    store._last_prune -= LogStore._PRUNE_INTERVAL_SECONDS
    store.append_many([("info", "four", "lifecycle")])
    assert pruned == [LOG_RETENTION_DAYS, LOG_RETENTION_DAYS]


def test_a_batch_prunes_once_not_once_per_row(tmp_path, monkeypatch) -> None:
    """append_many is the orchestrator's ingest path and a crash flush arrives
    as ~80 rows at a time."""
    store = LogStore(tmp_path / "sys.db")
    pruned: list[int] = []
    monkeypatch.setattr(LogStore, "prune_older_than", lambda _s, days: pruned.append(days) or 0)

    store.append_many([("info", f"line {i}", "node:edge-01") for i in range(50)])

    assert len(pruned) == 1


def test_a_write_is_never_undone_by_its_own_prune(tmp_path) -> None:
    """A shipped edge entry carries the time of the ORIGINAL event, so a crash
    flush from a node that has been offline for weeks can arrive already older
    than the cutoff.

    Pruning after the insert deleted exactly that row while append_many still
    returned a LogEntry for it, handing the SSE stream an id that is not in the
    table. The prune runs first instead; retention still bounds the table on
    the next hour's pass.
    """
    from client.log.store import LOG_RETENTION_DAYS

    store = LogStore(tmp_path / "orch.db")
    ancient = _iso(days=LOG_RETENTION_DAYS + 5)

    written = store.append_many([("error", "GGML_ASSERT failed", "node:edge-01", ancient)])

    assert [e.ts for e in written] == [ancient]
    assert [e.id for e in store.recent(limit=5)] == [written[0].id]


def test_a_failed_prune_never_breaks_a_write(tmp_path, monkeypatch) -> None:
    """Both writers of this store swallow exceptions silently
    (client/log/capture.py and entrypoint.py's _StoreHandler), so a raise here
    would kill the log path with no symptom at all."""
    store = LogStore(tmp_path / "sys.db")

    def boom(_self, _days):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(LogStore, "prune_older_than", boom)

    entry = store.append("error", "still recorded", "lifecycle")

    assert entry.text == "still recorded"
    assert [e.text for e in store.recent(limit=5)] == ["still recorded"]


# Sentinel, because None is a MEANINGFUL reporter value here — it is the
# no-logs-endpoint node the loop still has to run for.
_UNSET = object()


def _log_loop_client(bare_client, tmp_path, reporter=_UNSET):
    """A client scaffolded for _log_loop: a real store, a stubbed reporter."""
    return bare_client(
        _log_store=LogStore(tmp_path / "sys.db"),
        _log_reporter=(
            types.SimpleNamespace(push=lambda: None) if reporter is _UNSET else reporter
        ),
    )


def test_log_loop_prunes_the_local_buffer(bare_client, tmp_path, monkeypatch) -> None:
    """``logging.retention_days`` is enforced HERE and nowhere else.

    LogStore prunes itself at LOG_RETENTION_DAYS (30) as a backstop, which is
    the only trimming the orchestrator gets; the node's configured retention is
    the tighter of the two and reaches prune_older_than only through this loop.
    Without it that setting does nothing.
    """
    client = _log_loop_client(bare_client, tmp_path)
    pruned: list[int] = []
    monkeypatch.setattr(LogStore, "prune_older_than", lambda _s, days: pruned.append(days) or 0)
    # push_interval=1 means one sleep, and the sleep comes before the prune.
    monkeypatch.setattr(time, "sleep", lambda _s: setattr(client, "_shutdown", True))

    client._log_loop(1, 3)
    assert pruned == [3]


def test_log_loop_skips_pruning_when_retention_is_disabled(bare_client, tmp_path, monkeypatch) -> None:
    client = _log_loop_client(bare_client, tmp_path)
    pruned: list[int] = []
    monkeypatch.setattr(LogStore, "prune_older_than", lambda _s, days: pruned.append(days) or 0)
    monkeypatch.setattr(time, "sleep", lambda _s: setattr(client, "_shutdown", True))

    client._log_loop(1, 0)
    assert pruned == []


def test_log_loop_prunes_when_there_is_no_reporter(bare_client, tmp_path, monkeypatch) -> None:
    """Pruning is the loop's other job and must not depend on shipping.

    A node whose telemetry.remote carries no endpoint gets no LogReporter, and
    gating the whole loop on one left its hostPath-backed buffer trimmed only
    by LogStore's 30-day backstop, never by its configured retention.
    """
    client = _log_loop_client(bare_client, tmp_path, reporter=None)
    pruned: list[int] = []
    monkeypatch.setattr(LogStore, "prune_older_than", lambda _s, days: pruned.append(days) or 0)
    monkeypatch.setattr(time, "sleep", lambda _s: setattr(client, "_shutdown", True))

    client._log_loop(1, 3)
    assert pruned == [3]


def test_log_loop_pushes_when_there_is_a_reporter(bare_client, tmp_path, monkeypatch) -> None:
    """The counterpart to the test above: the push is skipped, not removed.

    The arguments are part of it. This loop is the same shape as the telemetry
    one — durable rows, a shared thread that also runs the hourly prune — so it
    drains a backlog (max_batches) and leaves the retry to the next cycle
    (attempts=1) rather than blocking here for the reporter's full backoff.
    A push() raising TypeError on an argument would be swallowed by the loop's
    own suppress and ship nothing at all, silently.
    """
    pushes: list[dict] = []
    client = _log_loop_client(
        bare_client, tmp_path,
        reporter=types.SimpleNamespace(push=lambda **kw: pushes.append(kw)),
    )
    monkeypatch.setattr(LogStore, "prune_older_than", lambda _s, _d: 0)
    monkeypatch.setattr(time, "sleep", lambda _s: setattr(client, "_shutdown", True))

    client._log_loop(1, 3)
    assert pushes == [{"max_batches": 10, "attempts": 1}]


def test_log_thread_starts_for_the_prune_alone(bare_client, tmp_path, monkeypatch) -> None:
    """_start_log_shipping gates on the STORE, not the reporter."""
    from client.entrypoint import AureaxClient

    client = bare_client(_shutdown=True, _log_store=LogStore(tmp_path / "sys.db"))

    ran: list[tuple[int, int]] = []
    monkeypatch.setattr(AureaxClient, "_log_loop", lambda _s, push, keep: ran.append((push, keep)))

    client._start_log_shipping()
    client._log_thread.join(5)
    assert ran == [(60, 7)], "the loop must run, or logging.retention_days does nothing"


def test_no_log_thread_without_a_store(bare_client, tmp_path, monkeypatch) -> None:
    """With no buffer there is nothing to push and nothing to prune."""
    from client.entrypoint import AureaxClient

    client = bare_client()
    monkeypatch.setattr(AureaxClient, "_log_loop", lambda *_a: pytest.fail("loop must not start"))

    client._start_log_shipping()
    assert client._log_thread is None


# --- shipping ----------------------------------------------------------------

class _FakeResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_shipper_advances_its_cursor_and_does_not_resend(tmp_path, monkeypatch) -> None:
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "boom", source="llama-server")

    sent: list[str] = []

    def fake_post(url, content, headers, timeout, verify):
        sent.append(content)
        return _FakeResp(200)

    monkeypatch.setattr("client.telemetry.reporter.httpx.post", fake_post)

    r = LogReporter(store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur")
    assert r.push() is True
    assert len(sent) == 1

    # Nothing new -> no second request.
    assert r.push() is True
    assert len(sent) == 1

    # A new entry ships, and only that entry.
    store.append("warn", "second", source="llama-server")
    assert r.push() is True
    assert len(sent) == 2
    assert "second" in sent[1]
    assert "boom" not in sent[1]


def test_drain_reaches_the_crash_context_behind_a_backlog(tmp_path, monkeypatch) -> None:
    """The post-mortem is appended LAST, so one FIFO batch never reaches it.

    A node that has been offline accumulates a backlog. `flush_crash_context`
    then appends the exit marker plus the run-up at the end of the table. A
    single batch ships only the oldest `batch_size` rows, so the batch that
    actually explains the death is left behind exactly when the process is
    about to exit and stop trying.
    """
    store = LogStore(tmp_path / "sys.db")
    for i in range(25):
        store.append("warn", f"backlog {i}", source="llama-server")
    store.append("error", "child exited with code 139 — last lines follow", source="llama-server")

    sent: list[str] = []
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: (sent.append(content), _FakeResp(200))[1],
    )

    r = LogReporter(store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur", batch_size=10)

    # One batch: the crash line is still sitting in the store.
    assert r.push() is True
    assert not any("code 139" in c for c in sent)

    # Draining catches up to it.
    assert r.drain() is True
    assert any("code 139" in c for c in sent)


def test_drain_does_not_sleep_through_the_retry_backoff(tmp_path, monkeypatch) -> None:
    """Shutdown must not block on the background retry policy.

    drain() runs on the main thread just before sys.exit. With the shipped
    values (3 attempts, 30s base) an unreachable orchestrator would hold the
    container Running for ~3 minutes after its child died.
    """
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "boom", source="llama-server")

    attempts: list[int] = []
    slept: list[float] = []

    def refuse(url, content, headers, timeout, verify):
        attempts.append(1)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("client.telemetry.reporter.httpx.post", refuse)
    monkeypatch.setattr("client.telemetry.reporter.time.sleep", lambda s: slept.append(s))

    r = LogReporter(
        store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur",
        max_attempts=3, backoff_base=30,
    )
    assert r.drain() is False
    assert len(attempts) == 1, "drain retried instead of failing fast"
    assert slept == [], "drain slept through a backoff on the shutdown path"

    # The background path keeps its retries.
    attempts.clear()
    assert r.push() is False
    assert len(attempts) == 3
    assert r.max_attempts == 3, "drain leaked its temporary attempt count"


def test_cursor_survives_a_restart(tmp_path, monkeypatch) -> None:
    """A pod restart must not re-ship the node's whole local history."""
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "boom", source="llama-server")
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post", lambda url, content, headers, timeout, verify: _FakeResp(200)
    )

    LogReporter(store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur").push()

    fresh = LogReporter(store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur")
    assert fresh._cursor > 0


def test_failed_push_keeps_the_cursor_so_entries_retry(tmp_path, monkeypatch) -> None:
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "boom", source="llama-server")

    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: _FakeResp(500),
    )
    monkeypatch.setattr("client.telemetry.reporter.time.sleep", lambda _s: None)

    r = LogReporter(store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur")
    assert r.push() is False
    assert r._cursor == 0, "a failed push must not advance the cursor"


class _Clock:
    """A monotonic clock the test advances by hand, standing in for ``time``."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = list(ticks)

    def monotonic(self) -> float:
        return self._ticks.pop(0) if len(self._ticks) > 1 else self._ticks[0]


def test_drain_stops_at_its_budget_instead_of_shipping_every_batch(tmp_path, monkeypatch) -> None:
    """drain runs before sys.exit, inside terminationGracePeriodSeconds (45s).

    Unbounded, 25 batches against a slow or black-holing orchestrator held the
    container Running with a dead child for minutes — and it would be SIGKILLed
    at 45s anyway, having shipped no more than a bounded drain would.
    """
    store = LogStore(tmp_path / "sys.db")
    for i in range(30):
        store.append("warn", f"backlog {i}", source="llama-server")

    sent: list[str] = []
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: (sent.append(content), _FakeResp(200))[1],
    )
    # Deadline, first batch (inside), second batch (spent).
    monkeypatch.setattr("client.log.reporter.time", _Clock([0.0, 0.0, 100.0]))

    r = LogReporter(store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur", batch_size=1)
    assert r.drain() is True
    assert len(sent) == 1, "kept shipping past the budget"


def test_drain_ships_even_when_the_logship_thread_holds_the_lock(
    tmp_path, monkeypatch, caplog
) -> None:
    """The contended branch is the one that actually runs on a dying pod.

    The background thread can be inside a 30s POST when the exit path calls
    drain(). If drain gave up there, the crash context that explains the exit
    would never leave the node — so it ships anyway, accepting at worst a
    duplicated batch.
    """
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "child exited with code 139", source="llama-server")

    sent: list[str] = []
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: (sent.append(content), _FakeResp(200))[1],
    )
    caplog.set_level(logging.WARNING, logger="aureax.logship")

    r = LogReporter(store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur")
    r._lock.acquire()  # stands in for the logship thread mid-POST
    try:
        assert r.drain(budget=0.05) is True
    finally:
        r._lock.release()

    assert any("code 139" in c for c in sent)
    assert r._cursor > 0, "shipped without advancing the cursor"
    assert any("still busy" in rec.getMessage() for rec in caplog.records)


def test_the_cursor_file_is_replaced_never_truncated(tmp_path, monkeypatch) -> None:
    """``write_text`` truncates in place, so a kill mid-write leaves it empty.

    _read_cursor then falls back to 0 and the node re-ships every row still
    inside logging.retention_days — putting that backlog in front of the fresh
    crash context drain() exists to reach. A SIGKILL here is expected, not
    exotic: terminationGracePeriodSeconds is 45.
    """
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "boom", source="llama-server")
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: _FakeResp(200),
    )

    written: list[str] = []
    real_write_text = Path.write_text

    def watch(self, data, *a, **kw):
        written.append(self.name)
        return real_write_text(self, data, *a, **kw)

    monkeypatch.setattr(Path, "write_text", watch)

    cursor = tmp_path / "cur"
    assert LogReporter(store, "https://o/api/v1/logs", "edge-01", cursor).push() is True

    assert written == ["cur.tmp"], "wrote into the live cursor file"
    assert cursor.read_text() == str(store.recent(limit=1)[0].id)
    assert not (tmp_path / "cur.tmp").exists(), "left the temp file behind"


def test_a_negative_cursor_on_disk_cannot_reship_the_table(tmp_path) -> None:
    """since_id(-1) returns everything; the read is clamped so it cannot."""
    store = LogStore(tmp_path / "sys.db")
    cursor = tmp_path / "cur"
    cursor.write_text("-5")
    assert LogReporter(store, "https://o/api/v1/logs", "edge-01", cursor)._cursor == 0


def test_a_cursor_ahead_of_the_store_restarts_instead_of_shipping_nothing(
    tmp_path, caplog
) -> None:
    """A system.db removed to recover from corruption restarts ids at 1.

    The cursor file is its sibling and survives that, so ``since_id`` returns
    nothing every cycle and ``push`` reports success: the node looks healthy
    and ships not one line for the life of the pod.
    """
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "after the wipe", source="llama-server")
    cursor = tmp_path / "cur"
    cursor.write_text("50000")
    caplog.set_level(logging.WARNING, logger="aureax.logship")

    r = LogReporter(store, "https://o/api/v1/logs", "edge-01", cursor)

    assert r._cursor == 0
    assert any("ahead of the store" in rec.getMessage() for rec in caplog.records)


def test_a_cursor_inside_the_store_is_left_alone(tmp_path) -> None:
    """The reconciliation must not re-ship history on an ordinary restart."""
    store = LogStore(tmp_path / "sys.db")
    entry = store.append("error", "boom", source="llama-server")
    cursor = tmp_path / "cur"
    cursor.write_text(str(entry.id))

    r = LogReporter(store, "https://o/api/v1/logs", "edge-01", cursor)
    assert r._cursor == entry.id


# --- orchestrator ingest ------------------------------------------------------

def test_ingest_tags_entries_by_node(tmp_path) -> None:
    store = LogStore(tmp_path / "orch.db")
    body = "\n".join(
        json.dumps({"node_id": "shard-w3", "level": "error", "text": "rpc died", "source": "rpc-server"})
        for _ in range(2)
    )
    assert ingest(body, store) == (2, 2)

    entries = store.recent(limit=10)
    assert all(e.source == "node:shard-w3" for e in entries)
    # The edge-side source is preserved in the text, not the source column, so
    # the column stays a stable per-node key the dashboard can filter on.
    assert all("[rpc-server]" in e.text for e in entries)


def test_ingest_skips_malformed_lines_without_dropping_the_batch(tmp_path) -> None:
    store = LogStore(tmp_path / "orch.db")
    body = "\n".join([
        "not json at all",
        json.dumps({"node_id": "n1", "level": "error"}),          # no text
        json.dumps({"node_id": "n1", "level": "error", "text": "good"}),
    ])
    assert ingest(body, store) == (1, 1)
    assert store.recent(limit=5)[0].text.endswith("good")


def test_unknown_levels_fall_back_rather_than_corrupt_the_store(tmp_path) -> None:
    store = LogStore(tmp_path / "orch.db")
    body = json.dumps({"node_id": "n1", "level": "bogus", "text": "x"})
    assert ingest(body, store) == (1, 1)
    assert store.recent(limit=1)[0].level == "info"


def test_shipped_logs_reach_the_live_console_not_just_the_database(tmp_path) -> None:
    """Ingest must go through SystemLogger, which fans out to SSE subscribers.

    Appending straight to the LogStore persists the row but broadcasts nothing,
    so a worker crash sat on disk and appeared in the dashboard only after a
    restart re-seeded the buffer — for a feature whose whole purpose is seeing
    the crash as it happens.
    """
    store = LogStore(tmp_path / "orch.db")
    system_log = SystemLogger(store=store)

    body = json.dumps(
        {"node_id": "shard-w1", "level": "error", "text": "GGML_ASSERT failed", "source": "rpc-server"}
    )
    assert ingest(body, store, system_log) == (1, 1)

    # Buffered for SSE replay, tagged per node, and persisted exactly once.
    lines = system_log.snapshot(since=0)
    assert any(line.source == "node:shard-w1" for line in lines)
    assert any("GGML_ASSERT failed" in line.text for line in lines)
    assert len(store.recent(limit=10)) == 1, "double-written to the store"


def test_ingest_keeps_the_time_the_node_said_it(tmp_path) -> None:
    """Shipped entries carry the time of the ORIGINAL event.

    A crash's run-up is shipped after the fact, sometimes minutes later behind
    a backlog. Stamping the whole batch with one ingest time put those lines
    after events that actually followed them, which is the opposite of what a
    post-mortem needs.
    """
    store = LogStore(tmp_path / "orch.db")
    shipped = "2026-01-02T03:04:05.000001+00:00"
    body = json.dumps(
        {"node_id": "n1", "ts": shipped, "level": "error", "text": "GGML_ASSERT failed"}
    )
    assert ingest(body, store) == (1, 1)
    assert store.recent(limit=1)[0].ts == shipped


def test_ingest_survives_the_system_logger_path_with_a_timestamp(tmp_path) -> None:
    """The live console path takes the same batch — including its timestamps."""
    store = LogStore(tmp_path / "orch.db")
    system_log = SystemLogger(store=store)
    shipped = "2026-01-02T03:04:05.000001+00:00"
    body = json.dumps({"node_id": "n1", "ts": shipped, "level": "warn", "text": "slow"})

    assert ingest(body, store, system_log) == (1, 1)
    assert store.recent(limit=1)[0].ts == shipped
    assert [ln.timestamp for ln in system_log.snapshot(since=0)] == [shipped]


def test_ingest_falls_back_when_the_shipped_timestamp_is_missing_or_junk(tmp_path) -> None:
    """The shape is checked, not just the type.

    The column is compared as text by retention and by the SSE ordering, so a
    number sorts wrong — and so does an arbitrary string, which a bare
    isinstance check let straight into the column every other writer fills
    with ISO values.
    """
    store = LogStore(tmp_path / "orch.db")
    body = "\n".join([
        json.dumps({"node_id": "n1", "level": "info", "text": "no ts"}),
        json.dumps({"node_id": "n1", "ts": 1767322445, "level": "info", "text": "numeric ts"}),
        json.dumps({"node_id": "n1", "ts": "hello", "level": "info", "text": "junk ts"}),
        json.dumps({"node_id": "n1", "ts": "9999", "level": "info", "text": "bare year"}),
    ])
    assert ingest(body, store) == (4, 4)

    for entry in store.recent(limit=4):
        assert entry.ts.startswith("20") and "T" in entry.ts, entry.ts


def test_ingest_bounds_the_row_it_actually_stores(tmp_path) -> None:
    """``origin`` used to be prepended AFTER the truncation, re-expanding the row.

    A huge "source" with a one-byte text stored a huge row, and node_id lands
    uncapped in the indexed source column. /api/v1/logs is unauthenticated on
    0.0.0.0, so neither field is under this fleet's control.
    """
    store = LogStore(tmp_path / "orch.db")
    body = json.dumps({
        "node_id": "n" * 5000,
        "source": "s" * 5000,
        "level": "error",
        "text": "t" * 10_000,
    })
    assert ingest(body, store) == (1, 1)

    entry = store.recent(limit=1)[0]
    assert len(entry.text) == 4000
    assert len(entry.source) == len("node:") + 120


def test_ingest_skips_a_bare_scalar_line_without_dropping_the_batch(tmp_path) -> None:
    """`"node_id" in 5` raises TypeError, which used to 500 the whole push."""
    store = LogStore(tmp_path / "orch.db")
    body = "\n".join([
        "5",
        "null",
        json.dumps({"node_id": "n1", "level": "error", "text": "good"}),
    ])
    assert ingest(body, store) == (1, 1)


def test_parse_jsonl_tolerates_blank_lines() -> None:
    body = "\n\n" + json.dumps({"node_id": "n", "level": "info", "text": "t"}) + "\n\n"
    assert len(parse_jsonl(body)) == 1


# --- mirroring the supervisor's own logging into the store --------------------
#
# These drive the REAL handler, by calling _init_log_store on a bare client, so
# they fail if production changes. An earlier version of this file re-declared a
# copy of the handler's guard and therefore could not fail no matter what the
# supervisor did.

# A resolved config's `tls:` block, in the shape the generator stamps into every
# node's config. _init_log_reporter builds its SSLContext from ca_file, so that
# path has to be a real, parseable CA bundle — certifi's is one, ships wherever
# httpx does, and its contents are irrelevant here: these tests never complete a
# handshake, they only check the context was threaded to the shipper at all.
_TLS_BLOCK = {
    "ca_file": certifi.where(),
    "cert_file": "/etc/aureax/tls/node.pem",
    "key_file": "/etc/aureax/tls/node.key",
}


@contextmanager
def _client_with_log_store(bare_client, tmp_path, config_extra: dict | None = None):
    """A bare AureaxClient with its real log store + handler installed."""
    client = bare_client(config={
        "logging": {"ship_logs": True, "log_db_path": str(tmp_path / "sys.db")},
        **(config_extra or {}),
    })

    root = logging.getLogger("aureax")
    before = list(root.handlers)
    try:
        client._init_log_store()
        yield client
    finally:
        # _init_log_store attaches to the process-global "aureax" logger.
        for h in list(root.handlers):
            if h not in before:
                root.removeHandler(h)


def test_shipper_failures_do_not_feed_back_into_the_store(bare_client, tmp_path) -> None:
    """A receiver outage must not fill the store with its own retry noise.

    Every failed push logs three warnings and an error. If those are stored
    they are then shipped, fail, and log again — so an outage generates work
    proportional to its own length. Observed for real after a dashboard
    restart: the first batch to arrive was almost entirely this.
    """
    with _client_with_log_store(bare_client, tmp_path) as client:
        logging.getLogger("aureax.logship").warning("Log push failed (refused)")
        logging.getLogger("aureax.reporter").warning("Push failed (refused)")
        texts = [e.text for e in client._log_store.recent(limit=10)]

    assert not any("Log push failed" in t for t in texts), "shipper logged into its own queue"
    # Telemetry's failures are still recorded — that is real information.
    assert any("Push failed" in t for t in texts)


def test_the_supervisors_own_logger_reaches_the_store(bare_client, tmp_path) -> None:
    """entrypoint's module logger must be a child of "aureax".

    The handler is attached to "aureax", and Python's logger hierarchy splits on
    "." only — so a hyphenated name like "aureax-client" is a sibling, not a
    child, and every supervisor diagnostic (rpc-server exited, peers
    unreachable, health timeout) silently bypasses shipping. Asserts through the
    real logger object rather than a name literal, so renaming it cannot pass.
    """
    from client.entrypoint import logger as supervisor_logger

    with _client_with_log_store(bare_client, tmp_path) as client:
        supervisor_logger.error("rpc-server exited with code 139")
        texts = [e.text for e in client._log_store.recent(limit=10)]

    assert any("rpc-server exited with code 139" in t for t in texts), (
        f"supervisor logger {supervisor_logger.name!r} does not propagate to the 'aureax' handler"
    )


def test_logs_ship_even_when_telemetry_is_disabled(bare_client, tmp_path) -> None:
    """Log shipping is not gated on telemetry.enabled.

    The two settings are independent, and a node that opts out of metrics still
    has to ship its tracebacks — an agent-loop node most of all. (Every node in
    the fleet, planner included, currently sets `telemetry.enabled: true`; the
    literal below is the OFF case this test exists to cover, not a description
    of any shipped config.)
    """
    extra = {
        "telemetry": {
            "enabled": False,
            "remote": {"endpoint": "https://10.77.0.1:7001/api/v1/telemetry"},
        },
        "tls": _TLS_BLOCK,
    }
    with _client_with_log_store(bare_client, tmp_path, extra) as client:
        client._init_log_reporter()
        assert client._log_reporter is not None, "telemetry.enabled gated log shipping"
        assert client._log_reporter.endpoint.endswith("/api/v1/logs")
        assert client._log_reporter.verify is not None, "the shipper got no CA context"


# --- the ingest endpoint must not claim success it did not achieve -----------

def test_ingest_reports_zero_when_the_store_write_fails(tmp_path) -> None:
    """A failed write must not look like a successful one.

    LogReporter advances its cursor on a 200, so an ingest that broadcast the
    lines but failed to persist them would move the shipper past entries that
    exist nowhere — and those are exactly the entries that explain a crash.
    """
    class _BrokenStore(LogStore):
        def append_many(self, records):
            raise sqlite3.OperationalError("database or disk is full")

    store = _BrokenStore(tmp_path / "orch.db")
    system_log = SystemLogger(store=store)

    body = json.dumps({"node_id": "n1", "level": "error", "text": "boom"})
    assert ingest(body, store, system_log) == (0, 1)

    # Still visible live, which is strictly better than losing sight of it.
    assert any("boom" in line.text for line in system_log.snapshot(since=0))


def test_ingest_reports_the_full_count_on_success(tmp_path) -> None:
    store = LogStore(tmp_path / "orch.db")
    system_log = SystemLogger(store=store)
    body = "\n".join(
        json.dumps({"node_id": "n1", "level": "warn", "text": f"line {i}"}) for i in range(3)
    )
    assert ingest(body, store, system_log) == (3, 3)


def test_line_index_is_monotonic_across_a_restart(tmp_path) -> None:
    """SSE resumption depends on line_index never going backwards.

    A per-process counter restarts at 0 while clients still hold a high-water
    mark from the previous process, so both the server's own skip and the
    browser's dedupe discard every subsequent line — the log console goes
    silently dead after a dashboard restart.
    """
    db = tmp_path / "s.db"
    first = SystemLogger(store=LogStore(db))
    for i in range(5):
        first.emit("info", f"before restart {i}")
    high_water = max(line.line_index for line in first.snapshot(since=0))

    second = SystemLogger(store=LogStore(db))  # dashboard restarts
    second.emit("info", "after restart")

    fresh = [ln for ln in second.snapshot(since=0) if ln.text == "after restart"]
    assert fresh, "the new line is missing from the buffer"
    assert fresh[0].line_index > high_water, (
        f"index went backwards: {fresh[0].line_index} <= {high_water}; "
        "every client would discard this line"
    )


# --- the shared transport, and the telemetry reporter that owns it -----------
#
# Both edge reporters POST their JSONL through client/telemetry/reporter.py's
# post_jsonl, so its retry policy is pinned here beside the log path that now
# depends on it.

def _telemetry_reporter(tmp_path, rows: int, **kwargs):
    from client.telemetry.reporter import TelemetryReporter
    from client.telemetry.storage import TelemetryStorage

    storage = TelemetryStorage(str(tmp_path / "t.db"))
    storage.insert([
        {"node_id": "n1", "metric_name": "cpu", "metric_value": float(i), "timestamp": _iso()}
        for i in range(rows)
    ])
    return storage, TelemetryReporter(storage, "https://o/api/v1/telemetry", **kwargs)


def test_telemetry_push_can_drain_a_backlog_in_bounded_batches(tmp_path, monkeypatch) -> None:
    """One batch per call left an outage backlog draining at ~39 rows/min.

    batch_size 50 per 60s push against the ~11 rows/min collect_all keeps
    producing means a day offline needs ~7 hours of catch-up, during which the
    dashboard shows values hours stale.
    """
    storage, r = _telemetry_reporter(tmp_path, 25, batch_size=10)
    sent: list[str] = []
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: (sent.append(content), _FakeResp(200))[1],
    )

    assert r.push() is True
    assert len(sent) == 1 and storage.count()["unpushed"] == 15

    assert r.push(max_batches=10) is True
    assert len(sent) == 3, "stopped before the buffer was empty"
    assert storage.count()["unpushed"] == 0


def test_telemetry_push_can_skip_the_retry_backoff(tmp_path, monkeypatch) -> None:
    """push() is called inline on the sampling thread, whose own sleep follows.

    With the shipped defaults a fully failing push spent 30+60s in backoff, so
    cpu and temperature samples thinned out during exactly the incident an
    operator would later reconstruct — and the rows are already durably
    buffered, so the sleeps bought no durability.
    """
    storage, r = _telemetry_reporter(tmp_path, 1, max_attempts=3, backoff_base=30)

    attempts: list[int] = []
    slept: list[float] = []
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: (attempts.append(1), _FakeResp(500))[1],
    )
    monkeypatch.setattr("client.telemetry.reporter.time.sleep", lambda s: slept.append(s))

    assert r.push(attempts=1) is False
    assert len(attempts) == 1 and slept == []
    assert storage.count()["pushed"] == 0, "a failed push must not mark rows shipped"

    attempts.clear()
    assert r.push() is False
    assert len(attempts) == 3
    assert r.max_attempts == 3, "the override leaked into the configured count"


def test_a_failed_telemetry_push_stores_one_row_not_four(tmp_path, monkeypatch, caplog) -> None:
    """_init_log_store mirrors this logger's WARNING/ERROR records into the store.

    Three per-attempt warnings plus a summary error meant four stored — then
    shipped — rows per failed push per node, and while the orchestrator is down
    those are the bulk of what floods it on recovery.
    """
    _storage, r = _telemetry_reporter(tmp_path, 1, max_attempts=3)
    caplog.set_level(logging.DEBUG, logger="aureax.reporter")
    monkeypatch.setattr(
        "client.telemetry.reporter.httpx.post",
        lambda url, content, headers, timeout, verify: _FakeResp(500),
    )
    monkeypatch.setattr("client.telemetry.reporter.time.sleep", lambda _s: None)

    assert r.push() is False

    stored = [
        rec for rec in caplog.records
        if rec.name == "aureax.reporter" and rec.levelno >= logging.WARNING
    ]
    assert [rec.levelno for rec in stored] == [logging.ERROR]
    # The per-attempt detail is still there, just below the mirroring threshold.
    assert sum(rec.levelno == logging.DEBUG for rec in caplog.records) == 3


# --- the transport verifies the orchestrator, and carries nothing else --------
#
# post_jsonl is the single POST both shippers go through, so the CA context has
# to survive two hops: reporter -> post_jsonl -> httpx. A drop at either hop
# leaves the node verifying against the public trust store, which no fleet
# certificate chains to — the push then fails every cycle and the rows pile up
# on the node, which is safe but silent.

def _capture_posts(monkeypatch) -> list[dict]:
    """Record every httpx.post call post_jsonl makes."""
    calls: list[dict] = []

    def fake_post(url, content, headers, timeout, verify):
        calls.append({"url": url, "headers": headers, "verify": verify})
        return _FakeResp(200)

    monkeypatch.setattr("client.telemetry.reporter.httpx.post", fake_post)
    return calls


def test_post_jsonl_forwards_the_ca_context_to_httpx(monkeypatch) -> None:
    calls = _capture_posts(monkeypatch)
    ctx = ssl.create_default_context()

    assert post_jsonl(
        "https://o/api/v1/logs", "{}", 1,
        backoff_base=0, backoff_max=0, log=logging.getLogger("t"), what="x",
        verify=ctx,
    ) is True

    assert calls[0]["verify"] is ctx


def test_post_jsonl_without_a_context_still_verifies(monkeypatch) -> None:
    """None means "no fleet CA supplied", never "skip the check".

    httpx resolves None to the public trust store, so a shipper that lost its
    context fails the handshake instead of pushing telemetry to whatever
    answered on :7001. Disabling verification is the thing that would make that
    silent, and it appears nowhere in the client code.
    """
    calls = _capture_posts(monkeypatch)

    post_jsonl(
        "https://o/api/v1/logs", "{}", 1,
        backoff_base=0, backoff_max=0, log=logging.getLogger("t"), what="x",
    )

    assert calls[0]["verify"] is None
    assert calls[0]["verify"] is not False


def test_the_push_carries_a_content_type_and_no_credential(monkeypatch) -> None:
    """The ingest routes are unauthenticated by design; TLS authenticates the
    SERVER only. Pinned so a future "just add the node key here" cannot land
    without also facing the ingest-auth decision the module docstring defers.
    """
    calls = _capture_posts(monkeypatch)

    post_jsonl(
        "https://o/api/v1/logs", "{}", 1,
        backoff_base=0, backoff_max=0, log=logging.getLogger("t"), what="x",
    )

    assert calls[0]["headers"] == {"Content-Type": "application/x-ndjson"}


def test_the_telemetry_reporter_threads_its_context_through(tmp_path, monkeypatch) -> None:
    ctx = ssl.create_default_context()
    _storage, r = _telemetry_reporter(tmp_path, 1, verify=ctx)
    calls = _capture_posts(monkeypatch)

    assert r.push() is True

    assert calls[0]["verify"] is ctx
    assert calls[0]["url"].startswith("https://")


def test_the_log_reporter_threads_its_context_through(tmp_path, monkeypatch) -> None:
    store = LogStore(tmp_path / "sys.db")
    store.append("error", "boom", source="llama-server")
    ctx = ssl.create_default_context()
    calls = _capture_posts(monkeypatch)

    r = LogReporter(
        store, "https://o/api/v1/logs", "edge-01", tmp_path / "cur", verify=ctx
    )
    assert r.push() is True

    assert calls[0]["verify"] is ctx
    assert calls[0]["url"].startswith("https://")
