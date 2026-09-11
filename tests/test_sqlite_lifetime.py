"""Every SQLite connection is closed on the way out, and a failed block rolls back.

`client/sqlite.py::connection` is the single place either guarantee is made,
and every telemetry, log and chat store opens through it. Long-lived processes
— the dashboard and the node supervisor — reopen these databases on a timer,
so a connection that survives its block is a file descriptor and a lock leaked
once per poll, and a half-applied write is a store that disagrees with itself.

It is also the single place the concurrency settings every store inherits are
made: WAL, so a writer does not lock out the readers on the other threads, and
the busy timeout a statement waits out before raising.
"""

import sqlite3

import pytest

import client.sqlite as sqlite_mod
from client.sqlite import connection


def test_sqlite_transaction_rollback_and_deterministic_close(tmp_path):
    db = tmp_path / "db"
    with connection(db) as conn:
        conn.execute("CREATE TABLE values_table(value)")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
    with pytest.raises(RuntimeError), connection(db) as conn:
        conn.execute("INSERT INTO values_table VALUES (1)")
        raise RuntimeError
    with connection(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM values_table").fetchone()[0] == 0


@pytest.mark.parametrize("module,kind", [
    ("client.telemetry.storage", "TelemetryStorage"), ("client.log.store", "LogStore"),
    ("orchestrator.telemetry.storage", "MetricsStorage"), ("orchestrator.telemetry.queries", "MetricsQueries"),
    ("client.chat.store", "SessionStore"),
])
def test_storage_factories_close_connections(tmp_path, module, kind):
    from importlib import import_module
    store = getattr(import_module(module), kind)(str(tmp_path / "db"))
    with store._conn() as conn:
        conn.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_wal_is_enabled_once_per_database_path(tmp_path, monkeypatch):
    """WAL is set on first contact with a file and never again in this process.

    journal_mode is a persistent property of the file, so re-running the pragma
    per connection would take the write lock it exists to avoid contending for.
    """
    monkeypatch.setattr(sqlite_mod, "_wal_paths", set())
    db = tmp_path / "db"

    with connection(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert {key[0] for key in sqlite_mod._wal_paths} == {str(db.resolve())}

    # A later connection inherits the mode from the file, not from the set.
    with connection(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    # And the set really is the gate: a file already in it is left alone, so a
    # never-converted file stays in the default rollback mode.
    other = tmp_path / "other"
    sqlite3.connect(other).close()
    st = other.stat()
    monkeypatch.setattr(sqlite_mod, "_wal_paths", {(str(other.resolve()), st.st_dev, st.st_ino)})
    with connection(other) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"

    # The key carries the inode: a file deleted and recreated at the same path
    # is a new database and gets converted again rather than assumed.
    other.unlink()
    with connection(other) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_connection_passes_a_busy_timeout_to_sqlite3(tmp_path, monkeypatch):
    """The lock wait is sqlite3.connect's own argument, not a per-store pragma."""
    seen: list[float] = []
    real_connect = sqlite3.connect

    def spy(target, **kw):
        seen.append(kw["timeout"])
        return real_connect(target, **kw)

    monkeypatch.setattr(sqlite_mod.sqlite3, "connect", spy)
    with connection(tmp_path / "db"):
        pass
    with connection(tmp_path / "slow", timeout=30.0):
        pass
    assert seen == [5.0, 30.0]
