"""SQLite transactions whose connections are always closed."""

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

# Databases opened through here are written by several threads of one process
# at once — on a node the telemetry loop, the hourly log loop and the
# supervisor's log handler all reach the same files, and on the orchestrator
# the ingest routes and the dashboard's queries do. In the rollback journal
# mode a writer locks readers out for the whole transaction, so a concurrent
# read waits out ``timeout`` and then raises OperationalError, which both log
# writers swallow — a dropped line with no symptom. WAL lets readers proceed
# against the last committed snapshot while a writer holds the file.
#
# ``journal_mode`` is a persistent property of the database file, so setting it
# is a header write that takes the very lock it exists to avoid contending for.
# Hence once per database FILE per process: the first connection converts the
# file and every later one inherits the mode from disk. The key carries the
# inode as well as the path, so a file deleted and recreated at the same path
# (a wiped store, a test fixture) is converted again rather than assumed.
_wal_paths: set[tuple[str, int, int]] = set()
_wal_lock = threading.Lock()


def _enable_wal(conn: sqlite3.Connection, key: tuple[str, int, int]) -> None:
    """Put ``key``'s database into WAL mode, at most once per process."""
    with _wal_lock:
        if key in _wal_paths:
            return
        # A busy file is the one case this can raise, and WAL is a throughput
        # property rather than a correctness one — failing the caller's whole
        # operation over it would be the worse outcome. The path stays out of
        # the set, so the next connection tries again.
        with suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode = WAL")
            _wal_paths.add(key)


@contextmanager
def connection(
    path: str | Path,
    *,
    rows: bool = False,
    foreign_keys: bool = False,
    timeout: float = 5.0,
) -> Iterator[sqlite3.Connection]:
    """A connection to ``path`` in WAL mode, closed when the block ends.

    ``timeout`` is how long a statement waits for a lock another connection
    holds before raising OperationalError; it is sqlite3's own default, named
    here so a store that ingests batches can raise its own ceiling.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        resolved = Path(path).resolve()
        st = os.stat(resolved)
        _enable_wal(conn, (str(resolved), st.st_dev, st.st_ino))
        if rows:
            conn.row_factory = sqlite3.Row
        if foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            yield conn
    finally:
        conn.close()
