"""SQLite-backed persistence for chat sessions and messages.

The orchestrator dashboard is the only side that constructs ``SessionStore``
today (``orchestrator/dashboard/app.py``). It lives under ``client.chat`` so an
edge-side agent loop (autonomous runs, local chat, future RAG context) can
reuse the schema later, each side against its own SQLite file — there is no
cross-process sync.

Schema: two tables.
- ``sessions`` — one row per conversation, keyed by UUID, scoped to a target
  (for the orchestrator: planner or node-<id>; for the edge client: a local
  agent name). Carries a ``status`` flag so UIs can mark in-flight sessions
  as "active".
- ``messages`` — append-only log per session. ``tool_calls`` stores the raw
  JSON for assistant messages that requested tool calls; ``tool_call_id``
  links a ``role='tool'`` row back to the assistant's request.

"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from client.sqlite import connection

logger = logging.getLogger("aureax.chat.store")


@dataclass
class Session:
    """One conversation, scoped to a ``target_id`` (planner / node-<id> / agent name).

    ``status`` is ``'idle'`` or ``'active'`` so UIs can highlight in-flight chats.
    """

    id: str
    target_id: str
    title: str
    status: str
    created_at: str
    updated_at: str


@dataclass
class Message:
    """One row in the append-only message log for a session.

    ``role`` is one of ``'system' | 'user' | 'assistant' | 'tool'``.
    ``tool_calls`` carries the assistant's raw tool-call list when applicable;
    ``tool_call_id`` links a ``role='tool'`` row back to that assistant request.
    """

    id: int
    session_id: str
    seq: int
    role: str
    content: str | None
    tool_calls: list | None
    tool_call_id: str | None
    created_at: str


class SessionStore:
    """SQLite store for chat sessions and messages."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(str(db_path)).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return connection(self.db_path, rows=True, foreign_keys=True)

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id         TEXT PRIMARY KEY,
                    target_id  TEXT NOT NULL,
                    title      TEXT NOT NULL DEFAULT '',
                    status     TEXT NOT NULL DEFAULT 'idle',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_target ON sessions(target_id);
                CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT NOT NULL,
                    seq          INTEGER NOT NULL,
                    role         TEXT NOT NULL,
                    content      TEXT,
                    tool_calls   TEXT,
                    tool_call_id TEXT,
                    created_at   TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS messages_session_seq
                    ON messages(session_id, seq);
                """
            )
        # Separate statement, and tolerant of failure: a DB written before
        # append_message took BEGIN IMMEDIATE may already hold duplicate
        # (session_id, seq) rows, and letting that raise here would break
        # dashboard startup outright. A new index name on purpose — reusing
        # messages_session_seq would make IF NOT EXISTS a silent no-op.
        with self._conn() as conn:
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS messages_session_seq_uniq "
                    "ON messages(session_id, seq)"
                )
            except sqlite3.IntegrityError:
                logger.warning(
                    "messages(session_id, seq) holds duplicates in %s — leaving the "
                    "uniqueness index off; ordering in those sessions is arbitrary.",
                    self.db_path,
                )

    # ---- Sessions -----------------------------------------------------------

    def create_session(self, target_id: str, title: str = "") -> Session:
        """Create a new ``idle`` session against ``target_id`` and return it."""
        now = _now()
        sid = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, target_id, title, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'idle', ?, ?)",
                (sid, target_id, title, now, now),
            )
        return Session(sid, target_id, title, "idle", now, now)

    def get_session(self, session_id: str) -> Session | None:
        """Fetch a session by ID; return ``None`` if it doesn't exist."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return _session_from_row(row) if row else None

    def list_sessions(self, target_id: str | None = None) -> list[Session]:
        """Return sessions newest-first; filtered by ``target_id`` if given."""
        with self._conn() as conn:
            if target_id:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE target_id = ? "
                    "ORDER BY updated_at DESC",
                    (target_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sessions ORDER BY updated_at DESC"
                ).fetchall()
        return [_session_from_row(r) for r in rows]

    def update_session_title(self, session_id: str, title: str) -> None:
        """Set a session's title and bump ``updated_at``."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), session_id),
            )

    def update_session_status(self, session_id: str, status: str) -> None:
        """Set a session's status (``'idle'`` or ``'active'``) and bump ``updated_at``."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, _now(), session_id),
            )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and (via ON DELETE CASCADE) all its messages."""
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def clear_session(self, session_id: str) -> None:
        """Drop all messages for a session but keep the session row."""
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_now(), session_id),
            )

    # ---- Messages -----------------------------------------------------------

    def list_messages(self, session_id: str) -> list[Message]:
        """Return all messages for a session in seq order (oldest first)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY seq ASC",
                (session_id,),
            ).fetchall()
        return [_message_from_row(r) for r in rows]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str | None = None,
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
    ) -> Message:
        """Append a message and bump the session's ``updated_at``.

        ``tool_calls`` is JSON-encoded into the ``messages.tool_calls`` column
        when provided. The returned ``Message`` carries the freshly assigned
        ``id`` and ``seq``.
        """
        now = _now()
        with self._conn() as conn:
            # Take the write lock BEFORE reading MAX(seq): python's sqlite3
            # begins a transaction only ahead of DML, so that SELECT would
            # otherwise run in autocommit and two appends against one session
            # could read the same value. Must be the first statement in the
            # block, or sqlite has already begun implicitly.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_seq = row[0]
            cur = conn.execute(
                "INSERT INTO messages "
                "(session_id, seq, role, content, tool_calls, tool_call_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    next_seq,
                    role,
                    content,
                    json.dumps(tool_calls) if tool_calls is not None else None,
                    tool_call_id,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            msg_id = cur.lastrowid
        return Message(
            id=msg_id,
            session_id=session_id,
            seq=next_seq,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            created_at=now,
        )


# ---- Helpers ----------------------------------------------------------------

def _now() -> str:
    return datetime.now(UTC).isoformat()


def _session_from_row(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        target_id=row["target_id"],
        title=row["title"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _message_from_row(row: sqlite3.Row) -> Message:
    tc: Any = row["tool_calls"]
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        seq=row["seq"],
        role=row["role"],
        content=row["content"],
        tool_calls=json.loads(tc) if tc else None,
        tool_call_id=row["tool_call_id"],
        created_at=row["created_at"],
    )


def messages_to_engine_format(messages: list[Message]) -> list[dict]:
    """Convert DB rows to the format PlannerEngine.messages expects.

    Every assistant tool_call is paired with a tool row, synthesizing an
    ``interrupted`` result for any that has none. The two rows are persisted at
    different points — the request when the ``tool_call`` event is yielded, the
    answer only after ``call_tool`` returns — so a dashboard SIGTERM (``aureax
    stop`` kills it) or a cancellation in between leaves the request stored and
    the answer not. Replaying that unanswered call to the model on every later
    ask is what this avoids; the assistant turn itself is kept.
    """
    answered = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}
    out: list[dict] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content or "",
                "tool_calls": m.tool_calls,
            })
            for tc in m.tool_calls:
                call_id = tc.get("id") if isinstance(tc, dict) else None
                if call_id and call_id not in answered:
                    out.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps({"error": "interrupted"}),
                    })
        elif m.role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content or "",
            })
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out
