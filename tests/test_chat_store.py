"""SessionStore persistence and replay invariants.

The dashboard replays a stored session to the model on every ask, so what is
missing from the log matters as much as what is in it.
"""

from __future__ import annotations

import sqlite3

from client.chat.store import SessionStore, messages_to_engine_format


def _store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "chat.db")


def test_append_message_assigns_unique_seq(tmp_path) -> None:
    """seq is read and written under one BEGIN IMMEDIATE, and pinned unique.

    list_messages orders by seq, so a duplicate replays the conversation in
    arbitrary order.
    """
    store = _store(tmp_path)
    session = store.create_session("node-edge-01")
    for i in range(5):
        store.append_message(session.id, "user", content=f"m{i}")

    seqs = [m.seq for m in store.list_messages(session.id)]
    assert seqs == [0, 1, 2, 3, 4]

    # The uniqueness index is real, not just an ordering convention.
    with sqlite3.connect(str(store.db_path)) as raw:
        try:
            raw.execute(
                "INSERT INTO messages (session_id, seq, role, created_at) VALUES (?, 0, 'user', '')",
                (session.id,),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("a duplicate (session_id, seq) was accepted")


def test_init_tolerates_a_db_that_already_holds_duplicates(tmp_path, caplog) -> None:
    """Adding the index must not break dashboard startup on an existing DB."""
    store = _store(tmp_path)
    session = store.create_session("node-edge-01")
    with sqlite3.connect(str(store.db_path)) as raw:
        raw.execute("DROP INDEX messages_session_seq_uniq")
        raw.executemany(
            "INSERT INTO messages (session_id, seq, role, created_at) VALUES (?, 0, 'user', '')",
            [(session.id,), (session.id,)],
        )

    # Reopening the same file must succeed, index or no index.
    with caplog.at_level("WARNING", logger="aureax.chat.store"):
        reopened = SessionStore(store.db_path)
    assert len(reopened.list_messages(session.id)) == 2
    assert "holds duplicates" in caplog.text, "the tolerant branch was never taken"


def test_connections_do_not_leak(tmp_path) -> None:
    """`with conn` ends the transaction; it does not close the handle.

    Every method opens its own connection, so a missing close held a file
    descriptor per call.
    """
    store = _store(tmp_path)
    session = store.create_session("node-edge-01")

    with store._conn() as conn:
        pass
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("_conn left the connection open")

    # And the store still works after that exit.
    assert store.get_session(session.id) is not None


def test_replay_synthesizes_results_for_unanswered_tool_calls() -> None:
    """A crash between the tool_call and tool_result rows leaves a half turn.

    The two rows are persisted at different points, with the call_tool await in
    between, so a dashboard SIGTERM or a cancellation stores the request and
    never the answer. Replaying that to the model on every later ask is what
    this avoids.
    """
    rows = _messages([
        ("user", "check both nodes", None, None),
        (
            "assistant",
            "",
            [
                {"id": "call_a", "function": {"name": "edge-01__get_health"}},
                {"id": "call_b", "function": {"name": "edge-02__get_health"}},
            ],
            None,
        ),
        ("tool", '{"status": "ok"}', None, "call_a"),
    ])

    out = messages_to_engine_format(rows)

    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool"]
    # The assistant turn is kept intact, not discarded.
    assert len(out[1]["tool_calls"]) == 2
    # The real answer stays where it was recorded; only call_b is synthesized.
    assert out[2] == {"role": "tool", "tool_call_id": "call_b", "content": '{"error": "interrupted"}'}
    assert out[3] == {"role": "tool", "tool_call_id": "call_a", "content": '{"status": "ok"}'}


def test_replay_leaves_a_fully_answered_turn_alone() -> None:
    rows = _messages([
        ("assistant", "", [{"id": "call_a", "function": {"name": "edge-01__get_health"}}], None),
        ("tool", '{"status": "ok"}', None, "call_a"),
    ])
    assert [m["role"] for m in messages_to_engine_format(rows)] == ["assistant", "tool"]


def _messages(specs) -> list:
    """Build Message rows without touching SQLite."""
    from client.chat.store import Message

    return [
        Message(
            id=i,
            session_id="s1",
            seq=i,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            created_at="2026-04-30T00:00:00+00:00",
        )
        for i, (role, content, tool_calls, tool_call_id) in enumerate(specs)
    ]
