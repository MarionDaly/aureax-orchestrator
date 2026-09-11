"""Guards for the edge client, run without a node.

Covers the supervisor's child-process handling (output teed to stdout and the
log file, the log fd released when the binary is missing), the telemetry
store's SQLite round trip and retention, and the chat engine's thinking-block
stripping, history trimming and per-node tool-call concurrency.

These don't exercise the live llama-server / MCP host stack; for that, run
``./scripts/test-all.sh`` with the cluster up.
"""

from __future__ import annotations

from client.inference.chat_client import strip_thinking
from client.mcp.pool import MCPPool
from client.sensors.simulator import SensorSimulator
from client.telemetry.storage import TelemetryStorage


def test_parse_tool_call_namespacing() -> None:
    pool = MCPPool()
    assert pool.parse_tool_call("edge-01__read_sensor") == ("edge-01", "read_sensor")
    assert pool.parse_tool_call("read_sensor") == ("", "read_sensor")
    # Only the first "__" splits — tool names may contain underscores.
    assert pool.parse_tool_call("edge-01__some__thing") == ("edge-01", "some__thing")


def test_strip_thinking_removes_blocks() -> None:
    assert strip_thinking("<think>plan</think>final") == "final"
    assert strip_thinking("plain text") == "plain text"
    assert strip_thinking("<think>a</think>x<think>b</think>y") == "xy"


def test_sensor_simulator_determinism(monkeypatch) -> None:
    # Pin time.time so two instances see identical _start_time + elapsed.
    monkeypatch.setattr("client.sensors.simulator.time.time", lambda: 1_700_000_000.0)
    a = SensorSimulator("edge-01").read_temperature()
    b = SensorSimulator("edge-01").read_temperature()
    assert a["value"] == b["value"]
    # Different node_id seeds the RNG differently, so the value must differ.
    c = SensorSimulator("edge-02").read_temperature()
    assert c["value"] != a["value"]


def test_spawn_child_tees_output_to_stdout_and_log(bare_client, tmp_path, capfd) -> None:
    """Child stdout+stderr must reach BOTH the log file and container stdout.

    The stdout half is load-bearing: `kubectl logs` and the orchestrator's
    crash-log fetchers only ever see container stdout, so a child logging
    solely to a node-local file reduces every crash to "exited with code N".

    stop() is what makes this deterministic: it joins the pump thread before
    closing the FD the thread writes to (client/entrypoint.py), so once it
    returns the log file holds everything the child ever wrote. Polling the
    wall clock for the last line instead raced the very drain the code under
    test guarantees — and, being after the assertions, left the daemon pump
    thread and the log FD open for the rest of the session on any failure.
    """
    client = bare_client()
    log_path = tmp_path / "logs" / "child.log"

    client._spawn_child(
        "llama-server",
        ["sh", "-c", "echo to-stdout; echo to-stderr >&2"],
        log_path,
    )
    assert client._child_proc.wait(timeout=10) == 0
    client.stop()
    assert client._child_log_file is None

    written = log_path.read_text()
    assert "to-stdout" in written
    assert "to-stderr" in written, "stderr must be merged into the same stream"

    captured = capfd.readouterr().out
    assert "[llama-server] to-stdout" in captured
    assert "[llama-server] to-stderr" in captured


def test_spawn_child_releases_log_fd_when_binary_missing(bare_client, tmp_path) -> None:
    """A missing binary must not leak the log FD (Popen raises after open)."""
    client = bare_client()
    log_path = tmp_path / "child.log"

    try:
        client._spawn_child("llama-server", ["/nonexistent/aureax-binary"], log_path)
    except (FileNotFoundError, OSError):
        pass
    else:
        raise AssertionError("expected spawn of a missing binary to raise")

    assert client._child_log_file is None
    assert client._child_proc is None


def test_llama_health_timeout_is_configurable(bare_client) -> None:
    """A sharded head needs minutes to load; 120s is only right for small models."""
    client = bare_client()
    assert client._llama_health_timeout() == 120

    client.config = {"model": {"server": {"health_timeout_seconds": 900}}}
    assert client._llama_health_timeout() == 900


def test_telemetry_storage_roundtrip(tmp_path) -> None:
    store = TelemetryStorage(str(tmp_path / "telemetry.db"))
    store.insert([
        {"node_id": "edge-01", "metric_name": "cpu_percent", "metric_value": 42.0, "timestamp": "2026-04-30T00:00:00+00:00"},
        {"node_id": "edge-01", "metric_name": "memory_percent", "metric_value": 50.0, "timestamp": "2026-04-30T00:00:01+00:00"},
        {"node_id": "edge-01", "metric_name": "disk_usage_percent", "metric_value": 12.0, "timestamp": "2026-04-30T00:00:02+00:00"},
    ])
    unpushed = store.get_unpushed(limit=10)
    assert len(unpushed) == 3
    store.mark_pushed([unpushed[0]["id"], unpushed[1]["id"]])
    counts = store.count()
    assert counts == {"total": 3, "pushed": 2, "unpushed": 1}


# --- planner engine -----------------------------------------------------------

class _StubLLM:
    """ChatLlamaClient stand-in that replays a scripted list of responses."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):
        # Copy: the engine may keep mutating its own list afterwards.
        self.seen.append([dict(m) for m in messages])
        return {"choices": [{"message": self._responses.pop(0)}]}


class _StubPool:
    """MCPPool stand-in recording call order and start/finish interleaving."""

    def __init__(self, delays: dict[str, float] | None = None) -> None:
        self.delays = delays or {}
        self.events: list[str] = []

    def get_openai_tools(self) -> list[dict]:
        return []

    def parse_tool_call(self, name: str):
        node, _, tool = name.partition("__")
        return node, tool

    async def call_tool(self, node_id: str, tool_name: str, args: dict) -> dict:
        import asyncio

        self.events.append(f"start:{node_id}")
        await asyncio.sleep(self.delays.get(node_id, 0))
        self.events.append(f"end:{node_id}")
        return {"node": node_id}


def _tool_call(call_id: str, name: str) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": "{}"}}


def _run(coro_iter) -> list[dict]:
    import asyncio

    async def drain():
        return [e async for e in coro_iter]

    return asyncio.run(drain())


def test_thinking_is_stripped_from_a_tool_calling_turn_too() -> None:
    """The dashboard persists engine.messages verbatim and rehydrates them, so
    a <think> block before a tool call reached the user AND was replayed to the
    model on every later turn — the exact failure the final-turn strip exists
    to prevent."""
    from client.engine import PlannerEngine

    llm = _StubLLM([
        {"role": "assistant", "content": "<think>plan</think>",
         "tool_calls": [_tool_call("c1", "edge-01__read")]},
        {"content": "done"},
    ])
    engine = PlannerEngine(llm, _StubPool(), "sys")

    _run(engine.arun("hello"))

    assistant = next(m for m in engine.messages if m.get("tool_calls"))
    assert "<think>" not in (assistant["content"] or "")


def test_tool_results_are_stored_compactly() -> None:
    """Stored history is replayed to the model every round and every later turn;
    the dashboard pretty-prints at render time (chat.js::pretty), so indent=2
    bought nothing but prompt tokens."""
    from client.engine import PlannerEngine

    llm = _StubLLM([
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("c1", "edge-01__read")]},
        {"content": "done"},
    ])
    engine = PlannerEngine(llm, _StubPool(), "sys")

    _run(engine.arun("hello"))

    tool_msg = next(m for m in engine.messages if m["role"] == "tool")
    assert "\n" not in tool_msg["content"], "tool result was pretty-printed into history"


def test_tool_calls_to_different_nodes_run_concurrently() -> None:
    """Each call is an HTTP round trip that runs a whole generation on a remote
    node, so a serial round cost the sum rather than the max. Same-node calls
    stay serial — the edge runs parallel_slots: 1."""
    from client.engine import PlannerEngine

    pool = _StubPool(delays={"edge-01": 0.05, "edge-02": 0.0})
    llm = _StubLLM([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _tool_call("c1", "edge-01__read"),
                _tool_call("c2", "edge-02__read"),
                _tool_call("c3", "edge-01__read"),
            ],
        },
        {"content": "done"},
    ])
    engine = PlannerEngine(llm, pool, "sys")

    events = _run(engine.arun("hello"))

    # edge-02 finishes while edge-01's first call is still running.
    assert pool.events.index("end:edge-02") < pool.events.index("end:edge-01")
    # ... but edge-01's two calls never overlap.
    first_end = pool.events.index("end:edge-01")
    assert pool.events[first_end + 1] == "start:edge-01"
    # Tool messages stay in the model's original call order: chat_service
    # persists engine.messages by index against the assistant's tool_calls.
    ids = [m["tool_call_id"] for m in engine.messages if m["role"] == "tool"]
    assert ids == ["c1", "c2", "c3"]
    results = [e["call_id"] for e in events if e["type"] == "tool_result"]
    assert results == ["c1", "c2", "c3"]


def test_history_sent_to_the_model_is_trimmed_but_never_the_stored_copy() -> None:
    """MAX_TOOL_ROUNDS bounds one turn, not the session: an unbounded history
    eventually overflows the served context and every later request fails, with
    the same oversized history reloaded next turn. The stored list must stay
    complete — chat_service persists it BY INDEX."""
    from client.engine import PlannerEngine, _fits_context

    llm = _StubLLM([{"content": "ok"}])
    engine = PlannerEngine(llm, _StubPool(), "sys")
    for i in range(40):
        engine.messages.append({"role": "user", "content": "u" * 5000 + str(i)})
        engine.messages.append({"role": "assistant", "content": "a" * 5000})

    before = len(engine.messages)
    _run(engine.arun("latest question"))

    sent = llm.seen[0]
    assert len(sent) < before, "the oversized history was sent verbatim"
    assert sent[0]["role"] == "system", "the system prompt must survive the trim"
    assert sent[-1]["content"] == "latest question"
    assert len(engine.messages) == before + 2, "the stored history must not be trimmed"

    # The cut lands on a user turn, so an assistant tool_calls message is never
    # separated from its role: tool replies.
    assert sent[1]["role"] == "user"
    assert _fits_context([]) == []


def test_the_newest_user_turn_survives_an_oversized_history() -> None:
    """Trimming must never drop the question being answered — a single huge
    tool result must not push the user's own message out of the request."""
    from client.engine import _fits_context

    trimmed = _fits_context(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "x" * 5_000},
            {"role": "user", "content": "the question"},
            {"role": "tool", "tool_call_id": "c1", "content": "y" * 5_000},
        ],
        max_chars=100,
    )
    assert trimmed[0]["role"] == "system"
    assert [m["content"] for m in trimmed[1:]] == ["the question", "y" * 5_000]


def test_history_budget_counts_tool_call_payloads() -> None:
    """An assistant turn's cost is its tool_calls, not its content.

    A message that calls a tool carries a stripped or empty content and the
    whole JSON arguments payload under tool_calls, so a budget that measures
    content alone keeps replaying turns it prices at zero — and overflows the
    served context it exists to respect."""
    import json

    from client.engine import _fits_context

    calls = [{
        "id": "c1",
        "type": "function",
        "function": {
            "name": "edge-01__run_inference",
            "arguments": json.dumps({"prompt": "p" * 5_000}),
        },
    }]
    trimmed = _fits_context(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "user", "content": "the question"},
        ],
        max_chars=1_000,
    )
    assert [m["role"] for m in trimmed] == ["system", "user"]
    assert trimmed[-1]["content"] == "the question"
