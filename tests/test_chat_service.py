"""ChatService keeps its cached engine and the database in agreement.

The engine is a cache in front of the session store: `_engine_for` rebuilds it
by replaying `store.list_messages`. Anything that leaves the in-memory message
list ahead of the DB silently changes what every later turn in that session is
conditioned on — and the divergence only becomes visible after a restart, when
the replay produces a different conversation than the one the user saw.
"""

from __future__ import annotations

import asyncio
import contextlib
import ssl
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from ruamel.yaml import YAML

from client.chat.store import SessionStore
from client.engine import MAX_HISTORY_CHARS, history_budget_chars
from client.inference.chat_client import ChatLlamaClient
from orchestrator.config.cache import load_node_config_resolved
from orchestrator.config.loader import generated_config_path
from orchestrator.dashboard import chat_api
from orchestrator.dashboard import chat_service as chat_service_mod
from orchestrator.dashboard.app import create_app
from orchestrator.dashboard.chat_service import ChatService
from tests.conftest import EXAMPLE_NETWORK_CONFIG, FakeLog


class _FakeEngine:
    """Stands in for PlannerEngine, with its message-ordering behaviour.

    The ordering is the point: arun() appends the user message BEFORE doing any
    work, so a failure in that work leaves the message on the engine.

    Constructed with PlannerEngine's signature (llm, pool, system_prompt,
    max_history_chars) so the REAL ChatService._engine_for builds and caches it
    — patching _engine_for itself would bypass the cache these tests are about,
    and every assertion on service._engines would pass for the wrong reason.
    """

    fail = False

    def __init__(
        self,
        _llm=None,
        _pool=None,
        system_prompt: str = "sys",
        max_history_chars: int = 0,
    ) -> None:
        self.max_history_chars = max_history_chars
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.fail = type(self).fail

    async def arun(self, user_message: str):
        self.messages.append({"role": "user", "content": user_message})
        if self.fail:
            raise RuntimeError("llama-server unreachable")
        self.messages.append({"role": "assistant", "content": "hi"})
        yield {"type": "final", "content": "hi"}


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A ChatService whose engine is fake but whose caching is real."""
    store = SessionStore(str(tmp_path / "chat.db"))
    svc = ChatService(EXAMPLE_NETWORK_CONFIG, store, system_log=None)

    async def _pool(_self, _target, _resolved=None):
        return object()

    async def _llm(_self, _target, _resolved=None):
        return object()

    async def _prompt(_self, _target, _resolved=None):
        return "sys"

    # _engine_for resolves (target, per-node config) once and hands the pair to
    # the three builders above. With those faked the pair is never read, so stub
    # the lookup rather than tie these tests to a node id in the example fleet.
    def _resolve(_self, _target_id):
        return None, {}

    monkeypatch.setattr(chat_service_mod, "PlannerEngine", _FakeEngine)
    monkeypatch.setattr(ChatService, "_target_for", _resolve)
    monkeypatch.setattr(ChatService, "_pool_for", _pool)
    monkeypatch.setattr(ChatService, "_llm_for", _llm)
    monkeypatch.setattr(ChatService, "_system_prompt_for", _prompt)
    return svc


@pytest.fixture(autouse=True)
def _reset_engine_mode():
    _FakeEngine.fail = False
    yield
    _FakeEngine.fail = False


# One recognisable context standing in for the fleet CA. tlsstore.ca_ssl_context
# deliberately RAISES when ~/.aureax/tls/ca.pem is absent rather than minting a
# CA no node was ever given, and every test here runs against a $HOME with no
# store — so minting a real one would cost an EC keygen per test for a value
# only identity is asserted on.
FLEET_CA_CONTEXT = ssl.create_default_context()


@pytest.fixture(autouse=True)
def fleet_ca(monkeypatch) -> ssl.SSLContext:
    """Every dashboard -> node client must verify against the fleet CA."""
    monkeypatch.setattr(
        chat_service_mod.tlsstore, "ca_ssl_context", lambda: FLEET_CA_CONTEXT,
    )
    return FLEET_CA_CONTEXT


def _engine_msgs(engine) -> list[tuple[str, str]]:
    return [(m.get("role", ""), m.get("content") or "") for m in engine.messages]


def _db_msgs(service, session_id) -> list[tuple[str, str]]:
    return [(m.role, m.content or "") for m in service.store.list_messages(session_id)]


def test_failed_ask_does_not_leave_the_engine_ahead_of_the_database(service):
    """A raise before the first yield must not strand an unpersisted message.

    The persist loop runs inside the `async for` body, so nothing is written
    when arun() raises immediately — while the user message is already on the
    engine. Left cached, that engine conditions every later turn on a message
    the database does not contain.
    """
    session = service.store.create_session(target_id="node-edge-01")
    _FakeEngine.fail = True

    asyncio.run(service.ask(session.id, "what is 2+2?"))

    assert _db_msgs(service, session.id) == [], "persisted a half-turn"
    assert session.id not in service._engines, (
        "kept the diverged engine cached — the next ask would build on a "
        "message the database has never seen"
    )


def test_next_ask_after_a_failure_replays_only_from_the_database(service):
    """The retry must not inherit the failed attempt's phantom message."""
    session = service.store.create_session(target_id="node-edge-01")

    _FakeEngine.fail = True
    asyncio.run(service.ask(session.id, "first try"))
    _FakeEngine.fail = False
    asyncio.run(service.ask(session.id, "second try"))

    db = _db_msgs(service, session.id)
    roles = [r for r, _ in db]
    contents = [c for _, c in db]
    assert roles == ["user", "assistant"], f"unexpected persisted history: {db}"
    assert "first try" not in contents, "the failed attempt leaked into history"
    assert "second try" in contents


def test_successful_ask_persists_every_engine_message_once(service):
    """The in-loop flush and the trailing drain must not double-write."""
    session = service.store.create_session(target_id="node-edge-01")

    asyncio.run(service.ask(session.id, "hello"))

    db = _db_msgs(service, session.id)
    assert db == [("user", "hello"), ("assistant", "hi")], db
    engine = service._engines[session.id]
    assert _engine_msgs(engine)[1:] == db, "engine and DB disagree after a clean run"


def test_reset_during_an_in_flight_ask_leaves_nothing_behind_the_wipe(service):
    """Reset mid-run must not have that turn's messages written back after it.

    `ask` holds its own reference to the engine and keeps draining new messages
    into the store, so a reset that only cleared the rows saw the running turn
    re-appended after the re-seeded system prompt — and broadcast, so the UI
    repopulated the view it had just cleared.
    """
    session = service.store.create_session(target_id="node-edge-01")

    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowEngine(_FakeEngine):
        async def arun(self, user_message: str):
            self.messages.append({"role": "user", "content": user_message})
            started.set()
            await release.wait()
            self.messages.append({"role": "assistant", "content": "hi"})
            yield {"type": "final", "content": "hi"}

    service._engines[session.id] = _SlowEngine()

    async def scenario():
        ask = asyncio.create_task(service.ask(session.id, "hello"))
        await started.wait()
        await service.reset_session(session.id)
        release.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await ask

    asyncio.run(scenario())

    roles = [role for role, _ in _db_msgs(service, session.id)]
    assert roles == ["system"], f"the in-flight turn survived the reset: {roles}"
    assert session.id not in service._engines


def test_delete_during_an_in_flight_ask_does_not_corrupt_the_store(service):
    """Deleting mid-run must not pull the session row out from under it.

    `ask` appends messages referencing session_id; deleting the row while it
    runs made the next append raise
    "IntegrityError: FOREIGN KEY constraint failed". Popping the ask lock made
    it worse — the running task held the old lock object, so a fresh one gave
    no mutual exclusion at all.
    """
    session = service.store.create_session(target_id="node-edge-01")

    started = asyncio.Event()
    release = asyncio.Event()

    class _SlowEngine(_FakeEngine):
        async def arun(self, user_message: str):
            self.messages.append({"role": "user", "content": user_message})
            started.set()
            await release.wait()          # hold the run open
            self.messages.append({"role": "assistant", "content": "hi"})
            yield {"type": "final", "content": "hi"}

    service._engines[session.id] = _SlowEngine()

    # ask() catches its own exceptions and only LOGS them, so the FK violation
    # is invisible to the store afterwards — record what it logged.
    recorder = FakeLog()
    service.system_log = recorder

    async def scenario():
        ask = asyncio.create_task(service.ask(session.id, "hello"))
        await started.wait()              # run is in flight
        await service.delete_session(session.id)   # cancels it, then deletes
        release.set()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await ask

    asyncio.run(scenario())

    assert service.get_session(session.id) is None, "session was not deleted"
    assert session.id not in service._engines
    assert session.id not in service._running

    errors = [t for lvl, t in recorder.lines if lvl == "error"]
    assert not any("FOREIGN KEY" in t or "IntegrityError" in t for t in errors), (
        f"the in-flight run wrote against a deleted session: {errors}"
    )


def test_an_idle_stream_writes_keepalives(service, monkeypatch):
    """An idle chat SSE connection must keep writing to the socket.

    A bare `await q.get()` puts nothing on the wire between events, so an idle
    session was dropped by anything in the path and every EventSource
    reconnect replayed the session's whole history. The keepalive also gives
    the HTTP layer a regular chance to notice the client is gone.
    """
    monkeypatch.setattr(chat_service_mod, "STREAM_KEEPALIVE_SECONDS", 0.02)
    session = service.store.create_session(target_id="node-edge-01")

    async def scenario() -> list[str]:
        seen: list[str] = []

        async def consume():
            async for event in service.stream(session.id):
                seen.append(event["type"])
                if seen.count("keepalive") >= 2:
                    return

        await asyncio.wait_for(consume(), timeout=5)
        return seen

    seen = asyncio.run(scenario())
    assert seen.count("keepalive") == 2, seen
    assert seen[0] == "session", "history replay must still come first"


def _expire_pool_backoff(monkeypatch) -> None:
    """Let the next ``_pool_for`` retry, without touching ChatService internals.

    Rewinding ``_pool_attempt_at`` instead encodes that the private dict stores
    a monotonic timestamp of the LAST ATTEMPT: storing a deadline, a failure
    count, or moving the state onto the pool object would break that line with
    no behaviour regression at all. ``_pool_for`` reads POOL_RETRY_SECONDS at
    call time, so zeroing the tunable makes the same request through the knob
    the backoff is actually configured by.
    """
    monkeypatch.setattr(chat_service_mod, "POOL_RETRY_SECONDS", 0.0)


def test_a_failed_pool_rebuild_is_not_retried_by_every_caller(tmp_path, monkeypatch):
    """A node that cannot answer must not cost a connect timeout per ask.

    _engine_for refreshes the pool on every cache hit, and MCPPool now evicts
    dead sessions — so against a sharded head that is still streaming tensors
    to its workers (minutes), each queued ask used to pay its own 30s connect.
    """
    from types import SimpleNamespace

    class _DeadPool:
        attempts = 0

        def __init__(self, verify=None) -> None:
            self.verify = verify
            self.connected_nodes: list[str] = []

        async def connect_url(self, _node_id, _url, api_key=None) -> bool:
            type(self).attempts += 1
            return False

    target = SimpleNamespace(node_id="edge-01", host="10.77.0.1", config_file="")
    monkeypatch.setattr(chat_service_mod, "MCPPool", _DeadPool)
    monkeypatch.setattr(ChatService, "_target_for", lambda _self, _tid: (target, {}))

    svc = ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))

    async def scenario():
        for _ in range(3):
            await svc._pool_for("node-edge-01")

    asyncio.run(scenario())
    assert _DeadPool.attempts == 1, "retried a known-dead pool on every call"

    # Once the window has passed, a retry is exactly what should happen: this
    # is a backoff, not a permanent write-off.
    _expire_pool_backoff(monkeypatch)
    asyncio.run(svc._pool_for("node-edge-01"))
    assert _DeadPool.attempts == 2, "backoff never expired"


def test_a_host_change_redials_the_pool_and_repoints_the_chat_client(
    tmp_path, monkeypatch
):
    """Both per-target caches are keyed by target id, which survives a remap.

    Editing `ssh.targets[].host` moves the node without changing its node_id,
    so a pool judged complete on member NAMES and a chat client whose base_url
    was frozen at construction both keep addressing the old host for the life
    of the process.
    """
    class _Pool:
        def __init__(self, verify=None) -> None:
            self.verify = verify
            self.connected_nodes: list[str] = []
            self.dialled: list[str] = []

        async def disconnect(self, node_id):
            if node_id in self.connected_nodes:
                self.connected_nodes.remove(node_id)

        async def connect_url(self, node_id, url, api_key=None) -> bool:
            self.dialled.append(url)
            if node_id not in self.connected_nodes:
                self.connected_nodes.append(node_id)
            return True

    target = SimpleNamespace(node_id="edge-01", host="10.77.0.1", config_file="")
    monkeypatch.setattr(chat_service_mod, "MCPPool", _Pool)
    monkeypatch.setattr(ChatService, "_target_for", lambda _self, _tid: (target, {}))
    svc = ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))

    pool = asyncio.run(svc._pool_for("node-edge-01"))
    llm = asyncio.run(svc._llm_for("node-edge-01"))
    assert pool.dialled == ["https://10.77.0.1:30900/mcp"]
    assert llm.base_url == "https://10.77.0.1:30800"

    # Unchanged host: both caches are reused untouched.
    assert asyncio.run(svc._pool_for("node-edge-01")) is pool
    assert asyncio.run(svc._llm_for("node-edge-01")) is llm
    assert len(pool.dialled) == 1

    target.host = "10.77.0.9"
    _expire_pool_backoff(monkeypatch)
    # Refreshed IN PLACE, both of them: engines built earlier hold references
    # to these two objects, so replacing either would leave those engines on
    # the old address.
    assert asyncio.run(svc._pool_for("node-edge-01")) is pool
    assert pool.dialled[-1] == "https://10.77.0.9:30900/mcp"
    assert asyncio.run(svc._llm_for("node-edge-01")) is llm
    assert llm.base_url == "https://10.77.0.9:30800"
    assert str(llm._client.base_url).rstrip("/") == "https://10.77.0.9:30800"

    asyncio.run(svc.close())


def test_the_history_budget_follows_the_targets_context_size(tmp_path, monkeypatch):
    """A fixed budget never trims for a target served at a small context.

    The engine replays its history on every turn, so once the stored session
    passes what the node serves, every later turn fails identically — the
    failure the budget exists to prevent.
    """
    target = SimpleNamespace(node_id="planner-01", host="10.77.0.4", config_file="")
    ncfg = {"model": {"server": {"context_size": 16384}}}

    async def _pool(_self, _target, _resolved=None):
        return object()

    async def _llm(_self, _target, _resolved=None):
        return object()

    monkeypatch.setattr(chat_service_mod, "PlannerEngine", _FakeEngine)
    monkeypatch.setattr(ChatService, "_target_for", lambda _self, _tid: (target, ncfg))
    monkeypatch.setattr(ChatService, "_pool_for", _pool)
    monkeypatch.setattr(ChatService, "_llm_for", _llm)
    svc = ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))

    session = svc.store.create_session(target_id="node-planner-01")
    engine = asyncio.run(svc._engine_for(session))
    assert engine.max_history_chars == history_budget_chars(16384)
    assert engine.max_history_chars < MAX_HISTORY_CHARS, (
        "a 16384-token target must get a smaller budget than the ceiling"
    )


# --- target resolution: what may be chatted with, and what may not ------------


def _write_chat_fleet(tmp_path, monkeypatch) -> Path:
    """A fleet with one of everything the target filter has to sort out: an
    edge node, a group head + shard worker, a planner, and a disabled node.

    The templates sit beside the fleet file, which is where ChatService reads
    each node's role from; $HOME is redirected so the generated-config
    directory load_node_config_resolved prefers is empty and the templates are
    used.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    yaml = YAML()

    def node_yaml(name: str, role: str, extra: dict | None = None) -> None:
        with open(configs / name, "w") as f:
            yaml.dump({"node": {"id": "${node_id}", "role": role}, **(extra or {})}, f)

    node_yaml("edge.yaml", "edge_executor")
    node_yaml("head.yaml", "edge_executor")
    node_yaml("worker.yaml", "shard_worker")
    node_yaml("planner.yaml", "cognitive_planner",
              {"agent": {"system_prompt": "plan things"}})

    network = {
        "network": {"name": "t", "version": "0.1.0"},
        "model_registry": {"models": []},
        "ssh": {"targets": [
            {"node_id": "edge-01", "host": "10.77.0.1", "config_file": "edge.yaml"},
            {"node_id": "head-01", "host": "10.77.0.2", "config_file": "head.yaml"},
            {"node_id": "worker-01", "host": "10.77.0.3", "config_file": "worker.yaml"},
            {"node_id": "planner-01", "host": "10.77.0.4", "config_file": "planner.yaml"},
            {"node_id": "off-01", "host": "10.77.0.5", "config_file": "edge.yaml",
             "enabled": False},
        ]},
        "node_groups": [{"name": "g1", "head": "head-01", "workers": ["worker-01"]}],
    }
    path = configs / "aureax-network.yaml"
    with open(path, "w") as f:
        yaml.dump(network, f)

    monkeypatch.setenv("HOME", str(tmp_path))
    return path


def test_targets_hides_shard_workers_and_disabled_nodes(tmp_path, monkeypatch):
    """A worker runs rpc-server only — no llama-server on :30800, no MCP on
    :30900 — so offering it as a chat target just hangs until the connect
    timeout. The role lookup fails OPEN, which is why it is tested."""
    path = _write_chat_fleet(tmp_path, monkeypatch)
    svc = ChatService(path, SessionStore(str(tmp_path / "chat.db")))

    entries = {e["id"]: e for e in svc.targets()}

    assert "node-worker-01" not in entries, "offered a shard worker as a chat target"
    assert "node-off-01" not in entries, "offered a disabled node"
    assert entries["node-head-01"]["kind"] == "node"
    assert entries["node-head-01"]["group"] == "g1"
    assert "g1" in entries["node-head-01"]["label"]
    assert entries["node-planner-01"]["kind"] == "planner"
    assert entries["node-edge-01"]["kind"] == "node"


def test_targets_reports_an_unreadable_config_instead_of_an_empty_fleet(tmp_path):
    """[] is indistinguishable from "no enabled targets", and every later
    request then failed as "unknown target" — the client blamed for a
    server-side config failure. chat_api turns this into a 503."""
    from orchestrator.config.loader import ConfigError

    broken = tmp_path / "aureax-network.yaml"
    broken.write_text("ssh: [[[\n")
    svc = ChatService(broken, SessionStore(str(tmp_path / "chat.db")))

    with pytest.raises((ConfigError, OSError)):
        svc.targets()


@pytest.mark.parametrize("node_id", ["worker-01", "off-01", "nope-01"])
def test_target_for_refuses_what_targets_hides(tmp_path, monkeypatch, node_id):
    """A session outlives its node being disabled or converted to a worker.
    _target_for scanning ALL targets let `ask` resolve one and dial :30900 on a
    node that runs only rpc-server — the hang the filter exists to prevent."""
    path = _write_chat_fleet(tmp_path, monkeypatch)
    svc = ChatService(path, SessionStore(str(tmp_path / "chat.db")))

    with pytest.raises(ValueError):
        svc._target_for(f"node-{node_id}")

    target, _ncfg = svc._target_for("node-head-01")
    assert target.node_id == "head-01"


# --- the llama API key a cached client keeps sending -------------------------


def _write_generated(path, api_key: str | None, mcp_api_key: str | None = None) -> None:
    """Write a resolved per-node config, bumping mtime so the (path, mtime)
    config cache actually re-reads it."""
    import os

    yaml = YAML()
    server: dict = {"host": "0.0.0.0", "port": 8080}
    if api_key is not None:
        server["api_key"] = api_key
    mcp_server: dict = {"enabled": True, "port": 9000}
    if mcp_api_key is not None:
        mcp_server["api_key"] = mcp_api_key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump({"node": {"id": "edge-01", "role": "edge_executor"},
                   "model": {"server": server},
                   "mcp": {"server": mcp_server}}, f)
    stamp = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(stamp, stamp))


@pytest.mark.parametrize(
    "value", [None, "", 123, "${api_key}", "prefix${api_key}"],
)
def test_an_unresolved_template_key_is_never_sent_as_a_bearer_token(value):
    """load_node_config_resolved falls back to the SOURCE TEMPLATE when no generated
    config exists (a fresh Mac, a wiped ~/.aureax), where this value is the
    literal "${api_key}" — sending it 401s every request with no clue why."""
    assert chat_service_mod._resolved_key({"api_key": value}) is None
    assert chat_service_mod._resolved_api_key(
        {"model": {"server": {"api_key": value}}}
    ) is None
    assert chat_service_mod._resolved_api_key({}) is None
    assert chat_service_mod._resolved_api_key({"model": {"server": {}}}) is None
    assert chat_service_mod._resolved_api_key(
        {"model": {"server": {"api_key": "aureax_sk_realkey1234567"}}}
    ) == "aureax_sk_realkey1234567"


@pytest.mark.parametrize(
    "value", [None, "", 123, "${mcp_api_key}", "prefix${mcp_api_key}"],
)
def test_an_unresolved_mcp_key_is_never_sent_either(value):
    """The MCP key is read out of the same resolved-or-template file and shares
    the ``aureax_sk_`` prefix, so it carries the identical trap. One helper
    serves both kinds precisely so a rule tightened for one cannot be missed
    for the other."""
    assert chat_service_mod._resolved_mcp_api_key(
        {"mcp": {"server": {"api_key": value}}}
    ) is None
    assert chat_service_mod._resolved_mcp_api_key({}) is None
    assert chat_service_mod._resolved_mcp_api_key({"mcp": {"server": {}}}) is None
    assert chat_service_mod._resolved_mcp_api_key(
        {"mcp": {"server": {"api_key": "aureax_sk_mcpkey1234567"}}}
    ) == "aureax_sk_mcpkey1234567"


async def test_a_rotated_key_reaches_the_client_a_cached_engine_holds(tmp_path, monkeypatch):
    """After `aureax apply --rotate-key`, a session with a cached engine kept
    sending the old bearer token and 401'd on every turn. The refresh has to
    MUTATE the client: cached PlannerEngines hold this object."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_path = generated_config_path("edge-01")
    _write_generated(cfg_path, "aureax_sk_AAAA")

    target = SimpleNamespace(node_id="edge-01", host="10.77.0.1", config_file="edge.yaml")
    monkeypatch.setattr(
        ChatService, "_target_for",
        lambda _self, _tid: (
            target,
            load_node_config_resolved(EXAMPLE_NETWORK_CONFIG.parent, target),
        ),
    )
    svc = ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))

    first = await svc._llm_for("node-edge-01")
    assert first._client.headers["Authorization"] == "Bearer aureax_sk_AAAA"

    _write_generated(cfg_path, "aureax_sk_BBBB")
    again = await svc._llm_for("node-edge-01")
    assert again is first, "replaced the client a cached engine still points at"
    assert again.api_key == "aureax_sk_BBBB"
    assert again._client.headers["Authorization"] == "Bearer aureax_sk_BBBB"

    _write_generated(cfg_path, None)
    assert (await svc._llm_for("node-edge-01")).api_key is None
    assert "Authorization" not in first._client.headers
    await svc.close()


# --- cancellation: nothing keeps writing into a session that is going away ----


class _HoldingEngine(_FakeEngine):
    """An engine whose turn stays open until `release` is set."""

    started: asyncio.Event
    release: asyncio.Event

    async def arun(self, user_message: str):
        self.messages.append({"role": "user", "content": user_message})
        type(self).started.set()
        await type(self).release.wait()
        self.messages.append({"role": "assistant", "content": "hi"})
        yield {"type": "final", "content": "hi"}


def _holding_engine() -> type[_HoldingEngine]:
    _HoldingEngine.started = asyncio.Event()
    _HoldingEngine.release = asyncio.Event()
    return _HoldingEngine


async def _until(predicate, timeout: float = 2.0) -> None:
    """Yield to the loop until ``predicate()`` holds, or fail saying it never did.

    The alternative — a fixed ``await asyncio.sleep(0.02)`` and then assert —
    is a guess at how long the scheduler needs, so it fails on a loaded machine
    for no reason and would pass a regression that merely got slower.
    ``sleep(0)`` reschedules without consuming wall clock, so this is bounded
    by the timeout rather than paced by it.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition never became true"
        await asyncio.sleep(0)


async def test_close_cancels_in_flight_asks_before_tearing_down_their_clients(service):
    """app.py awaits close() from the lifespan. A run left going kept appending
    to the store against an httpx client that had just been closed."""
    session = service.store.create_session(target_id="node-edge-01")
    engine = _holding_engine()
    service._engines[session.id] = engine()

    ask = asyncio.create_task(service.ask(session.id, "hello"))
    await engine.started.wait()
    assert session.id in service._running

    await service.close()

    assert ask.done(), "close() returned with the ask still running"
    assert service._running == {}
    roles = [role for role, _ in _db_msgs(service, session.id)]
    assert "assistant" not in roles, "the cancelled run kept persisting"


async def test_delete_also_cancels_an_ask_queued_behind_the_running_one(service):
    """chat_api spawns a task per POST with no dedupe, so a second ask sits on
    the session's lock. Registered only inside that lock, it was invisible to
    Delete — which then blocked on the lock and let the queued run persist its
    turn into a session the operator had deleted."""
    session = service.store.create_session(target_id="node-edge-01")
    engine = _holding_engine()
    service._engines[session.id] = engine()

    first = asyncio.create_task(service.ask(session.id, "first"))
    await engine.started.wait()
    second = asyncio.create_task(service.ask(session.id, "second"))
    await _until(lambda: len(service._running.get(session.id, ())) == 2)

    await asyncio.wait_for(service.delete_session(session.id), timeout=2)

    assert first.done() and second.done()
    assert service.get_session(session.id) is None
    assert session.id not in service._running
    assert service.store.list_messages(session.id) == []


# --- the per-session caches stay bounded without breaking what is in use ------


async def test_evict_engines_bounds_the_caches_and_spares_what_is_in_use(service):
    """A dropped Broadcaster with a live subscriber leaves that browser on a
    spinner, and dropping a HELD ask lock hands the next ask a fresh lock that
    excludes nobody — the mistake delete_session's docstring records."""
    from orchestrator.dashboard.broadcaster import Broadcaster

    for i in range(40):
        sid = f"s{i:02d}"
        service._engines[sid] = object()
        service._broadcasters[sid] = Broadcaster()
        service._ask_locks[sid] = asyncio.Lock()

    watched = service._broadcasters["s00"]
    watched.subscribe()
    held = service._ask_locks["s01"]
    await held.acquire()
    service._running["s01"] = {object()}
    # A run whose SSE client is momentarily absent: no subscriber, but live.
    in_flight_bc = service._broadcasters["s02"]
    service._running["s02"] = {object()}

    service._evict_engines()

    assert len(service._engines) == ChatService._MAX_CACHED_ENGINES
    assert "s00" not in service._engines, "kept the oldest engine"
    assert service._broadcasters["s00"] is watched, "evicted a subscribed broadcaster"
    assert service._broadcasters["s02"] is in_flight_bc, "evicted a live run's broadcaster"
    assert service._ask_locks["s01"] is held and held.locked()
    held.release()


# --- MCP pools: repaired, rate-limited, and reported --------------------------


class _ScriptedPool:
    """MCPPool stand-in. Connects only the names in `reachable`, records every
    dial, and tracks how many ran at once."""

    reachable: ClassVar[set] = set()
    dialled: ClassVar[list] = []
    keys: ClassVar[dict] = {}
    urls: ClassVar[dict] = {}
    in_flight = 0
    max_in_flight = 0
    dial_delay = 0.0
    dialling = None

    def __init__(self, verify=None) -> None:
        self.verify = verify
        self.connected_nodes: list[str] = []

    async def disconnect(self, node_id):
        if node_id in self.connected_nodes:
            self.connected_nodes.remove(node_id)

    async def connect_url(self, node_id: str, _url: str, api_key=None) -> bool:
        cls = type(self)
        cls.dialled.append(node_id)
        cls.keys[node_id] = api_key
        cls.urls[node_id] = _url
        cls.in_flight += 1
        cls.max_in_flight = max(cls.max_in_flight, cls.in_flight)
        if cls.dialling is not None:
            cls.dialling.set()
        await asyncio.sleep(cls.dial_delay)
        cls.in_flight -= 1
        if node_id in cls.reachable:
            self.connected_nodes.append(node_id)
            return True
        return False


@pytest.fixture
def scripted_pool(monkeypatch):
    _ScriptedPool.reachable = set()
    _ScriptedPool.dialled = []
    _ScriptedPool.keys = {}
    _ScriptedPool.urls = {}
    _ScriptedPool.in_flight = 0
    _ScriptedPool.max_in_flight = 0
    _ScriptedPool.dial_delay = 0.0
    _ScriptedPool.dialling = None
    monkeypatch.setattr(chat_service_mod, "MCPPool", _ScriptedPool)
    return _ScriptedPool


def _planner_remote(name: str, key: str | None) -> dict:
    """One ``mcp.client.remote_servers`` entry as ${mcp_endpoints} resolves it."""
    entry = {"name": name, "url": f"https://{name}:30900/mcp"}
    if key is not None:
        entry["api_key"] = key
    return entry


def _planner_service(
    tmp_path, monkeypatch, remotes: list[str], keys: dict | None = None,
) -> ChatService:
    keys = keys or {}
    target = SimpleNamespace(node_id="planner-01", host="10.77.0.4", config_file="p.yaml")
    ncfg = {
        "node": {"role": "cognitive_planner"},
        "mcp": {"client": {"remote_servers": [
            _planner_remote(n, keys.get(n)) for n in remotes
        ]}},
    }
    monkeypatch.setattr(ChatService, "_target_for", lambda _self, _tid: (target, ncfg))
    return ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))


async def test_a_partly_connected_planner_pool_is_repaired(tmp_path, monkeypatch, scripted_pool):
    """"Any node connected" read as healthy, so one dead remote's tools were
    absent for the life of the process — including after call_tool evicted a
    node on a transport error."""
    svc = _planner_service(tmp_path, monkeypatch, ["edge-a", "edge-b", "edge-c"])
    scripted_pool.reachable = {"edge-a"}

    pool = await svc._pool_for("node-planner-01")
    assert sorted(scripted_pool.dialled) == ["edge-a", "edge-b", "edge-c"]

    # Backoff expires, and the other two come up.
    _expire_pool_backoff(monkeypatch)
    scripted_pool.reachable = {"edge-a", "edge-b", "edge-c"}
    scripted_pool.dialled = []

    again = await svc._pool_for("node-planner-01")

    assert again is pool, "replaced a pool cached engines hold a reference to"
    assert sorted(scripted_pool.dialled) == ["edge-b", "edge-c"], (
        "redialled a healthy remote, or never repaired the dead ones"
    )
    assert sorted(pool.connected_nodes) == ["edge-a", "edge-b", "edge-c"]
    # Fully connected now: no further dialling, and no backoff needed.
    scripted_pool.dialled = []
    await svc._pool_for("node-planner-01")
    assert scripted_pool.dialled == []


async def test_planner_remotes_are_dialled_concurrently(tmp_path, monkeypatch, scripted_pool):
    """Each dial of a cold node runs to the full connect timeout, so a serial
    pass cost (unreachable remotes) x that while the ask waited."""
    svc = _planner_service(tmp_path, monkeypatch, ["edge-a", "edge-b", "edge-c"])
    scripted_pool.dial_delay = 0.02

    await svc._pool_for("node-planner-01")

    assert scripted_pool.max_in_flight == 3, "dialled the remotes one at a time"


async def test_a_failed_mcp_connect_reaches_the_dashboard_log(tmp_path, monkeypatch, scripted_pool):
    """The one condition an operator needs to see — this chat has no tools —
    only ever went to the Python logger."""
    svc = _planner_service(tmp_path, monkeypatch, ["edge-a"])
    recorder = FakeLog()
    svc.system_log = recorder

    await svc._pool_for("node-planner-01")

    warnings = [text for level, text in recorder.lines if level == "warn"]
    assert any("edge-a" in t and "FAILED" in t for t in warnings), recorder.lines


async def test_the_retry_window_starts_when_the_attempt_finishes(
    tmp_path, monkeypatch, scripted_pool,
):
    """The stamp used to be taken BEFORE the dial, so an attempt slower than
    the window came back with its own backoff already expired and the next ask
    retried immediately — the cost the backoff exists to prevent."""
    monkeypatch.setattr(chat_service_mod, "POOL_RETRY_SECONDS", 0.05)
    svc = _planner_service(tmp_path, monkeypatch, ["edge-a"])
    scripted_pool.dial_delay = 0.1

    await svc._pool_for("node-planner-01")
    await svc._pool_for("node-planner-01")

    assert scripted_pool.dialled == ["edge-a"], "retried a dial that just failed"


async def test_a_concurrent_rebuild_waits_for_reconciled_membership(tmp_path, monkeypatch, scripted_pool):
    """Concurrent requests wait for reconciled membership without duplicate dials."""
    svc = _planner_service(tmp_path, monkeypatch, ["edge-a"])
    await svc._pool_for("node-planner-01")          # first build, fails
    _expire_pool_backoff(monkeypatch)
    scripted_pool.dialled = []
    scripted_pool.dial_delay = 0.2
    scripted_pool.dialling = asyncio.Event()

    rebuild = asyncio.create_task(svc._pool_for("node-planner-01"))
    await scripted_pool.dialling.wait()
    monkeypatch.setattr(chat_service_mod, "POOL_RETRY_SECONDS", 60)
    pending = asyncio.create_task(svc._pool_for("node-planner-01"))
    await asyncio.sleep(0)
    assert not pending.done()
    await rebuild
    second = await pending

    assert second is svc._pools["node-planner-01"]
    assert scripted_pool.dialled == ["edge-a"], "two callers rebuilt the same pool"


# --- TLS and MCP bearer keys on every dialled endpoint ------------------------


def _ssl_context_of(client) -> ssl.SSLContext | None:
    """The SSLContext httpx actually built its connection pool with.

    A reach into httpx internals on purpose: httpx exposes no getter, and what
    it pins is exactly the failure a public attribute could not — a transport
    rebuilt by set_base_url that kept the address and silently dropped the CA.
    """
    return client._transport._pool._ssl_context


def _edge_service(tmp_path, monkeypatch, ncfg: dict) -> ChatService:
    target = SimpleNamespace(node_id="edge-01", host="10.77.0.1", config_file="edge.yaml")
    monkeypatch.setattr(ChatService, "_target_for", lambda _self, _tid: (target, ncfg))
    return ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))


async def test_both_per_target_clients_verify_against_the_fleet_ca(
    tmp_path, monkeypatch, scripted_pool, fleet_ca,
):
    """Every node listens with a leaf signed by the fleet CA and nothing else,
    so the system trust store verifies none of them. There is no plaintext
    fallback: an absent CA raises out of tlsstore rather than downgrading."""
    scripted_pool.reachable = {"edge-01"}
    svc = _edge_service(tmp_path, monkeypatch, {})

    pool = await svc._pool_for("node-edge-01")
    llm = await svc._llm_for("node-edge-01")

    assert pool.verify is fleet_ca
    assert llm.verify is fleet_ca
    assert _ssl_context_of(llm._client) is fleet_ca
    assert llm.base_url == "https://10.77.0.1:30800"
    assert scripted_pool.urls["edge-01"] == "https://10.77.0.1:30900/mcp"
    await svc.close()


async def test_a_missing_fleet_ca_fails_the_chat_instead_of_downgrading_it(
    tmp_path, monkeypatch, scripted_pool,
):
    """tlsstore.ca_ssl_context raises when ~/.aureax/tls/ca.pem is absent. That
    error has to reach the caller: a chat that cannot verify the node must not
    connect to it anyway, and the message names `aureax generate`."""
    def _no_ca():
        raise RuntimeError("no fleet CA at /x/ca.pem — run `aureax generate`")

    monkeypatch.setattr(chat_service_mod.tlsstore, "ca_ssl_context", _no_ca)
    svc = _edge_service(tmp_path, monkeypatch, {})

    with pytest.raises(RuntimeError, match="aureax generate"):
        await svc._pool_for("node-edge-01")
    with pytest.raises(RuntimeError, match="aureax generate"):
        await svc._llm_for("node-edge-01")

    assert scripted_pool.dialled == [], "dialled a node it could not verify"


async def test_the_edge_branch_dials_with_the_nodes_own_mcp_key(
    tmp_path, monkeypatch, scripted_pool,
):
    """The key comes from the target's RESOLVED config (mcp.server.api_key),
    the same source and the same staleness semantics as the llama key — not
    from the keystore, which would hand out a key rotated but not yet applied.
    """
    scripted_pool.reachable = {"edge-01"}
    svc = _edge_service(
        tmp_path, monkeypatch, {"mcp": {"server": {"api_key": "aureax_sk_EDGEKEY"}}},
    )

    await svc._pool_for("node-edge-01")

    assert scripted_pool.keys["edge-01"] == "aureax_sk_EDGEKEY"


async def test_a_node_without_an_mcp_key_is_dialled_without_one(
    tmp_path, monkeypatch, scripted_pool,
):
    """Auth is one template line, so a node whose template has none serves an
    open :30900 — dialling it with a bearer header would be harmless, but
    inventing a key here would hide that the port is open."""
    scripted_pool.reachable = {"edge-01"}
    svc = _edge_service(tmp_path, monkeypatch, {"mcp": {"server": {}}})

    await svc._pool_for("node-edge-01")

    assert scripted_pool.keys["edge-01"] is None


async def test_each_planner_remote_is_dialled_with_its_own_key(
    tmp_path, monkeypatch, scripted_pool,
):
    """${mcp_endpoints} resolves to one entry per peer carrying THAT peer's
    key, so a planner holds N credentials and no two members share one. Sending
    one member's key to another 401s that whole remote's tools away."""
    svc = _planner_service(
        tmp_path, monkeypatch, ["edge-a", "edge-b"],
        keys={"edge-a": "aureax_sk_AAAA", "edge-b": "aureax_sk_BBBB"},
    )
    scripted_pool.reachable = {"edge-a", "edge-b"}

    await svc._pool_for("node-planner-01")

    assert scripted_pool.keys == {
        "edge-a": "aureax_sk_AAAA", "edge-b": "aureax_sk_BBBB",
    }
    assert scripted_pool.urls["edge-a"] == "https://edge-a:30900/mcp"


async def test_a_planner_remote_still_holding_the_placeholder_is_dialled_unkeyed(
    tmp_path, monkeypatch, scripted_pool,
):
    """A fresh Mac has no generated planner config, so the resolver falls back
    to the SOURCE TEMPLATE where every remote's key is the literal
    "${mcp_api_key}" — sending that as a Bearer token 401s with no clue why."""
    svc = _planner_service(
        tmp_path, monkeypatch, ["edge-a"], keys={"edge-a": "${mcp_api_key}"},
    )
    scripted_pool.reachable = {"edge-a"}

    await svc._pool_for("node-planner-01")

    assert scripted_pool.keys["edge-a"] is None


async def test_a_rotated_mcp_key_forces_a_reconnect(tmp_path, monkeypatch, scripted_pool):
    """The MCP twin of the llama rotation above. `aureax apply --rotate-key`
    rotates both kinds and leaves the URL untouched, so a pool judged on the
    address alone reads as complete while every tool call on it 401s — until
    the dashboard restarts."""
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_path = generated_config_path("edge-01")
    _write_generated(cfg_path, None, mcp_api_key="aureax_sk_MCPAAAA")

    target = SimpleNamespace(node_id="edge-01", host="10.77.0.1", config_file="edge.yaml")
    monkeypatch.setattr(
        ChatService, "_target_for",
        lambda _self, _tid: (
            target,
            load_node_config_resolved(EXAMPLE_NETWORK_CONFIG.parent, target),
        ),
    )
    scripted_pool.reachable = {"edge-01"}
    svc = ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))

    pool = await svc._pool_for("node-edge-01")
    assert scripted_pool.keys["edge-01"] == "aureax_sk_MCPAAAA"

    # Unchanged config: the cached pool is reused, nothing is redialled.
    scripted_pool.dialled = []
    assert await svc._pool_for("node-edge-01") is pool
    assert scripted_pool.dialled == []

    _write_generated(cfg_path, None, mcp_api_key="aureax_sk_MCPBBBB")
    _expire_pool_backoff(monkeypatch)

    again = await svc._pool_for("node-edge-01")

    assert again is pool, "replaced a pool cached engines hold a reference to"
    assert scripted_pool.dialled == ["edge-01"], "kept the session holding the dead key"
    assert scripted_pool.keys["edge-01"] == "aureax_sk_MCPBBBB"
    await svc.close()


# --- the chat client every cached engine holds a reference to -----------------


def test_the_chat_client_keeps_the_ca_and_the_key_across_a_repoint(fleet_ca):
    """set_base_url rebuilds the httpx transport from scratch. When it did that
    inline it was one edit away from dropping whatever the constructor had
    configured, and a host edit would leave the client verifying against the
    system trust store — where nothing the fleet CA signed validates."""
    client = ChatLlamaClient(
        "https://10.77.0.1:30800", api_key="aureax_sk_AAAA", verify=fleet_ca,
    )
    assert _ssl_context_of(client._client) is fleet_ca

    client.set_base_url("https://10.77.0.9:30800")

    assert str(client._client.base_url).rstrip("/") == "https://10.77.0.9:30800"
    assert _ssl_context_of(client._client) is fleet_ca, "the repoint dropped the fleet CA"
    assert client._client.headers["Authorization"] == "Bearer aureax_sk_AAAA"
    client.close()


def test_the_chat_client_never_builds_an_unverified_transport():
    """No plaintext and no unverified transport anywhere: a client constructed
    without an explicit context must still check the chain and the hostname, so
    a mistake shows up as a certificate error rather than as a silently
    unauthenticated connection."""
    client = ChatLlamaClient("https://10.77.0.1:30800")

    ctx = _ssl_context_of(client._client)

    assert ctx is not None, "httpx was left to build the context lazily"
    assert ctx.verify_mode is ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    client.close()


# --- the HTTP surface in front of all this ------------------------------------

def _api_client(tmp_path, network_config_path) -> TestClient:
    app = create_app(
        metrics_db_path=str(tmp_path / "metrics.db"),
        network_config_path=str(network_config_path),
        chat_db_path=str(tmp_path / "chat.db"),
        log_db_path=str(tmp_path / "log.db"),
    )
    return TestClient(app)


@pytest.fixture
def api_client(tmp_aureax_home, tmp_path) -> TestClient:
    return _api_client(tmp_path, EXAMPLE_NETWORK_CONFIG)


def test_an_unreadable_network_config_is_a_503_not_an_empty_target_list(
    tmp_aureax_home, tmp_path,
):
    broken = tmp_path / "aureax-network.yaml"
    broken.write_text("ssh: [[[\n")

    resp = _api_client(tmp_path, broken).get("/api/chat/targets")

    assert resp.status_code == 503, resp.text
    assert "config unreadable" in resp.json()["detail"]


def test_an_oversized_body_is_refused_before_it_is_parsed(api_client):
    """Whatever the body carries is persisted to sqlite and replayed into every
    later turn of the session, so the cap runs before the parse."""
    session = api_client.app.state.chat.store.create_session(target_id="node-edge-01")

    resp = api_client.post(
        f"/api/chat/sessions/{session.id}/ask",
        json={"prompt": "x" * (chat_api.MAX_ASK_BYTES + 1)},
    )

    assert resp.status_code == 413, resp.text


def test_an_oversized_chunked_body_is_refused_too(api_client):
    """The cap has to hold on the read itself.

    A chunked request carries no Content-Length at all, so a header check would
    let the whole body into RAM — on a bind reachable by anything on the LAN,
    that is a one-request OOM taking the lifecycle manager, the chat MCP pools
    and every SSE stream with it.
    """
    session = api_client.app.state.chat.store.create_session(target_id="node-edge-01")

    def chunked():
        yield b'{"prompt": "'
        for _ in range(chat_api.MAX_ASK_BYTES // 4096 + 2):
            yield b"x" * 4096
        yield b'"}'

    resp = api_client.post(
        f"/api/chat/sessions/{session.id}/ask",
        content=chunked(),
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 413, resp.text


def test_an_oversized_prompt_is_refused(api_client):
    session = api_client.app.state.chat.store.create_session(target_id="node-edge-01")

    resp = api_client.post(
        f"/api/chat/sessions/{session.id}/ask",
        json={"prompt": "x" * (chat_api.MAX_PROMPT_CHARS + 1)},
    )

    assert resp.status_code == 400, resp.text
    assert "prompt" in resp.json()["detail"]


def test_a_supplied_session_title_is_capped(api_client):
    """AUTO_TITLE_MAX_LEN capped only auto-generated titles; a supplied one was
    stored whole and rendered in the session list."""
    targets = api_client.get("/api/chat/targets").json()
    assert targets, "the example fleet config offers no chat targets"

    resp = api_client.post(
        "/api/chat/sessions",
        json={"target_id": targets[0]["id"], "title": "T" * 500},
    )

    assert resp.status_code == 201, resp.text
    assert len(resp.json()["title"]) == chat_service_mod.AUTO_TITLE_MAX_LEN


def test_an_ask_is_refused_once_too_many_are_in_flight(api_client):
    """Asks on one session serialise on ChatService's per-session lock, so an
    unbounded loop of POSTs parks one task per request — each holding its
    prompt until it finally drives a full inference against the node. The
    dashboard binds 0.0.0.0, so the caller need not be the browser."""
    session = api_client.app.state.chat.store.create_session(target_id="node-edge-01")
    at_limit = {object() for _ in range(chat_api._MAX_INFLIGHT_ASKS)}
    api_client.app.state.chat_tasks = at_limit

    resp = api_client.post(
        f"/api/chat/sessions/{session.id}/ask", json={"prompt": "one too many"},
    )

    assert resp.status_code == 429, resp.text
    assert len(at_limit) == chat_api._MAX_INFLIGHT_ASKS, "the refused ask was still spawned"


async def test_cached_engine_rejects_a_now_disabled_target(service, monkeypatch):
    session = service.store.create_session("node-edge-01")
    service._engines[session.id] = object()
    def disabled(*args):
        raise ValueError("unknown or disabled node")
    monkeypatch.setattr(service, "_target_for", disabled)
    with pytest.raises(ValueError, match="disabled"):
        await service._engine_for(session)


async def test_removed_peer_is_evicted_even_inside_retry_backoff(tmp_path, monkeypatch, scripted_pool):
    svc = _planner_service(tmp_path, monkeypatch, ["edge-a", "edge-b"])
    scripted_pool.reachable = {"edge-a", "edge-b"}
    pool = await svc._pool_for("node-planner-01")
    target, config = svc._target_for("node-planner-01")
    config["mcp"]["client"]["remote_servers"] = [_planner_remote("edge-a", None)]
    same_pool = await svc._pool_for("node-planner-01", (target, config))
    assert same_pool is pool
    assert pool.connected_nodes == ["edge-a"]
    assert "edge-b" not in svc._pool_member_urls["node-planner-01"]


async def test_an_ipv6_node_address_is_bracketed_in_both_chat_urls(
    tmp_path, monkeypatch, scripted_pool, fleet_ca,
):
    """An IPv6 literal's own colons read as the port separator in a URL
    authority, so a bare one is unparseable and httpx refuses it. Both builders
    route the host through the tree's single bracketing rule; the fleet CA's
    name constraints admit fc00::/7, so such a host is a supported shape."""
    target = SimpleNamespace(node_id="edge-v6", host="fd00::5", config_file="edge.yaml")
    monkeypatch.setattr(ChatService, "_target_for", lambda _self, _tid: (target, {}))
    svc = ChatService(EXAMPLE_NETWORK_CONFIG, SessionStore(str(tmp_path / "chat.db")))
    scripted_pool.reachable = {"edge-v6"}

    await svc._pool_for("node-edge-v6")
    llm = await svc._llm_for("node-edge-v6")

    assert scripted_pool.urls["edge-v6"] == "https://[fd00::5]:30900/mcp"
    assert llm.base_url == "https://[fd00::5]:30800"
    await svc.close()


def test_the_stream_replays_history_off_the_event_loop(service):
    """list_messages is an unbounded synchronous SELECT over every turn a
    session ever held, and nothing prunes a session. Replaying it inline stalls
    node telemetry/log ingest and every other SSE stream on this loop — once
    per EventSource reconnect, which the browser does on its own."""
    session = service.store.create_session(target_id="node-edge-01")
    service.store.append_message(session.id, "user", "hello")
    seen_threads: list[int] = []
    real = service.store.list_messages

    def recording(session_id):
        seen_threads.append(threading.get_ident())
        return real(session_id)

    service.store.list_messages = recording

    async def scenario() -> int:
        loop_thread = threading.get_ident()
        async for event in service.stream(session.id):
            if event["type"] == "status":
                break
        return loop_thread

    loop_thread = asyncio.run(scenario())
    assert seen_threads, "history was never replayed"
    assert loop_thread not in seen_threads, "the replay ran on the event loop"
