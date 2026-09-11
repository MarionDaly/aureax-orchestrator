"""MCPPool connection lifecycle, and the credentials it carries.

Five invariants:

1. A FAILED connect leaves nothing behind. The transport's __aenter__ does no
   network I/O, so real failures surface later, at initialize() — after the
   task group and httpx client exist. Callers retry cold nodes on a timer, so
   anything not unwound here accumulates for as long as the node is down.

2. A transport error EVICTS the session, in every shape it arrives in — a
   plain exception, the SDK task group's BaseExceptionGroup, or this task's
   own cancellation. connected_nodes is the only liveness signal the
   dashboard's reconnect check reads; a dead node that stays listed makes that
   check permanently unable to fire.

3. A SUCCESSFUL connect over a live entry unwinds the predecessor first, and
   close() unwinds everything. Both are release paths.

4. Cancellation is not a connect failure, and a parse problem is not a
   transport failure.

5. The bearer key and the fleet CA context reach the httpx client that carries
   the whole session, and the key never reaches a log record.

Every pool below is populated by driving ``connect_url``, never by assigning
into the pool's private dicts by hand, so the tests see what ``connect_url``
actually stores rather than a hand-built imitation of it.
"""

from __future__ import annotations

import asyncio
import logging
import ssl
from types import SimpleNamespace

import httpx
import pytest

from client.mcp import pool as pool_mod
from client.mcp.pool import MCPPool

URL = "https://10.77.0.9:30900/mcp"
KEY = "aureax_sk_" + "P" * 43

# Captured before any test patches httpx.AsyncClient. _FakeHttpClient replays
# the recorded kwargs into the real class, and patching is done on the httpx
# module itself (pool_mod.httpx IS httpx), so reading the name at call time
# would hand the fake back to itself.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _FakeHttpClient:
    """Records what connect_url passed, and proves httpx accepts it.

    The kwargs are replayed into a REAL ``httpx.AsyncClient``: asserting on the
    recorded dict alone would pass just as happily for a misspelled ``header=``
    or a ``verify`` value httpx rejects, and that failure would surface on the
    fleet rather than here.
    """

    def __init__(self, *_a, **kw) -> None:
        self.kwargs = kw
        self.real = _REAL_ASYNC_CLIENT(**kw)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
        await self.real.aclose()


class _TransportCtx:
    """Stands in for streamable_http_client: enters cheaply, records exit."""

    def __init__(self) -> None:
        self.exited = False

    async def __aenter__(self):
        return ("read", "write", None)

    async def __aexit__(self, *_exc) -> bool:
        self.exited = True
        return False


class _Session:
    """ClientSession stand-in that completes connect_url's handshake.

    Subclasses supply ``call_tool``; the two handshake methods are here so a
    test only has to say what the live session does with a call.
    """

    async def initialize(self) -> None:
        return None

    async def list_tools(self):
        return SimpleNamespace(tools=[])


class _FailingSessionCtx:
    """ClientSession stand-in whose initialize() fails like a dead node."""

    # connect_url passes read_timeout_seconds; **_kw keeps the stub honest
    # without pinning the keyword's name here.
    exc: BaseException = ConnectionError("node is down")

    def __init__(self, _read, _write, **_kw) -> None:
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc) -> bool:
        self.exited = True
        return False

    async def initialize(self) -> None:
        raise _FailingSessionCtx.exc


class _LiveSessionCtx:
    """ClientSession stand-in that initializes and reports one tool."""

    def __init__(self, _read, _write, **_kw) -> None:
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc) -> bool:
        self.exited = True
        return False

    async def initialize(self) -> None:
        return None

    async def list_tools(self):
        return SimpleNamespace(
            tools=[SimpleNamespace(name="get_health", description="", inputSchema={})]
        )


class _Stubs:
    """Everything connect_url builds, in creation order.

    Recorded per call rather than on the stub classes: a class attribute
    outlives the test that set it, so the next test would read a stale object
    and its assertion would silently stop meaning anything.
    """

    def __init__(self) -> None:
        self.transports: list[_TransportCtx] = []
        self.http_clients: list[_FakeHttpClient] = []
        self.session_ctxs: list = []


def _patch_transport(monkeypatch, session_cls, transport_cls=_TransportCtx) -> _Stubs:
    """Drive connect_url against stubs; returns what it built, in order.

    ``transport_cls`` exists because the SDK reports a transport-level failure
    from the transport context's teardown, not from the call that failed.
    """
    stubs = _Stubs()

    def make_http(*a, **kw) -> _FakeHttpClient:
        stubs.http_clients.append(_FakeHttpClient(*a, **kw))
        return stubs.http_clients[-1]

    def make_transport(_url, http_client, terminate_on_close):
        stubs.transports.append(transport_cls())
        return stubs.transports[-1]

    def make_session(read, write, **kw):
        stubs.session_ctxs.append(session_cls(read, write, **kw))
        return stubs.session_ctxs[-1]

    monkeypatch.setattr(pool_mod.httpx, "AsyncClient", make_http)
    monkeypatch.setattr(pool_mod, "streamable_http_client", make_transport)
    monkeypatch.setattr(pool_mod, "ClientSession", make_session)
    return stubs


def _session_ctx_class(session):
    """A ClientSession stand-in that hands connect_url the given session."""

    class _Ctx:
        def __init__(self, _read, _write, **_kw) -> None:
            self.exited = False

        async def __aenter__(self):
            return session

        async def __aexit__(self, *_exc) -> bool:
            self.exited = True
            return False

    return _Ctx


async def _connected_pool(monkeypatch, session, *node_ids: str) -> tuple[MCPPool, _Stubs]:
    """A pool holding a live entry per node id, built through connect_url."""
    stubs = _patch_transport(monkeypatch, _session_ctx_class(session))
    pool = MCPPool()
    for node_id in node_ids or ("n1",):
        assert await pool.connect_url(node_id, URL) is True, f"{node_id} did not connect"
    return pool, stubs


async def test_failed_connect_leaves_nothing_in_the_pool(monkeypatch) -> None:
    stubs = _patch_transport(monkeypatch, _FailingSessionCtx)

    pool = MCPPool()
    ok = await pool.connect_url("n1", URL)

    assert ok is False
    assert pool.connected_nodes == []
    assert pool._conns == {}
    assert stubs.transports[0].exited, "the transport's task group was left running"
    assert stubs.session_ctxs[0].exited, "the session context was left entered"


async def test_transport_error_evicts_the_session(monkeypatch) -> None:
    class _DeadSession(_Session):
        async def call_tool(self, _name, _args):
            raise ConnectionError("connection reset")

    pool, stubs = await _connected_pool(monkeypatch, _DeadSession())

    result = await pool.call_tool("n1", "get_health", {})

    assert "error" in result
    assert pool.connected_nodes == []
    assert stubs.transports[0].exited and stubs.http_clients[0].closed


async def test_exception_group_mid_call_evicts_the_session(monkeypatch) -> None:
    """The SDK's task group wraps a mid-call transport death in a group.

    A group carrying the cancellation the dying group delivered alongside the
    real cause is a BaseExceptionGroup, not an ExceptionGroup, so a narrower
    catch would let the dead session stay in the pool while connected_nodes
    keeps advertising it.
    """

    class _GroupSession(_Session):
        async def call_tool(self, _name, _args):
            raise BaseExceptionGroup(
                "unhandled errors in a TaskGroup",
                [asyncio.CancelledError(), ConnectionError("connection reset")],
            )

    pool, stubs = await _connected_pool(monkeypatch, _GroupSession())

    result = await pool.call_tool("n1", "get_health", {})

    # _describe unwraps the group; the group's own str names nothing.
    assert "connection reset" in result["error"]
    assert pool.connected_nodes == []
    assert stubs.transports[0].exited and stubs.http_clients[0].closed


def _cancelling_exit_ctx_class(session):
    """A session context whose __aexit__ re-raises a cancellation.

    That is what anyio does once the transport's task group has died under an
    in-flight call: the SESSION context's teardown raises CancelledError before
    the TRANSPORT context has been exited. An eviction that stops there leaves
    the transport's group and the httpx client open and unreachable.
    """

    class _Ctx(_session_ctx_class(session)):
        async def __aexit__(self, *exc) -> bool:
            await super().__aexit__(*exc)
            raise asyncio.CancelledError

    return _Ctx


async def test_cancelled_call_evicts_and_propagates(monkeypatch) -> None:
    """A cancelled call still evicts EVERYTHING, and the cancellation reaches the caller."""

    class _CancelledSession(_Session):
        async def call_tool(self, _name, _args):
            raise asyncio.CancelledError

    stubs = _patch_transport(monkeypatch, _cancelling_exit_ctx_class(_CancelledSession()))
    pool = MCPPool()
    assert await pool.connect_url("n1", URL) is True

    try:
        await pool.call_tool("n1", "get_health", {})
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("call_tool swallowed the cancellation")

    assert pool.connected_nodes == []
    assert pool._conns == {}
    assert stubs.session_ctxs[0].exited, "the session context was left entered"
    assert stubs.transports[0].exited, "the transport's task group was left running"
    assert stubs.http_clients[0].closed, "the httpx client was left open"


async def test_a_cancellation_wrapped_in_a_group_still_propagates(monkeypatch) -> None:
    """The SDK delivers a caller's cancellation inside a BaseExceptionGroup;
    with no other cause beside it, it must not come back as an {"error": ...}
    dict that reads as if the node answered."""

    class _GroupCancelledSession(_Session):
        async def call_tool(self, _name, _args):
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [asyncio.CancelledError()])

    stubs = _patch_transport(monkeypatch, _cancelling_exit_ctx_class(_GroupCancelledSession()))
    pool = MCPPool()
    assert await pool.connect_url("n1", URL) is True

    with pytest.raises(BaseExceptionGroup):
        await pool.call_tool("n1", "get_health", {})

    assert pool.connected_nodes == []
    assert stubs.transports[0].exited and stubs.http_clients[0].closed


async def test_an_interpreter_exit_is_never_turned_into_an_error_dict(monkeypatch) -> None:
    class _ExitingSession(_Session):
        async def call_tool(self, _name, _args):
            raise SystemExit(3)

    pool, _stubs = await _connected_pool(monkeypatch, _ExitingSession())
    with pytest.raises(SystemExit):
        await pool.call_tool("n1", "get_health", {})


async def test_tool_level_error_does_not_evict(monkeypatch) -> None:
    """isError comes from a live server answering — the session is fine."""

    class _LiveSession(_Session):
        async def call_tool(self, _name, _args):
            return SimpleNamespace(
                isError=True, content=[SimpleNamespace(text="tool blew up")]
            )

    pool, _stubs = await _connected_pool(monkeypatch, _LiveSession())

    result = await pool.call_tool("n1", "get_health", {})

    assert result == {"error": "tool blew up"}
    assert pool.connected_nodes == ["n1"]


async def test_cancelled_connect_propagates(monkeypatch) -> None:
    """CancelledError is not a connect failure.

    Soft-failing it consumed the caller's cancel: ChatService.delete_session
    cancels an ask parked in initialize() and then awaits the task, which ran
    on to a full inference instead of stopping.
    """
    stubs = _patch_transport(monkeypatch, _FailingSessionCtx)
    monkeypatch.setattr(_FailingSessionCtx, "exc", asyncio.CancelledError())

    pool = MCPPool()
    try:
        await pool.connect_url("n1", URL)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("connect_url swallowed the cancellation")

    # Still cleaned up on the way out.
    assert pool._conns == {}
    assert stubs.transports[0].exited


# --- how a transport-level failure actually arrives --------------------------
#
# Shapes verified against a live streamable-http server answering 401: the
# POST's raise_for_status fires inside the SDK's anyio task group, the group
# cancels the task parked in initialize() (so connect_url's own `except` sees a
# CancelledError, not the status error), the SESSION context's __aexit__
# re-raises that cancellation, and only the TRANSPORT context's __aexit__ hands
# back the real cause, wrapped in a BaseExceptionGroup.


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", URL)
    return httpx.HTTPStatusError(
        f"Client error '{code}' for url '{URL}'",
        request=request,
        response=httpx.Response(code, request=request),
    )


class _ScopeCancelledSessionCtx:
    """Session context that fails, and cancels again on the way out."""

    def __init__(self, _read, _write, **_kw) -> None:
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc) -> bool:
        self.exited = True
        raise asyncio.CancelledError("Cancelled via cancel scope 0x1")

    async def initialize(self) -> None:
        raise asyncio.CancelledError("Cancelled via cancel scope 0x1")


def _transport_reporting(exc: BaseException):
    """Transport context whose teardown reports ``exc``, as anyio's does."""

    class _Ctx(_TransportCtx):
        async def __aexit__(self, *_exc) -> bool:
            self.exited = True
            raise BaseExceptionGroup("unhandled errors in a TaskGroup", [exc])

    return _Ctx


async def test_a_rejected_key_is_reported_once_and_is_not_a_cancellation(
    monkeypatch, caplog,
) -> None:
    """The most likely failure of MCP auth used to produce ZERO log output.

    The teardown's own CancelledError replaced the real one, so the caller got
    CancelledError('') — a value both consumers wrap in gather(...) and drop —
    and the 401 was never unwrapped by anything. A planner holding a stale key
    then retried on a 60 s backoff forever, in silence.
    """
    stubs = _patch_transport(
        monkeypatch, _ScopeCancelledSessionCtx, _transport_reporting(_status_error(401)),
    )

    pool = MCPPool()
    with caplog.at_level(logging.WARNING, logger="aureax.mcp.pool"):
        ok = await pool.connect_url("shard-head", URL, KEY)

    assert ok is False, "a rejected key must be a soft failure, not a CancelledError"
    assert pool.connected_nodes == []
    assert stubs.transports[0].exited and stubs.http_clients[0].closed
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1, messages
    assert "MCP key rejected (401)" in messages[0]
    assert "shard-head" in messages[0] and URL in messages[0]
    assert "aureax apply" in messages[0]
    assert KEY not in messages[0], "the key must never reach a log record"


async def test_a_transport_failure_is_named_rather_than_reported_as_a_cancel(
    monkeypatch, caplog,
) -> None:
    """Every other transport failure arrives the same way, so it was equally
    silent. The group's own str is "unhandled errors in a TaskGroup", which
    names nothing — the leaf is what the operator needs."""
    stubs = _patch_transport(
        monkeypatch,
        _ScopeCancelledSessionCtx,
        _transport_reporting(httpx.ConnectError("All connection attempts failed")),
    )

    pool = MCPPool()
    with caplog.at_level(logging.WARNING, logger="aureax.mcp.pool"):
        ok = await pool.connect_url("edge-01", URL)

    assert ok is False
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 1, messages
    assert "Failed to connect to edge-01" in messages[0]
    assert "ConnectError: All connection attempts failed" in messages[0]
    assert stubs.http_clients[0].closed


async def test_a_teardown_that_reports_nothing_still_re_raises_a_real_cancel(
    monkeypatch,
) -> None:
    """The other half of the same rule: with no cause behind it, a
    CancelledError IS the caller cancelling, and must still reach them."""
    stubs = _patch_transport(monkeypatch, _ScopeCancelledSessionCtx)

    pool = MCPPool()
    try:
        await pool.connect_url("n1", URL)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("connect_url swallowed the cancellation")

    assert pool._conns == {}
    assert stubs.transports[0].exited and stubs.http_clients[0].closed


async def test_successful_connect_unwinds_the_predecessor(monkeypatch) -> None:
    """The reconnect path must release the dead entry before replacing it.

    Callers retry cold nodes on a timer, so a leak here accumulates for as long
    as the node keeps flapping.
    """
    stubs = _patch_transport(monkeypatch, _LiveSessionCtx)

    pool = MCPPool()
    assert await pool.connect_url("n1", URL) is True
    old_conn = pool._conns["n1"]

    assert await pool.connect_url("n1", URL) is True

    assert stubs.transports[0].exited, "the predecessor's task group was left running"
    assert stubs.http_clients[0].closed, "the predecessor's httpx client was left open"
    assert pool._conns["n1"] is not old_conn
    assert pool.connected_nodes == ["n1"]


async def test_close_unwinds_every_connection(monkeypatch) -> None:
    """close() is the dashboard's only session-release path (app.py shutdown).

    Clearing the dicts without unwinding — the original bug — leaves every
    transport task group and httpx client open for the process's life.
    """
    pool, stubs = await _connected_pool(monkeypatch, _Session(), "n1", "n2")
    assert len(stubs.transports) == 2

    await pool.close()

    assert all(c.exited for c in stubs.transports)
    assert all(h.closed for h in stubs.http_clients)
    assert pool.connected_nodes == []
    assert pool._conns == {}
    assert pool._tools == {}


async def test_non_text_content_block_does_not_evict(monkeypatch) -> None:
    """A result we cannot parse is a bug on THIS side of the wire.

    Evicting for it tore down a live session and cost the caller a reconnect
    plus the POOL_RETRY_SECONDS backoff.
    """

    class _ImageSession(_Session):
        async def call_tool(self, _name, _args):
            # ImageContent has .data, not .text.
            return SimpleNamespace(
                isError=False, content=[SimpleNamespace(data="…", mimeType="image/png")]
            )

    pool, _stubs = await _connected_pool(monkeypatch, _ImageSession())

    result = await pool.call_tool("n1", "get_health", {})

    assert result == {"raw": ""}
    assert pool.connected_nodes == ["n1"]


async def test_list_result_is_wrapped_in_a_dict(monkeypatch) -> None:
    """read_sensor("all") answers with a JSON array; -> dict must stay true."""

    class _ArraySession(_Session):
        async def call_tool(self, _name, _args):
            return SimpleNamespace(
                isError=False, content=[SimpleNamespace(text='[{"sensor_type": "temperature"}]')]
            )

    pool, _stubs = await _connected_pool(monkeypatch, _ArraySession())

    result = await pool.call_tool("n1", "read_sensor", {"sensor_type": "all"})

    assert result == {"result": [{"sensor_type": "temperature"}]}
    # The contract every caller relies on: .get("error") never raises.
    assert result.get("error") is None


# --- the credentials one connection carries ----------------------------------

async def test_the_bearer_key_rides_the_client_that_carries_the_session(monkeypatch) -> None:
    """The SDK's streamable_http_client takes no headers argument.

    So the key has to be set on the pre-built httpx client, which is what every
    leg of one MCP session reuses — the POST, the background SSE GET and the
    DELETE. Set it per request instead and the GET goes out unauthenticated,
    which a bearer-checking server answers with a 401 the SDK reports as a dead
    transport.
    """
    stubs = _patch_transport(monkeypatch, _LiveSessionCtx)

    assert await MCPPool().connect_url("n1", URL, api_key=KEY) is True

    built = stubs.http_clients[0]
    assert built.kwargs["headers"] == {"Authorization": f"Bearer {KEY}"}
    # Replayed into the real httpx client: proves the kwarg is one httpx honours.
    assert built.real.headers["authorization"] == f"Bearer {KEY}"


async def test_no_authorization_header_is_sent_for_an_unkeyed_node(monkeypatch) -> None:
    """Auth-off nodes are the schema default, and the rollback pair stays there.

    Sending `Bearer None` — the shape a naive f-string produces — would be a
    credential-looking header on a server that does not check one, and a real
    401 on one that does.
    """
    stubs = _patch_transport(monkeypatch, _LiveSessionCtx)

    assert await MCPPool().connect_url("n1", URL) is True

    built = stubs.http_clients[0]
    assert built.kwargs["headers"] is None
    assert "authorization" not in built.real.headers


async def test_the_fleet_ca_context_reaches_the_client(monkeypatch) -> None:
    """MCP over TLS is verified against the fleet CA, not the public roots.

    The pool holds one context for every node it dials; dropping it here would
    leave the connection verifying against certifi, which no fleet certificate
    chains to — so the failure is loud, but only if the context is threaded at
    all.
    """
    stubs = _patch_transport(monkeypatch, _LiveSessionCtx)
    ctx = ssl.create_default_context()

    assert await MCPPool(verify=ctx).connect_url("n1", URL) is True

    assert stubs.http_clients[0].kwargs["verify"] is ctx


async def test_a_pool_without_a_context_still_verifies(monkeypatch) -> None:
    """verify=None must mean "no fleet CA supplied", never "do not check".

    httpx resolves None to the public trust store, so a caller that forgets the
    context gets a certificate failure against a fleet node instead of an
    unverified connection. Pinned because the obvious "fix" for that failure is
    to disable verification, which is the one thing this fleet must never do.
    """
    stubs = _patch_transport(monkeypatch, _LiveSessionCtx)

    assert await MCPPool().connect_url("n1", URL) is True

    built = stubs.http_clients[0]
    assert built.kwargs["verify"] is None
    # Asserted on the transport rather than the kwarg, because this is where a
    # disabled client would be visible: httpx turns a falsy `verify` into a
    # CERT_NONE context and stores it right here, while None is left for
    # httpcore to resolve into a verifying default at connect time. So the two
    # arms below are "not configured" and "configured to verify" — never
    # "configured to skip".
    stored = built.real._transport._pool._ssl_context
    assert stored is None or stored.verify_mode is ssl.CERT_REQUIRED, (
        "the pool built a client that would accept any certificate"
    )


async def test_the_key_never_reaches_a_log_record(monkeypatch, caplog) -> None:
    """Both log lines carry the URL only.

    Pool records travel under the "aureax" logger hierarchy into the shipped
    log store, which the dashboard serves unauthenticated on :7001 — a key in
    one of these lines is readable by anyone who can reach the LAN.
    """
    caplog.set_level(logging.DEBUG, logger="aureax.mcp.pool")

    _patch_transport(monkeypatch, _LiveSessionCtx)
    assert await MCPPool().connect_url("n1", URL, api_key=KEY) is True

    _patch_transport(monkeypatch, _FailingSessionCtx)
    monkeypatch.setattr(_FailingSessionCtx, "exc", ConnectionError("401 Unauthorized"))
    assert await MCPPool().connect_url("n2", URL, api_key=KEY) is False

    messages = [rec.getMessage() for rec in caplog.records]
    assert messages, "connect_url logged nothing at all"
    assert any(URL in m for m in messages), "the URL stopped being logged"
    for message in messages:
        assert KEY not in message
        assert "Bearer" not in message
