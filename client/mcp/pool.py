"""MCP client session pool — connects to remote MCP servers and manages tool dispatch.

Reuses the same streamable-http pattern proven in the harness CLI. Two
consumers, both of which drive the same cold-start path: the planner pod's
MCP-host runtime (``client/entrypoint.py::_run_planner_host``, gated on
``mcp.client.enabled``) and the dashboard chat
(``orchestrator/dashboard/chat_service.py::_pool_for``, which refreshes this
object in place rather than replacing it).
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger("aureax.mcp.pool")

# MCP-level deadline for one tool call. The httpx read timeout below is NOT a
# backstop: the edge MCP server wraps every stream in sse-starlette's
# EventSourceResponse, which pings every 15s, so bytes keep arriving while a
# tool hangs and the read leg never expires. Sits above the edge's own
# inference budget (client.inference.llama_client.LLAMA_HTTP_TIMEOUT_SECONDS,
# 600s) so a real long generation still completes; kept local to avoid a
# cross-module import for one constant.
MCP_CALL_TIMEOUT_SECONDS = 660.0


def _unauthorized(exc: BaseException | None) -> bool:
    """Whether a connect failure is the peer rejecting our bearer key.

    Recursive because the status error never arrives bare: the SDK raises it
    inside its anyio task group, which hands it back wrapped in a
    BaseExceptionGroup (nested one deeper again when the group aborts).
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 401
    if isinstance(exc, BaseExceptionGroup):
        return any(_unauthorized(sub) for sub in exc.exceptions)
    return False


def _is_cancellation(exc: BaseException) -> bool:
    """True when ``exc`` is nothing but cancellation: bare, or a group whose every leaf is one.

    A group that also carries a real cause (a ConnectionError beside the
    CancelledError anyio adds when the transport's task dies) is a transport
    failure the caller should hear about as an error, not a cancellation to
    propagate.
    """
    if isinstance(exc, asyncio.CancelledError):
        return True
    return (isinstance(exc, BaseExceptionGroup) and bool(exc.exceptions)
            and all(_is_cancellation(sub) for sub in exc.exceptions))


def _describe(exc: BaseException) -> str:
    """One log-line rendering of a transport failure, groups unwrapped.

    A BaseExceptionGroup's own str is "unhandled errors in a TaskGroup (1
    sub-exception)" — true, and useless to whoever is reading the log to find
    out why a node will not connect.
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(_describe(sub) for sub in exc.exceptions) or type(exc).__name__
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class _NodeConn:
    """Everything one node's connection owns, so it can be torn down as a unit."""

    def __init__(self, contexts: list, http_client: httpx.AsyncClient) -> None:
        # Entered context managers, outermost first (transport, then session).
        self.contexts = contexts
        self.http_client = http_client

    async def unwind(self, *, tolerate_cancel: bool = False) -> list[BaseException]:
        """Exit the contexts innermost-first, then close the HTTP client.

        Returns what each step raised and this swallowed, newest step first —
        the only place a connect failure's real cause is ever visible (see
        connect_url), and ignored by every other caller.

        Suppresses Exception and BaseExceptionGroup per step: anyio TaskGroup
        cleanup raises BaseExceptionGroup (and, when unwound from a task other
        than the one that entered it, RuntimeError). Leaking one object beats
        aborting the teardown of the rest.

        Those two only, never bare BaseException: both steps are await points,
        so a cancel delivered during teardown has to reach the caller —
        MCPPool.close() is awaited from a dashboard lifespan shutdown and from
        _run_planner_host's finally, and both must stop when cancelled.

        ``tolerate_cancel`` swallows CancelledError per step as well; the two
        failure paths — connect_url's and call_tool's — pass it, close() does
        not. When a background transport task
        dies (a 401, a refused connection), anyio cancels the awaiting task and
        the SESSION context's __aexit__ re-raises that cancellation — before
        the TRANSPORT context, the one holding the group with the real cause,
        has been exited at all. Stopping there is what made a rejected key
        silent: the caller got CancelledError('') and the 401 was never
        unwrapped by anything.
        """
        caught: tuple[type[BaseException], ...] = (Exception, BaseExceptionGroup)
        if tolerate_cancel:
            caught = (*caught, asyncio.CancelledError)
        suppressed: list[BaseException] = []
        for cm in reversed(self.contexts):
            try:
                await cm.__aexit__(None, None, None)
            except caught as e:
                suppressed.append(e)
        try:
            await self.http_client.aclose()
        except caught as e:
            suppressed.append(e)
        return suppressed


class MCPPool:
    """Pool of MCP client sessions to remote servers."""

    def __init__(self, verify: ssl.SSLContext | None = None) -> None:
        """Initialize empty pool.

        ``verify`` is the fleet CA context every connection is made under —
        ``tlsstore.ca_ssl_context()`` on the Mac, ``client.tls.ssl_context()``
        in the planner pod. Held on the pool rather than passed per call
        because every node in one pool is dialled against the same CA.

        Leaving it None is NOT "do not verify": httpx resolves None to the
        public trust store, so a caller that forgets to thread the context
        fails the handshake against a node's fleet-CA certificate instead of
        quietly connecting to something unauthenticated.
        """
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, list] = {}
        self._conns: dict[str, _NodeConn] = {}
        self._verify = verify

    async def connect_url(self, node_id: str, url: str, api_key: str | None = None) -> bool:
        """Connect to a remote MCP server by URL and discover its tools.

        Returns True on success, False on failure.

        ``api_key`` is the target node's ``mcp.server.api_key``. It rides the
        httpx client rather than a per-request kwarg because the SDK's
        ``streamable_http_client`` takes no headers argument — the pre-built
        client is the supported way to authenticate every leg of one session
        (the POST, the background SSE GET, the DELETE), and a header set once
        at construction cannot be forgotten on one of them.
        """
        entered: list = []
        http_client = httpx.AsyncClient(
            # connect=5s, set explicitly: an unready MCP pod behind the
            # NodePort DROPs the SYN rather than refusing it, so the connect
            # leg always runs to its full budget and every dial of a cold node
            # costs whatever this says. A LAN handshake is sub-second and 5s
            # still covers two SYN retransmits. write/pool stay at 30s.
            timeout=httpx.Timeout(30.0, connect=5.0, read=300.0),
            follow_redirects=True,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            verify=self._verify,
        )
        try:
            # terminate_on_close=False: the SDK would otherwise send the
            # session-ending DELETE from whichever task exits the context, and
            # unwind() runs from a different task than the one that entered it
            # (MCPPool.close() from a lifespan shutdown, _evict() from a tool
            # call). A DELETE there costs up to the 5s connect budget per node
            # inside the pod's termination grace, on a path whose whole job is
            # to let go. The server-side session is left to age out instead.
            ctx = streamable_http_client(url, http_client=http_client, terminate_on_close=False)
            # NOTE: this __aenter__ does no network I/O — it starts the SDK's
            # task group and returns. A dead node fails later, at initialize().
            read, write, _ = await ctx.__aenter__()
            entered.append(ctx)

            session_ctx = ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=MCP_CALL_TIMEOUT_SECONDS)
            )
            session = await session_ctx.__aenter__()
            entered.append(session_ctx)

            await session.initialize()
            tools_result = await session.list_tools()

            # Replacing a dead predecessor (reconnect path): release what it
            # still holds before dropping the last reference to it.
            await self._evict(node_id)

            self._conns[node_id] = _NodeConn(entered, http_client)
            self._sessions[node_id] = session
            self._tools[node_id] = tools_result.tools

            tool_names = [t.name for t in tools_result.tools]
            # URL only, never the key or the client's headers. Both pool
            # consumers run under the "aureax" logger hierarchy, whose records
            # are mirrored into the shipped log store and served by the
            # dashboard's unauthenticated log console — a bearer token in this
            # line would be readable by anyone who can reach :7001.
            logger.info("Connected to %s (%s) — %d tools: %s", node_id, url, len(tool_names), tool_names)
            return True

        except BaseException as e:
            # BaseException (not Exception) — anyio TaskGroup raises
            # BaseExceptionGroup on connect failure, which bypasses
            # `except Exception`. Treating a connect failure as a soft False
            # keeps cold-start callers (planner pod, dashboard chat)
            # crash-free; cancellation is re-raised below.
            #
            # Unwind exactly what was entered. Callers retry a cold node on a
            # timer, so anything left behind here — the transport's task
            # group, the httpx client — accumulates for as long as the node
            # is down.
            suppressed = await _NodeConn(entered, http_client).unwind(tolerate_cancel=True)
            # `e` itself is a CancelledError for EVERY transport-level failure,
            # not just a real cancel: anyio cancels this task's await when a
            # background task in the SDK's group dies, so the httpx error that
            # actually killed the connect only ever surfaces from the teardown
            # above. Looked for only when `e` says nothing — an exception the
            # connect itself raised is the more direct cause.
            cause = e
            if isinstance(e, asyncio.CancelledError):
                cause = next(
                    (x for x in suppressed if not isinstance(x, asyncio.CancelledError)), e,
                )
            # Cancellation is not a connect failure, so it is re-raised: a
            # caller that cancels an in-flight connect must actually stop.
            # ChatService.delete_session cancels an ask parked in initialize()
            # and then awaits the task. Only when the teardown found no other
            # cause: otherwise the cancel came from the SDK's own group, and
            # the failure below is the thing worth reporting.
            if cause is e and isinstance(e, asyncio.CancelledError):
                raise
            if _unauthorized(cause):
                # The named remedy matters: nothing else in the fleet writes a
                # rejected MCP key down. Both consumers retry on a timer, so
                # without this line a stale key is an endless silent retry.
                logger.warning(
                    "MCP key rejected (401) by %s (%s) — the peer's key changed or this "
                    "side's key is stale; run `aureax apply` to push the current one",
                    node_id, url,
                )
                return False
            # URL only here too — see the success line above. _describe unwraps
            # the exception group, whose own str ("unhandled errors in a
            # TaskGroup") names nothing; neither carries the token that was
            # sent.
            logger.warning("Failed to connect to %s (%s): %s", node_id, url, _describe(cause))
            return False

    async def disconnect(self, node_id: str) -> None:
        """Remove a peer and its tools before accepting further calls."""
        await self._evict(node_id)

    async def _evict(self, node_id: str, *, tolerate_cancel: bool = False) -> None:
        """Drop a node's session and release everything it owns.

        ``tolerate_cancel`` is for a failure path: the conn is popped from the
        pool BEFORE it is unwound, so a cancellation re-raised by the session
        context's __aexit__ (what anyio does when the transport's group has
        died) would otherwise escape here with the transport context and the
        httpx client still open and no longer reachable from anywhere.
        """
        conn = self._conns.pop(node_id, None)
        self._sessions.pop(node_id, None)
        self._tools.pop(node_id, None)
        if conn is not None:
            await conn.unwind(tolerate_cancel=tolerate_cancel)

    async def call_tool(self, node_id: str, tool_name: str, args: dict) -> dict:
        """Call a tool on a specific remote server. Returns parsed result."""
        session = self._sessions.get(node_id)
        if not session:
            return {"error": f"No session for node '{node_id}'"}

        try:
            result = await session.call_tool(tool_name, args)
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as e:
            # Transport-level failure: evict, so connected_nodes stops
            # reporting a dead node and callers' reconnect checks can fire.
            # BaseException for the same reason connect_url catches it: the
            # transport's anyio task group is still running behind this await,
            # so a transport that dies mid-call arrives as a
            # BaseExceptionGroup, or cancels this await outright — neither of
            # which `except Exception` sees, and both of which would otherwise
            # leave a dead session in the pool. Eviction is the load-bearing
            # half, and it tolerates the cancellation the teardown re-raises
            # for the reason _evict's docstring gives.
            await self._evict(node_id, tolerate_cancel=True)
            logger.warning(
                "Evicted %s from the pool after a transport error: %s", node_id, _describe(e),
            )
            # Cancellation keeps propagating: a caller that cancels an in-flight
            # tool call must actually stop, not receive an {"error": ...} that
            # reads like the node answered. A cancellation the SDK wrapped in
            # a group is still a cancellation.
            if _is_cancellation(e):
                raise
            return {"error": _describe(e)}

        # Parsing sits OUTSIDE that try: only the RPC can fail the transport.
        # A first content block that is not TextContent (image, embedded
        # resource, structured-only) is a bug on this side of the wire, and
        # evicting a live session for it costs the caller a reconnect plus the
        # POOL_RETRY_SECONDS backoff — so it degrades to the {"raw": ""} path.
        text = getattr(result.content[0], "text", "") if result.content else ""

        # isError is a tool-level failure from a live server — the
        # session is fine, so the node stays in the pool.
        if result.isError:
            return {"error": text or "unknown error"}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
        # read_sensor("all") answers with a JSON array, every other tool with an
        # object. Wrapping keeps the -> dict annotation and the {"error": ...}
        # contract true for all of them; engine.py re-serializes either shape.
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    def get_openai_tools(self) -> list[dict]:
        """Convert all MCP tool schemas to OpenAI function-calling format.

        Tools are namespaced by node_id: "edge-01__read_sensor".
        The double underscore separates node_id from tool_name.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": f"{node_id}__{tool.name}",
                    "description": f"[{node_id}] {tool.description or tool.name}",
                    "parameters": tool.inputSchema,
                },
            }
            for node_id, tools in self._tools.items()
            for tool in tools
        ]

    def parse_tool_call(self, function_name: str) -> tuple[str, str]:
        """Parse a namespaced tool name into (node_id, tool_name).

        A name with no "__" yields ("", name), which call_tool turns into a
        "No session for node ''" error — the model emitted an un-namespaced
        tool name.
        """
        if "__" in function_name:
            node_id, tool_name = function_name.split("__", 1)
            return node_id, tool_name
        return "", function_name

    @property
    def connected_nodes(self) -> list[str]:
        """List of connected node IDs."""
        return list(self._sessions.keys())

    async def close(self) -> None:
        """Close all sessions and HTTP clients."""
        for conn in reversed(list(self._conns.values())):
            await conn.unwind()
        self._sessions.clear()
        self._tools.clear()
        self._conns.clear()
