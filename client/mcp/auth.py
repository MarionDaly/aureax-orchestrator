"""Bearer-token gate in front of the edge MCP app.

The MCP port is reachable twice — directly as ``hostNetwork`` :9000 and through
the NodePort Service on :30900 (``k8s/daemonset-edge.yaml``'s
``containerPort: 9000``, ``k8s/service-mcp.yaml``'s ``nodePort: 30900``). Both
doors lead into the *same* uvicorn process
(``client/entrypoint.py::_start_mcp_server``), so one ASGI wrapper
around the app covers both; there is no second listener to secure separately.
That matters because ``run_inference`` forwards to llama-server with the node's
llama bearer key attached, which made an unauthenticated MCP port a way to
reach key-protected inference through the proxy.

``/health`` stays exempt deliberately, not by oversight. The deploy and verify
scripts poll it with a plain unauthenticated ``curl``
(``scripts/lib/common.sh::mcp_health_wait``, called from ``setup-all.sh``,
``update-all.sh`` and ``test-all.sh``), and those run before any key is in hand
on a freshly imaged node. Requiring a key there would turn every provisioning
run into a manual step.

Written as a **pure ASGI middleware**, not a ``BaseHTTPMiddleware`` subclass.
BaseHTTPMiddleware buffers the response through an anyio stream and only sees
``http`` scopes, which breaks this app three ways: the MCP transport holds a
long-lived SSE GET open, POST replies are streamed
(``sse_starlette.EventSourceResponse``), and the ``lifespan`` scope must reach
the inner app untouched because the app's lifespan *is*
``session_manager.run()`` — swallow it and every session dies at startup.
"""

from __future__ import annotations

import hmac
import logging

from starlette.responses import JSONResponse

logger = logging.getLogger("aureax.mcp.auth")

# Realm string in the WWW-Authenticate challenge. Names the fleet component so
# an operator staring at a 401 in a browser or curl -v knows which key to reach
# for — the node holds two (llama and MCP) and they are not interchangeable.
AUTH_REALM = "aureax-mcp"

# One WARNING per process, everything after it at DEBUG. An open port on a LAN
# collects scanners, and a per-request WARNING would flood SystemLogger's
# in-memory ring and the log store that ships to the orchestrator — pushing the
# real startup lines out of both. The first line is the one that tells an
# operator "your key is wrong"; the thousandth adds nothing.
_warned = False


def reset_rejection_log_state() -> None:
    """Re-arm the once-per-process WARNING. For tests only.

    The flag is module state, so without this a test that asserts on the first
    rejection would pass or fail depending on which test ran before it.
    """
    global _warned
    _warned = False


def _presented_token(headers: list[tuple[bytes, bytes]]) -> bytes:
    """Extract the bearer token from raw ASGI headers, as **bytes**.

    Bytes all the way through on purpose: this value is attacker-controlled and
    ``hmac.compare_digest`` raises TypeError on a ``str`` containing non-ASCII,
    which would turn a garbage token into a 500 (and a stack trace in the log)
    instead of a 401. The raw ASGI header is already bytes, so never decode it.

    Returns ``b""`` for a missing header, a non-Bearer scheme or an empty
    token — all of which then fail the comparison against a non-empty key.
    """
    for name, value in headers:
        # ASGI guarantees lower-cased header names, but a hand-built scope in a
        # test (or a future non-uvicorn server) need not, so fold anyway.
        if name.lower() != b"authorization":
            continue
        scheme, _, token = value.partition(b" ")
        # RFC 7235: the scheme is case-insensitive. Clients send "Bearer",
        # "bearer" and "BEARER" in the wild; rejecting on case would be a 401
        # that no amount of checking the key would explain.
        if scheme.lower() != b"bearer":
            return b""
        return token.strip()
    return b""


def _normalise_path(path: str) -> str:
    """Fold a trailing slash so ``/health/`` matches ``/health``; ``/`` stays ``/``."""
    return path.rstrip("/") or "/"


def _log_rejection(scope, path: str) -> None:
    """Log a rejected request — path and peer only, never the presented token.

    The token is a credential even when it is wrong: it may be another node's
    live key pasted at the wrong host, and this log is shipped to the
    orchestrator and kept for ``logging.retention_days``.
    """
    global _warned
    client = scope.get("client")
    peer = client[0] if client else "unknown"
    message = "MCP auth rejected %s from %s — missing or invalid bearer token"
    if _warned:
        logger.debug(message, path, peer)
        return
    _warned = True
    logger.warning(
        message + " (further rejections are logged at DEBUG)",
        path,
        peer,
    )


class BearerAuthMiddleware:
    """Require ``Authorization: Bearer <key>`` on every path but the exempt set.

    Args:
        app: the inner ASGI app (``mcp.streamable_http_app()``).
        api_key: the node's MCP key, from its resolved config.
        exempt_paths: paths served without a key. An **exact set**, not a
            prefix match — ``/healthz`` must not ride in on ``/health``'s
            exemption, and a prefix rule would hand a future ``/health-admin``
            route away for free.
    """

    def __init__(
        self,
        app,
        api_key: str,
        exempt_paths: frozenset[str] = frozenset({"/health"}),
    ) -> None:
        if not api_key:
            # An empty key would make compare_digest(b"", b"") true for a
            # request with no Authorization header at all — auth silently off
            # while the log says it is on. Callers gate on a key being set
            # (client/mcp/server.py::build_app); this is the backstop for a
            # config that resolved `api_key:` to an empty string.
            raise ValueError("BearerAuthMiddleware requires a non-empty api_key")
        self.app = app
        self._expected = api_key.encode()
        # Normalise the exemptions the same way request paths are normalised
        # below, so passing "/health/" configures the same route as "/health".
        self._exempt = frozenset(_normalise_path(p) for p in exempt_paths)

    async def __call__(self, scope, receive, send) -> None:
        # Anything that is not an HTTP request — above all ``lifespan`` —
        # passes through untouched. The app's lifespan is
        # ``session_manager.run()``: intercept or drop it and the MCP session
        # manager never starts, so every /mcp call fails with no request ever
        # having been rejected.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = _normalise_path(scope.get("path") or "/")
        if path in self._exempt:
            await self.app(scope, receive, send)
            return

        if hmac.compare_digest(_presented_token(scope.get("headers") or []), self._expected):
            await self.app(scope, receive, send)
            return

        _log_rejection(scope, path)
        response = JSONResponse(
            {"error": "unauthorized"},
            status_code=401,
            # Without this header a client cannot tell "wrong key" from "wrong
            # URL"; with it, curl -v and every HTTP library name the scheme.
            headers={"WWW-Authenticate": f'Bearer realm="{AUTH_REALM}"'},
        )
        await response(scope, receive, send)
