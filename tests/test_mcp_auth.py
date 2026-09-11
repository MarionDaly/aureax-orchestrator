"""Bearer gate in front of the edge MCP app (``client/mcp/auth.py``).

The middleware is the single enforcement point for the MCP port, which is
reachable both as ``hostNetwork`` :9000 and as NodePort :30900. Every case here
is a way to get that gate silently wrong: a 500 where a 401 belongs, a prefix
match that hands ``/healthz`` away with ``/health``, a swallowed ``lifespan``
scope that kills the MCP session manager without ever rejecting a request, or a
rejected token echoed into a log that ships to the orchestrator.

Driven through ``starlette.testclient.TestClient`` over a stub inner app rather
than the real ``mcp.streamable_http_app()``: the gate sits above the whole app,
so what the inner app is must not matter — and the stub lets each test assert
whether the request actually reached it.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from starlette.testclient import TestClient

from client.mcp import auth

KEY = "aureax_sk_" + "a" * 43


class _StubApp:
    """Minimal ASGI app in place of ``mcp.streamable_http_app()``.

    Records what reached it, so "the middleware passed this through" is an
    assertion about the inner app being called, not about a status code that a
    401 response could also produce.
    """

    def __init__(self) -> None:
        self.http_paths: list[str] = []
        self.lifespan_messages: list[str] = []

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                self.lifespan_messages.append(message["type"])
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        self.http_paths.append(scope["path"])
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"ok":true}'})


@pytest.fixture(autouse=True)
def _reset_warning_flag():
    """The once-per-process WARNING is module state — re-arm it around every test."""
    auth.reset_rejection_log_state()
    yield
    auth.reset_rejection_log_state()


@pytest.fixture
def stub() -> _StubApp:
    return _StubApp()


@pytest.fixture
def client(stub: _StubApp) -> TestClient:
    return TestClient(auth.BearerAuthMiddleware(stub, KEY))


# ---- the 401 matrix ---------------------------------------------------------

def test_no_authorization_header_is_401(client: TestClient, stub: _StubApp) -> None:
    """The default case for anyone who finds the port with a scanner."""
    response = client.get("/mcp")
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    # The request must not have reached the MCP app at all: an inner app that
    # saw the scope could have already opened a session or spent inference.
    assert stub.http_paths == []


def test_wrong_scheme_is_401(client: TestClient) -> None:
    """Basic auth with the key as a password is not a bearer token."""
    response = client.get("/mcp", headers={"Authorization": f"Basic {KEY}"})
    assert response.status_code == 401


def test_wrong_key_is_401(client: TestClient) -> None:
    response = client.get("/mcp", headers={"Authorization": "Bearer aureax_sk_not-the-key"})
    assert response.status_code == 401


def test_empty_token_is_401(client: TestClient) -> None:
    """``Bearer `` with nothing after it must not compare equal to anything.

    Guards the shape of the empty-key backstop: if the expected key were ever
    empty, this request would be the one that sailed through.
    """
    response = client.get("/mcp", headers={"Authorization": "Bearer "})
    assert response.status_code == 401


def test_401_carries_the_www_authenticate_challenge(client: TestClient) -> None:
    """Without the challenge header a caller cannot tell a wrong key from a wrong URL."""
    response = client.get("/mcp")
    assert response.headers["www-authenticate"] == 'Bearer realm="aureax-mcp"'


def test_correct_key_reaches_the_inner_app(client: TestClient, stub: _StubApp) -> None:
    response = client.get("/mcp", headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 200
    assert stub.http_paths == ["/mcp"]


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER", "BeArEr"])
def test_scheme_match_is_case_insensitive(scheme: str, stub: _StubApp) -> None:
    """RFC 7235 makes the scheme case-insensitive; clients send all three spellings.

    A case-sensitive check would produce a 401 that no amount of checking the
    key explains — the worst kind of auth bug to debug over SSH.
    """
    client = TestClient(auth.BearerAuthMiddleware(stub, KEY))
    response = client.get("/mcp", headers={"Authorization": f"{scheme} {KEY}"})
    assert response.status_code == 200


def test_non_ascii_token_is_401_not_500(client: TestClient, stub: _StubApp) -> None:
    """A UTF-8 token must be rejected, not crash the request.

    ``hmac.compare_digest`` raises TypeError on a ``str`` with a codepoint above
    U+00FF, so a middleware that decoded the header would turn an attacker-chosen
    token into a 500 plus a stack trace in the shipped log. The comparison runs
    on the raw ASGI bytes precisely to keep this a 401.

    Sent as raw bytes because httpx refuses to encode a non-ASCII ``str`` header
    value — but nothing stops a hand-rolled client (or ``curl``) from putting
    exactly these bytes on the wire, which is the case being pinned.
    """
    response = client.get("/mcp", headers={b"Authorization": "Bearer schlüssel-🔑".encode()})
    assert response.status_code == 401
    assert stub.http_paths == []


# ---- the exemption ----------------------------------------------------------

def test_health_is_open_without_a_key(client: TestClient, stub: _StubApp) -> None:
    """scripts/lib/common.sh::mcp_health_wait polls this with a plain curl."""
    response = client.get("/health")
    assert response.status_code == 200
    assert stub.http_paths == ["/health"]


def test_health_with_trailing_slash_is_open(client: TestClient, stub: _StubApp) -> None:
    """``/health/`` is the same route; a redirect-then-retry would break the poll loop."""
    response = client.get("/health/")
    assert response.status_code == 200
    assert stub.http_paths == ["/health/"]


def test_healthz_is_not_exempt(client: TestClient, stub: _StubApp) -> None:
    """The exemption is an exact set, not a prefix — a prefix rule would give
    away every future path that happens to start with ``/health``."""
    response = client.get("/healthz")
    assert response.status_code == 401
    assert stub.http_paths == []


def test_root_is_not_exempt(client: TestClient) -> None:
    """``/`` normalises to ``/`` and must not collapse into the exempt set."""
    assert client.get("/").status_code == 401


def test_custom_exempt_set_replaces_the_default(stub: _StubApp) -> None:
    """An explicit exempt set is honoured verbatim — /health loses its pass."""
    client = TestClient(auth.BearerAuthMiddleware(stub, KEY, frozenset({"/ping"})))
    assert client.get("/ping").status_code == 200
    assert client.get("/health").status_code == 401


# ---- scopes that are not HTTP ----------------------------------------------

def test_lifespan_scope_passes_through(stub: _StubApp) -> None:
    """The app's lifespan IS ``session_manager.run()``.

    Swallow or answer it in the middleware and the MCP session manager never
    starts, so every /mcp call fails without a single request having been
    rejected — the failure mode a status-code test would never see.
    """
    with TestClient(auth.BearerAuthMiddleware(stub, KEY)):
        assert stub.lifespan_messages == ["lifespan.startup"]
    assert stub.lifespan_messages == ["lifespan.startup", "lifespan.shutdown"]


def test_non_http_scope_is_forwarded_verbatim(stub: _StubApp) -> None:
    """Any non-http scope (here a bare lifespan) reaches the inner app untouched."""
    middleware = auth.BearerAuthMiddleware(stub, KEY)
    sent: list[dict] = []
    messages = iter([{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}])

    async def receive() -> dict:
        return next(messages)

    async def send(message: dict) -> None:
        sent.append(message)

    asyncio.run(middleware({"type": "lifespan"}, receive, send))
    assert [m["type"] for m in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


# ---- construction guard -----------------------------------------------------

def test_empty_api_key_is_refused(stub: _StubApp) -> None:
    """An empty expected key would make ``compare_digest(b"", b"")`` true for a
    request with no Authorization header — auth off while the log says ENABLED."""
    with pytest.raises(ValueError, match="non-empty api_key"):
        auth.BearerAuthMiddleware(stub, "")


# ---- logging ----------------------------------------------------------------

def test_rejections_never_log_the_presented_token(client: TestClient, caplog) -> None:
    """A wrong token is still a credential — it may be another node's live key.

    Also pins the once-per-process rule: three rejections produce exactly one
    WARNING, because an open port on a LAN collects scanners and a per-request
    WARNING would push the real startup lines out of the in-memory ring and the
    log store that ships to the orchestrator.
    """
    token = "aureax_sk_leakcanary9zZ"
    with caplog.at_level(logging.DEBUG, logger="aureax.mcp.auth"):
        for _ in range(3):
            assert client.get("/mcp", headers={"Authorization": f"Bearer {token}"}).status_code == 401

    records = [r for r in caplog.records if r.name == "aureax.mcp.auth"]
    assert len(records) == 3
    assert [r.levelno for r in records] == [logging.WARNING, logging.DEBUG, logging.DEBUG]

    # No substring of the token — not the whole thing, not the part after the
    # shared aureax_sk_ prefix — may appear anywhere in the emitted text.
    rendered = "\n".join(r.getMessage() for r in records)
    assert token not in rendered
    assert "leakcanary9zZ" not in rendered
    assert "/mcp" in rendered


def test_first_rejection_names_the_path_and_peer(client: TestClient, caplog) -> None:
    """The one WARNING has to be actionable on its own: which path, which caller."""
    with caplog.at_level(logging.DEBUG, logger="aureax.mcp.auth"):
        client.get("/tools")

    warning = next(r for r in caplog.records if r.levelno == logging.WARNING)
    message = warning.getMessage()
    assert "/tools" in message
    # TestClient's default client address; the field exists so a real rejection
    # names the source LAN address.
    assert "testclient" in message


def test_authorised_requests_log_nothing(client: TestClient, caplog) -> None:
    """A working fleet must not write a line per tool call into the shipped store."""
    with caplog.at_level(logging.DEBUG, logger="aureax.mcp.auth"):
        assert client.get("/mcp", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
        assert client.get("/health").status_code == 200

    assert [r for r in caplog.records if r.name == "aureax.mcp.auth"] == []
