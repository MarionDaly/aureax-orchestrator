"""A foreign web page cannot drive the dashboard, and a local one needs no token.

The dashboard is deliberately credential-free, so the only boundary it has is
the browser's: `BrowserBoundaryMiddleware` rejects a request whose Host or
Origin says it was issued by a page the operator did not open on this machine
— DNS rebinding, a cross-site form post, an opaque origin — before any route
runs. A request that came from the dashboard itself is served unchanged, and
never cached, so a key read stays out of the browser's disk cache.
See ARCHITECTURE.md, "Security".
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orchestrator.dashboard.browser_boundary import BrowserBoundaryMiddleware


@pytest.mark.parametrize("headers", [
    {"host": "attacker.example"}, {"origin": "https://attacker.example"},
    {"origin": "null"}, {"sec-fetch-site": "cross-site"}, {"origin": "http://localhost:99"},
])
def test_foreign_browser_requests_never_invoke_administration(headers):
    app = FastAPI()
    invoked = []
    @app.post("/api/stop")
    def stop():
        invoked.append(True)
        return {}
    app.add_middleware(BrowserBoundaryMiddleware)
    response = TestClient(app, base_url="http://localhost").post("/api/stop", headers=headers, content="{}")
    assert response.status_code == 403
    assert not invoked


def test_trusted_dashboard_requests_need_no_credentials_and_are_not_cached():
    app = FastAPI()
    @app.get("/api/key")
    def key():
        return {"key": "synthetic"}
    app.add_middleware(BrowserBoundaryMiddleware)
    response = TestClient(app, base_url="http://localhost").get("/api/key", headers={"origin": "http://localhost"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def _ingest_app(trusted_hosts=()):
    """A stand-in for the dashboard's node-ingest route behind the boundary."""
    app = FastAPI()

    @app.post("/api/v1/telemetry")
    def telemetry():
        return {"ok": True}

    app.add_middleware(BrowserBoundaryMiddleware, trusted_hosts=trusted_hosts)
    # base_url is the accepted socket's address, so scope["server"] never
    # matches the Host header under test — only trusted_hosts can.
    return TestClient(app, base_url="https://127.0.0.1:7001")


@pytest.mark.parametrize("bind, host_header", [
    ("mac-orchestrator.local", "mac-orchestrator.local:7001"),
    ("MAC-ORCHESTRATOR.local", "mac-orchestrator.local:7001"),
    ("fd00::1", "[fd00::1]:7001"),
    ("192.0.2.7", "192.0.2.7:7001"),
])
def test_a_node_reaches_ingest_by_the_address_the_fleet_was_given(bind, host_header):
    """`${orchestrator_url}` stamps the address `aureax generate` was given into
    every node's telemetry and log endpoint, and a hostname is legal there — so
    a node's push arrives with that name as its Host. The bind address run()
    passes through is what makes it a name this dashboard answers to; without
    it a hostname-addressed fleet has every push rejected while a browser on
    the IP keeps working."""
    response = _ingest_app(trusted_hosts=(bind,)).post(
        "/api/v1/telemetry", headers={"host": host_header}, content="{}")

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("host_header", [
    "mac-orchestrator.local:7001", "[fd00::1]:7001", "attacker.example:7001",
])
def test_an_address_this_dashboard_was_not_started_on_is_still_refused(host_header):
    """Threading the bind address through widens the allowlist by exactly one
    name. Anything else — including the hostname a differently-configured fleet
    would use — is still a rebinding attempt."""
    response = _ingest_app().post(
        "/api/v1/telemetry", headers={"host": host_header}, content="{}")

    assert response.status_code == 403


def test_a_wildcard_bind_is_not_a_name_the_dashboard_answers_to():
    """`server.dashboard.host` defaults to 0.0.0.0, which run() resolves to a
    real address before binding. Should one reach here anyway it is a bind
    directive, never something a client can put in a Host header."""
    for wildcard in ("0.0.0.0", "::"):
        response = _ingest_app(trusted_hosts=(wildcard,)).post(
            "/api/v1/telemetry", headers={"host": f"{wildcard}:7001"}, content="{}")
        assert response.status_code == 403, wildcard
