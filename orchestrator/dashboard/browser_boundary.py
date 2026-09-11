"""Browser isolation for the credential-free, trusted-host dashboard.

This prevents foreign web pages and DNS rebinding from reaching administration.
It is not caller authentication: local programs and trusted LAN clients retain access.
"""

import socket
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse


def _origin(value: str):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("invalid origin")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("invalid origin")
    return parsed.scheme, parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == "https" else 80)


def _host(value: str) -> str:
    """A Host-header comparison key: lowercased, brackets off an IPv6 literal."""
    return value.strip().strip("[]").lower()


class BrowserBoundaryMiddleware:
    def __init__(self, app, trusted_hosts=()):
        self.app = app
        # ``trusted_hosts`` is the address this dashboard was started on, which
        # is also the address every node was told to ship to: it is stamped
        # into each node's telemetry and log endpoint at generation time, and a
        # hostname is a legal value there (loader.validate_orchestrator_host).
        # A node dialing an mDNS or DNS name sends that name as Host, which is
        # neither loopback nor the socket's own IP literal — so without this a
        # hostname-addressed fleet has every push rejected while a browser on
        # the IP keeps working.
        self.hosts = {"localhost", "127.0.0.1", "::1", socket.gethostname().lower(),
                      *(_host(h) for h in trusted_hosts)}
        # A wildcard bind is not a name anything can be addressed by.
        self.hosts -= {"", "0.0.0.0", "::"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        try:
            if len(headers.getlist("host")) != 1 or len(headers.getlist("origin")) > 1:
                raise ValueError("ambiguous headers")
            current = _origin(f'{scope["scheme"]}://{headers["host"]}')
            # The accepted socket's local address is supplied by ASGI, never
            # by a proxy header or the request's Host value.
            server = scope.get("server")
            addresses = self.hosts | ({server[0].lower()} if server else set())
            addresses -= {"0.0.0.0", "::"}
            if current[1] not in addresses:
                raise ValueError("untrusted host")
            if headers.get("sec-fetch-site") == "cross-site":
                raise ValueError("cross-site request")
            if "origin" in headers and _origin(headers["origin"]) != current:
                raise ValueError("foreign origin")
        except (ValueError, KeyError):
            await PlainTextResponse("Dashboard requests must use its own host and origin", status_code=403)(
                scope, receive, send)
            return

        async def no_store(message):
            if message["type"] == "http.response.start" and scope["path"].startswith("/api/"):
                message["headers"] = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                message["headers"].append((b"cache-control", b"no-store"))
            await send(message)

        await self.app(scope, receive, no_store)
