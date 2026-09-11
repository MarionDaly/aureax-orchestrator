"""Shared HTTP helpers for dashboard routers — JSON parsing, SSE framing,
plus the two things every router needs from ``app.state``: the network config
path and the system log, and the ``{node_id}`` path-parameter gate."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from fastapi import HTTPException, Request

from orchestrator.config.cache import load_network_config_cached

# Cap on a request body read through read_capped_text. ~10x the largest
# legitimate batch (200 log entries x 4000 chars is ~800 KB).
MAX_BODY_BYTES = 8 * 1024 * 1024

# Default cap on a request body read through read_json. The control routes
# carry a short JSON object — an edited config, a node id, a chat prompt — so a
# ceiling far under the ingest one covers every legitimate caller. A route with
# its own published limit passes that as ``max_bytes`` instead of relying on
# this number happening to match (chat_api.MAX_ASK_BYTES does exactly that).
MAX_JSON_BYTES = 1024 * 1024


async def read_capped_text(request: Request, max_bytes: int = MAX_BODY_BYTES) -> str:
    """Read a request body as text, bounded and decoded leniently.

    ``request.body()`` buffers the whole request into RAM with no Content-Length
    check — on the ingest routes, which are reachable from anywhere on the LAN,
    one multi-GB POST OOM-kills the dashboard process and takes
    the lifecycle manager, the chat MCP pools and every SSE stream with it.

    ``errors="replace"``, matching the client's own read side: a strict decode
    of invalid UTF-8 raises before any of the tolerant per-line parsing runs,
    and the resulting 500 makes the shipper retry the same permanently doomed
    bytes forever.
    """
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > max_bytes:
            raise HTTPException(413, f"request body exceeds {max_bytes} bytes")
    return buf.decode("utf-8", errors="replace")


async def read_json(request: Request, max_bytes: int = MAX_JSON_BYTES) -> dict:
    """Parse the request body as JSON. Returns ``{}`` for empty bodies.

    Consumes ``request.stream()``, so call it once per request: a second body
    read of the same request raises Starlette's "Stream consumed".

    Streamed under ``max_bytes`` for the same reason read_capped_text is: this
    is the body reader for every state-changing route on a server that binds
    0.0.0.0, and an unbounded read is a one-request OOM that takes the
    lifecycle manager, the chat MCP pools and every SSE stream with it. The cap
    holds on the READ, not on Content-Length, which is client-supplied and
    absent altogether from a chunked body.

    Raises ``HTTPException(413)`` past the cap, ``HTTPException(400)`` for a
    body that is not a JSON object.
    """
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > max_bytes:
            raise HTTPException(413, f"request body exceeds {max_bytes} bytes")
    if not buf:
        return {}
    try:
        body = json.loads(buf.decode("utf-8"))
    # UnicodeDecodeError and json.JSONDecodeError are unrelated ValueError
    # subclasses, and a body of raw bytes raises the first: both mean the
    # client sent something no route here can read, so both are a 400.
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise HTTPException(400, "invalid JSON body") from e
    # Valid JSON is not necessarily a JSON *object*. Every caller goes straight
    # to body.get(...), so `[]`, `"x"` or `null` would raise AttributeError
    # inside the handler — a client's malformed request reported as a server
    # fault.
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body must be an object")
    return body


def sse(event: str, data: dict) -> str:
    """Format a Server-Sent Events frame.

    ``default=str`` because callers pass payloads of arbitrary shape — chat
    forwards planner engine events and tool results straight through — and a
    non-JSON-native value raising here would break an SSE response already on
    the wire, mid-frame.
    """
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def emit(request: Request, level: str, text: str, *, source: str) -> None:
    """Write one line to the dashboard's system log, if there is one.

    Every router logs the same way and for the same reason — the operator's
    only record of what the dashboard did on their behalf — so the getattr, the
    None check and the suppression live here once. ``source`` is per router
    ("config", "access", "chat", "lifecycle"); it is the column the log console
    filters on.

    Suppressed rather than raised: this is a side effect of a route that has
    already done its work, and a SQLite hiccup must not turn a successful save
    into a 500. Callers that must not lose the record — the routes that reveal
    a plaintext key, ``access_api._reveal`` — also log through ``logging``.
    Naming the helper rather than counting the routes keeps the two files from
    drifting apart as kinds of key are added.
    """
    log = getattr(request.app.state, "system_log", None)
    if log is not None:
        with contextlib.suppress(Exception):
            log.emit(level, text, source=source)


def network_config_path(request: Request, *, fallback: Path | None = None) -> Path:
    """The network config this dashboard was started against, resolved.

    The lifecycle manager owns the path, so routers do not each decide where
    the fleet is described. ``fallback`` is for the router that must keep
    working without a manager (config_api lists and validates files whether or
    not lifecycle came up); it is passed in rather than defaulted here so this
    module never has to import a router's CONFIGS_DIR back.

    Without a fallback, a missing manager is a 503 — the honest answer for the
    routes that go on to touch the fleet.
    """
    mgr = getattr(request.app.state, "lifecycle", None)
    if mgr is not None:
        return Path(mgr.network_config_path).resolve()
    if fallback is not None:
        return fallback.resolve()
    raise HTTPException(503, "Lifecycle manager not initialized")


def target_for(request: Request, node_id: str):
    """Resolve a ``{node_id}`` path parameter to ``(network config, target)``.

    The gate every ``/api/.../{node_id}`` route opens with, shared by the
    access and node-detail routers so a node id that is unknown, disabled or
    unreadable answers identically on both: 404 for a node this fleet does not
    deploy to, 503 when the fleet file itself cannot be read. It also bounds
    the key space of anything the caller keys on ``node_id`` downstream — a
    cache, a keystore path — to ids that are actually in the config.

    ``enabled_targets``, not ``targets``: a disabled node is documented but not
    deployed to, so every per-node view of it would be about a node that has no
    pods, no keys and no certificate.
    """
    path = network_config_path(request)
    try:
        cfg = load_network_config_cached(path)
    except Exception as e:
        raise HTTPException(503, f"network config unreadable: {type(e).__name__}") from e
    for t in cfg.ssh.enabled_targets:
        if t.node_id == node_id:
            return cfg, t
    raise HTTPException(404, f"unknown or disabled node: {node_id}")
