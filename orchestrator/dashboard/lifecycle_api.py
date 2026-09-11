"""Dashboard API routes for lifecycle control (start/stop + log streaming)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from orchestrator.dashboard._http_utils import read_json, sse
from orchestrator.lifecycle import BusyError, LifecycleManager

logger = logging.getLogger("aureax.dashboard.lifecycle")

router = APIRouter()


def _manager(request: Request) -> LifecycleManager:
    mgr = getattr(request.app.state, "lifecycle", None)
    if mgr is None:
        raise HTTPException(503, "Lifecycle manager not initialized")
    return mgr


@router.get("/api/lifecycle/state")
def get_state(request: Request) -> dict:
    return _manager(request).status()


@router.get("/api/lifecycle/nodes")
def list_configured_nodes(request: Request) -> list[dict]:
    """Configured node IDs for the per-node buttons, with role and group.

    Both extra fields are what the UI needs to not lie to the operator: a shard
    worker serves no MCP and is not a chat target, and per-node Stop on any
    group member fans out to the whole group — so a confirmation that names one
    node understates what is about to happen.
    """
    from orchestrator.config.loader import load_network_config
    from orchestrator.lifecycle import node_role

    mgr = _manager(request)
    try:
        cfg = load_network_config(str(mgr.network_config_path))
    except Exception as e:
        # An unparseable or half-written config must not surface as an
        # unhandled 500 with a stack trace — this endpoint is polled, and the
        # dashboard binds to the LAN with no auth. Observed for real while the
        # file was being rewritten underneath a running server.
        logger.warning("Could not read network config: %s", e)
        raise HTTPException(
            status_code=503,
            detail=f"network config unreadable: {type(e).__name__}",
        ) from e
    out = []
    for t in cfg.ssh.enabled_targets:
        group = cfg.group_for_node(t.node_id)
        out.append({
            "node_id": t.node_id,
            "display_name": t.display_name or t.node_id,
            # node_role answers None for a config file it cannot read, which
            # the UI would render as the string "None" beside a node id.
            "role": node_role(t, mgr.network_config_path) or "",
            "group": group.name if group else None,
            "group_members": list(group.members) if group else [],
        })
    return out


def _apply_in_flight(request: Request) -> bool:
    """Mirror image of access_api's lifecycle-busy guard: a start/stop racing
    a config apply would relabel nodes mid-push. Lives in the dashboard layer
    on purpose — LifecycleManager's state machine has its own invariants and
    tests, and knows nothing about applies."""
    runner = getattr(request.app.state, "access_apply", None)
    return bool(runner is not None and runner.running)


async def _busy(mgr: LifecycleManager, detail: str | None = None) -> JSONResponse:
    """The 409 body: the busy marker plus a fresh status snapshot.

    ``status()`` opens a socket to the API server and reads the network config,
    so it is blocking work — on the event loop that also serves every SSE
    stream.
    """
    body = {"error": "busy", **await asyncio.to_thread(mgr.status)}
    if detail:
        body["detail"] = detail
    return JSONResponse(body, status_code=409)


def _node_id(body: dict) -> str | None:
    """The optional ``node`` field of a start/stop body, type-checked.

    ``try_start``/``try_stop`` take ``str | None``; anything else truthy — a
    number, an object — reaches group resolution and fails deep inside the run,
    where the manager records State.ERROR and a last_error for what was only a
    malformed request. Every sibling route type-checks its body the same way.
    """
    node = body.get("node")
    if node is not None and not isinstance(node, str):
        raise HTTPException(400, "'node' must be a string")
    return node


@router.post("/api/lifecycle/start")
async def post_start(request: Request) -> JSONResponse:
    mgr = _manager(request)
    # Body FIRST: the apply-busy check and try_start (which claims `busy`
    # synchronously) have to sit in one uninterrupted stretch. Check before the
    # await and a Start suspended on a slow client's body read lets an Apply see
    # busy=False and dispatch its config push — the interleaving the guard
    # exists to prevent.
    body = await read_json(request)
    if _apply_in_flight(request):
        return await _busy(mgr, "a config apply is running")
    node = _node_id(body)
    try:
        run_id = mgr.try_start(node=node)
    except BusyError:
        return await _busy(mgr)
    return JSONResponse({"run_id": run_id, "state": mgr.state.value}, status_code=202)


@router.post("/api/lifecycle/stop")
async def post_stop(request: Request) -> JSONResponse:
    mgr = _manager(request)
    body = await read_json(request)  # see post_start: read before the guard
    if _apply_in_flight(request):
        return await _busy(mgr, "a config apply is running")
    node = _node_id(body)
    try:
        run_id = mgr.try_stop(node=node, skip_self_kill=True)
    except BusyError:
        return await _busy(mgr)
    return JSONResponse({"run_id": run_id, "state": mgr.state.value}, status_code=202)


# Each open stream holds a 1000-slot queue (broadcaster.MAX_QUEUE_SIZE), and
# SystemLogger.emit writes every line to a store this process does not prune —
# so the streams are capped, and only the transitions that change whether
# anyone is watching at all are recorded. Per-connection lines would let a
# connect/disconnect loop append two durable rows per TCP connection.
_MAX_LOG_STREAMS = 32
_log_streams = 0


@router.get("/api/lifecycle/logs")
async def stream_logs(request: Request, since: int = 0) -> StreamingResponse:
    mgr = _manager(request)
    client = request.client.host if request.client else "?"
    # Admission check only. The counter itself moves inside the generator: a
    # response whose body is never iterated would otherwise leak a slot
    # permanently, where this can at worst admit a couple over the cap.
    if _log_streams >= _MAX_LOG_STREAMS:
        raise HTTPException(503, f"too many log streams open (max {_MAX_LOG_STREAMS})")

    async def gen():
        global _log_streams

        # Subscribe BEFORE replaying the snapshot, not after: a line emitted
        # while the replay is in flight would otherwise land in neither — it is
        # too late for the snapshot and too early for the queue. Dedupe on
        # line_index covers the overlap the other way round.
        q = mgr.logger.subscribe()
        _log_streams += 1
        try:
            last_index = since - 1
            # The replay can be thousands of frames. A client that closes
            # mid-replay (curl --max-time, a tab switch) would otherwise have
            # every remaining frame written into a dead socket before the loop
            # below ever checks — one uvicorn "socket.send() raised exception"
            # line per frame. Checking every so often keeps that bounded
            # without paying the await on each frame.
            for n, line in enumerate(mgr.logger.snapshot(since=since)):
                if n % 200 == 199 and await request.is_disconnected():
                    return
                last_index = line.line_index
                yield sse("log", asdict(line))
            if _log_streams == 1:
                mgr.logger.emit(
                    "info", f"Stream: log subscriber connected ({client})", source="stream"
                )
            while True:
                if await request.is_disconnected():
                    return
                try:
                    line = await asyncio.wait_for(q.get(), timeout=15.0)
                    if line.line_index <= last_index:
                        continue  # already sent in the snapshot above
                    last_index = line.line_index
                    yield sse("log", asdict(line))
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            mgr.logger.unsubscribe(q)
            _log_streams -= 1
            if _log_streams == 0:
                with contextlib.suppress(Exception):
                    mgr.logger.emit(
                        "info", f"Stream: log subscriber left ({client})", source="stream"
                    )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
