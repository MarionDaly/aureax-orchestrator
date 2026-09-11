"""Dashboard API routes for session-based agentic chat.

Flow:
1. Client calls ``GET /api/chat/targets`` to pick a target.
2. Client calls ``GET /api/chat/sessions?target_id=X`` to list sessions,
   or ``POST /api/chat/sessions`` to create a new one for that target.
3. Client calls ``GET /api/chat/sessions/{id}/stream`` — SSE that replays
   history and then streams live events.
4. To ask a question, client calls ``POST /api/chat/sessions/{id}/ask``.
   The handler returns 202 immediately and fires the agentic loop as a
   background task; events land on the stream opened in step 3.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from orchestrator.config.loader import ConfigError
from orchestrator.dashboard._http_utils import emit, read_json, sse
from orchestrator.dashboard.chat_service import AUTO_TITLE_MAX_LEN, ChatService

router = APIRouter()

# Bounds on what a single request may carry. These values are persisted to
# sqlite and replayed into every later turn of the session, so an accidental
# paste of a whole log file is worth refusing before it becomes permanent
# context.
MAX_ASK_BYTES = 1024 * 1024
MAX_PROMPT_CHARS = 200_000

# Strong refs to in-flight ask() tasks so the event loop doesn't GC them
# mid-flight (Python's asyncio holds only weak references).
#
# Bounded for the same reason lifecycle_api bounds its concurrent log streams:
# this bind is reachable by anything on the LAN, asks on one session serialise
# on ChatService's per-session lock, and every queued task holds its prompt
# (up to MAX_PROMPT_CHARS) until it finally drives a full inference against the
# node. A loop of POSTs is refused here rather than left to accumulate.
_MAX_INFLIGHT_ASKS = 32


def _inflight(app) -> set[asyncio.Task]:
    """Asks in flight on THIS app.

    Kept on ``app.state`` rather than at module level so the cap is per
    dashboard process: a suite that builds several apps must not count one
    app's parked asks against another's, and in production a module-level set
    would be one fleet-wide limit rather than one per server.
    """
    tasks = getattr(app.state, "chat_tasks", None)
    if tasks is None:
        tasks = set()
        app.state.chat_tasks = tasks
    return tasks


def _service(request: Request) -> ChatService:
    svc = getattr(request.app.state, "chat", None)
    if svc is None:
        raise HTTPException(503, "Chat service not initialized")
    return svc


@router.get("/api/chat/targets")
def list_targets(request: Request) -> list[dict]:
    try:
        return _service(request).targets()
    except (ConfigError, OSError) as e:
        # Same degradation as access_api.access_nodes: an empty list would
        # read as "this fleet has no chat targets", which is a different
        # problem from "the network config cannot be read right now".
        raise HTTPException(503, f"config unreadable: {type(e).__name__}") from e


# ---- Sessions ---------------------------------------------------------------


@router.get("/api/chat/sessions")
def list_sessions(request: Request, target_id: str | None = None) -> list[dict]:
    return _service(request).list_sessions(target_id)


@router.post("/api/chat/sessions")
async def create_session(request: Request) -> JSONResponse:
    body = await read_json(request)
    target_id = (body.get("target_id") or "").strip()
    if not target_id:
        raise HTTPException(400, "'target_id' is required")
    # Truncated rather than rejected, at the same bound auto-titling uses: a
    # title is a label in a list, and nothing downstream caps a supplied one.
    title = (body.get("title") or "").strip()[:AUTO_TITLE_MAX_LEN]
    try:
        session = await _service(request).create_session(target_id, title)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except (ConfigError, OSError) as e:
        raise HTTPException(503, f"config unreadable: {type(e).__name__}") from e
    return JSONResponse(session, status_code=201)


@router.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> JSONResponse:
    """Delete a session. Deleting an unknown id is a no-op by design — unlike
    reset, which 404s: the browser fires this from a card it may already have
    dropped, and a delete that has nothing left to do has succeeded."""
    await _service(request).delete_session(session_id)
    return JSONResponse({"ok": True})


@router.post("/api/chat/sessions/{session_id}/reset")
async def reset_session(session_id: str, request: Request) -> JSONResponse:
    try:
        await _service(request).reset_session(session_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return JSONResponse({"ok": True})


# ---- Ask + stream -----------------------------------------------------------


@router.post("/api/chat/sessions/{session_id}/ask")
async def ask(session_id: str, request: Request) -> JSONResponse:
    """Start an ask on this session. Returns 202 immediately; events stream
    via the session's /stream endpoint.
    """
    svc = _service(request)
    # Bounded on the READ, not on Content-Length: that header is client-supplied
    # and a chunked body carries none at all, so a header check alone leaves the
    # body buffered into RAM before any of the limits below can look at it.
    body = await read_json(request, MAX_ASK_BYTES)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "'prompt' is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"'prompt' exceeds {MAX_PROMPT_CHARS} characters")
    if svc.get_session(session_id) is None:
        raise HTTPException(404, f"not found: {session_id}")
    tasks = _inflight(request.app)
    if len(tasks) >= _MAX_INFLIGHT_ASKS:
        raise HTTPException(
            429,
            f"too many asks in flight (limit {_MAX_INFLIGHT_ASKS}) — "
            "wait for one to finish",
        )
    task = asyncio.create_task(svc.ask(session_id, prompt))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return JSONResponse({"ok": True}, status_code=202)


@router.get("/api/chat/sessions/{session_id}/stream")
async def stream(session_id: str, request: Request) -> StreamingResponse:
    svc = _service(request)
    if svc.get_session(session_id) is None:
        raise HTTPException(404, f"not found: {session_id}")
    client = request.client.host if request.client else "?"
    emit(request, "info", f"Stream: chat subscriber joined {session_id[:8]} ({client})",
         source="stream")

    async def gen():
        try:
            async for event in svc.stream(session_id):
                if await request.is_disconnected():
                    return
                # An SSE comment: it keeps the connection (and the
                # disconnect check above) alive on an idle session without
                # reaching the client's event handlers.
                if event.get("type") == "keepalive":
                    yield ": keepalive\n\n"
                    continue
                yield sse(event.get("type", "message"), event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            emit(request, "info",
                 f"Stream: chat subscriber left {session_id[:8]} ({client})", source="stream")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


