"""Chat orchestration — session-keyed PlannerEngine + MCPPool instances.

Each chat session (UUID) has its own engine. Engines are rehydrated from
``SessionStore`` on demand. Every message emitted by the agentic loop is
persisted and broadcast over a per-session fanout channel, so multiple
browsers can observe an in-flight session in real time.

Targets are derived from the ENABLED ``ssh.targets`` in the network config,
minus shard workers — a worker serves neither :30800 nor :30900, so chatting
with one would only hang. Each remaining target becomes ``node-<node_id>``; the
planner is just the target whose per-node config has
``node.role == "cognitive_planner"`` (it dispatches to other nodes via its own
``mcp.client.remote_servers``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path

from client.chat.store import (
    Message,
    Session,
    SessionStore,
    messages_to_engine_format,
)
from client.engine import PlannerEngine, history_budget_chars
from client.inference.chat_client import ChatLlamaClient
from client.mcp.pool import MCPPool
from orchestrator.config import tlsstore
from orchestrator.config.cache import load_network_config_cached, load_node_config_resolved
from orchestrator.config.schema import (
    LLAMA_NODE_PORT,
    MCP_NODE_PORT,
    SHARD_WORKER_ROLE,
    SSHTarget,
    node_role,
)
from orchestrator.config.schema import PLANNER_ROLE as COGNITIVE_PLANNER_ROLE

# The tree's single IPv6-bracketing rule for a URL authority: a bare literal's
# own colons read as the port separator, so every URL builder routes its host
# through this one helper rather than growing its own copy.
from orchestrator.config.templating import _url_host
from orchestrator.dashboard.broadcaster import Broadcaster

logger = logging.getLogger("aureax.chat")

GENERIC_NODE_SYSTEM_PROMPT = (
    "You are an AI agent running on an Aureax edge node. You have access to "
    "local tools via MCP (sensors, health, inference). Use tools when useful "
    "and answer the user concisely."
)
AUTO_TITLE_MAX_LEN = 48

# How long a stream sits on an idle queue before writing a keepalive. Chosen to
# match lifecycle_api's log stream, which faces the same intermediaries.
STREAM_KEEPALIVE_SECONDS = 15.0

# Minimum gap between attempts to rebuild a pool that failed to connect. A
# failed dial costs at least MCPPool's 5s connect timeout — and up to its 300s
# read timeout when the server accepts but never answers — while _engine_for
# refreshes the pool on every cache hit, so without this each queued ask
# against a sharded head still streaming tensors to its workers paid that
# again.
POOL_RETRY_SECONDS = 15.0


def _resolved_key(block: dict | None) -> str | None:
    """The ``api_key`` of a resolved config block, or None if it is not usable.

    load_node_config_resolved falls back to the SOURCE TEMPLATE when no
    generated config exists (see its docstring). On a template this value is
    the literal string "${api_key}" / "${mcp_api_key}", and sending that as a
    Bearer token 401s every request — the same shape as the ${mcp_endpoints}
    bug recorded in load_node_config_resolved.

    One helper for both key kinds on purpose: llama and MCP keys share the
    ``aureax_sk_`` prefix and the same template-fallback trap, so a rule
    tightened for one must not be missable for the other.
    """
    key = (block or {}).get("api_key")
    if not isinstance(key, str) or not key or "${" in key:
        return None
    return key


def _resolved_api_key(ncfg: dict) -> str | None:
    """model.server.api_key from a node config, or None."""
    return _resolved_key((ncfg.get("model") or {}).get("server"))


def _resolved_mcp_api_key(ncfg: dict) -> str | None:
    """mcp.server.api_key from a node config, or None.

    Read from the node's RESOLVED config rather than the keystore, exactly as
    the llama key is: what the node is actually serving with is what the
    generated file says, and a keystore read would hand out a key that has been
    rotated but not yet applied.
    """
    return _resolved_key((ncfg.get("mcp") or {}).get("server"))


class ChatService:
    """Owns the session store, engines, pools, and broadcasters."""

    def __init__(
        self,
        network_config_path: Path,
        store: SessionStore,
        system_log=None,
    ) -> None:
        self.network_config_path = network_config_path
        # The templates that describe this fleet sit beside the fleet file, so
        # the directory follows whatever ``aureax --config`` was pointed at —
        # never the repo checkout, and never the process CWD. targets() reads
        # these YAMLs to learn each node's role, and a directory that does not
        # hold them makes every lookup return "", which fails the shard-worker
        # filter OPEN and offers workers as chat targets. Handed to
        # load_node_config_resolved, which prefers `aureax generate`'s resolved
        # copy and falls back to the template here.
        self.configs_dir = Path(network_config_path).parent
        self.store = store
        self.system_log = system_log
        self._engines: dict[str, PlannerEngine] = {}       # session_id -> engine
        self._pools: dict[str, MCPPool] = {}               # target_id -> pool
        # target_id -> {member name: the (URL, bearer key) it was last connected
        # with}. MCPPool keys its sessions by name only, so this is what lets
        # _pool_covers see a member that is connected AT A STALE ADDRESS — or,
        # since MCP gained bearer auth, one holding a key that has since been
        # rotated. The key is half of the tuple because a rotate leaves the URL
        # untouched: without it the pool reads as complete and every tool call
        # in that session 401s until the dashboard restarts.
        self._pool_member_urls: dict[str, dict[str, tuple[str, str | None]]] = {}
        self._llm_clients: dict[str, ChatLlamaClient] = {}  # target_id -> chat client
        self._pool_attempt_at: dict[str, float] = {}       # target_id -> monotonic
        self._broadcasters: dict[str, Broadcaster] = {}    # session_id -> broadcaster
        self._build_lock = asyncio.Lock()
        self._ask_locks: dict[str, asyncio.Lock] = {}
        self._pool_locks: dict[str, asyncio.Lock] = {}   # target_id -> rebuild lock
        # A SET per session, not one task: asks queue on the per-session lock,
        # and a queued run that delete/reset cannot see is one they neither
        # cancel nor outrun.
        self._running: dict[str, set[asyncio.Task]] = {}  # session_id -> in-flight asks

    # ---- Targets ------------------------------------------------------------

    def targets(self) -> list[dict]:
        from orchestrator.config.loader import ConfigError

        try:
            cfg = load_network_config_cached(self.network_config_path)
        except (ConfigError, OSError):
            # Raised, not swallowed into []: an empty list is indistinguishable
            # from a fleet with no enabled targets, and every later request
            # then failed as "unknown target" — blaming the client for a
            # server-side config failure. chat_api turns this into a 503 that
            # names the cause.
            logger.exception("failed to load network config %s", self.network_config_path)
            raise
        entries: list[dict] = []
        for t in cfg.ssh.enabled_targets:
            ncfg = load_node_config_resolved(self.configs_dir, t)
            role = node_role(ncfg)

            # A shard worker runs rpc-server only — no llama-server on 30800
            # and no MCP server on 30900 — so offering it as a chat target
            # would just hang. A group is reachable through its head, which
            # appears here as an ordinary node.
            if role == SHARD_WORKER_ROLE:
                continue

            group = cfg.group_for_node(t.node_id)
            label = t.display_name or t.node_id
            if group is not None:
                label = f"{label} — {group.name} ({len(group.members)} nodes)"

            entries.append({
                "id": f"node-{t.node_id}",
                "label": label,
                "kind": "planner" if role == COGNITIVE_PLANNER_ROLE else "node",
                "node_id": t.node_id,
                "host": t.host,
                "group": group.name if group else None,
            })
        return entries

    def _target_for(self, target_id: str) -> tuple[SSHTarget, dict]:
        """Resolve a chat target_id (``node-<node_id>``) to its SSH target.

        Returns (target, per_node_config_dict). Raises ValueError on unknown id.

        ``_engine_for`` resolves this once per build and hands the pair to
        ``_pool_for``/``_llm_for``/``_system_prompt_for``, so one engine costs
        one read of the network config and one of the node's YAML rather than
        three of each.

        Resolves exactly what ``targets()`` offers, and must keep doing so: a
        session outlives its node being disabled or converted to a shard
        worker, and a resolved worker means dialling :30900 on a node that runs
        only rpc-server — the hang the target filter exists to prevent. The
        ValueError surfaces as a chat error event instead.
        """
        if not target_id.startswith("node-"):
            raise ValueError(f"unknown target: {target_id}")
        node_id = target_id[len("node-"):]
        cfg = load_network_config_cached(self.network_config_path)
        target = next((t for t in cfg.ssh.enabled_targets if t.node_id == node_id), None)
        if target is None:
            raise ValueError(f"unknown or disabled node: {node_id}")
        ncfg = load_node_config_resolved(self.configs_dir, target)
        if node_role(ncfg) == SHARD_WORKER_ROLE:
            raise ValueError(f"{node_id} is a shard worker — it serves no chat endpoint")
        return target, ncfg

    # ---- Sessions -----------------------------------------------------------

    def list_sessions(self, target_id: str | None = None) -> list[dict]:
        return [_session_dict(s) for s in self.store.list_sessions(target_id)]

    async def create_session(self, target_id: str, title: str = "") -> dict:
        if not any(t["id"] == target_id for t in self.targets()):
            raise ValueError(f"unknown target: {target_id}")
        session = self.store.create_session(target_id, title)
        system_prompt = await self._system_prompt_for(target_id)
        if system_prompt:
            self.store.append_message(session.id, "system", system_prompt)
        self._log("info", f"Chat: new session for {target_id} ({session.id[:8]})")
        return _session_dict(session)

    def get_session(self, session_id: str) -> dict | None:
        s = self.store.get_session(session_id)
        return _session_dict(s) if s else None

    async def delete_session(self, session_id: str) -> None:
        """Delete a session, cancelling any run still in flight.

        Deleting the row under a live `ask` makes that run append a message
        referencing a session that is gone: "IntegrityError: FOREIGN KEY
        constraint failed". Drop the ask lock only at the very end of this
        order — every run cancelled and awaited, the row deleted, and only then
        the lock popped. A lock popped while a task still holds it is not
        released: the next ask gets a fresh object that excludes nobody.

        Cancel first rather than simply awaiting the lock: an in-flight run can
        be a multi-minute inference, and Delete should not hang for it.
        """
        await self._cancel_running(session_id)

        lock = self._ask_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            self._engines.pop(session_id, None)
            self._broadcasters.pop(session_id, None)
            self._running.pop(session_id, None)
            self.store.delete_session(session_id)
        self._ask_locks.pop(session_id, None)
        self._log("info", f"Chat: deleted session {session_id[:8]}")

    async def reset_session(self, session_id: str) -> None:
        """Clear a session's history, cancelling any run still in flight.

        The same hazard delete_session documents, with a quieter failure. `ask`
        holds its own reference to the engine and keeps draining new messages
        into the store, so a Reset that only wiped the rows had that turn's
        messages appended straight back after the wipe — and broadcast, so the
        UI repopulated the view it had just cleared, on top of a re-seeded
        system prompt.
        """
        session = self.store.get_session(session_id)
        if not session:
            raise ValueError(f"unknown session: {session_id}")

        await self._cancel_running(session_id)

        lock = self._ask_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            self.store.clear_session(session_id)
            # Re-seed system prompt and discard the engine so it rehydrates fresh.
            prompt = await self._system_prompt_for(session.target_id)
            if prompt:
                self.store.append_message(session_id, "system", prompt)
            self._engines.pop(session_id, None)
            self._bc(session_id).emit({"type": "reset"})

    # ---- Ask (fire-and-stream) ---------------------------------------------

    async def ask(self, session_id: str, prompt: str) -> None:
        """Run the agentic loop for this session. Persists + broadcasts each
        new message. Runs in the event loop's task scheduler — the caller
        spawns this with ``asyncio.create_task``.

        Registered in ``_running`` BEFORE the lock is acquired: chat_api spawns
        a task per POST with no dedupe, so a second ask sits on the lock. One
        that delete_session cannot see is one it neither cancels nor outruns —
        it would take the lock the moment Delete released it and persist its
        turn into a session that no longer exists.
        """
        task = asyncio.current_task()
        running = self._running.setdefault(session_id, set())
        if task is not None:
            running.add(task)
        try:
            await self._ask_locked(session_id, prompt)
        finally:
            self._discard_running(session_id, task)

    async def _ask_locked(self, session_id: str, prompt: str) -> None:
        """One ask, serialised per session. See :meth:`ask`."""
        lock = self._ask_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = None
            try:
                session = self.store.get_session(session_id)
                if session is None:
                    raise ValueError(f"unknown session: {session_id}")

                engine = await self._engine_for(session)
                bc = self._bc(session_id)
                tag = f"[{session.target_id}|{session_id[:8]}]"
                self._log("info", f"Chat {tag}: ask → {prompt[:80].strip()!r}")

                # Auto-title from the first user prompt
                if not session.title.strip():
                    title = prompt.strip().splitlines()[0][:AUTO_TITLE_MAX_LEN] or "Untitled"
                    self.store.update_session_title(session_id, title)
                    bc.emit({"type": "session_updated", "title": title})

                self.store.update_session_status(session_id, "active")
                bc.emit({"type": "status", "status": "active"})

                # Track which messages on the engine have been persisted so we
                # can diff after each yield. Everything on the engine now is
                # already in the store — replayed at build time or written by
                # an earlier turn on this cached engine — except the
                # constructor-seeded system prompt, which is deliberately never
                # persisted (pinned by tests/test_chat_service.py).
                last_persisted = len(engine.messages)

                async for event in engine.arun(prompt):
                    last_persisted = self._drain_engine(session_id, engine, bc, last_persisted)
                    # Event passthrough (UI uses tool_call for pending-state display)
                    bc.emit(event)
                    if event.get("type") == "tool_call":
                        self._log("info", f"Chat {tag}: tool {event.get('node_id','?')}.{event.get('tool','?')}")
                    elif event.get("type") == "tool_result":
                        result = event.get("result")
                        self._log("info", f"Chat {tag}: ← result ({_pretty_short(result)})")
                    elif event.get("type") == "final":
                        self._log("success", f"Chat {tag}: answered")

                # Drain any trailing append (e.g. after the final yield)
                self._drain_engine(session_id, engine, bc, last_persisted)
            except Exception as e:
                logger.exception("chat.ask failed for session %s", session_id)
                tag = f"[{session.target_id}|{session_id[:8]}]" if session else f"[{session_id[:8]}]"
                self._log("error", f"Chat {tag}: failed — {type(e).__name__}: {e}")
                self._bc(session_id).emit({
                    "type": "error",
                    "message": f"{type(e).__name__}: {e}",
                })
                # Discard the cached engine so the next ask rebuilds it from the
                # database.
                #
                # PlannerEngine.arun appends the user message to engine.messages
                # BEFORE its first llm.chat call, and the persist loop runs only
                # once the generator yields. A failure in that first call — an
                # unreachable node, a model still loading — therefore leaves the
                # cached engine holding a message the DB never saw, so it has to
                # go: otherwise every later turn is conditioned on a history
                # that a restart would not replay.
                self._engines.pop(session_id, None)
            finally:
                # The session may have been deleted while this ran; both calls
                # are no-ops against a row that is gone.
                with contextlib.suppress(Exception):
                    self.store.update_session_status(session_id, "idle")
                    self._bc(session_id).emit({"type": "status", "status": "idle"})

    # ---- Stream (observe) ---------------------------------------------------

    async def stream(self, session_id: str) -> AsyncIterator[dict]:
        """Yield a session's history + live events. Replays from DB then
        subscribes to the per-session broadcaster.

        Emits:
          {"type": "session", "session": {...}}            — once, up front
          {"type": "message", "message": {...}}            — per message (replay + live)
          {"type": "tool_call"|"tool_result"|"final"|...}  — live engine events
          {"type": "status",  "status": "active"|"idle"}   — status changes
          {"type": "keepalive"}                            — idle heartbeat
          {"type": "error",   "message": "..."}            — failures
        """
        session = self.store.get_session(session_id)
        if session is None:
            yield {"type": "error", "message": f"unknown session: {session_id}"}
            return

        # 1. Snapshot session + messages. Subscribe BEFORE iterating so we
        #    don't miss any messages appended during replay.
        q = self._bc(session_id).subscribe()
        try:
            yield {"type": "session", "session": _session_dict(session)}
            # Off the loop: list_messages is an unbounded synchronous SELECT
            # over every turn a session ever held, nothing prunes a session,
            # and an EventSource reconnects on its own — so replaying inline
            # would stall node telemetry/log ingest and every other chat
            # stream on this loop for the length of that read.
            for msg in await asyncio.to_thread(self.store.list_messages, session_id):
                yield {"type": "message", "message": _message_dict(msg)}
            yield {"type": "status", "status": session.status}

            # 2. Live events. Already-persisted messages may also arrive again
            #    via the broadcast channel; the client should dedupe by id.
            #
            #    The timeout is what keeps an idle session's connection alive:
            #    a bare `await q.get()` writes nothing to the socket between
            #    events, so an idle chat was dropped by anything in the path
            #    and every EventSource reconnect replayed the whole history.
            #    It also gives the HTTP layer a regular chance to notice the
            #    client is gone and unsubscribe.
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=STREAM_KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield {"type": "keepalive"}
                    continue
                yield event
        finally:
            self._bc(session_id).unsubscribe(q)

    async def _cancel_running(self, session_id: str) -> None:
        """Cancel and await every ask on this session, queued ones included.

        Awaited, not just cancelled: the run has to reach its own `finally`
        (and release the ask lock) before the caller may touch the session's
        rows, or the cancelled task's last writes race the delete.
        """
        for task in list(self._running.get(session_id) or ()):
            if task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def _discard_running(self, session_id: str, task) -> None:
        """Drop one finished ask; the key goes when the last one does, so
        ``sid in self._running`` still means "a run is in flight"."""
        running = self._running.get(session_id)
        if running is None:
            return
        running.discard(task)
        if not running:
            self._running.pop(session_id, None)

    async def close(self) -> None:
        # In-flight asks first. They hold their own references to the engines,
        # pools and llm clients torn down below, so a run left going kept
        # appending to the store against a closed httpx client while the
        # process was shutting down (app.py awaits this from the lifespan).
        for session_id in list(self._running):
            await self._cancel_running(session_id)
        self._running.clear()

        for pool in self._pools.values():
            with contextlib.suppress(Exception):
                await pool.close()
        # ChatLlamaClient wraps an httpx.Client (a connection pool); its close()
        # releases that pool, which nothing else here does.
        for llm in self._llm_clients.values():
            with contextlib.suppress(Exception):
                llm.close()
        self._engines.clear()
        self._pools.clear()
        self._pool_member_urls.clear()
        self._llm_clients.clear()
        self._pool_attempt_at.clear()
        self._pool_locks.clear()
        self._broadcasters.clear()
        self._ask_locks.clear()

    # ---- Internals ----------------------------------------------------------

    def _log(self, level: str, text: str) -> None:
        if self.system_log is not None:
            with contextlib.suppress(Exception):
                self.system_log.emit(level, text, source="chat")

    def _bc(self, session_id: str) -> Broadcaster:
        """Broadcaster for a session, created on demand.

        A cached entry is never created for a session the store does not know:
        this is reached from the SSE subscribe path, so a client still holding a
        deleted session's id would otherwise re-add an entry that delete_session
        had just removed and nothing would ever clean up again. Such a caller
        gets a live, unregistered Broadcaster — it simply receives nothing.
        """
        bc = self._broadcasters.get(session_id)
        if bc is None:
            bc = Broadcaster()
            if self.store.get_session(session_id) is not None:
                self._broadcasters[session_id] = bc
        return bc

    def _persist_engine_msg(self, session_id: str, raw: dict) -> Message:
        role = raw.get("role", "")
        content = raw.get("content")
        # Normalize non-string content (some LLMs return lists) to string
        if content is not None and not isinstance(content, str):
            content = str(content)
        tool_calls = raw.get("tool_calls")
        tool_call_id = raw.get("tool_call_id")
        return self.store.append_message(
            session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )

    async def _system_prompt_for(
        self, target_id: str, resolved: tuple[SSHTarget, dict] | None = None
    ) -> str:
        if resolved is None:
            if not target_id.startswith("node-"):
                return ""
            try:
                resolved = self._target_for(target_id)
            except ValueError:
                return ""
        _, ncfg = resolved
        role = node_role(ncfg)
        if role == COGNITIVE_PLANNER_ROLE:
            return (ncfg.get("agent") or {}).get("system_prompt", "") or ""
        return GENERIC_NODE_SYSTEM_PROMPT

    def _drain_engine(self, session_id: str, engine, bc, last_persisted: int) -> int:
        """Persist and broadcast engine messages not yet written. Returns the new mark.

        One implementation for the in-loop flush and the trailing drain, so the
        two cannot diverge — and so the failure path can reuse it.
        """
        while last_persisted < len(engine.messages):
            raw = engine.messages[last_persisted]
            persisted = self._persist_engine_msg(session_id, raw)
            bc.emit({"type": "message", "message": _message_dict(persisted)})
            last_persisted += 1
        return last_persisted

    async def _engine_for(self, session: Session) -> PlannerEngine:
        resolved = self._target_for(session.target_id)
        engine = self._engines.get(session.id)
        if engine is not None:
            await self._llm_for(session.target_id, resolved)
            await self._pool_for(session.target_id, resolved)
            return engine
        async with self._build_lock:
            if session.id in self._engines:
                return self._engines[session.id]

            # Resolve the network config and the node's YAML once for all three
            # builders below. An unknown target raises ValueError out of
            # _target_for, which is still the first thing to run.
            #
            # Deliberately re-resolved rather than reusing the value read
            # before the lock: acquiring _build_lock awaits, so a config edit
            # made while this call queued is visible now and the engine is
            # built from it. The loads are mtime-cached, so the repeat costs a
            # dict lookup when nothing changed.
            resolved = self._target_for(session.target_id)
            pool = await self._pool_for(session.target_id, resolved)
            llm = await self._llm_for(session.target_id, resolved)
            sys_prompt = await self._system_prompt_for(session.target_id, resolved)
            # Sized from what THIS target serves, not from a fleet-wide
            # constant: targets range from a sharded head at 131072 down to a
            # planner at 16384 and the standalone edge template at 4096, and a
            # budget larger than the context never trims, so the session
            # overflows and every later turn fails identically.
            _, ncfg = resolved
            ctx_size = ((ncfg.get("model") or {}).get("server") or {}).get("context_size")
            engine = PlannerEngine(
                llm, pool, sys_prompt, max_history_chars=history_budget_chars(ctx_size)
            )

            # Rehydrate conversation from DB. If the DB already has a system
            # message we use it (honours user edits); otherwise the engine's
            # constructor-seeded one stays.
            db_msgs = self.store.list_messages(session.id)
            if db_msgs:
                engine.messages = messages_to_engine_format(db_msgs)

            self._engines[session.id] = engine
            self._evict_engines()
            return engine

    # Engines hold a session's whole message list, including every tool result,
    # and nothing but an explicit delete or reset drops one — so the cache has
    # to be bounded or a long-running dashboard keeps one per session ever
    # opened. Eviction is free of consequence: _engine_for rehydrates from the
    # store, so it costs a re-read on next use and loses nothing.
    _MAX_CACHED_ENGINES = 32

    def _evict_engines(self) -> None:
        """Drop least-recently-created per-session state beyond the bound.

        Engines, broadcasters and ask locks are all keyed by session id and all
        grew for the life of the process; one pass bounds the three together.
        Entries still in use are skipped rather than dropped: a broadcaster
        with subscribers is feeding a live SSE stream, and dropping a held ask
        lock would hand the next ask a fresh lock object that excludes nobody —
        the rule delete_session's docstring states.
        """
        # dicts preserve insertion order, so the first key is the oldest.
        while len(self._engines) > self._MAX_CACHED_ENGINES:
            self._engines.pop(next(iter(self._engines)), None)

        for sid in list(self._broadcasters):
            if len(self._broadcasters) <= self._MAX_CACHED_ENGINES:
                break
            # In-flight guard, like the ask locks below: `ask` holds this
            # object for the whole turn, so evicting it sends the rest of that
            # turn's events into a Broadcaster nobody can subscribe to — a
            # browser that reconnects gets a fresh one from _bc and sits on a
            # spinner until it reloads and replays from the DB.
            if not self._broadcasters[sid].has_subscribers and sid not in self._running:
                self._broadcasters.pop(sid, None)

        for sid in list(self._ask_locks):
            if len(self._ask_locks) <= self._MAX_CACHED_ENGINES:
                break
            if not self._ask_locks[sid].locked() and sid not in self._running:
                self._ask_locks.pop(sid, None)

    async def _pool_for(
        self, target_id: str, resolved: tuple[SSHTarget, dict] | None = None
    ) -> MCPPool:
        """Cached MCP pool for a target, reconnected if any member is missing.

        The connectedness check is the point. A first chat against a node whose
        MCP server isn't up yet caches an EMPTY pool, and a sharded head is
        unreachable for minutes while it streams tensors to its workers — so
        without the recheck the session stays toolless until the dashboard
        restarts, long after the group came up. The check is per EXPECTED
        member, not "any node connected": a planner dispatches to several
        remotes, so a single connected one must not stand for the rest, whether
        they never came up or call_tool evicted them on a transport error.

        Reconnection refreshes the SAME MCPPool object rather than replacing it:
        engines built earlier hold a reference to it.

        Retries are rate-limited per target. The node this reconnects to is
        typically one that cannot answer yet, so each attempt costs at least the
        connect timeout — and every queued ask, plus every engine cache hit,
        comes through here.
        """
        target, ncfg = resolved if resolved is not None else self._target_for(target_id)
        role = node_role(ncfg)
        if role == COGNITIVE_PLANNER_ROLE:
            # Planner is an MCP CLIENT — its pool is built from the remote
            # MCP servers it dispatches to (the edges).
            remotes = ((ncfg.get("mcp") or {}).get("client") or {}).get("remote_servers") or []
            from orchestrator.config.templating import mcp_endpoints, template_references_mcp_endpoints
            template = self.configs_dir / target.config_file if target.config_file else None
            if template is not None and template.exists() and template_references_mcp_endpoints(template):
                # Generated YAML may predate disabling/removing a fleet member.
                # Reconcile eligibility and addresses against today's fleet;
                # keep the deployed config's keys until regeneration/apply.
                cfg = load_network_config_cached(self.network_config_path)
                eligible = {entry["name"]: entry for entry in mcp_endpoints(cfg, self.configs_dir, target.node_id)}
                remotes = [{**entry, "url": eligible[entry["name"]]["url"]}
                           for entry in remotes if entry.get("name") in eligible]
            # Each remote carries its OWN key: `${mcp_endpoints}` resolves to
            # one entry per peer with that peer's key, so a planner holds N
            # credentials and no two members share one.
            members = [
                (srv.get("name", "unknown"), srv.get("url", ""), _resolved_key(srv))
                for srv in remotes if srv.get("url")
            ]
        else:
            members = [(
                target.node_id,
                f"https://{_url_host(target.host)}:{MCP_NODE_PORT}/mcp",
                _resolved_mcp_api_key(ncfg),
            )]

        pool = self._pools.get(target_id)
        if pool is not None and self._pool_covers(target_id, pool, members):
            return pool

        lock = self._pool_locks.setdefault(target_id, asyncio.Lock())
        async with lock:
            # Double-checked: the peer we queued behind may have fixed it.
            pool = self._pools.get(target_id)
            if pool is not None and self._pool_covers(target_id, pool, members):
                return pool

            desired = {name: (url, key) for name, url, key in members}
            dialled = self._pool_member_urls.setdefault(target_id, {})
            if pool is not None:
                for name in set(pool.connected_nodes) | set(dialled):
                    if name not in desired or dialled.get(name) != desired[name]:
                        await pool.disconnect(name)
                        dialled.pop(name, None)

            last_attempt = self._pool_attempt_at.get(target_id)
            if (
                pool is not None
                and last_attempt is not None
                and time.monotonic() - last_attempt < POOL_RETRY_SECONDS
            ):
                return pool
            self._pool_attempt_at[target_id] = time.monotonic()

            if pool is None:
                # Every node serves MCP over TLS with a leaf signed by the
                # fleet CA, so the pool verifies against that CA and nothing
                # else. Raises when ~/.aureax/tls/ca.pem is absent, naming
                # `aureax generate` — a chat that cannot verify is a chat that
                # must not connect, not one that falls back to plaintext.
                pool = MCPPool(verify=tlsstore.ca_ssl_context())
            # Re-read rather than trusting the binding taken above: the
            # disconnect pass awaits, and close() drops the whole map on
            # shutdown, so the dict this name pointed at may no longer be the
            # one _pool_covers will consult.
            dialled = self._pool_member_urls.setdefault(target_id, {})
            # A member counts as missing when it is not connected OR is
            # connected at a different URL or with a different bearer key than
            # the config now names — connect_url evicts the stale session
            # before storing the new one.
            missing = [
                (n, u, k) for n, u, k in members
                if n not in pool.connected_nodes or dialled.get(n) != (u, k)
            ]
            # Concurrent, mirroring client/entrypoint.py's connect_pass: each
            # dial of a cold node runs to the full connect timeout, so a serial
            # pass over a planner's remotes multiplied that by their number
            # while the ask waited.
            results = await asyncio.gather(
                *(pool.connect_url(name, url, api_key=key) for name, url, key in missing),
                return_exceptions=True,
            )
            # Every line below carries the URL and never the key: these land in
            # the dashboard's system log, which is readable by anyone who can
            # open the unauthenticated :7001 bind.
            for (name, url, key), ok in zip(missing, results, strict=True):
                if ok is True:
                    dialled[name] = (url, key)
                    logger.info("%s pool: %s -> OK", target_id, url)
                else:
                    dialled.pop(name, None)
                    logger.info("%s pool: %s -> FAILED (%r)", target_id, url, ok)
                    # The Python log is not where an operator looks: this is
                    # why a chat has no tools. connect_url returns only a bool,
                    # so the reason stays in client/mcp/pool.py's own log.
                    self._log("warn", f"Chat: MCP connect FAILED for {name} ({url}) — "
                                      "no tools from it this session (see the pool log)")

            # Stamped again at the END of the attempt: a dial that takes longer
            # than POOL_RETRY_SECONDS would otherwise return with its own
            # window already expired, and the very next ask would retry
            # immediately — the cost this backoff exists to prevent.
            self._pool_attempt_at[target_id] = time.monotonic()
            self._pools[target_id] = pool
            return pool

    def _pool_covers(
        self, target_id: str, pool: MCPPool, members: list[tuple[str, str, str | None]]
    ) -> bool:
        """True when every member is connected AT THE EXPECTED URL AND KEY.

        The URL half is what makes an address change visible. Member names are
        node ids (or a planner's ``remote_servers`` names) and do not move when
        ``ssh.targets[].host`` is edited, so a pool judged on names alone reads
        as complete after a remap and is never redialled — every tool call in
        that session keeps going to the old host until the dashboard restarts.
        The key half is the same failure one rotation later: `aureax apply
        --rotate-key` leaves the URL identical, so a pool judged on the address
        alone keeps a session authenticated with a dead key and every tool call
        401s. MCPPool keys its sessions by name only, so both come from this
        service's own record of what each member was last dialled with.
        """
        connected = set(pool.connected_nodes)
        dialled = self._pool_member_urls.get(target_id, {})
        return connected == {name for name, _, _ in members} and all(
            name in connected and dialled.get(name) == (url, key)
            for name, url, key in members
        )

    async def _llm_for(
        self, target_id: str, resolved: tuple[SSHTarget, dict] | None = None
    ) -> ChatLlamaClient:
        """Cached chat client for a target.

        Cached per target, like the MCP pools, because it owns an httpx
        connection pool that nothing else closes: one client per engine build
        would leave every evicted or deleted session's sockets to the garbage
        collector. ChatService.close() closes these.

        The cache-hit branch re-checks BOTH the API key and the address on
        every call, because the cache key — the target id — is derived from
        node_id and survives a change to either. A rotate + apply changes the
        key under a cached client; editing ``ssh.targets[].host`` moves the
        node, and the mtime-keyed config cache makes both visible here
        immediately. Without the address check every chat on that target keeps
        dialling the old host for the life of the process.

        Both are updated IN PLACE (set_api_key / set_base_url) for the same
        reason _pool_for refreshes an MCPPool in place — cached PlannerEngines
        hold their own reference to this client object.
        """
        target, ncfg = resolved if resolved is not None else self._target_for(target_id)
        api_key = _resolved_api_key(ncfg)
        base_url = f"https://{_url_host(target.host)}:{LLAMA_NODE_PORT}"
        client = self._llm_clients.get(target_id)
        if client is not None:
            if client.api_key != api_key:
                client.set_api_key(api_key)
                logger.info("chat: refreshed llama API key for %s", target_id)
            if client.base_url != base_url:
                client.set_base_url(base_url)
                logger.info("chat: repointed %s at %s", target_id, base_url)
            return client
        # Same CA as the MCP pool, and for the same reason: llama-server holds
        # the node's fleet-signed leaf, so this is the only trust anchor that
        # can verify it. Raises when the store is empty rather than falling
        # back — see _pool_for.
        client = ChatLlamaClient(base_url, api_key=api_key, verify=tlsstore.ca_ssl_context())
        self._llm_clients[target_id] = client
        return client


def _session_dict(session: Session) -> dict:
    return asdict(session)


def _message_dict(msg: Message) -> dict:
    return asdict(msg)


def _pretty_short(value, limit: int = 80) -> str:
    try:
        import json as _json
        s = value if isinstance(value, str) else _json.dumps(value)
    except Exception:
        s = str(value)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 1] + "…"
