"""Lifecycle orchestration for the Aureax stack.

Extracted from cli.py so the dashboard can invoke the same operations
and stream their output. The CLI uses this module directly via thin
adapters; the dashboard wraps it in LifecycleManager (state machine + a
``busy`` flag checked synchronously on the event loop, which rejects a second
operation rather than queueing it) and streams log lines via SSE.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from orchestrator.config.loader import DEFAULT_NETWORK_CONFIG
from orchestrator.config.schema import LLAMA_NODE_PORT, PLANNER_ROLE, SHARD_WORKER_ROLE
from orchestrator.config.schema import node_role as role_from_config

logger = logging.getLogger("aureax.lifecycle")

SETUP_ALL_SCRIPT = Path("scripts/setup-all.sh")
# DEFAULT_NETWORK_CONFIG (re-exported from orchestrator.config.loader) is the
# fallback for callers that pass no path. Relative to the cwd, like
# SETUP_ALL_SCRIPT: every entry point runs from the project root.


class State(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class LogLine:
    line_index: int
    level: str    # "info" | "warn" | "error" | "success" | "hint"
    text: str
    timestamp: str
    # lifecycle | access | chat | telemetry | pods | nodes | drift | config |
    # stream, or "node:<id>" for a line shipped up from an edge node.
    source: str = "lifecycle"


class BusyError(RuntimeError):
    """Raised when a lifecycle operation is already in flight."""


class CancelledOp(Exception):
    """Raised inside a worker thread when the user cancels via Stop.

    Caught by ``LifecycleManager._wrap``; transitions state back to IDLE
    rather than ERROR. The lifecycle log records what was cancelled.
    """


class SystemLogger:
    """Thread-safe ring buffer of log lines with asyncio fan-out.

    - ``emit`` may be called from any thread; subscribers are per-loop
      asyncio queues populated via ``call_soon_threadsafe``.
    - When constructed with a ``LogStore``, every emit is also persisted.
      On startup, the most-recent rows are loaded into the buffer so SSE
      replay survives dashboard restarts.

    Fan-out is delegated to ``Broadcaster``; this class owns the ring
    buffer + persistence + LogLine indexing.
    """

    def __init__(self, maxlen: int = 2000, store=None, seed_recent: int = 500) -> None:
        from orchestrator.dashboard.broadcaster import Broadcaster

        self._buffer: deque[LogLine] = deque(maxlen=maxlen)
        self._broadcaster = Broadcaster()
        self._lock = threading.Lock()
        self._next_index = 0
        self._store = store
        # One-shot latch for the store-write warning; see _note_store_failure.
        self._store_failed = False

        if store is not None and seed_recent > 0:
            recent = list(reversed(store.recent(seed_recent)))
            with self._lock:
                for entry in recent:
                    self._buffer.append(LogLine(
                        line_index=entry.id,
                        level=entry.level,
                        text=entry.text,
                        timestamp=entry.ts,
                        source=entry.source,
                    ))
                    self._next_index = max(self._next_index, entry.id + 1)

    def _claim_index(self, stored_id: int | None) -> int:
        """line_index for a line, preferring the store's row id.

        The row id is what makes the index STABLE ACROSS RESTARTS, and SSE
        resumption depends on that. A per-process counter restarts at 0 while
        clients still hold a high-water mark from the previous process, so both
        the server's own `line_index <= last_index` skip and the browser's
        dedupe discard every line until the counter climbs back past it — the
        log console silently goes dead after a dashboard restart.

        Falls back to the counter when there is no store, or when the write
        failed; the counter is kept above every id seen so the two never
        collide.
        """
        if stored_id is not None:
            self._next_index = max(self._next_index, stored_id + 1)
            return stored_id
        idx = self._next_index
        self._next_index += 1
        return idx

    def _note_store_ok(self) -> None:
        """Clear the failure latch after a store write succeeds again."""
        if self._store_failed:
            self._store_failed = False
            logger.warning("log store writes recovered")

    def _note_store_failure(self, e: Exception) -> None:
        """Name the first store-write failure of an outage, then stay quiet.

        Both emit paths fall back to the per-process counter for ``line_index``,
        which ``_claim_index`` explains breaks SSE resumption across a restart —
        so a silent failure here is invisible until the log console goes dead.
        Warned once per outage, not once per line: emitting is high-rate, and
        stdlib logging cannot recurse back into SystemLogger.
        """
        if not self._store_failed:
            self._store_failed = True
            logger.warning("log store write failed (%s): %s", type(e).__name__, e)

    def emit(self, level: str, text: str, source: str = "lifecycle") -> None:
        # The store write, the index claim, the buffer append and the broadcast
        # all happen under ONE lock hold. ``line_index`` is the store's row id,
        # and both consumers drop any line whose index is not above the highest
        # they have seen (the SSE skip in dashboard/lifecycle_api.py and the
        # browser's own dedupe). Claiming an id outside the lock lets two
        # emitters — a lifecycle worker thread and a pod watcher on the event
        # loop — reach the buffer and the fan-out in the reverse of their id
        # order, and the lower line is then gone from the live view for good.
        # The window is real: a store append is an INSERT plus a commit.
        #
        # Persisting first also means a crash during fan-out still leaves the
        # row on disk.
        with self._lock:
            stored_id: int | None = None
            if self._store is not None:
                try:
                    stored_id = self._store.append(level, text, source=source).id
                    self._note_store_ok()
                except Exception as e:
                    self._note_store_failure(e)
            line = LogLine(
                line_index=self._claim_index(stored_id),
                level=level,
                text=text,
                timestamp=datetime.now(UTC).isoformat(),
                source=source,
            )
            self._buffer.append(line)
            self._broadcaster.emit(line)

    def emit_many(self, records: list[tuple]) -> int:
        """Emit a batch of ``(level, text, source[, ts])``. Returns rows DURABLY stored.

        The optional fourth element is the entry's own timestamp, passed
        straight through to the store — shipped edge entries carry the time of
        the original event, not of their arrival here.

        Same visible behaviour as calling :meth:`emit` per record — one buffer
        entry and one broadcast each — but a single commit instead of one per
        row. Used by the log-ingest endpoint, which hands whole shipped batches
        to a worker thread; hence one commit rather than one per row.

        The return value is load-bearing for that caller: the edge shipper
        advances its cursor on a 200, so "we broadcast it but the write failed"
        must not read as success or those entries are lost for good. Lines are
        still buffered and broadcast either way — a live view of logs we could
        not persist is better than nothing.
        """
        if not records:
            return 0
        ts = datetime.now(UTC).isoformat()
        # One lock hold over the store write, the index claims, the buffer and
        # the fan-out, for the ordering reason spelled out in :meth:`emit`: a
        # batch that claims ids under the lock but writes them outside it can
        # still be overtaken by a concurrent emit, and every line of the batch
        # then falls below the consumers' high-water mark.
        with self._lock:
            stored: list = []
            if self._store is not None:
                try:
                    stored = self._store.append_many(records)
                    self._note_store_ok()
                except Exception as e:
                    # Whatever the caller reports, only the count reaches it
                    # (orchestrator/logs/receiver.py::ingest), so without
                    # this the cause of a disk-full or locked-SQLite outage is
                    # recorded nowhere.
                    self._note_store_failure(e)
            for i, record in enumerate(records):
                level, text, source = record[:3]
                entry = stored[i] if i < len(stored) else None
                shipped_ts = record[3] if len(record) > 3 and record[3] else None
                line = LogLine(
                    line_index=self._claim_index(entry.id if entry else None),
                    level=level,
                    text=text,
                    timestamp=entry.ts if entry else (shipped_ts or ts),
                    source=source,
                )
                self._buffer.append(line)
                self._broadcaster.emit(line)
        # No store configured means there is nothing to be durable against, so
        # the caller's own persistence path is the authority.
        return len(records) if self._store is None else len(stored)

    def subscribe(self) -> asyncio.Queue:
        return self._broadcaster.subscribe()

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._broadcaster.unsubscribe(q)

    def snapshot(self, since: int = 0) -> list[LogLine]:
        with self._lock:
            return [line for line in self._buffer if line.line_index >= since]


class LifecycleManager:
    """State machine for a single in-flight lifecycle op.

    Stored on ``app.state.lifecycle`` (one instance per dashboard process).
    ``try_start`` / ``try_stop`` raise ``BusyError`` synchronously if an op
    is already in flight, otherwise kick off a background task and return
    immediately with a run_id.
    """

    def __init__(
        self,
        network_config_path: Path,
        logger: SystemLogger | None = None,
    ) -> None:
        self.network_config_path = network_config_path
        self.state: State = State.IDLE
        self.run_id: str | None = None
        self.started_at: str | None = None
        self.last_error: str | None = None
        self.busy: bool = False
        self.logger = logger if logger is not None else SystemLogger()
        # Cancel signal — set by .cancel() (Stop button while busy). Worker
        # threads check this via _sleep_or_cancel and raise CancelledOp.
        self.cancel_event: threading.Event = threading.Event()
        # In-flight subprocess (e.g. setup-all.sh) so .cancel() can kill it.
        # Spawned with start_new_session=True so we can SIGTERM the whole
        # process group, not just the bash parent.
        self.subprocess: subprocess.Popen | None = None
        # Strong ref to the asyncio task running the current operation, so
        # the event loop doesn't GC it mid-flight.
        self._task: asyncio.Task | None = None

    def status(self) -> dict:
        return {
            "state": self.state.value,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "last_error": self.last_error,
            "nodes": _nodes_status(self.network_config_path),
        }

    def try_start(self, node: str | None = None) -> str:
        """Kick off a start. Returns the new run_id. Raises BusyError if busy."""
        if self.busy:
            raise BusyError(self.state.value)
        return self._begin(State.STARTING, self._run_start, node)

    def try_stop(self, node: str | None = None, skip_self_kill: bool = True) -> str:
        """Kick off a stop. Returns the new run_id — except on the cancel path
        below, which returns the run_id of the start it aborted. The caller
        tells the two apart by the state, still STARTING after a cancel.

        If the manager is currently STARTING, this *cancels* the in-flight
        start instead of raising BusyError or queueing a stop — Stop while
        Starting means "abort whatever is happening." After cancellation
        completes (state → IDLE), the user can click Stop again to actually
        suspend the cluster.
        """
        if self.state == State.STARTING and self.busy:
            self.cancel()
            return self.run_id or ""
        if self.busy:
            raise BusyError(self.state.value)
        return self._begin(State.STOPPING, self._run_stop, node, skip_self_kill)

    async def wait(self, timeout: float | None = None) -> None:
        """Await the in-flight operation, at most ``timeout`` seconds.

        The public way to join the current run, so the dashboard's shutdown
        drain (``orchestrator/dashboard/app.py::_drain_lifecycle``) never has to
        reach in for ``_task`` — a private attribute is not another module's
        contract.

        Shielded, so a timeout here does not cancel the worker: ``_run_stop``
        cancelled mid-flight leaves the DaemonSet selectors patched and the
        control plane up. Returns immediately when nothing has ever run.
        Propagates whatever the operation raised — ``_wrap`` already records it
        in ``last_error`` and emits it, so callers that only want to drain
        suppress it.
        """
        task = self._task
        if task is None:
            return
        if timeout is None:
            await asyncio.shield(task)
            return
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    def cancel(self) -> None:
        """Signal the in-flight worker thread to abort.

        Sets ``cancel_event`` (worker wait-loops poll it via _sleep_or_cancel)
        and SIGTERMs any tracked subprocess group. The worker transitions to
        IDLE asynchronously inside ``_wrap``.

        NOTE: this is called from the event loop, or from the worker thread
        itself right after it spawns a subprocess, and can block the caller for
        up to 5s waiting for that subprocess to die on SIGTERM. That is a
        deliberate trade for now — escalating to SIGKILL needs the wait, and
        moving it off the loop means the Stop button can return before the
        process group is actually gone. It is bounded and only happens on an
        explicit Stop.
        """
        self.cancel_event.set()
        proc = self.subprocess
        if proc is not None and proc.poll() is None:
            self.logger.emit("info", "  Cancellation requested — terminating subprocess")
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.emit("warn", "  Subprocess didn't exit on SIGTERM — sending SIGKILL")
                    os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError) as e:
                # Process already gone, or we don't own it — nothing to do.
                self.logger.emit("info", f"  Subprocess already gone ({type(e).__name__})")

    def _begin(self, op_state: State, coro_fn, *args) -> str:
        self.busy = True
        self.state = op_state
        self.run_id = str(uuid.uuid4())
        self.started_at = datetime.now(UTC).isoformat()
        self.last_error = None
        # Reset the cancel signal for the new op.
        self.cancel_event.clear()
        self.subprocess = None
        self._task = asyncio.create_task(self._wrap(coro_fn(*args)))
        return self.run_id

    async def _wrap(self, coro) -> None:
        try:
            await coro
            self.state = State.IDLE
        except CancelledOp as e:
            # User pressed Stop. State returns to IDLE so they can interact
            # with the dashboard normally — they can click Stop again to
            # actually suspend the cluster.
            self.last_error = f"cancelled: {e}" if str(e) else "cancelled by user"
            self.state = State.IDLE
            self.logger.emit("warn", f"Cancelled: {e}" if str(e) else "Cancelled by user")
        except Exception as e:
            self.last_error = str(e)
            self.state = State.ERROR
            self.logger.emit("error", f"operation failed: {e}")
        finally:
            self.busy = False
            self.subprocess = None

    async def _run_start(self, node: str | None) -> None:
        from orchestrator.config.loader import load_network_config

        cfg = load_network_config(str(self.network_config_path))
        if node:
            await asyncio.to_thread(
                do_start_node, cfg, node, self.logger, self.network_config_path
            )
        else:
            await asyncio.to_thread(do_start, cfg, self.logger, mgr=self)

    async def _run_stop(self, node: str | None, skip_self_kill: bool) -> None:
        from orchestrator.config.loader import load_network_config

        cfg = load_network_config(str(self.network_config_path))
        if node:
            await asyncio.to_thread(
                do_stop_node, cfg, node, self.logger, self.network_config_path
            )
        else:
            await asyncio.to_thread(
                do_stop, cfg, self.logger, skip_self_kill=skip_self_kill, mgr=self
            )


def listeners_on_port(port: int) -> tuple[list[str], str | None]:
    """(pids LISTENING on a TCP port, why the list is empty when it is).

    ``-sTCP:LISTEN`` is load-bearing: it matches only the listener. Without it
    ``lsof -i :N`` matches any socket whose local *or remote* port is N, so
    stopping the dashboard would also kill every client connected to it — the
    operator's own browser included.

    Pids come back as lsof printed them rather than as ints: the one caller
    that signals them reports a non-numeric line as its own warning, and the
    other only quotes them in an error message. The second caller is the
    dashboard's bind preflight (``orchestrator/dashboard/app.py``), which turns
    "[Errno 48] address already in use" into a sentence naming the pid to
    stop — hence a *lookup* helper rather than more of the kill path.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"TCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
            # lsof stats every open file and blocks indefinitely on a wedged
            # mount, and this is the last step of `aureax stop`. TimeoutExpired
            # is a SubprocessError, so it degrades to the error return below.
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        return [], f"{type(e).__name__}: {e}"
    # De-duplicated, order kept. `lsof -t` already prints each pid once — even
    # for the dashboard, which holds THREE listening sockets on this port (the
    # LAN HTTPS one plus the two loopback ones) — but a repeat would mean
    # signalling a dying process again and naming one dashboard three times in
    # the bind error below, and that is not worth depending on lsof for.
    return list(dict.fromkeys(result.stdout.strip().split())), None


def _kill_process_on_port(port: int, log: SystemLogger, label: str = "Process") -> None:
    """Kill whatever is LISTENING on a TCP port."""
    pids, lookup_error = listeners_on_port(port)
    if lookup_error is not None:
        log.emit("warn", f"  Could not check port {port} ({lookup_error})")
        return
    if not pids:
        log.emit("info", f"  {label} not running (port {port})")
        return

    # Signalling can fail two ways, and the exception type names which:
    # PermissionError (EPERM, another user's listener) and ProcessLookupError
    # (ESRCH, the pid exited between lsof and here). Reporting "killed" past
    # either leaves the dashboard serving on the port a stop just claimed to
    # free. Caught as OSError so no other errno can escape the last step of a
    # stop; ValueError covers a pid lsof did not print as a number.
    # os.kill rather than the `kill` binary: no extra process, no PATH lookup.
    signalled: list[str] = []
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (OSError, ValueError) as e:
            log.emit("warn", f"  kill {pid} failed ({type(e).__name__}: {e})")
            continue
        signalled.append(pid)
    if signalled:
        # A delivered signal is not a dead process; SIGTERM is what was asked
        # for, and the exit is the listener's own business.
        log.emit("info", f"  Sent SIGTERM to {label.lower()} process on port {port}")
    else:
        log.emit("warn", f"  {label} is still listening on port {port} — no pid could be signalled")


# node_id -> (hostname, expiry on the monotonic clock). Every miss costs a full
# paramiko connect, and three independent paths resolve hostnames:
# _split_hostnames_by_role (whole-fleet start, once per enabled target),
# _resolve_group_members (per-node start/stop) and _resolve_ssh_target_for_node
# (crash diagnostics), which scans every enabled target per lookup — so one
# uncached name costs a connect per target there, and that path can overlap a
# fleet start. A node's hostname changes about as often as its hardware, so a
# short TTL is plenty — and bounded, so a rename is still picked up without a
# restart.
_HOSTNAME_TTL = 300.0
_hostname_cache: dict[str, tuple[str | None, float]] = {}


def _resolve_node_hostname(config, node_id: str) -> str | None:
    """Resolve a node_id to its K8s hostname via SSH (memoized ~5 min).

    Returns None for unknown, disabled, or unreachable nodes. Swallowing the
    connect error matters: a single unreachable target must not abort a whole
    start/stop run, and `enabled: false` entries are expected to be down.
    """
    from orchestrator.deploy.ssh import conn_kwargs, run_command

    target = next((t for t in config.ssh.targets if t.node_id == node_id), None)
    if not target or not target.enabled:
        return None

    cached = _hostname_cache.get(node_id)
    if cached is not None and time.monotonic() < cached[1]:
        return cached[0]

    try:
        rc, hostname, _ = run_command(
            target.host, "hostname", **conn_kwargs(target, config.ssh.key_store, timeout=10)
        )
    except Exception as e:  # an unreachable node is not fatal here
        logger.warning("Could not resolve hostname for %s (%s): %s", node_id, target.host, e)
        # Deliberately NOT cached: a node that was merely unreachable this pass
        # would otherwise stay unresolvable for the whole TTL, and callers treat
        # "unresolvable" as a reason to refuse a group operation.
        return None

    if rc != 0:
        # run_command reports the real remote status (124 on timeout). Caching
        # the stdout of a failed command as this node's K8s name feeds it to
        # `kubectl label node` and to the pod wait's expected list, where it
        # never matches and burns the full timeout.
        logger.warning("`hostname` failed on %s (%s): exit %s", node_id, target.host, rc)
        return None

    resolved = _hostname_from_output(hostname)
    _cache_hostname(node_id, resolved)
    return resolved


def _hostname_from_output(raw: str | None) -> str | None:
    """The node name in a remote ``hostname``'s stdout, or None.

    A login banner or an rc-file ``echo`` lands on stdout ahead of the answer,
    so the LAST non-empty line is the hostname and the first is not. A value
    with embedded whitespace is no hostname at all and is refused rather than
    cached.
    """
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    return None if any(c.isspace() for c in candidate) else candidate


def _cache_hostname(node_id: str, hostname: str | None) -> None:
    """Memoize a resolved hostname, keeping the TTL rule in one place.

    Also called with a hostname learned from another SSH round-trip, so the
    next resolution does not pay a second handshake for it.
    """
    if hostname:
        _hostname_cache[node_id] = (hostname, time.monotonic() + _HOSTNAME_TTL)


# path -> (mtime_ns, role). _nodes_status calls this once per node on each
# refresh (the 1.5s state poll, rate-limited to ~3s by _NODE_STATUS_TTL via the
# cached _nodes_status), and each miss is a full YAML read + parse. The mtime
# lives in the VALUE, not the key: an edited config is still picked up
# immediately, but its new row replaces the old one instead of adding to it, so
# the map stays bounded by the number of config files however often the
# dashboard's Access tab rewrites a template.
_role_cache: dict[str, tuple[int, str]] = {}


def node_role(target, network_config_path: Path) -> str | None:
    """`node.role` from a target's per-node config file.

    Public because it is imported across a module boundary: the dashboard's
    /api/lifecycle/nodes route calls it to label every target. An underscore
    name reads as "safe to change freely", and a signature drift here 500s that
    live endpoint.

    Returns "" when the target names no config_file — a legitimate "no role
    set", which every caller reads as edge — and None when it names one that
    cannot be read or parsed. The two must stay distinct: "" for an unreadable
    file silently demotes a shard worker to edge, so per-node stop strips
    ``aureax.io/role`` (absent, hence "ok") and reports success while the
    worker's pod keeps running.
    """
    from client.config.parser import load_edge_config_raw

    if not target.config_file:
        return ""
    per_node_path = Path(network_config_path).parent / target.config_file
    try:
        mtime = per_node_path.stat().st_mtime_ns
    except OSError as e:
        logger.warning("Could not read config for %s (%s): %s", target.node_id, per_node_path, e)
        return None

    key = str(per_node_path)
    cached = _role_cache.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        role = role_from_config(load_edge_config_raw(str(per_node_path)))
    except Exception as e:
        logger.warning("Could not read role for %s: %s", target.node_id, e)
        # Deliberately not cached: the row would keep the old mtime, so the
        # next call re-reads rather than trusting a failed parse.
        return None

    _role_cache[key] = (mtime, role)
    return role


def _split_hostnames_by_role(
    config, network_config_path: Path
) -> tuple[list[str], list[str], list[str]]:
    """Return (edge_hostnames, shard_worker_hostnames, planner_hostnames).

    Hostnames are resolved via SSH `hostname`; per-node role comes from the
    target's config_file.

    An unresolvable target is counted under its OWN role, using its node_id as a
    placeholder, rather than being dropped: dropping it shrinks the expected pod
    count to match, so the readiness wait passes trivially with fewer nodes than
    intended and reports success. The consequence of keeping it is that the wait
    for that role never completes — the node_id never matches a K8s node name —
    which is the intended outcome: a node we cannot even reach over SSH is the
    problem, and the run should stop on it rather than declare victory.
    Placeholders are safe to compare AGAINST real node names — they never
    match, which is exactly what the label passes rely on: those iterate
    kubectl.get_nodes() and act only on names present in these lists. What must
    never happen is passing one to kubectl AS a node name; see the planner guard
    in _start_phase_label_and_wait_pods.
    """
    edges: list[str] = []
    workers: list[str] = []
    planners: list[str] = []
    # An unreadable per-node config is collected rather than raised on the spot:
    # every offender is named in one error, and it stays distinct from the
    # deliberate "one unreachable node must not abort the run" above.
    unreadable: list[str] = []

    for target in config.ssh.enabled_targets:
        hostname = _resolve_node_hostname(config, target.node_id)
        role = node_role(target, network_config_path)
        if role is None:
            unreadable.append(
                f"{target.node_id} ({Path(network_config_path).parent / target.config_file})"
            )
            continue

        if not hostname:
            logger.warning(
                "Could not resolve hostname for %s; counting it as an expected node",
                target.node_id,
            )
            if role == SHARD_WORKER_ROLE:
                workers.append(target.node_id)
            elif role == PLANNER_ROLE:
                # Also counted, so the planner is still labelled and waited for.
                # _start_phase_label_and_wait_pods raises on a planner name that
                # is not a K8s node, which is what makes this one fatal.
                planners.append(target.node_id)
            else:
                edges.append(target.node_id)
            continue

        if role == PLANNER_ROLE:
            planners.append(hostname)
        elif role == SHARD_WORKER_ROLE:
            workers.append(hostname)
        else:
            edges.append(hostname)

    if unreadable:
        msg = (
            f"could not read the per-node config of {len(unreadable)} node(s): "
            f"{', '.join(unreadable)} — their role is unknown, and treating that "
            "as edge labels a shard worker with the edge key"
        )
        logger.error("%s", msg)
        raise RuntimeError(msg)

    return edges, workers, planners


# Ceiling on any single readiness wait. Not a tuned value — it exists so the
# wait cannot run forever. A sharded head's own worst case is 2700s (peer wait
# plus model load, see ARCHITECTURE.md), so this leaves real margin while still
# terminating. The dashboard can cancel sooner via the cancel_event; the CLI
# passes mgr=None and has no cancel path at all, which is exactly why a
# deadline is needed rather than optional.
_POD_WAIT_TIMEOUT = 3600


def _wait_pods_on_hosts(kubectl, namespace: str, hostnames: list[str],
                        expected_count: int, *, log: SystemLogger,
                        mgr, label: str, app: str = "aureax-edge",
                        timeout_seconds: int = _POD_WAIT_TIMEOUT) -> None:
    """Wait until at least `expected_count` pods on the given hosts are Ready.

    `app` filters by pod-name prefix, and it is load-bearing once more than
    one DaemonSet exists in the namespace: without it, Ready `aureax-shard`
    worker pods count toward the edge total, so the wait reports success while
    the head's llama-server has never loaded.

    Emits progress every 10s and per-pod diagnostics every 30s after a 30s
    grace period. Honors the lifecycle cancel_event via _sleep_or_cancel, and
    raises RuntimeError once `timeout_seconds` elapses.
    """
    if not hostnames:
        return
    host_set = set(hostnames)
    elapsed = 0
    last_diag = 0
    # (pod_uid, container, restart_count) whose stdout has already been dumped
    # in THIS wait. lastState.terminated persists for a pod's whole life, so
    # without this the same 60 lines are re-fetched and re-emitted every 30s for
    # the rest of the wait — thousands of duplicate rows at the timeout ceiling.
    # Scoped to one wait, not the module: the dashboard process is long-lived,
    # and a later start must dump the crash again.
    dumped_crashes: set[tuple[str, str, int]] = set()
    while True:
        _sleep_or_cancel(mgr, 2)
        elapsed += 2
        pods = kubectl.get_pods(namespace) or []
        relevant = [
            p for p in pods
            if p.get("node") in host_set and str(p.get("name", "")).startswith(app)
        ]
        ready = [p for p in relevant if p.get("ready")]
        if len(ready) >= expected_count:
            log.emit("success", f"  {len(ready)} {label} pod(s) Ready ({elapsed}s)")
            return
        if elapsed % 10 == 0:
            log.emit("info",
                     f"  ...still waiting ({elapsed}s, {len(ready)}/{expected_count} {label} pod(s) Ready)")
        if elapsed >= 30 and elapsed - last_diag >= 30:
            last_diag = elapsed
            not_ready = [p for p in relevant if not p.get("ready")]
            if not_ready:
                # One kubectl pair for the whole round, not two per pod.
                pods_json, events_by_pod = _diagnostic_snapshot(namespace, log)
                for p in not_ready:
                    name = str(p.get("name", ""))
                    _log_pod_diagnostics(
                        name, namespace, log, mgr=mgr,
                        pod=pods_json.get(name),
                        events=events_by_pod.get(name, []),
                        dumped=dumped_crashes,
                    )

        if elapsed >= timeout_seconds:
            msg = (
                f"timed out after {elapsed}s waiting for {expected_count} {label} pod(s) "
                f"to be Ready ({len(ready)} ready). The pods above are the ones still "
                f"not Ready — check `kubectl -n {namespace} describe pod <name>`."
            )
            log.emit("error", f"  {msg}")
            raise RuntimeError(msg)


def _patch_daemonset_selector(
    kubectl, namespace: str, daemonset: str, label: str, value: str, log: SystemLogger | None = None
) -> str:
    """Point a DaemonSet's nodeSelector at `label=value`.

    Returns one of three outcomes — callers MUST distinguish them:
      "patched" — the selector now points at `value`.
      "absent"  — the DaemonSet is not deployed. Not an error for stop (a
                  stack may be partially deployed), fatal for start (waiting
                  for its pods would burn the whole pod-wait timeout for
                  nothing).
      "failed"  — the patch did not land, so the selector still points at the
                  old value. Reporting "stopped"/"started" past this is a lie.

    Suspending by selector rather than deleting keeps the DaemonSet, Services
    and labels intact, so a restart is a one-line patch instead of a redeploy.
    """
    patch = json.dumps(
        {"spec": {"template": {"spec": {"nodeSelector": {label: value}}}}}
    )
    try:
        kubectl.patch("daemonset", daemonset, patch, namespace)
        return "patched"
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "").strip() or f"exit {e.returncode}"
        if "NotFound" in detail or "not found" in detail:
            if log:
                log.emit("info", f"  DaemonSet {daemonset} not deployed — nothing to patch")
            return "absent"
        if log:
            log.emit("error", f"  Failed to patch DaemonSet {daemonset}: {detail}")
        else:
            logger.error("Failed to patch DaemonSet %s: %s", daemonset, detail)
        return "failed"


def _selector_patch_error(daemonset: str, outcome: str) -> RuntimeError:
    """The error every start-side selector patch raises on a non-"patched" outcome.

    Three call sites raise it, so one outcome reads the same wherever it
    happens. Neither may be continued past: entering a pod wait behind a
    selector that never landed burns the whole timeout on pods that cannot
    schedule.
    """
    if outcome == "absent":
        return RuntimeError(f"{daemonset} DaemonSet is not deployed — run `aureax deploy` first")
    return RuntimeError(f"{daemonset} DaemonSet selector patch failed — see the log above")


def _wait_pods_terminated(kubectl, namespace: str, log: SystemLogger,
                          mgr: LifecycleManager | None,
                          timeout_seconds: int = 90) -> None:
    """Wait for the suspended DaemonSets' pods to actually terminate.

    Bounded by the pods' 45s terminationGracePeriodSeconds plus margin. An
    expired wait is a warning naming the nodes still holding pods — never an
    error, because refusing to finish a stop is worse than an unclean one.
    """
    elapsed = 0
    remaining: list[dict] = []
    api_lost = False
    while elapsed < timeout_seconds:
        remaining = [
            p for p in (kubectl.get_pods(namespace) or [])
            if str(p.get("name", "")).startswith(("aureax-edge", "aureax-shard"))
        ]
        if not remaining:
            # get_pods returns [] on ANY API error, so "drained" and "could not
            # ask" arrive identically. Only confirm the drain when the API is
            # actually answering — otherwise the control plane goes down while a
            # node that never saw the deletion keeps its containers running.
            if kubectl.api_reachable():
                log.emit("info", f"  All pods terminated ({elapsed}s)")
                return
            if not api_lost:
                api_lost = True
                log.emit("warn", "  Kubernetes API not answering — cannot confirm the pods drained")
        else:
            api_lost = False
            if elapsed and elapsed % 10 == 0:
                log.emit("info", f"  ...waiting for {len(remaining)} pod(s) to terminate ({elapsed}s)")
        _sleep_or_cancel(mgr, 2)
        elapsed += 2

    if not remaining:
        # Nothing to name: either the API never answered, or the caller passed a
        # timeout the loop could not run in.
        detail = ("the Kubernetes API stopped answering" if api_lost
                  else f"the wait was given only {timeout_seconds}s")
        log.emit("warn",
                 f"  Could not confirm the pods terminated after {elapsed}s — {detail}; "
                 "stopping the control plane anyway")
        return

    nodes = sorted({p.get("node", "?") for p in remaining})
    log.emit("warn",
             f"  {len(remaining)} pod(s) still terminating after {elapsed}s "
             f"(on: {', '.join(nodes)}) — stopping the control plane anyway; "
             "they will be reaped when the node next reconnects")


def do_stop(config, log: SystemLogger, *, skip_self_kill: bool,
            mgr: LifecycleManager | None = None) -> None:
    """Stop the entire Aureax stack. When skip_self_kill=True, keep the dashboard alive.

    Public, like node_role above and for the same reason: ``aureax stop``
    imports it across a module boundary. An underscore name reads as "safe to
    change freely", and this signature has a keyword-only argument the CLI must
    keep passing.
    """
    from orchestrator.cluster import k3d, kubectl

    namespace = config.kubernetes.namespace
    k8s = config.kubernetes

    log.emit("info", "Stopping Aureax...")

    # Asked once: each cluster_running() shells out to `k3d cluster list`, and
    # nothing between here and the stop below can change the answer.
    cluster_up = k3d.cluster_running(k8s.cluster_name)

    if cluster_up:
        # Head/standalone pods first, then workers. The reverse order makes the
        # head's llama-server abort on a vanished RPC peer and crash-loop while
        # we are trying to shut it down gracefully — indistinguishable from a
        # real failure in the dashboard.
        #
        # Note this ordering is best-effort: both patches return immediately and
        # K8s drains the two DaemonSets concurrently. It biases the race the
        # right way rather than enforcing anything, which is enough — the head
        # aborting during a shutdown we asked for is noise, not damage.
        edge_patch = _patch_daemonset_selector(
            kubectl, namespace, "aureax-edge", k8s.node_label, k8s.stopped_label_value, log
        )
        if edge_patch == "patched":
            log.emit("info", "  Edge pods terminating")

        # Attempted unconditionally, NOT gated on `config.node_groups`. The
        # start side decides a node is a shard worker from that node's own
        # `node.role`, so a worker config with no matching group still gets
        # rpc-server scheduled; gating the stop on node_groups would then take
        # the control plane down with those pods never asked to terminate.
        # "absent" already covers the not-deployed case.
        shard_patch = _patch_daemonset_selector(
            kubectl, namespace, "aureax-shard",
            k8s.shard_label, k8s.stopped_label_value, log,
        )
        if shard_patch == "patched":
            log.emit("info", "  Shard worker pods terminating")

        # A stop must never refuse to stop — a failed patch is warned about,
        # not raised, and the control plane still goes down.
        failed = [ds for ds, outcome in (("aureax-edge", edge_patch),
                                         ("aureax-shard", shard_patch))
                  if outcome == "failed"]
        if failed:
            log.emit("warn",
                     f"  {', '.join(failed)}: selector patch failed — their pods were "
                     "NOT asked to terminate and keep running on the nodes")

        # Pod deletion is driven by the API server; stopping k3d immediately
        # after the patches means a node that has not yet seen the deletion
        # keeps its llama-server/rpc-server running with no control plane left
        # to ever deliver it. Bounded (grace period is 45s) and cancellable.
        if edge_patch == "patched" or shard_patch == "patched":
            _wait_pods_terminated(kubectl, namespace, log, mgr=mgr)

        if k3d.stop_cluster(k8s.cluster_name):
            log.emit("info", "  k3d cluster stopped")
        else:
            log.emit("warn", "  k3d cluster stop FAILED — control plane may still be running; "
                             "check `k3d cluster list`")
    else:
        log.emit("info", "  k3d cluster already stopped")

    if not skip_self_kill:
        dashboard_port = config.server.dashboard.port
        _kill_process_on_port(dashboard_port, log, label="Dashboard")
    else:
        log.emit("info", "  Dashboard kill skipped (invoked from dashboard)")

    log.emit("success", "Stopped. Run 'aureax start' to bring everything back.")


def _group_member_ids(config, node_id: str) -> tuple[list[str], str | None]:
    """(node_ids to act on, group name).

    A sharded group serves ONE model across all its members, so acting on a
    single member is never what the user wants: dropping a worker makes the
    head abort mid-inference and crash-loop, and dropping the head leaves the
    workers idling with their share of the weights resident. Per-node
    operations therefore fan out to the whole group.
    """
    group = config.group_for_node(node_id)
    if group is None:
        return [node_id], None
    # Head last on the way in, first on the way out — handled by the callers.
    return list(group.members), group.name


def _label_key_for_node(config, node_id: str, network_config_path: Path) -> str:
    """Which label key gates this node's workload.

    Group membership answers this without reading anything: a node listed under
    a group's ``workers`` runs rpc-server by definition. Only a node outside
    every group falls back to its config file, and an unreadable one raises
    rather than defaulting to the edge key — stripping the wrong label reports
    "stopped" over a pod that is still running.
    """
    k8s = config.kubernetes
    target = next((t for t in config.ssh.targets if t.node_id == node_id), None)
    if target is None:
        return k8s.node_label
    group = config.group_for_node(node_id)
    if group is not None and node_id in group.workers:
        return k8s.shard_label
    role = node_role(target, network_config_path)
    if role is None:
        raise RuntimeError(
            f"node {node_id!r}: could not read its per-node config "
            f"({target.config_file}) — refusing to guess which label gates it"
        )
    return k8s.shard_label if role == SHARD_WORKER_ROLE else k8s.node_label


def do_stop_node(config, node_id: str, log: SystemLogger,
                 network_config_path: Path | None = None) -> None:
    """Stop a single node — or its whole group, if it belongs to one.

    Public for the reason node_role is: ``aureax stop --node`` imports it.
    """
    from orchestrator.cluster import kubectl

    cfg_path = network_config_path or DEFAULT_NETWORK_CONFIG
    members, group_name = _group_member_ids(config, node_id)
    if group_name:
        log.emit(
            "info",
            f"Node '{node_id}' belongs to group '{group_name}' — stopping all "
            f"{len(members)} member(s); the group serves one model and cannot "
            "run with a member missing.",
        )

    # `_group_member_ids` returns head-first, which is the order we want on the
    # way down: stopping a worker first makes the head abort on a vanished peer
    # and crash-loop while we are trying to shut it down gracefully.
    resolved = _resolve_group_members(config, node_id, members, group_name, log)

    failed: list[str] = []
    for member, hostname in resolved:
        label = _label_key_for_node(config, member, cfg_path)
        log.emit("info", f"Removing {label} from {hostname}...")
        ok, detail = kubectl.remove_label(hostname, label)
        if not ok:
            log.emit("error", f"  {member} ({hostname}): label removal failed — {detail}")
            failed.append(member)

    if failed:
        # The label is what gates the workload: while it is still on the node,
        # the pods keep running no matter what we report.
        raise RuntimeError(
            f"could not remove the gating label from {len(failed)}/{len(resolved)} "
            f"node(s): {', '.join(failed)} — their pods are still running"
        )

    log.emit("success", f"Stopped {len(resolved)} node(s) — pods will be terminated by K8s.")
    log.emit("info", f"Run 'aureax start --node {node_id}' to bring them back.")


def _resolve_group_members(
    config, node_id: str, members: list[str], group_name: str | None, log: SystemLogger
) -> list[tuple[str, str]]:
    """Resolve every member to a hostname, or refuse the whole operation.

    Resolution happens BEFORE anything is mutated, and a group operation is
    all-or-nothing. Acting on the reachable subset and reporting
    "Started 3 node(s)" is worse than failing: a group serves one model across
    all its members, so the head comes up, waits out its peer timeout on the
    member we silently skipped, and crash-loops — after the CLI said it worked.
    """
    resolved: list[tuple[str, str]] = []
    unresolved: list[str] = []
    for member in members:
        hostname = _resolve_node_hostname(config, member)
        if hostname:
            resolved.append((member, hostname))
        else:
            unresolved.append(member)

    if not unresolved:
        return resolved

    reasons = {member: _unresolved_reason(config, member) for member in unresolved}
    for member, reason in reasons.items():
        log.emit("error", f"Could not resolve hostname for node '{member}' — {reason}")

    detail = ", ".join(f"{member} ({reason})" for member, reason in reasons.items())
    if group_name:
        raise RuntimeError(
            f"group {group_name!r}: only {len(resolved)}/{len(members)} member(s) could be "
            f"resolved ({detail}) — refusing a partial group operation; the group serves "
            "one model and cannot run with a member missing"
        )
    raise RuntimeError(f"node {node_id!r}: {reasons[node_id]}")


def _unresolved_reason(config, node_id: str) -> str:
    """Why a node_id has no hostname — the three cases point at three fixes.

    A typo in the network YAML, an `enabled: false` someone set deliberately
    and a powered-off node need three different fixes; one "unknown node" for
    all three points the operator at the config for a problem in the rack. The
    CLI shows
    only the per-member log line (cli.py discards the RuntimeError text), so the
    distinction has to be made there as well as in the raise.
    """
    target = next((t for t in config.ssh.targets if t.node_id == node_id), None)
    if target is None:
        return "unknown node_id: not in ssh.targets"
    if not target.enabled:
        return "disabled — set enabled: true to act on it"
    return "unreachable over SSH"


def do_start(
    config,
    log: SystemLogger,
    mgr: LifecycleManager | None = None,
    network_config_path: Path | None = None,
) -> None:
    """Start the entire Aureax stack.

    Every wait loop has a hard ceiling (_NODE_WAIT_TIMEOUT for the node
    join, _POD_WAIT_TIMEOUT for pod readiness) and raises RuntimeError when
    it expires; the dashboard can end a wait sooner via ``mgr.cancel_event``
    (raises CancelledOp). While waiting, if any pod is unhealthy the actual
    failure reason (container state, last termination, recent K8s events)
    is logged so the user doesn't have to guess.

    The function delegates to four phase-specific helpers, each testable in
    isolation.

    Public for the reason node_role is: ``aureax start`` imports it across a
    module boundary, and a signature drift here breaks that command at call
    time with no naming signal that it is anyone else's API.
    """
    from orchestrator.cluster import kubectl

    log.emit("info", "Starting Aureax...")
    # The same path resolution _start_phase_label_and_wait_pods does, and for
    # the same reason: `aureax -c other.yaml start` must validate the file it
    # was pointed at, not the default one.
    _start_phase_validate_config(
        config, log,
        mgr.network_config_path if mgr else (network_config_path or DEFAULT_NETWORK_CONFIG),
    )
    if not _start_phase_cluster(config, log, mgr):
        return
    if not _start_phase_verify_nodes(config, log, mgr):
        return
    _start_phase_wait_k8s_nodes(config, log, mgr, kubectl)
    _start_phase_label_and_wait_pods(config, log, mgr, kubectl, network_config_path)
    log.emit("success", "Started. Run 'aureax status --verbose' to verify.")


def _start_phase_validate_config(
    config, log: SystemLogger, network_config_path: Path | None
) -> None:
    """Re-run the generate-time group validation before the start touches anything.

    ``validate_groups``/``validate_planners`` run only inside ``aureax
    generate`` (orchestrator/config/templating.py). ``aureax start`` does not
    generate, so a fleet file edited since the last generate — a worker added
    to a group, a head's role flipped, a ``tensor_split`` that no longer
    matches, a planner whose ``remote_servers`` was hardcoded — reaches pod
    scheduling unchecked, and the symptoms are the silent ones: a head serving
    the whole model alone, or hanging at the RPC handshake for the full
    startupProbe budget. This is the cheapest possible place to catch it: no
    SSH, no kubectl, just the network config and the per-node templates read
    through the (path, mtime) cache.

    What it can and cannot see: it validates the SOURCE templates, i.e. the
    operator's intent. It does not read the resolved configs already on the
    nodes. The baseline comparison below is what covers the gap between the
    two — and it only warns, because bringing a fleet back up after an edit
    that has not been deployed yet is legitimate and common. Refusing there
    would make `aureax start` unusable exactly when the operator is trying to
    get the cluster back before regenerating.
    """
    from orchestrator.config import group_state
    from orchestrator.config.templating import validate_groups, validate_planners

    cfg_path = Path(network_config_path or DEFAULT_NETWORK_CONFIG)
    configs_dir = cfg_path.parent
    problems = (
        validate_groups(config, configs_dir) + validate_planners(config, configs_dir)
    )
    if problems:
        for problem in problems:
            log.emit("error", f"  {problem}")
        raise RuntimeError(
            "node_groups validation failed before anything was labelled:\n  "
            + "\n  ".join(problems)
        )

    if group_state.baseline_differs(cfg_path, config):
        log.emit(
            "warn",
            "  the fleet's node_groups no longer match the last `aureax generate`, so the "
            "configs on the nodes are stale — the head will dial the peers it was last "
            "generated with, not the ones this file names. Run `aureax generate` and "
            "`aureax apply --all` (or `aureax deploy`) before starting.",
        )


def _start_phase_cluster(config, log: SystemLogger, mgr: LifecycleManager | None) -> bool:
    """Ensure the k3d cluster is running. Returns True to continue do_start.

    Returns False when first-time setup ran end-to-end (it already finished
    everything do_start would do, so the caller exits early).
    """
    from orchestrator.cluster import k3d, kubectl

    k8s = config.kubernetes
    if k3d.cluster_running(k8s.cluster_name):
        log.emit("info", "  k3d cluster already running")
        return True
    if k3d.cluster_exists(k8s.cluster_name):
        log.emit("info", "  Starting k3d cluster...")
        # The return is load-bearing: unchecked, a failed start (api_port
        # conflict, wedged container) reads as "started", and the run then
        # burns the whole node-join wait before blaming the edge nodes.
        if not k3d.start_cluster(k8s.cluster_name):
            log.emit("error", "  k3d cluster start FAILED — check `k3d cluster list` and Docker Desktop")
            raise RuntimeError(f"k3d cluster {k8s.cluster_name!r} failed to start")
        # `k3d cluster start` can exit 0 before the API server serves, which
        # reproduces the same blind wait one phase later.
        if not kubectl.wait_api_ready(timeout=120):
            log.emit("error", "  k3d cluster started but its API server never answered within 120s")
            raise RuntimeError("k3d cluster API server did not become ready")
        log.emit("info", "  k3d cluster started")
        return True
    log.emit("info", "  k3d cluster does not exist — running first-time setup")
    log.emit("info", "  This builds images on each node and downloads models — 5-15 min")
    _run_setup_all(log, mgr=mgr, config=config)
    log.emit("success", "First-time setup complete. Stack is up — verify with 'aureax status --verbose'.")
    return False


def _start_phase_verify_nodes(config, log: SystemLogger, mgr: LifecycleManager | None) -> bool:
    """Drift + image-presence check. Re-bootstraps via setup-all.sh if needed.

    Returns True if all nodes are joined, point at the current Mac IP, and
    have the aureax-client:dev image. Returns False when bootstrap ran
    (caller exits). Raises RuntimeError on unreachable node(s) (checked
    first), a node whose k3s-agent config is present but unreadable, a stale
    K3S_URL, or an undetectable Mac IP.
    """
    from orchestrator.cluster import k3d
    from orchestrator.deploy.k3s_agent import AGENT_ENV_PATH, _parse_k3s_url
    from orchestrator.deploy.ssh import conn_kwargs, probe_facts

    k8s = config.kubernetes

    expected_url = k3d.get_server_url(k8s.k3d.api_port, k3d.probe_host(config))
    if expected_url.startswith(("https://127.", "https://0.")):
        log.emit(
            "error",
            f"  ERROR: could not detect the orchestrator LAN IP (got {expected_url}). The address comes "
            "from the routing table's answer for the first enabled node — check you are on "
            "the same network as the nodes.",
        )
        raise RuntimeError(f"could not detect the orchestrator LAN IP: {expected_url}")

    log.emit("info", f"  Expected K3s server URL: {expected_url}")
    unjoined: list[str] = []
    stale: list[str] = []
    unreachable: list[str] = []
    no_image: list[str] = []
    sudo_failed: list[str] = []
    for t in config.ssh.enabled_targets:
        _check_cancel(mgr)
        # Every fact in ONE SSH round-trip per node — the handshake against an
        # edge node's small CPU is the expensive part of this whole pass.
        #
        # Unreachable is its own bucket, NOT folded into `unjoined`: unjoined
        # triggers a full setup-all.sh re-bootstrap, which is the wrong answer
        # to a transient SSH blip on an already-provisioned node.
        try:
            facts = probe_facts(
                t.host,
                {
                    "k3s_env": f"sudo cat {AGENT_ENV_PATH}",
                    # Needs no sudo (only search permission on the directory),
                    # and answers exactly what k3s_env cannot: probe_facts wraps
                    # every snippet in `2>/dev/null || true`, so a sudo failure
                    # and a missing file both come back as "".
                    "agent_env_present": f"test -e {AGENT_ENV_PATH} && echo yes",
                    "image": (
                        "sudo k3s ctr images list -q | "
                        "grep -qx 'docker.io/library/aureax-client:dev' && echo yes"
                    ),
                    # Free here, and the next phase resolves every node's
                    # hostname over SSH — that was a second full handshake.
                    "hostname": "hostname",
                },
                **conn_kwargs(t, config.ssh.key_store, timeout=15),
            )
        except Exception as e:
            log.emit("error", f"  [{t.node_id}] unreachable over SSH ({type(e).__name__}: {e})")
            unreachable.append(t.node_id)
            continue

        # The SSH path stays authoritative: _HOSTNAME_TTL can expire before a
        # slow node-join wait ends, which costs a connect and nothing else.
        _cache_hostname(t.node_id, _hostname_from_output(facts["hostname"]))

        if not facts["k3s_env"] and facts["agent_env_present"] == "yes":
            # The file is there and sudo could not read it. Reading that as
            # "not joined" re-bootstraps the WHOLE fleet (image builds + model
            # downloads) because one node's sudo stopped working — and the
            # image fact is sudo too, so it would also read as missing.
            log.emit("error",
                     f"  [{t.node_id}] {AGENT_ENV_PATH} exists but could not be read — sudo failed")
            sudo_failed.append(t.node_id)
            continue

        configured = _parse_k3s_url(facts["k3s_env"]) if facts["k3s_env"] else None
        if configured is None:
            # Could not read k3s-agent env file — most likely the K3s agent
            # was never installed on this node (first-time bootstrap of a
            # newly-added target). Distinct from "joined but stale URL".
            log.emit("warn", f"  [{t.node_id}] no k3s-agent config (not joined yet, or sudo failed)")
            unjoined.append(t.node_id)
        elif configured != expected_url:
            log.emit("error", f"  [{t.node_id}] stale K3S_URL: {configured} (expected {expected_url})")
            stale.append(t.node_id)

        # A node can be joined but missing the aureax-client:dev image (e.g. a
        # previous setup-all.sh failed mid-build). Without it, pods stay
        # Pending forever: the DaemonSet uses imagePullPolicy: Never.
        if facts["image"] != "yes":
            log.emit("warn", f"  [{t.node_id}] aureax-client:dev image not present in K3s containerd")
            no_image.append(t.node_id)

    if unreachable:
        log.emit("hint", "  Check power and network on the node(s) above, then re-run start.")
        raise RuntimeError(
            f"{len(unreachable)} node(s) unreachable over SSH: {', '.join(unreachable)}"
        )

    if sudo_failed:
        log.emit("hint", "  Check passwordless sudo for the SSH user on the node(s) above, "
                         "then re-run start.")
        raise RuntimeError(
            f"{len(sudo_failed)} node(s) refused sudo: {', '.join(sudo_failed)} — their "
            "k3s-agent config is present but unreadable, and treating that as 'not joined' "
            "would re-bootstrap the whole fleet"
        )

    if unjoined:
        log.emit("info", f"  {len(unjoined)} node(s) not joined: {', '.join(unjoined)} — running setup-all.sh to bootstrap")
        _run_setup_all(log, mgr=mgr, config=config)
        log.emit("success", "Bootstrap complete. Stack is up.")
        return False

    if stale:
        log.emit("error", f"ERROR: {len(stale)} edge node(s) have stale K3S_URL: {', '.join(stale)}")
        log.emit("info", "Your Mac's LAN IP changed; edge nodes can't reach the new IP.")
        log.emit("hint", "Run: uv run aureax heal-nodes   (rewrites env + restarts k3s-agent on each node)")
        raise RuntimeError(f"{len(stale)} node(s) have stale K3S_URL")

    # Re-running setup-all.sh is idempotent: it skips already-done steps and
    # only builds where the image is missing.
    if no_image:
        log.emit("info", f"  {len(no_image)} node(s) missing image: {', '.join(no_image)} — running setup-all.sh to bootstrap")
        _run_setup_all(log, mgr=mgr, config=config)
        log.emit("success", "Bootstrap complete. Stack is up.")
        return False

    return True


# Ceiling on the node-join wait. A powered-on node that was verified over SSH
# moments ago (phase order guarantees that) reconnects its k3s-agent well
# inside this; what this actually catches is an API server that answers with
# nothing forever — kubectl.get_nodes() returns [] on any API error, which
# without a deadline spun this wait for good. Same rationale as
# _POD_WAIT_TIMEOUT: the CLI passes mgr=None and has no cancel path.
_NODE_WAIT_TIMEOUT = 600


def _start_phase_wait_k8s_nodes(config, log: SystemLogger, mgr: LifecycleManager | None, kubectl,
                                timeout_seconds: int = _NODE_WAIT_TIMEOUT) -> None:
    """Block until every enabled ssh.target's K8s node reports Ready.

    Allow-list, not a count. ``kubectl get nodes`` lists every machine joined to
    the cluster — targets marked ``enabled: false`` among them, and any machine
    that joined for other reasons — so counting non-control-plane Ready nodes
    lets a stranger stand in for a configured node that never joined. The miss
    then surfaces only when the pod wait two phases later burns its full
    timeout on a host that is not in the cluster. The label pass filters by the
    same allow-list, for the same reason.

    A target whose hostname cannot be resolved over SSH is carried by its
    node_id, which never matches a K8s node name — so this wait fails on it
    here rather than passing and stalling in the pod wait.

    Raises RuntimeError once `timeout_seconds` elapses.
    """
    # Cache hits: _start_phase_verify_nodes runs first and stamps every
    # target's hostname from the facts round-trip it already makes.
    resolved = [
        _resolve_node_hostname(config, t.node_id) or t.node_id
        for t in config.ssh.enabled_targets
    ]
    expected = set(resolved)
    if len(expected) < len(resolved):
        # K8s node names are unique, so two targets answering the same
        # hostname can only ever be one cluster node — count them as one, but
        # say so, because it usually means a cloned image or a duplicated
        # ssh.targets entry rather than two machines.
        dupes = sorted({h for h in resolved if resolved.count(h) > 1})
        log.emit(
            "warn",
            f"  {len(resolved) - len(expected)} enabled target(s) share a hostname with "
            f"another ({', '.join(dupes)}) — counted as one node",
            source="nodes",
        )
    expected_nodes = len(expected)
    log.emit("info", f"  Waiting for {expected_nodes} edge node(s) to connect{_cancel_hint(mgr)}")
    elapsed = 0
    while True:
        _sleep_or_cancel(mgr, 2)
        elapsed += 2
        nodes = kubectl.get_nodes()
        ready_names = {
            n.get("name") for n in nodes
            if "control-plane" not in n.get("roles", "")
            and n.get("status") == "True"
        }
        edge_ready = expected & ready_names
        if len(edge_ready) >= expected_nodes:
            log.emit("success", f"  {len(edge_ready)} edge node(s) Ready ({elapsed}s)")
            return
        if elapsed % 10 == 0:
            log.emit("info", f"  ...still waiting ({elapsed}s, {len(edge_ready)}/{expected_nodes} nodes Ready)")
            # Surface any configured node that's been not-Ready for a while
            for n in nodes:
                if n.get("name") not in expected:
                    continue
                if n.get("status") != "True":
                    log.emit("warn", f"  [{n.get('name', '?')}] node not Ready (status={n.get('status', '?')})", source="nodes")

        if elapsed >= timeout_seconds:
            joined = {n.get("name") for n in nodes}
            not_ready = sorted(n for n in expected - edge_ready if n in joined)
            absent = sorted(n for n in expected - edge_ready if n not in joined)
            detail = "; ".join(
                part for part in (
                    f"not Ready: {', '.join(not_ready)}" if not_ready else "",
                    f"never joined the cluster: {', '.join(absent)}" if absent else "",
                ) if part
            )
            msg = (
                f"timed out after {elapsed}s waiting for {expected_nodes} edge node(s) "
                f"to be Ready ({len(edge_ready)} ready; {detail}). "
                "Check the nodes' k3s-agent and `kubectl get nodes`."
            )
            log.emit("error", f"  {msg}")
            raise RuntimeError(msg)


def _preflight_double_labels(
    nodes: list[dict], k8s, claimed: set[str], log: SystemLogger,
) -> None:
    """Report — or refuse — nodes already carrying BOTH gating labels.

    Both label passes only ever ADDED, so a node promoted from shard worker to
    head (or demoted) kept the label it was leaving behind: it matches both
    DaemonSet selectors and runs llama-server and rpc-server side by side on
    one machine, a whole model and a weight shard competing for the same RAM.
    Nothing else in the stack notices two pods on one node.

    ``claimed`` is the set of K8s node names this run will label: every enabled
    target for a fleet start, the group's members for ``do_start_node``. One of
    those is only reported — the strip that follows each label write fixes it a
    few lines later. Any OTHER double-labelled node is fatal: nothing in this
    run will touch it, and it is running two pods right now, on both paths (a
    per-node start restores the DaemonSet selectors, which are fleet-wide).

    Free of API calls: it reads the ``labels`` dict of the listing the caller
    already fetched.
    """
    from orchestrator.cluster import labels as cluster_labels

    for name in cluster_labels.double_labelled_nodes(nodes, k8s):
        if name in claimed:
            log.emit(
                "error",
                f"  {name} carries both {k8s.node_label} and {k8s.shard_label} — it matches "
                "both DaemonSet selectors and has been running two pods. Removing the label "
                "this start does not claim.",
            )
            continue
        raise RuntimeError(
            f"node {name!r} carries BOTH {k8s.node_label} and {k8s.shard_label}, so it "
            "matches both DaemonSet selectors and runs llama-server and rpc-server side by "
            "side on one machine. This run does not label it — it is in none of the nodes "
            "this start claims — so nothing here will fix it. Remove the wrong label — "
            f"`kubectl label node {name} {k8s.shard_label}-` for an edge node, "
            f"`kubectl label node {name} {k8s.node_label}-` for a shard worker. "
            "`aureax deploy` is no remedy on its own: both of its label passes walk only "
            "the targets this fleet file marks enabled, so a node that is disabled here, "
            "or joined the cluster for other reasons, keeps both labels however often it "
            "is run."
        )


def _strip_conflicting_label_or_raise(
    node_name: str, label: str, k8s, log: SystemLogger
) -> None:
    """Remove the other pass's label from a node this start just claimed.

    Fatal here, where the deploy path logs and continues
    (``Deployer._strip_conflicting_label``). The asymmetry is deliberate: a
    deploy is normally followed by a start, which gets another chance at the
    same node, but start is the LAST writer before pods schedule — a label it
    failed to remove is a second pod on that node for the whole run.
    """
    from orchestrator.cluster import labels as cluster_labels

    ok, detail = cluster_labels.strip_conflicting_label(node_name, label)
    if ok:
        return
    if label == k8s.shard_label:
        daemonset, doubled = "aureax-shard", "rpc-server alongside its llama-server"
    else:
        daemonset, doubled = "aureax-edge", "llama-server alongside its rpc-server"
    msg = (
        f"could not strip {label} from {node_name}: {detail or 'no detail from kubectl'} — "
        f"the node still matches the {daemonset} DaemonSet selector and would run {doubled}. "
        f"Remove it with `kubectl label node {node_name} {label}-` and re-run `aureax start`."
    )
    log.emit("error", f"  {msg}")
    raise RuntimeError(msg)


def _claim_gating_label(
    kubectl, node_name: str, label: str, value: str, k8s, log: SystemLogger,
    *, existing: dict | None = None,
) -> None:
    """Give a node exactly ONE gating label: write this run's, strip the other.

    Every writer of a gating label goes through here — the fleet start's worker,
    edge and planner loops and ``do_start_node`` — because adding without
    removing is what left a role-flipped node matching both DaemonSet selectors,
    running llama-server and rpc-server side by side on one machine. Per-node
    start is the path that flips a role in practice: demoting a head into an RPC
    slot is the documented rollback manoeuvre, and it is `aureax start --node`
    that is reached for afterwards.

    ``existing`` is the node's current labels, when the caller already holds a
    listing: an already-correct label is not rewritten. The strip runs either
    way — a node that already carries the right key is exactly the one a role
    flip left carrying both, and it is the node the pre-flight promised would be
    fixed here.
    """
    if existing is None or existing.get(label) != value:
        kubectl.label_node(node_name, label, value)
    other = k8s.node_label if label == k8s.shard_label else k8s.shard_label
    _strip_conflicting_label_or_raise(node_name, other, k8s, log)


# One SSH round trip per group member, all of them concurrent, with a hard
# ceiling: this is a pre-start gate, not a diagnosis. `group doctor` is the
# diagnosis and asks the same question with the same probe.
_BUILD_GATE_TIMEOUT = 60


def _assert_group_builds_match(config, log: SystemLogger) -> None:
    """Fail the start if a group's members run different llama.cpp builds.

    This is the "hangs at the RPC handshake" failure made loud. The wire format
    is version-locked, so a head that dials a peer built from another commit
    connects and then stops with no error on either side; the only symptom is an
    edge pod sitting 0/1 Running until its startupProbe budget (3015s) expires —
    50 minutes of looking like a slow model load.

    Asks each member's IMAGE, over SSH, for the llama.cpp commit it was built
    from, its architecture and the sha256 of ``llama-server`` and
    ``rpc-server``: the commit must be the head's on every member, and the
    binaries must be the head's on every member of the head's architecture (an
    aarch64 and an x86_64 image are never byte-identical, and the protocol is
    architecture-neutral). Not the pods' image digests — the digest version of
    this gate failed every start on the real fleet: every node builds
    ``aureax-client:dev`` natively (scripts/build-on-node.sh), so the digests
    differ by construction while the binaries are identical. See
    ``orchestrator/cluster/buildcheck.py`` for the full account.

    Fails OPEN on a member that could not be probed — an SSH blip must not
    refuse a start — and CLOSED on two commits or two same-architecture
    checksums that differ.
    """
    if not config.node_groups:
        return

    from orchestrator.cluster import buildcheck

    # enabled_targets, like every other SSH pass in this module: this gate must
    # not become the one that dials a node the operator switched off, paying a
    # connect and up to _BUILD_GATE_TIMEOUT for an answer about a group that
    # node does not serve. A disabled member does not validate today
    # (schema.NetworkConfig._check_node_groups refuses one), so this is
    # agreement with the rest of the module, not a second gate.
    by_id = {t.node_id: t for t in config.ssh.enabled_targets}
    members = [by_id[m] for g in config.node_groups for m in g.members if m in by_id]
    facts = buildcheck.probe_group(
        members, key_store=config.ssh.key_store, timeout=_BUILD_GATE_TIMEOUT
    )

    problems: list[str] = []
    for group in config.node_groups:
        group_problems, notes = buildcheck.compare_group(group, facts)
        problems.extend(group_problems)
        for note in notes:
            # A skipped check is a warning, not information: "not checked" and
            # "checked, all good" must never look the same in the start log.
            # Prefixed with the group, which `group doctor` does not need (it
            # prints a verdict block per group) but a flat start log does: two
            # groups otherwise emit the same OK line twice with nothing to say
            # which fleet half each one covered.
            log.emit(
                "warn" if note.startswith("SKIP") else "info", f"  {group.name}: {note}"
            )

    if problems:
        for problem in problems:
            log.emit("error", f"  {problem}")
        raise RuntimeError(
            "the members of a sharded group do not run the same llama.cpp build (one "
            "commit everywhere, identical binaries within the head's architecture):\n  "
            + "\n  ".join(problems)
        )


def _start_phase_label_and_wait_pods(
    config,
    log: SystemLogger,
    mgr: LifecycleManager | None,
    kubectl,
    network_config_path: Path | None = None,
) -> None:
    """Label nodes and wait for pods, in dependency order.

    Sequencing is load-bearing throughout:
      A-. the group build gate — a group whose members were built from
          different llama.cpp commits hangs at the RPC handshake, so it must be
          caught before anything is labelled or scheduled.
      A0. shard workers — a head's llama-server exits immediately if an RPC
          peer is unreachable at load time, so workers must be Ready first.
      A.  edges/heads — the model load itself.
      B.  planner — its MCP host must be able to reach edge MCP servers on
          first connect, or it cold-start crash-loops.
    """
    namespace = config.kubernetes.namespace
    k8s = config.kubernetes

    # The dashboard supplies the manager's path; the CLI passes the one behind
    # its --config flag. Falling straight through to the default meant
    # `aureax -c other.yaml start` resolved every role against the wrong file
    # and silently found none.
    cfg_path = (
        mgr.network_config_path if mgr
        else (network_config_path or DEFAULT_NETWORK_CONFIG)
    )
    edge_hostnames, worker_hostnames, planner_hostnames = _split_hostnames_by_role(
        config, cfg_path
    )

    nodes = kubectl.get_nodes()
    if not nodes and (worker_hostnames or edge_hostnames):
        # get_nodes() returns [] on ANY API error, and the previous phase just
        # watched these nodes report Ready — so an empty snapshot here is the
        # API server, not the fleet. Left alone it labels nothing, patches the
        # selector anyway, and burns the whole pod wait blaming the pods.
        raise RuntimeError(
            "kubectl returned no nodes seconds after the node-join wait passed — the API "
            "server is unreachable; check `kubectl get nodes`"
        )

    _preflight_double_labels(
        nodes, k8s,
        set(edge_hostnames) | set(worker_hostnames) | set(planner_hostnames),
        log,
    )

    # BEFORE any label is written, not between the edge patch and the edge wait
    # where the imageID version of this gate had to sit: it asks the nodes
    # themselves, so it no longer depends on a pod existing to have something to
    # compare. Nothing should be labelled or scheduled on a group whose members
    # cannot talk to each other — and once the edge wait starts, a build
    # mismatch costs the full startupProbe budget (3015s) in silence.
    _assert_group_builds_match(config, log)

    # --- Phase A0: shard workers, before anything that depends on them ------
    if worker_hostnames:
        log.emit("info", "  ===  Shard worker phase  ===")
        worker_set = set(worker_hostnames)
        for n in nodes:
            if n["name"] not in worker_set:
                continue
            _claim_gating_label(
                kubectl, n["name"], k8s.shard_label, k8s.shard_label_value, k8s, log,
                existing=n.get("labels", {}),
            )

        current = _get_daemonset_node_selector(namespace, k8s.shard_label, "aureax-shard")
        if current == k8s.shard_label_value:
            log.emit("info", f"  Shard DaemonSet selector already '{k8s.shard_label_value}' — no change")
        else:
            outcome = _patch_daemonset_selector(
                kubectl, namespace, "aureax-shard", k8s.shard_label, k8s.shard_label_value, log
            )
            if outcome == "patched":
                log.emit("info", f"  Shard DaemonSet selector → '{k8s.shard_label_value}' — workers scheduling")
            else:
                raise _selector_patch_error("aureax-shard", outcome)

        log.emit(
            "info",
            f"  Waiting for {len(worker_hostnames)} shard worker(s) to be Ready "
            f"before starting the head{_cancel_hint(mgr)}",
        )
        _wait_pods_on_hosts(
            kubectl, namespace, worker_hostnames, len(worker_hostnames),
            log=log, mgr=mgr, label="shard worker", app="aureax-shard",
        )

    # Allow-list, not deny-list. `nodes` is every node joined to the cluster,
    # so labelling "everything that isn't control-plane" hands an edge pod to
    # targets marked `enabled: false` — which the config says not to deploy to
    # — and to any machine that joined the cluster for other reasons.
    # edge_hostnames comes from enabled_targets with planners and workers
    # already split out (planners are labelled after edges below; workers use
    # the shard key), so it is exactly the set that was asked for.
    #
    # Unresolvable targets are carried in edge_hostnames as node_id
    # placeholders (see _split_hostnames_by_role); those never match a real K8s
    # node name, so they simply select nothing here.
    wanted_edges = set(edge_hostnames)
    for n in nodes:
        if "control-plane" in n.get("roles", ""):
            continue
        if n["name"] not in wanted_edges:
            continue
        _claim_gating_label(
            kubectl, n["name"], k8s.node_label, k8s.edge_label_value, k8s, log,
            existing=n.get("labels", {}),
        )

    # Only patch the DaemonSet if its current selector differs from the
    # desired value. Patching with the same value is a no-op for K8s, but
    # it's misleading in the dashboard log if we always say "DaemonSet
    # restored — pods scheduling" when nothing actually changed.
    current_selector = _get_daemonset_node_selector(namespace, k8s.node_label)
    if current_selector == k8s.edge_label_value:
        log.emit("info", f"  DaemonSet selector already '{k8s.edge_label_value}' — no change (existing pods stay)")
    else:
        outcome = _patch_daemonset_selector(
            kubectl, namespace, "aureax-edge", k8s.node_label, k8s.edge_label_value, log
        )
        if outcome == "patched":
            log.emit("info", f"  DaemonSet selector '{current_selector or 'unset'}' → '{k8s.edge_label_value}' — pods will be (re)scheduled")
        else:
            raise _selector_patch_error("aureax-edge", outcome)

    # --- Phase A: wait for edge pods only -----------------------------------
    expected_edges = len(edge_hostnames)
    if config.node_groups:
        log.emit(
            "info",
            "  The sharded head streams every tensor to its workers over the LAN "
            "on first load — expect minutes, not seconds.",
        )
    log.emit("info", f"  Waiting for {expected_edges} edge pod(s) to be Ready{_cancel_hint(mgr)}")
    _wait_pods_on_hosts(
        kubectl, namespace, edge_hostnames, expected_edges,
        log=log, mgr=mgr, label="edge", app="aureax-edge",
    )

    # --- Phase B: schedule planner (if any) and wait for it -----------------
    # Only real K8s node names reach kubectl. _split_hostnames_by_role puts a
    # target's node_id into these lists as a PLACEHOLDER when SSH could not
    # resolve its hostname — deliberately, so the count still reflects the node
    # and the wait does not pass trivially. An edge or worker placeholder is
    # made fatal by its pod wait, which the placeholder can never satisfy; a
    # planner has no such wait to fail it, so the check below raises instead —
    # skipping it leaves the planner unlabelled and unwaited-for while the run
    # reports success. The message states only what this check establishes —
    # the name is absent from the cluster — because an SSH failure and a node
    # that never joined both land here.
    known_nodes = {n["name"] for n in nodes}
    if planner_hostnames:
        log.emit("info", "  ===  Cognitive planner phase  ===")
    for planner_host in planner_hostnames:
        if planner_host not in known_nodes:
            raise RuntimeError(
                f"planner '{planner_host}' is not a node in the cluster — either its "
                "hostname could not be resolved over SSH, or the node never joined. "
                "Check `kubectl get nodes` and the planner's SSH reachability, then "
                "re-run start."
            )
        log.emit("info", f"  Scheduling planner pod on {planner_host}.")
        log.emit("info", "  Planner is the largest workload — a large model on GPU.")
        log.emit("info", "  First start typically: ~10-30s CUDA kernel autotune, then model load,")
        log.emit("info", f"  then MCP host connects to {expected_edges} edge MCP server(s).")
        _claim_gating_label(
            kubectl, planner_host, k8s.node_label, k8s.edge_label_value, k8s, log
        )
        log.emit("info", f"  Labeled {planner_host} — planner pod scheduling now.")
        _wait_pods_on_hosts(
            kubectl, namespace, [planner_host], 1,
            log=log, mgr=mgr, label="planner",
        )


def do_start_node(config, node_id: str, log: SystemLogger,
                  network_config_path: Path | None = None) -> None:
    """Start a single node — or its whole group, if it belongs to one.

    Public for the reason node_role is: ``aureax start --node`` imports it.
    """
    from orchestrator.cluster import kubectl

    k8s = config.kubernetes
    cfg_path = network_config_path or DEFAULT_NETWORK_CONFIG
    # Before any label is written, for the same reason as in do_start: this
    # path fans out to a whole group, so an edit that broke the group is worse
    # here than fleet-wide — it schedules the members one at a time.
    _start_phase_validate_config(config, log, cfg_path)
    members, group_name = _group_member_ids(config, node_id)
    if group_name:
        log.emit(
            "info",
            f"Node '{node_id}' belongs to group '{group_name}' — starting all "
            f"{len(members)} member(s).",
        )

    # Workers before the head: the head's llama-server exits if a peer is
    # unreachable when it loads. _group_member_ids returns head-first, so
    # reverse it here.
    order = list(reversed(members)) if group_name else members

    # Resolve every member before labelling any of them — see
    # _resolve_group_members for why a partial group start is worse than none.
    resolved = _resolve_group_members(config, node_id, order, group_name, log)

    # The fleet start's pre-flight, on the fleet start's terms — a node this
    # run does not claim is fatal here too, because the selector restore below
    # is fleet-wide: it schedules BOTH pods on any machine that still carries
    # both keys, however few members were asked for. get_nodes() is [] on any
    # API error, which makes this a no-op and leaves the first label write
    # below to fail loudly.
    _preflight_double_labels(
        kubectl.get_nodes(), k8s, {hostname for _member, hostname in resolved}, log
    )

    # No emptiness guard: _resolve_group_members raises rather than returning
    # an empty or partial list, so `resolved` always covers every member here.
    used_labels: set[str] = set()
    labelled: list[str] = []
    for member, hostname in resolved:
        label = _label_key_for_node(config, member, cfg_path)
        value = k8s.shard_label_value if label == k8s.shard_label else k8s.edge_label_value
        log.emit("info", f"Adding {label}={value} to {hostname}...")
        try:
            # Writes this role's label and strips the other one: a member whose
            # role was flipped since the last deploy reaches this path carrying
            # both, and `aureax start --node` is how the documented head/worker
            # demotion is brought back up. A failed strip raises out of here
            # rather than being caught below — it is not a labelling failure,
            # and its own message names the node and the fix.
            _claim_gating_label(kubectl, hostname, label, value, k8s, log)
        except subprocess.CalledProcessError as e:
            # Stop at the first failure rather than collecting: `order` is
            # workers-first so the head never boots without its peers, and
            # labelling the head anyway schedules a llama-server that waits out
            # worker_wait_timeout_seconds and crash-loops.
            log.emit("error", f"  {member} ({hostname}): kubectl label node failed — {e}")
            raise RuntimeError(
                f"could not label {member} ({hostname}) with {label}={value}: {e} — "
                f"already labelled: {', '.join(labelled) or 'none'}"
            ) from e
        used_labels.add(label)
        labelled.append(member)

    # Labelling a node is only half of it. After a whole-stack `aureax stop`
    # the DaemonSet's nodeSelector points at the "stopped" value that no node
    # carries, so the label alone schedules nothing and reporting success would
    # leave the operator watching for a pod that is never going to appear.
    # Restore the selector for whichever DaemonSets the labels above apply to.
    namespace = k8s.namespace
    # Shard selector before the edge one, for the same reason the label loop
    # above runs workers-first: the edge patch is what schedules the head.
    # Not sorted(): alphabetically aureax.io/role precedes aureax.io/shard,
    # which is exactly backwards. Like the ordering in do_stop this only
    # biases the race (both patches return immediately and
    # nothing waits for a worker to be Ready); the head's own peer wait remains
    # the actual guard.
    resumed_fleet = False
    for label in (k8s.shard_label, k8s.node_label):
        if label not in used_labels:
            continue
        if label == k8s.shard_label:
            daemonset, wanted = "aureax-shard", k8s.shard_label_value
        else:
            daemonset, wanted = "aureax-edge", k8s.edge_label_value
        current = _get_daemonset_node_selector(namespace, label, daemonset)
        if current == wanted:
            continue
        if current == k8s.stopped_label_value:
            # A whole-stack stop suspends by selector and leaves every node's
            # gating label in place, so restoring the selector re-schedules
            # every still-labelled node — not only the one(s) asked for.
            resumed_fleet = True
            log.emit(
                "warn",
                f"  DaemonSet {daemonset} is suspended by a whole-stack stop — restoring its "
                "selector resumes EVERY node that still carries the gating label, not just "
                "the requested one(s).",
            )
        log.emit(
            "info",
            f"  DaemonSet {daemonset} selector '{current or 'unset'}' → '{wanted}' "
            "(was suspended by a previous stop)",
        )
        outcome = _patch_daemonset_selector(kubectl, namespace, daemonset, label, wanted, log)
        if outcome != "patched":
            # The label alone schedules nothing while the selector points at
            # "stopped" — reporting success here is the exact lie this
            # restore step exists to prevent.
            raise _selector_patch_error(daemonset, outcome)

    if resumed_fleet:
        log.emit(
            "success",
            f"Started {len(resolved)} node(s) — and resumed every other node still carrying "
            "a gating label, because the DaemonSet selector is fleet-wide.",
        )
    else:
        log.emit("success", f"Started {len(resolved)} node(s) — pods will be scheduled by K8s.")


_NODE_STATUS_TTL = 3.0  # seconds
_node_status_cache: dict = {"ts": 0.0, "path": None, "data": []}
# Held across the TTL check, the probe pass and the store. /api/lifecycle/state
# is a sync route, so Starlette threadpools it, and the browser re-polls every
# 1.5s without awaiting the previous response — without this every overlapping
# viewer redoes the whole pass. Same shape as dashboard/api.py::_cached.
_node_status_lock = threading.Lock()


def _nodes_status(network_config_path: Path) -> list[dict]:
    """Per-node reachability summary:
    [{node_id, display_name, host, enabled, role, group, running}, ...].

    Probes each node's llama NodePort (30800). Result is cached for ~3s so
    the 1.5s state-poll interval doesn't hammer the LAN.
    """
    with _node_status_lock:
        return _refresh_nodes_status(network_config_path)


def _refresh_nodes_status(network_config_path: Path) -> list[dict]:
    """The cached body of :func:`_nodes_status`. Call it holding the lock."""
    # The path check comes first: a `ts` of 0.0 means "boot" under monotonic,
    # so on a freshly-booted machine the TTL comparison alone would read the
    # sentinel as fresh. `path: None` never matches, which is what saves it.
    if (
        _node_status_cache["path"] == str(network_config_path)
        and time.monotonic() - _node_status_cache["ts"] < _NODE_STATUS_TTL
    ):
        return _node_status_cache["data"]

    from orchestrator.config.loader import load_network_config

    try:
        cfg = load_network_config(str(network_config_path))
    except Exception as e:
        # Stamped so a config that stays broken warns once per TTL rather than
        # once per 1.5s poll. The sibling endpoint (dashboard/lifecycle_api.py)
        # 503s on the same condition; here the dots simply go stale.
        logger.warning("Could not read network config for node status: %s", e)
        if _node_status_cache["path"] == str(network_config_path):
            _node_status_cache["ts"] = time.monotonic()
            return _node_status_cache["data"]
        # A path that has never loaded is stamped too, empty. Stamping only
        # `ts` leaves the entry guard's path check failing forever, so the TTL
        # never engages and every poll re-reads and re-warns.
        _node_status_cache["ts"] = time.monotonic()
        _node_status_cache["path"] = str(network_config_path)
        _node_status_cache["data"] = []
        return []

    hosts_by_id = cfg.host_by_node_id()
    probe_cache: dict[tuple[str, int], bool] = {}

    def probe(host: str, port: int) -> bool:
        """Probe once per (host, port) per refresh — members share a head."""
        key = (host, port)
        if key not in probe_cache:
            probe_cache[key] = tcp_reachable(host, port, timeout=0.3)
        return probe_cache[key]

    out: list[dict] = []
    for t in cfg.ssh.targets:
        group = cfg.group_for_node(t.node_id)
        # None (config_file unreadable) becomes "" for this display row. That
        # also routes the probe below to the node's own :30800 instead of its
        # head's, so an unreadable worker config shows a red dot rather than
        # the group's real state — visible, unlike the label paths, which
        # refuse to guess.
        role = node_role(t, network_config_path) or ""

        # A shard worker runs rpc-server and no llama-server, so it has no
        # NodePort 30800 of its own (service-llama selects app=aureax-edge, and
        # externalTrafficPolicy: Local means only nodes with such a pod answer).
        #
        # It is NOT probed on its RPC port either. rpc-server accepts one client
        # with a listen backlog of 1, so once the head is attached an off-node
        # dial goes unaccepted: the dot would flap red while the worker does
        # exactly its job, and each probe leaves a queued connection for
        # rpc-server to pick up later. daemonset-shard.yaml carries the same
        # rule (see ARCHITECTURE.md, "Invariants and how each is enforced"), and
        # test-all.sh checks the listener on the node rather than dialling it.
        #
        # Instead a worker reports its GROUP's state, read from the head's 30800
        # — the group serves one model across all its members, so "up" is a
        # property of the group, not of any one machine.
        probe_host, probe_port = t.host, LLAMA_NODE_PORT
        if role == SHARD_WORKER_ROLE and group is not None:
            probe_host = hosts_by_id.get(group.head, t.host)

        out.append({
            "node_id": t.node_id,
            "display_name": t.display_name or t.node_id,
            "host": t.host,
            "enabled": t.enabled,
            "role": role,
            "group": group.name if group else None,
            # A disabled target is deliberately not deployed, so probing it
            # would just render a permanent red dot for expected absence.
            "running": t.enabled and probe(probe_host, probe_port),
        })
    # Stamped AFTER the probe pass, not before it: a pass that took a second
    # would otherwise arrive already a second into its own 3s TTL.
    _node_status_cache["ts"] = time.monotonic()
    _node_status_cache["path"] = str(network_config_path)
    _node_status_cache["data"] = out
    return out


def tcp_reachable(host: str, port: int, timeout: float = 0.3) -> bool:
    """Is anything listening at host:port? One reachability gate for the whole
    orchestrator.

    Public because it is used across a module boundary: the node-status column
    here and the Access tab's live-auth probe (access_service.probe_one) ask
    the same question and must give the same answer. One helper, not one per
    caller: an AF_INET-only implementation on either side makes the two
    disagree about an IPv6 node. ``create_connection`` resolves the name and
    tries every address family, which is the behaviour both callers want.

    Never raises: a name that does not resolve, a host with no route and a
    closed port are all "not reachable" to both callers.
    """
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def detect_mac_lan_ip(config=None) -> str:
    """Return the orchestrator LAN IP that edge nodes can actually reach us on."""
    from orchestrator.cluster.k3d import mac_lan_ip, probe_host

    return mac_lan_ip(probe_host(config) if config is not None else None)


def _docker_responsive() -> bool:
    """True if `docker info` succeeds within 5s."""
    try:
        r = subprocess.run(
            ["docker", "info"], capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ensure_docker_running(log: SystemLogger, mgr: LifecycleManager | None = None, max_wait: int = 60) -> None:
    """Ensure Docker Desktop is responsive. Auto-launches on Mac if not.

    Raises RuntimeError if Docker doesn't come up within ``max_wait`` seconds,
    or CancelledOp if the user clicked Stop while we were waiting.
    """
    if _docker_responsive():
        return
    log.emit("info", "  Docker not responding — launching Docker Desktop")
    with contextlib.suppress(FileNotFoundError, subprocess.TimeoutExpired):
        subprocess.run(["open", "-a", "Docker"], check=False, timeout=10)

    waited = 0
    while waited < max_wait:
        _sleep_or_cancel(mgr, 2)
        waited += 2
        if _docker_responsive():
            log.emit("success", f"  Docker ready (waited ~{waited}s)")
            return
        if waited % 10 == 0:
            log.emit("info", f"  Still waiting for Docker ({waited}s/{max_wait}s)...")

    log.emit("error", f"Docker Desktop did not start within {max_wait}s.")
    retry = "click Start again." if mgr is not None else "re-run `aureax start`."
    log.emit("hint", f"Open Docker Desktop manually, wait for it to be ready, then {retry}")
    raise RuntimeError("Docker not running")


def _run_setup_all(log: SystemLogger, mgr: LifecycleManager | None = None, config=None) -> None:
    """Invoke ``scripts/setup-all.sh`` end-to-end, streaming output to ``log``.

    Auto-detects the orchestrator LAN IP and passes it via ``$MAC_IP``, which
    ``detect_mac_ip`` in scripts/lib/common.sh takes in place of its own
    routing-table probe — validating it identically — so both halves of a
    dashboard-driven run agree on one address. Raises RuntimeError on
    non-zero exit, or CancelledOp if the user pressed Stop.

    The subprocess is launched in its own session group so ``mgr.cancel()``
    can SIGTERM the whole tree (bash + ssh/scp/docker children), not just
    the bash parent.
    """
    if not SETUP_ALL_SCRIPT.exists():
        log.emit("error", f"setup-all.sh not found at {SETUP_ALL_SCRIPT} (cwd={Path.cwd()})")
        raise RuntimeError(f"setup-all.sh not found at {SETUP_ALL_SCRIPT}")

    # Docker Desktop must be running for the k3d cluster step.
    _ensure_docker_running(log, mgr=mgr)

    mac_ip = detect_mac_lan_ip(config)
    if not mac_ip:
        log.emit(
            "error",
            "Could not auto-detect the orchestrator LAN IP that reaches the nodes.",
        )
        log.emit("hint", "Connect to a network and retry, or override detection: MAC_IP=<addr> ./scripts/setup-all.sh")
        raise RuntimeError("could not auto-detect the orchestrator LAN IP")
    log.emit("info", f"  Orchestrator LAN IP: {mac_ip}")

    env = os.environ.copy()
    env["MAC_IP"] = mac_ip

    # A Stop that landed earlier in this phase must not spawn the bootstrap at
    # all — cheaper than starting it and killing it.
    _check_cancel(mgr)

    proc = subprocess.Popen(
        ["bash", str(SETUP_ALL_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Nothing may prompt: a `read` with no terminal hangs the run forever.
        stdin=subprocess.DEVNULL,
        text=True,
        # Explicit, because bare text=True decodes with the locale encoding —
        # not guaranteed UTF-8 for a service-spawned dashboard — and strictly,
        # so one bad byte would raise out of the read loop below and leave
        # proc.wait() blocked on a pipe nobody is draining.
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        start_new_session=True,  # so cancel() can killpg() the whole tree
    )
    if mgr is not None:
        mgr.subprocess = proc
        # Assign, then re-check: cancel() sets the event and then reads
        # .subprocess, so a Stop landing between _begin (which clears it) and
        # this assignment would signal nothing and let the whole 5-15 min
        # bootstrap run to completion.
        if mgr.cancel_event.is_set():
            mgr.cancel()

    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            level = _classify_setup_line(line)
            log.emit(level, line)
    finally:
        # Close before waiting: ANY exception out of the loop leaves the pipe
        # undrained, and the child then blocks writing into it forever.
        proc.stdout.close()
        rc = proc.wait()
        if mgr is not None:
            mgr.subprocess = None

    # If the user pressed Stop, the subprocess was SIGTERM'd by cancel() and
    # exited with a signal — surface this as cancellation, not failure.
    if mgr is not None and mgr.cancel_event.is_set():
        raise CancelledOp(f"setup-all.sh terminated by user (exit {rc})")
    if rc != 0:
        log.emit("error", f"setup-all.sh exited with code {rc}")
        raise RuntimeError(f"setup-all.sh failed (exit {rc})")


def _classify_setup_line(line: str) -> str:
    """Map setup-all.sh output prefixes to log levels."""
    if "[FAIL]" in line or line.startswith("ERROR") or " ERROR " in line:
        return "error"
    if "[WARN]" in line or line.startswith("WARNING"):
        return "warn"
    if "[OK]" in line:
        return "success"
    return "info"


# ---- Cancellation + diagnostics helpers --------------------------------------

def _check_cancel(mgr: LifecycleManager | None) -> None:
    """Raise CancelledOp if the user clicked Stop. No-op when mgr is None."""
    if mgr is not None and mgr.cancel_event.is_set():
        raise CancelledOp("operation cancelled by user")


def _cancel_hint(mgr: LifecycleManager | None) -> str:
    """How the operator ends a long wait, on whichever front end is running.

    The dashboard has a Stop button that sets ``cancel_event``. The CLI passes
    mgr=None and has no cancel path at all, so its waits are a plain
    ``time.sleep`` that only Ctrl-C interrupts — telling that operator to click
    a button names one that is not there.
    """
    return " — click Stop to cancel." if mgr is not None else " — press Ctrl-C to cancel."


def _sleep_or_cancel(mgr: LifecycleManager | None, seconds: float) -> None:
    """Sleep for up to ``seconds``; raise CancelledOp early if cancelled.

    Worker threads use this in place of ``time.sleep`` so a cancel signal
    from the main loop wakes them within milliseconds.
    """
    if mgr is None:
        time.sleep(seconds)
        return
    if mgr.cancel_event.wait(timeout=seconds):
        raise CancelledOp("operation cancelled by user")


def _diagnostic_snapshot(namespace: str, log: SystemLogger) -> tuple[dict[str, dict], dict[str, list]]:
    """One `get pods` + one `get events` for a whole diagnostic round.

    Fetching both per not-Ready pod costs two kubectl processes per pod every
    30s, run serially on the polling thread. Returns (pod item by name, events
    by involved pod name); either map is empty when its fetch failed, which the
    callers already treat as "nothing to report".
    """
    pods: dict[str, dict] = {}
    events: dict[str, list] = {}

    pod_data = _kubectl_json(["get", "pods"], namespace, log, what="pods")
    for item in (pod_data or {}).get("items", []):
        name = (item.get("metadata") or {}).get("name", "")
        if name:
            pods[name] = item

    event_data = _kubectl_json(["get", "events"], namespace, log, what="events")
    for item in (event_data or {}).get("items", []):
        name = (item.get("involvedObject") or {}).get("name", "")
        if name:
            events.setdefault(name, []).append(item)
    # A bulk read cannot use kubectl's `--sort-by .lastTimestamp`, and callers
    # take the last few, so sort here. Either timestamp may be absent.
    for items in events.values():
        items.sort(key=lambda e: e.get("lastTimestamp") or e.get("eventTime") or "")
    return pods, events


def _kubectl_json(args: list[str], namespace: str, log: SystemLogger, *, what: str) -> dict | None:
    """`kubectl -n <ns> <args> -o json`, or None with a warn on any failure."""
    try:
        result = subprocess.run(
            ["kubectl", "-n", namespace, *args, "-o", "json"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        log.emit("warn", f"  kubectl get {what} failed ({type(e).__name__}: {e})", source="pods")
        return None
    if result.returncode != 0:
        log.emit("warn", f"  kubectl get {what} failed: {result.stderr.strip()[:200]}", source="pods")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _log_pod_diagnostics(
    pod_name: str,
    namespace: str,
    log: SystemLogger,
    mgr: LifecycleManager | None = None,
    *,
    pod: dict | None = None,
    events: list | None = None,
    dumped: set[tuple[str, str, int]] | None = None,
) -> None:
    """Log the actual reason a pod isn't Ready: container state, K8s events,
    and — when a container has just crashed — the container's own stdout/stderr.

    ``pod`` and ``events`` come from :func:`_diagnostic_snapshot`, which reads
    the whole namespace once per round. ``dumped`` carries the
    (pod_uid, container, restart_count) triples already dumped in this wait, so
    a crash is reported once rather than every 30s for the rest of it.

    The container-stdout fetch is two-tier:
      1. ``kubectl logs <pod> --previous --tail=60`` (primary).
      2. SSH to the node running the pod and tail the newest
         ``/var/log/pods/<ns>_<pod>_<uid>/<container>/*.log`` (fallback used
         when kubectl returns 502/proxy errors — a node's kubelet API port is
         not always reachable from the k3d server, which surfaces as a 502 from
         the proxy).

    Container-stdout requires ``mgr`` (for the network config / SSH key store).
    Without ``mgr``, only the kubectl path is attempted; SSH fallback is skipped.
    """
    if not pod_name or pod is None:
        return

    phase = pod.get("status", {}).get("phase", "Unknown")
    node_name = (pod.get("spec") or {}).get("nodeName", "")
    pod_uid = (pod.get("metadata") or {}).get("uid", "")
    log.emit("info", f"  [{pod_name}] phase={phase} on node={node_name or '?'}", source="pods")

    # Containers whose lastState shows a previous termination, with the restart
    # count that identifies THAT crash; their stdout is fetched once each, after
    # the event dump below.
    crashed_containers: list[tuple[str, int]] = []

    for cs in pod.get("status", {}).get("containerStatuses", []) or []:
        cname = cs.get("name", "?")
        if cs.get("ready", False):
            continue  # only diagnose unhealthy containers
        restarts = cs.get("restartCount", 0)
        state = cs.get("state", {}) or {}
        if "running" in state:
            started = state["running"].get("startedAt", "?")
            log.emit("warn", f"  [{pod_name}/{cname}] running since {started}, but readiness probe not passing yet (restarts={restarts})", source="pods")
        elif "waiting" in state:
            w = state["waiting"]
            reason = w.get("reason", "unknown")
            msg = (w.get("message") or "")[:200]
            log.emit("error", f"  [{pod_name}/{cname}] waiting: {reason} — {msg} (restarts={restarts})", source="pods")
        elif "terminated" in state:
            t = state["terminated"]
            log.emit("error", f"  [{pod_name}/{cname}] terminated: {t.get('reason', '?')} (exit {t.get('exitCode', '?')}) — {(t.get('message') or '')[:200]} (restarts={restarts})", source="pods")
        last = (cs.get("lastState") or {}).get("terminated") or {}
        if last:
            log.emit("warn", f"  [{pod_name}/{cname}] previously terminated: {last.get('reason', '?')} (exit {last.get('exitCode', '?')})", source="pods")
            crashed_containers.append((cname, restarts))

    # Recent K8s events for this pod — readiness probe failures, image pulls,
    # scheduling decisions, OOMKilled signals. Trim to the last 3 to avoid spam.
    for e in (events or [])[-3:]:
        etype = e.get("type", "?")  # Normal | Warning
        reason = e.get("reason", "?")
        message = (e.get("message") or "")[:240]
        level = "warn" if etype == "Warning" else "info"
        log.emit(level, f"  [{pod_name}] event {etype}/{reason}: {message}", source="pods")

    # For each crashed container, fetch its previous stdout/stderr — the
    # diagnostic that otherwise means SSH'ing into the node by hand.
    for cname, restarts in crashed_containers:
        key = (pod_uid or pod_name, cname, restarts)
        if dumped is not None and key in dumped:
            continue
        ok = _log_crashed_container_stdout(
            pod_name=pod_name,
            pod_uid=pod_uid,
            container=cname,
            namespace=namespace,
            node_name=node_name,
            mgr=mgr,
            log=log,
        )
        # Recorded only after a dump actually landed: a transient `kubectl logs`
        # failure must not suppress the real crash logs for this restart. A new
        # crash increments restartCount, so it is a new key.
        if ok and dumped is not None:
            dumped.add(key)


def _log_crashed_container_stdout(
    *,
    pod_name: str,
    pod_uid: str,
    container: str,
    namespace: str,
    node_name: str,
    mgr: LifecycleManager | None,
    log: SystemLogger,
) -> bool:
    """Fetch container stdout for a recently-crashed container.

    Tries kubectl logs --previous first; falls back to SSH-reading the newest
    /var/log/pods/<ns>_<pod>_<uid>/<container>/*.log file on the node. The
    latter is needed when a node's kubelet API is not reachable from the k3d
    server, which surfaces as a 502 Bad Gateway from the proxy.

    Returns True when stdout was actually emitted, so the caller only records
    this crash as dumped once it really has been.
    """
    # 1) Primary: kubectl logs --previous
    try:
        rc_result = subprocess.run(
            ["kubectl", "-n", namespace, "logs", pod_name, "-c", container,
             "--previous", "--tail=60"],
            capture_output=True, text=True, check=False, timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        # A diagnostic must never end the operation it is diagnosing, and an
        # orchestrator without kubectl on PATH is a configuration to report,
        # not a traceback out of a crash dump: fall through to the SSH read.
        rc_result = None

    if rc_result is not None and rc_result.returncode == 0 and rc_result.stdout.strip():
        log.emit("error", f"  [{pod_name}/{container}] last-instance stdout (kubectl logs --previous, tail=60):", source="pods")
        for line in rc_result.stdout.splitlines()[-60:]:
            log.emit("error", f"    {line}", source="pods")
        return True

    # 2) Fallback: SSH-read the newest
    #    /var/log/pods/<ns>_<pod>_<uid>/<container>/<latest>.log
    err_hint = ""
    if rc_result is not None:
        err_hint = rc_result.stderr.strip()[:160]
    if mgr is None:
        log.emit("warn", f"  [{pod_name}/{container}] kubectl logs failed ({err_hint or 'no output'}); SSH fallback unavailable (no mgr)", source="pods")
        return False

    log.emit("info", f"  [{pod_name}/{container}] kubectl logs failed ({err_hint or 'no output'}) — falling back to SSH /var/log/pods on {node_name or '?'}", source="pods")
    return _log_via_ssh(
        pod_name=pod_name,
        pod_uid=pod_uid,
        container=container,
        namespace=namespace,
        node_name=node_name,
        mgr=mgr,
        log=log,
    )


def _log_via_ssh(
    *,
    pod_name: str,
    pod_uid: str,
    container: str,
    namespace: str,
    node_name: str,
    mgr: LifecycleManager,
    log: SystemLogger,
) -> bool:
    """SSH-read /var/log/pods/<ns>_<pod>_<uid>/<container>/<latest>.log.

    K3s's kubelet writes container stdout there as JSON-lines; we strip the
    JSON wrapper before logging. Returns True only when lines were emitted.
    """
    if not pod_uid:
        log.emit("warn", f"  [{pod_name}/{container}] SSH fallback skipped (no pod UID)", source="pods")
        return False
    if not node_name:
        log.emit("warn", f"  [{pod_name}/{container}] SSH fallback skipped (no node name)", source="pods")
        return False

    from orchestrator.config.loader import load_network_config
    from orchestrator.deploy.ssh import conn_kwargs, run_command

    try:
        cfg = load_network_config(str(mgr.network_config_path))
    except Exception as e:
        log.emit("warn", f"  [{pod_name}/{container}] SSH fallback: could not load network config — {e}", source="pods")
        return False

    target = _resolve_ssh_target_for_node(cfg, node_name)
    if target is None:
        log.emit("warn", f"  [{pod_name}/{container}] SSH fallback: no SSH target matches K8s node '{node_name}'", source="pods")
        return False

    log_glob = f"/var/log/pods/{namespace}_{pod_name}_{pod_uid}/{container}"
    # Pick the most recently modified .log (the previous container's log if it
    # has rotated, or the current one). `tail -60` keeps it compact.
    #
    # `namespace` is an unconstrained config string, so the directory is quoted
    # and the whole script quoted again rather than interpolated into a
    # `sudo bash -c '...'` literal. The `/*.log` stays outside the quoting so
    # the remote shell still expands the glob.
    inner = (
        f"f=$(ls -t {shlex.quote(log_glob)}/*.log 2>/dev/null | head -1); "
        '[ -n "$f" ] && tail -60 "$f" || echo NO_LOGS'
    )
    cmd = "sudo bash -c " + shlex.quote(inner)
    try:
        rc, stdout, stderr = run_command(
            target.host, cmd, **conn_kwargs(target, cfg.ssh.key_store, timeout=15)
        )
    except Exception as e:
        # A diagnostic must never end the operation it is diagnosing. This runs
        # inside the pod-readiness wait, against a node that has already
        # crash-looped once and may itself have gone unreachable since.
        log.emit("warn", f"  [{pod_name}/{container}] SSH fallback failed: {e}", source="pods")
        return False
    if rc != 0:
        log.emit("warn", f"  [{pod_name}/{container}] SSH fallback failed (rc={rc}): {(stderr or '').strip()[:200]}", source="pods")
        return False
    if not stdout or stdout.strip() == "NO_LOGS":
        log.emit("warn", f"  [{pod_name}/{container}] SSH fallback: no log files at {log_glob}/*.log", source="pods")
        return False

    log.emit("error", f"  [{pod_name}/{container}] last-instance stdout (via SSH on {target.host}, tail=60):", source="pods")
    for raw in stdout.splitlines()[-60:]:
        # Each line is JSON: {"log":"...","stream":"stdout","time":"..."}.
        # Fall back to raw if it doesn't parse.
        try:
            entry = json.loads(raw)
            line = (entry.get("log") or "").rstrip("\n")
        except (json.JSONDecodeError, AttributeError):
            line = raw
        log.emit("error", f"    {line}", source="pods")
    return True


_NODE_TO_TARGET_CACHE: dict[str, str] = {}  # K8s node hostname → ssh target node_id

def _resolve_ssh_target_for_node(config, node_name: str):
    """Find the ssh.target whose host's `hostname` matches the K8s node name.

    K8s node names are the OS hostname of the joined node (e.g. "debian",
    "node1"), while the network YAML has hosts as IPs — so the match goes
    through each node's SSH `hostname`. Built on _resolve_node_hostname so
    both resolution paths share ONE hostname cache — a second SSH loop with a
    cache of its own can disagree with the first. The map here only remembers
    which target's node_id a name matched.

    The node_id, not the target object: the caller reloads the network config
    on every call precisely to pick up edits, so caching the object would keep
    dialling the host from whichever load first populated this map — a host
    remap in the YAML would never reach a long-running dashboard.
    """
    if not node_name:
        return None
    cached_id = _NODE_TO_TARGET_CACHE.get(node_name)
    if cached_id is not None:
        target = next((t for t in config.ssh.enabled_targets if t.node_id == cached_id), None)
        if target is not None:
            return target
        # Removed, renamed or since disabled — rescan rather than leaving this
        # node permanently unresolvable over one config edit.

    for target in config.ssh.enabled_targets:
        if _resolve_node_hostname(config, target.node_id) == node_name:
            _NODE_TO_TARGET_CACHE[node_name] = target.node_id
            return target

    # Deliberately not caching the miss: a node that was merely unreachable
    # this pass would otherwise stay unresolvable for the whole process
    # lifetime — unlike _hostname_cache, this map has no TTL, so a cached miss
    # would never expire.
    return None


def _get_daemonset_node_selector(
    namespace: str, label: str, daemonset: str = "aureax-edge"
) -> str:
    """Read ``spec.template.spec.nodeSelector[label]`` on a DaemonSet.

    Returns "" if unset or on error. Used to make Start idempotent: if the
    selector already matches, don't re-patch (a no-op for K8s, but misleading
    in the dashboard log).

    Through ``orchestrator.cluster.kubectl`` rather than its own subprocess
    call: that module is where kubectl's argv, its ``--request-timeout`` and
    the "{} on any failure" contract live, and a hung API request bounded only
    by a local process timeout is exactly what the shared wrapper avoids. A
    missing kubectl binary arrives as ``CommandError``, which is still "cannot
    tell", i.e. "".
    """
    from orchestrator.cluster import kubectl
    from orchestrator.cluster._subprocess import CommandError

    try:
        ds = kubectl.get(f"daemonset/{daemonset}", namespace)
    except (CommandError, ValueError):
        return ""
    try:
        return (ds.get("spec", {}).get("template", {}).get("spec", {}).get("nodeSelector", {}) or {}).get(label, "")
    except AttributeError:
        return ""
