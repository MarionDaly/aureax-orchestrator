"""REST API routes for the Aureax dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from itertools import islice

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from orchestrator.cluster import kubectl
from orchestrator.config.cache import load_node_config_resolved
from orchestrator.config.schema import mcp_server_enabled, node_role
from orchestrator.dashboard._http_utils import (
    network_config_path,
    read_capped_text,
    target_for,
)
from orchestrator.logs.receiver import ingest as ingest_logs
from orchestrator.telemetry.queries import MetricsQueries
from orchestrator.telemetry.receiver import ingest as ingest_metrics
from orchestrator.telemetry.storage import MetricsStorage

logger = logging.getLogger("aureax.dashboard")

router = APIRouter()

# The two routes edge nodes push to. Kept on their own router because their
# ingest is the only reason this server binds 0.0.0.0 — grouping them keeps
# that visible. Nothing else belongs here.
ingest_router = APIRouter()

# Every /api/cluster and /api/pods hit shells out to kubectl, and /api/nodes
# opens a SQLite connection. The Overview tab polls all three on a 10s timer,
# so N open tabs cost N times that work per cycle for an answer identical
# across them. A short TTL collapses concurrent viewers onto one call without
# making the view meaningfully staler — same approach, and roughly the same
# window, as lifecycle._nodes_status.
_KUBECTL_TTL = 3.0
_kubectl_cache: dict[str, tuple[float, object]] = {}
_kubectl_lock = threading.Lock()
_kubectl_key_locks: dict[str, threading.Lock] = {}


def _fresh(key: str):
    """The cache entry for ``key`` if still inside its TTL, else None.

    Caller must hold _kubectl_lock.
    """
    hit = _kubectl_cache.get(key)
    if hit is not None and time.monotonic() - hit[0] < _KUBECTL_TTL:
        return hit
    return None


def _cached(key: str, fetch):
    """Return `fetch()`'s result, reused for up to _KUBECTL_TTL seconds.

    Handlers here are sync, so FastAPI runs them in a threadpool and several
    can land at once. Single-flight is PER KEY: the fetch runs under a lock so
    concurrent misses collapse onto one kubectl (the duplication this cache
    exists to remove), while "nodes", "pods", "readyz" and the SQLite-backed
    "nodes-metrics:<db>" never queue behind each other. One global lock would
    make the Overview's parallel /api/cluster + /api/pods + /api/nodes cost
    their SUM, serialising them behind whichever key is slowest. Each fetch is
    itself bounded — the kubectl reads this cache makes (get_nodes, get_pods,
    api_reachable) all carry --request-timeout — so the per-key lock is about
    parallelism, not about surviving a wedged one.

    The entry is stamped AFTER the fetch returns, so a call that took a second
    does not arrive already a second into its own TTL.
    """
    with _kubectl_lock:
        hit = _fresh(key)
        if hit is not None:
            return hit[1]
        key_lock = _kubectl_key_locks.setdefault(key, threading.Lock())
    with key_lock:
        with _kubectl_lock:
            # Re-check: whoever held this key's lock has just published.
            hit = _fresh(key)
            if hit is not None:
                return hit[1]
        value = fetch()
        with _kubectl_lock:
            _kubectl_cache[key] = (time.monotonic(), value)
        return value


def _queries(request: Request) -> MetricsQueries | None:
    return getattr(request.app.state, "metrics_queries", None)


def _storage(request: Request) -> MetricsStorage | None:
    return getattr(request.app.state, "metrics_storage", None)


@router.get("/api/nodes")
def list_nodes(request: Request) -> list[dict]:
    """List all nodes with status and latest metrics.

    TTL-cached like the kubectl siblings it is fetched alongside: each call
    opens a SQLite connection and scans the metrics table, and every open tab
    asks for the same answer every 10 seconds. Keyed by database path so two
    apps in one process (tests) never read each other's cache.
    """
    queries = _queries(request)
    if not queries:
        return []
    return _cached(f"nodes-metrics:{queries.db_path}", lambda: _collect_nodes(queries))


def _collect_nodes(queries: MetricsQueries) -> list[dict]:
    """Every reporting node with its newest value per metric.

    ONE connection and ONE query for the whole fleet: this route is polled by
    every open dashboard tab, over a table an active fleet grows by thousands
    of rows an hour.
    """
    by_node: dict[str, dict] = {}
    # Rows arrive ordered by node_id, which is what lets a single pass group
    # them — and, since dicts keep insertion order, what orders the fleet
    # listing the operator sees.
    for row in queries.get_latest_all():
        node = by_node.get(row["node_id"])
        if node is None:
            node = by_node[row["node_id"]] = {
                "node_id": row["node_id"], "last_seen": None, "metrics": {},
            }
        node["metrics"][row["metric_name"]] = row["metric_value"]
        # MAX over `received_at`, and never over the node-supplied `timestamp`.
        #
        # MAX because these rows are the newest value PER METRIC ordered by
        # metric_name: row 0 is whichever metric sorts first alphabetically,
        # not the most recent sample, so a node that stops reporting one metric
        # while still pushing the others would report the stale one.
        #
        # `received_at` because the dashboard's "online" dot is a 120s
        # freshness check on this field, and `timestamp` is copied verbatim
        # from the pushed JSONL: a node whose clock runs ahead pins the dot
        # green until its rows age out, and a junk value pins it red. The
        # ingest column is server-stamped, so no poster can move it — the same
        # reason MetricsStorage.rotate prunes on it.
        if node["last_seen"] is None or row["received_at"] > node["last_seen"]:
            node["last_seen"] = row["received_at"]
    return list(by_node.values())


_UNREACHABLE = {"status": "error", "message": "kubectl API server unreachable"}


def api_reachable_cached() -> bool:
    """Cached /readyz probe. THE one in this process.

    ``kubectl.get`` runs with check=False and returns ``{}`` for any failure,
    so an unreachable API server is indistinguishable from an empty cluster at
    that call — and reporting it as "ok, zero nodes" tells the operator the
    fleet is gone rather than that we could not ask. The background watchers
    guard the same ambiguity and call this rather than caching it themselves:
    a second cache would mean a second TTL, a second lock, and a tick that
    could still spawn two `kubectl get --raw /readyz` at once.

    Public for that reason. Blocking — the watchers run it via ``to_thread``.
    ``_cached`` locks per key, so a watcher tick waits only on another readyz
    probe, never on the "nodes"/"pods" fetches a dashboard tab is driving.
    """
    return bool(_cached("readyz", kubectl.api_reachable))


@router.get("/api/cluster")
def cluster_status() -> dict:
    """Get K3s cluster status."""
    try:
        if not api_reachable_cached():
            return dict(_UNREACHABLE)
        data = _cached("nodes", lambda: kubectl.get("nodes"))
    except Exception as e:
        return {"status": "error", "message": str(e)}
    if not data or "items" not in data:
        return {"status": "ok", "nodes": []}

    nodes = [
        {
            "name": node.get("metadata", {}).get("name", ""),
            "status": _ready_status(node),
            "labels": node.get("metadata", {}).get("labels", {}),
        }
        for node in data.get("items", [])
    ]
    return {"status": "ok", "nodes": nodes}


def _ready_status(node: dict) -> str:
    """A K8s node's Ready condition: "Ready" | "NotReady" | "Unknown".

    Null-safe at every level, which is a behaviour change for this listing: a
    node object with no ``status`` block, or a condition with no ``type``,
    raised a KeyError from inside the loop — which the handler's try does not
    cover, so ONE malformed node 500'd the whole listing while every other node
    was fine. It now reads "Unknown", which is what that field already means
    for a node whose Ready condition has not been reported yet.
    """
    for cond in (node.get("status") or {}).get("conditions", []):
        if cond.get("type") == "Ready":
            return "Ready" if cond.get("status") == "True" else "NotReady"
    return "Unknown"


@router.get("/api/pods")
def pod_status() -> dict:
    """Get Aureax pod status from the 'aureax' namespace.

    Returns per-pod details including:
    - phase: the K8s pod phase (Running, Pending, Failed, etc.)
    - ready: whether all containers passed readiness probes
    - restarts: total container restart count (high count = crash loop)
    - node: which K3s node the pod is scheduled on
    - containers: per-container {name, ready, restarts, state, last_error},
      where last_error is the last terminated reason + exit code. It is a
      per-CONTAINER field: there is no pod-level last_error key.

    This gives visibility into WHY pods aren't working — e.g. model file
    missing, shared library errors, or llama-server crashes.
    """
    try:
        if not api_reachable_cached():
            return dict(_UNREACHABLE)
        data = _cached("pods", lambda: kubectl.get("pods", namespace="aureax"))
    except Exception as e:
        return {"status": "error", "message": str(e)}
    if not data or "items" not in data:
        return {"status": "ok", "pods": []}

    pods = [_parse_pod(p) for p in data.get("items", [])]
    return {"status": "ok", "pods": pods}


def _parse_pod(pod: dict) -> dict:
    """Extract useful status info from a raw K8s pod object.

    Focuses on the fields that help diagnose startup failures:
    - container state (waiting/running/terminated)
    - restart count and last exit code
    - reason and message for crashes
    """
    metadata = pod.get("metadata", {})
    spec = pod.get("spec", {})
    status = pod.get("status", {})

    # Aggregate container statuses
    containers = []
    for cs in status.get("containerStatuses", []):
        container = {
            "name": cs.get("name", ""),
            "ready": cs.get("ready", False),
            "restarts": cs.get("restartCount", 0),
            "state": "unknown",
            "last_error": None,
        }

        # Current state: one of waiting, running, or terminated
        state = cs.get("state", {})
        if "running" in state:
            container["state"] = "running"
        elif "waiting" in state:
            # Waiting = not started yet. Reason tells us why.
            # Common reasons: CrashLoopBackOff, ImagePullBackOff, ContainerCreating
            waiting = state["waiting"]
            container["state"] = waiting.get("reason", "waiting")
            container["last_error"] = waiting.get("message", None)
        elif "terminated" in state:
            # Terminated = exited. Reason + exit code tell us what happened.
            term = state["terminated"]
            container["state"] = f"exited ({term.get('exitCode', '?')})"
            container["last_error"] = term.get("reason", None)

        # Also check lastState for the most recent crash info
        # (when a container restarts, current state becomes "running" or "waiting"
        #  but lastState preserves why it died last time)
        last_state = cs.get("lastState", {})
        if "terminated" in last_state and not container["last_error"]:
            term = last_state["terminated"]
            exit_code = term.get("exitCode", "?")
            reason = term.get("reason", "unknown")
            container["last_error"] = f"{reason} (exit {exit_code})"

        containers.append(container)

    # Overall pod readiness: all containers must be ready
    all_ready = all(c["ready"] for c in containers) if containers else False
    total_restarts = sum(c["restarts"] for c in containers)

    return {
        "name": metadata.get("name", ""),
        "node": spec.get("nodeName", ""),
        "phase": status.get("phase", "Unknown"),
        "ready": all_ready,
        "restarts": total_restarts,
        "containers": containers,
    }


# --- per-node detail ----------------------------------------------------------
# One node's whole story on one page, joined from four sources that are ALREADY
# collected: the config caches, the "nodes"/"pods" kubectl entries the Overview
# grids read, the fleet metrics listing /api/nodes serves, and the log store.
# The join key is the node id — telemetry reports under ssh.targets[].node_id
# because templating resolves ${node_id} into the config the collector reads.

# How many of a node's shipped log lines the detail view carries. The store
# holds 30 days of them and the whole payload is re-sent on every 10 s poll, so
# this is a tail, not an archive: the Lifecycle console is where the full
# fleet-wide stream lives.
_NODE_LOG_TAIL = 100

# The saturation half of a node's condition list — the USE questions about the
# box itself. Ready is reported separately as `status`; nothing else in that
# list says anything about this node's capacity to keep running its pod.
_PRESSURE_CONDITIONS = ("MemoryPressure", "DiskPressure", "PIDPressure")


@router.get("/api/nodes/{node_id}")
def node_detail(request: Request, node_id: str) -> dict:
    """Identity, K8s placement, metric history and shipped logs for ONE node.

    No new fleet I/O: the kubectl reads go through the SAME cache keys the
    Overview grids fill, and the latest metrics come from the same fleet
    listing /api/nodes serves, so an open detail tab costs one SQLite history
    query beyond the grids it was opened from.

    Each of the four sections carries its own status. A stopped cluster or an
    unopened store then blanks one card instead of 500ing the page — which is
    exactly when an operator opens this view.

    Cached per node like its siblings: the tab polls on a 10 s timer. Keying
    the cache on a path parameter is safe only because the gate runs FIRST —
    an unknown or disabled id is a 404 and never reaches the cache, so no
    caller can grow the key space. Qualified by the fleet file like the
    "nodes-metrics:<db>" key beside it, so two apps in one process (tests)
    never read each other's answer for the same node id.
    """
    cfg, target = target_for(request, node_id)
    config_path = network_config_path(request)
    return _cached(f"node-detail:{config_path}:{node_id}", lambda: {
        "node_id": node_id,
        "identity": _identity(cfg, target, config_path.parent),
        "k8s": _k8s_detail(node_id, target.host),
        "metrics": _node_metrics(request, node_id),
        "logs": _node_logs(request, node_id),
    })


def _resolved(value):
    """A config value, or None when it is still an unresolved ``${...}``.

    ``load_node_config_resolved`` falls back to the SOURCE TEMPLATE when a
    fleet has never run `aureax generate`, and a template spells its per-node
    values as placeholders. Rendering the literal "${context_size}" as this
    node's context is worse than rendering nothing: it reads as a fact.
    """
    return None if isinstance(value, str) and "${" in value else value


def _identity(cfg, target, configs_dir) -> dict:
    """Who this node is, entirely from config — no fleet I/O at all.

    Host, display name and tags come from the fleet file rather than from the
    node's own config, because that is where they are literals: the per-node
    config carries them as ${display_name} / ${tags}, resolved FROM the target.

    The model block is shown for every role, workers included. A worker holds a
    share of exactly that model's weights, so its template naming it is not a
    mistake to hide — it is the group's model seen from the worker's side.
    """
    node_cfg = load_node_config_resolved(configs_dir, target)
    model = node_cfg.get("model") or {}
    server = model.get("server") or {}
    group = cfg.group_for_node(target.node_id)
    return {
        "display_name": target.display_name,
        "host": target.host,
        "tags": list(target.tags),
        # "" when the config could not be read at all, matching node_role's own
        # contract; every real role is one of the three schema literals.
        "role": node_role(node_cfg),
        "mcp_enabled": mcp_server_enabled(node_cfg),
        "group": None if group is None else {
            "name": group.name,
            "head": group.head,
            "workers": list(group.workers),
        },
        "model": {
            "name": _resolved(model.get("name")),
            "filename": _resolved(model.get("filename")),
            "quantization": _resolved(model.get("quantization")),
            "context_size": _resolved(server.get("context_size")),
            "gpu_layers": _resolved(server.get("gpu_layers")),
        },
    }


def _k8s_detail(node_id: str, host: str) -> dict:
    """This node as K8s sees it: the node object, and the pods on it.

    Reads the SAME two cache entries as /api/cluster and /api/pods, so opening
    a detail view adds no kubectl calls to a tab already showing the grids.
    """
    try:
        if not api_reachable_cached():
            return {**_UNREACHABLE, "matched_by": None, "node": None, "pods": []}
        nodes = (_cached("nodes", lambda: kubectl.get("nodes")) or {}).get("items", [])
        pods = (_cached("pods", lambda: kubectl.get("pods", namespace="aureax")) or {}).get("items", [])
    except Exception as e:
        return {"status": "error", "message": str(e),
                "matched_by": None, "node": None, "pods": []}

    node, k8s_name, matched_by = _match_k8s_node(nodes, pods, node_id, host)
    return {
        "status": "ok",
        "message": None,
        "matched_by": matched_by,
        "node": None if node is None else _node_summary(node),
        "pods": [_pod_detail(p) for p in pods if _pod_is_on(p, k8s_name, host)],
    }


def _match_k8s_node(nodes: list, pods: list, node_id: str, host: str):
    """Which K8s node a target IS. Returns ``(node | None, name | None, how | None)``.

    There is no field linking the two: a K8s node is named by its hostname and
    an SSH target by its address, and resolving one to the other otherwise
    costs an SSH round trip (lifecycle does exactly that, over minutes). The IP
    is free and correct for this fleet — both DaemonSets are hostNetwork and
    every target is addressed by IP.

    Three attempts, most to least certain. The pod fallback exists because a
    node can be missing from the node listing while its pods are still
    reported, and it yields no node object by design: what matched was a pod's
    hostIP, which says where the pods run and nothing about the node.
    ``matched_by`` is returned so the view can say which of the three it was
    rather than presenting a guess as an identification.
    """
    for node in nodes:
        for addr in (node.get("status") or {}).get("addresses", []):
            if addr.get("type") == "InternalIP" and addr.get("address") == host:
                return node, node.get("metadata", {}).get("name", ""), "internal-ip"
    for pod in pods:
        if (pod.get("status") or {}).get("hostIP") == host:
            return None, (pod.get("spec") or {}).get("nodeName", ""), "host-ip"
    for node in nodes:
        name = node.get("metadata", {}).get("name", "")
        if name in (node_id, host):
            return node, name, "name"
    return None, None, None


def _pod_is_on(pod: dict, k8s_name: str | None, host: str) -> bool:
    """Whether a pod runs on the matched node.

    Either half alone misses a real case: a pod that has been scheduled but has
    no hostIP yet (Pending, ContainerCreating) is known only by nodeName, and a
    node identified by its pods' hostIP is the one case where the name may be
    empty. ``k8s_name`` is checked truthy first, or an unmatched node would
    collect every pod whose nodeName is likewise absent.
    """
    spec = pod.get("spec") or {}
    return (bool(k8s_name) and spec.get("nodeName") == k8s_name) or (
        (pod.get("status") or {}).get("hostIP") == host
    )


def _node_summary(node: dict) -> dict:
    """The parts of a K8s node object worth showing: what it runs on, how big
    it is, and whether the kubelet is under pressure.

    Everything here is already in the cached listing and thrown away by
    /api/cluster, which keeps only name, Ready and labels for its grid.
    """
    status = node.get("status") or {}
    info = status.get("nodeInfo") or {}
    capacity = status.get("capacity") or {}
    return {
        "name": node.get("metadata", {}).get("name", ""),
        "status": _ready_status(node),
        "kernel_version": info.get("kernelVersion", ""),
        "os_image": info.get("osImage", ""),
        "architecture": info.get("architecture", ""),
        "kubelet_version": info.get("kubeletVersion", ""),
        "container_runtime": info.get("containerRuntimeVersion", ""),
        "cpu_capacity": capacity.get("cpu", ""),
        "memory_capacity": capacity.get("memory", ""),
        "pressure": [
            cond.get("type") for cond in status.get("conditions", [])
            if cond.get("type") in _PRESSURE_CONDITIONS and cond.get("status") == "True"
        ],
    }


def _pod_detail(pod: dict) -> dict:
    """``_parse_pod`` plus the timing the fleet grid has no room for.

    The grid answers "is it up"; this answers "since when, and what killed it
    last time" — the pair that separates a pod up since deploy from one in a
    crash loop, which look identical at Running/0-ready.

    Deliberately no image or imageID: every node builds ``aureax-client:dev``
    natively, so those differ between nodes by construction and reading them as
    drift is a false positive (see ARCHITECTURE.md, "Invariants and how each is
    enforced"). No podIP either — both DaemonSets are hostNetwork, so it only
    ever repeats host_ip.
    """
    status = pod.get("status") or {}
    detail = _parse_pod(pod)
    detail["host_ip"] = status.get("hostIP", "")
    detail["start_time"] = status.get("startTime", "")
    by_name = {cs.get("name", ""): cs for cs in status.get("containerStatuses", [])}
    for container in detail["containers"]:
        cs = by_name.get(container["name"], {})
        state = cs.get("state") or {}
        # The CURRENT process's start, which is what "up 4h" means and what the
        # pod's own startTime stops tracking the moment a container restarts.
        # Only a running or terminated state carries one; a waiting container
        # has not started anything yet.
        started = state.get("running") or state.get("terminated") or {}
        container["started_at"] = started.get("startedAt", "")
        # WHEN the previous run ended, and only that: its reason and exit code
        # are already in ``last_error``, which is the one thing that string
        # cannot say. None rather than "" so the view has a single absent test.
        last_term = (cs.get("lastState") or {}).get("terminated") or {}
        container["last_finished_at"] = last_term.get("finishedAt") or None
    return detail


def _node_metrics(request: Request, node_id: str) -> dict:
    """This node's newest value per metric, plus the last hour of samples.

    ``latest`` and ``last_seen`` are lifted out of the same fleet listing
    /api/nodes serves rather than re-queried per node: it is already cached,
    and taking them from anywhere else would let the detail view's freshness
    dot disagree with the Overview card the operator clicked to get here.

    A database that goes bad under a running dashboard — locked, truncated,
    mid-write on a machine that lost power — reports as "unavailable" like an
    unopened one. Blanking this card is what the route promises; raising would
    take identity and the K8s placement down with it, which is the pair an
    operator opened this view to read.
    """
    queries = _queries(request)
    if queries is not None:
        try:
            entry = next((n for n in list_nodes(request) if n["node_id"] == node_id), {})
            return {
                "status": "ok",
                "last_seen": entry.get("last_seen"),
                "latest": entry.get("metrics", {}),
                "history": queries.get_metric_history(node_id),
            }
        except Exception as e:
            logger.warning("metrics unreadable for %s: %s", node_id, e)
    return {"status": "unavailable", "last_seen": None, "latest": {}, "history": {}}


def _node_logs(request: Request, node_id: str) -> dict:
    """The tail of what this node shipped, newest first.

    ``node:<id>`` is the source the log receiver tags every ingested entry
    with, and for a shard worker this is the ONLY log surface there is: it runs
    rpc-server and no HTTP server, and `kubectl logs` cannot reach a LAN-joined
    node's kubelet from the k3d server container (see ARCHITECTURE.md,
    "Observability: how a node's failure reaches the orchestrator").

    A store fault reports as "unavailable", the same as an unopened one: this
    card is the least of the four, and losing the whole page to it would hide
    the identity and placement the operator came here for.
    """
    store = getattr(request.app.state, "log_store", None)
    if store is not None:
        try:
            entries = store.recent(_NODE_LOG_TAIL, source=f"node:{node_id}")
            return {
                "status": "ok",
                "entries": [
                    {"ts": e.ts, "level": e.level, "text": e.text} for e in entries
                ],
            }
        except Exception as e:
            logger.warning("logs unreadable for %s: %s", node_id, e)
    return {"status": "unavailable", "entries": []}


@ingest_router.post("/api/v1/telemetry")
async def receive_telemetry(request: Request) -> JSONResponse:
    """Receive JSONL telemetry from edge nodes."""
    storage = _storage(request)
    if not storage:
        return JSONResponse({"error": "storage not initialized"}, status_code=503)

    text = await read_capped_text(request)
    log = getattr(request.app.state, "system_log", None)
    # Off the loop: parsing the batch and the INSERT (plus the hourly prune it
    # can trigger) are blocking work, and this handler runs on the event loop
    # that also serves every SSE stream.
    count = await asyncio.to_thread(_ingest_telemetry, text, storage, log)
    return JSONResponse({"ingested": count})


def _ingest_telemetry(text: str, storage: MetricsStorage, log) -> int:
    """Insert a shipped batch, then summarise it. Returns the count ingested.

    The rollup is built from the receiver's own records, so the body is JSON-
    parsed once. Safe off the loop: SystemLogger.emit is thread-safe and fans
    out via call_soon_threadsafe.
    """
    count, records = ingest_metrics(text, storage)
    _emit_telemetry_rollup(log, records)
    return count


@ingest_router.post("/api/v1/logs")
async def receive_logs(request: Request) -> JSONResponse:
    """Ingest JSONL log entries shipped by an edge node.

    Deliberately a sibling of /api/v1/telemetry rather than something clever:
    it is the one transport from edge to orchestrator that is known to work in
    this topology. `kubectl logs` proxies through each node's kubelet on :10250,
    which the k3d server container cannot reach for any LAN-joined node, and a
    shard worker serves no HTTP of its own — so without this a worker crash is
    only visible over SSH.
    """
    store = getattr(request.app.state, "log_store", None)
    if not store:
        return JSONResponse({"error": "log store not initialized"}, status_code=503)

    text = await read_capped_text(request)
    # Through the SystemLogger so shipped lines also reach the live SSE console,
    # not just the database behind it. Off the loop, and both counts come back
    # from the one parse — `expected` from a second pass would JSON-parse every
    # push twice.
    count, expected = await asyncio.to_thread(
        ingest_logs, text, store, getattr(request.app.state, "system_log", None)
    )

    # 500 on a partial write, so the shipper does NOT advance its cursor.
    # LogReporter treats any non-200 as a failure and retries the same batch;
    # answering 200 for entries we failed to persist loses them permanently,
    # and they are exactly the entries that explain a crash.
    if count < expected:
        return JSONResponse({"ingested": count, "expected": expected}, status_code=500)
    return JSONResponse({"ingested": count})


# Bound on how many summary lines one push can produce. Every emit is a
# durable row (a connection and an fsync each) plus an SSE frame to every open
# console, and /api/v1/telemetry is unauthenticated on 0.0.0.0 with only a body
# cap above it — so without this, one accepted 8 MB batch of distinct node ids
# buys tens of thousands of transactions on the ingest thread. 32 covers any
# real fleet; the rest are counted in one trailing line.
_MAX_ROLLUP_NODES = 32


def _emit_telemetry_rollup(log, records: list[dict]) -> None:
    """One summary line per node per push: 'edge-01 reported 6 metrics (cpu 24%)'."""
    if log is None:
        return
    per_node: dict[str, dict] = {}
    for r in records:
        nid = r.get("node_id", "?")
        slot = per_node.setdefault(nid, {"count": 0, "hints": {}})
        slot["count"] += 1
        name = r.get("metric_name", "")
        value = r.get("metric_value")
        # Pick a couple of human-useful hints for the log line
        if isinstance(value, (int, float)) and name in ("cpu_percent", "memory_percent", "temperature_c"):
            slot["hints"][name] = value
    for nid, slot in islice(per_node.items(), _MAX_ROLLUP_NODES):
        hints = slot["hints"]
        bits = []
        if "cpu_percent" in hints:
            bits.append(f"cpu {hints['cpu_percent']:.0f}%")
        if "memory_percent" in hints:
            bits.append(f"mem {hints['memory_percent']:.0f}%")
        if "temperature_c" in hints:
            bits.append(f"temp {hints['temperature_c']:.0f}°C")
        suffix = f" ({', '.join(bits)})" if bits else ""
        # Best-effort emit — a logger backend hiccup shouldn't drop telemetry.
        with contextlib.suppress(Exception):
            log.emit("info", f"Telemetry: {nid} reported {slot['count']} metrics{suffix}", source="telemetry")
    remaining = len(per_node) - _MAX_ROLLUP_NODES
    if remaining > 0:
        with contextlib.suppress(Exception):
            log.emit(
                "info",
                f"Telemetry: +{remaining} more node(s) in this batch",
                source="telemetry",
            )
