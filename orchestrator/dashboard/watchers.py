"""Background watchers that push cluster/system events to the system log.

Each watcher is a long-running asyncio task. They poll the same helpers the
dashboard uses (``kubectl`` wrappers + SSH drift check), track last-seen
state in memory, and only emit on transitions so a healthy cluster stays
quiet.

Started from ``app.py`` via the ``lifespan`` hook.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

# The dashboard's kubectl TTL cache, not a second one here: both watchers and
# the /api/cluster + /api/pods handlers ask the same `/readyz` question, so one
# probe per TTL answers all of them. A second cache would mean a second TTL, a
# second lock, and a tick that could still spawn two `kubectl get --raw
# /readyz` at once.
from orchestrator.dashboard.api import api_reachable_cached

logger = logging.getLogger("aureax.watchers")


async def poll_loop(
    log,
    label: str,
    source: str,
    interval: float,
    tick: Callable[[], Awaitable[None]],
) -> None:
    """Run ``tick`` every ``interval`` seconds, forever, announcing pauses once.

    The shared frame for all three watchers, so the loop, the interval and the
    pause announcement are described in one place.

    ONE warning per outage, not one per tick: a watcher polls every 10 s and an
    unreachable cluster is a normal state to sit in for minutes, so an
    unfiltered warning would fill the operator's log console with the same
    line. The flag clears on the next successful tick, so a second outage is
    announced again.

    Nothing is re-raised: a watcher that dies stops watching silently for the
    life of the process. CancelledError is a BaseException and passes through,
    which is what lets app.py's shutdown cancel these tasks.
    """
    warned = False
    while True:
        try:
            await tick()
            warned = False
        except Exception as e:
            if not warned:
                log.emit("warn", f"{label} paused: {e}", source=source)
                warned = True
        await asyncio.sleep(interval)


class PodWatcher:
    """Emit pod phase / readiness / restart transitions."""

    def __init__(self, system_log, namespace: str = "aureax", interval_seconds: int = 10) -> None:
        self.log = system_log
        self.namespace = namespace
        self.interval = interval_seconds
        self.last: dict[str, tuple[str, bool, int]] = {}

    async def run(self) -> None:
        await poll_loop(self.log, "Pod watcher", "pods", self.interval, self._tick)

    async def _tick(self) -> None:
        from orchestrator.cluster import kubectl

        # `get_pods` returns [] both for "namespace is empty" and for "the API
        # server is unreachable", and _diff reads an empty dict as every pod
        # having vanished — a dropped network would report the whole namespace
        # as removed. Reachability is asked separately rather than inferred
        # from the result.
        if not await asyncio.to_thread(api_reachable_cached):
            raise RuntimeError("kubectl API server unreachable")
        pods = await asyncio.to_thread(kubectl.get_pods, self.namespace)
        self._diff({
            p["name"]: (
                # kubectl.get_pods exposes the pod phase under the key
                # "status"; NodeWatcher's "status" is a different thing (the
                # Ready condition).
                p.get("status", "?"),
                bool(p.get("ready", False)),
                int(p.get("restarts", 0)),
            )
            for p in pods
        })

    def _diff(self, current: dict[str, tuple[str, bool, int]]) -> None:
        for name, state in current.items():
            prev = self.last.get(name)
            if prev is None:
                phase, ready, _ = state
                level = "success" if ready else "info"
                self.log.emit(
                    level,
                    f"Pod appeared: {name} ({phase}{', ready' if ready else ''})",
                    source="pods",
                )
            elif prev != state:
                phase, ready, restarts = state
                pprev, rprev, restprev = prev
                parts = []
                if pprev != phase:
                    parts.append(f"phase {pprev}→{phase}")
                if rprev != ready:
                    parts.append("became ready" if ready else "became NotReady")
                if restarts > restprev:
                    # The only signal a crash-looping pod produces: phase stays
                    # Running and ready stays False, so nothing else transitions.
                    parts.append(f"restarted ({restprev}→{restarts})")
                if not parts:
                    continue
                level = "success" if (not rprev and ready) else ("warn" if (rprev and not ready) else "info")
                if restarts > restprev:
                    level = "error"
                self.log.emit(level, f"Pod {name}: {', '.join(parts)}", source="pods")
        for name in list(self.last):
            if name not in current:
                self.log.emit("warn", f"Pod removed: {name}", source="pods")
        self.last = current


class NodeWatcher:
    """Emit K3s node Ready/NotReady transitions."""

    def __init__(self, system_log, interval_seconds: int = 10) -> None:
        self.log = system_log
        self.interval = interval_seconds
        self.last: dict[str, str] = {}

    async def run(self) -> None:
        await poll_loop(self.log, "Node watcher", "nodes", self.interval, self._tick)

    async def _tick(self) -> None:
        from orchestrator.cluster import kubectl

        # `get_nodes` returns [] both when the cluster is empty and when it
        # cannot be reached, and _diff reads the second as every node having
        # been removed — the same ambiguity PodWatcher guards.
        if not await asyncio.to_thread(api_reachable_cached):
            raise RuntimeError("kubectl API server unreachable")
        nodes = await asyncio.to_thread(kubectl.get_nodes)
        self._diff({n["name"]: n.get("status", "Unknown") for n in nodes})

    def _diff(self, current: dict[str, str]) -> None:
        for name, status in current.items():
            prev = self.last.get(name)
            if prev is None:
                pretty = "Ready" if status == "True" else ("NotReady" if status == "False" else status)
                level = "success" if status == "True" else "warn"
                self.log.emit(level, f"Node appeared: {name} ({pretty})", source="nodes")
            elif prev != status:
                level = "success" if status == "True" else "warn"
                pretty_prev = "Ready" if prev == "True" else ("NotReady" if prev == "False" else prev)
                pretty_cur = "Ready" if status == "True" else ("NotReady" if status == "False" else status)
                self.log.emit(level, f"Node {name}: {pretty_prev} → {pretty_cur}", source="nodes")
        for name in list(self.last):
            if name not in current:
                self.log.emit("warn", f"Node removed: {name}", source="nodes")
        self.last = current


class DriftChecker:
    """Periodically check each edge node's K3S_URL against the orchestrator's
    current LAN IP."""

    def __init__(
        self,
        system_log,
        network_config_path: Path,
        interval_seconds: int = 300,
        warmup_seconds: int = 15,
    ) -> None:
        self.log = system_log
        self.network_config_path = network_config_path
        self.interval = interval_seconds
        # Warmup avoids racing the server's startup logs. A parameter only so a
        # test can drive one pass without waiting it out.
        self.warmup = warmup_seconds
        self.last_drift: set[str] = set()

    async def run(self) -> None:
        # The warmup is outside poll_loop: it avoids racing the server's
        # startup logs once, not once per pass.
        await asyncio.sleep(self.warmup)
        await poll_loop(self.log, "Drift checker", "drift", self.interval, self._tick)

    async def _tick(self) -> None:
        from orchestrator.cluster import k3d
        from orchestrator.config.loader import load_network_config

        cfg = load_network_config(str(self.network_config_path))
        expected = await asyncio.to_thread(
            k3d.get_server_url, cfg.kubernetes.k3d.api_port, k3d.probe_host(cfg)
        )
        # The same guard cli.py's _expected_server_url and heal_nodes apply:
        # get_server_url falls back to 127.0.0.1 when LAN-IP detection fails,
        # and comparing every node against loopback flags the whole fleet. One
        # "paused" line, not nine false drift alarms every five minutes.
        if expected.startswith(("https://127.", "https://0.")):
            raise RuntimeError(f"could not detect the orchestrator LAN IP (got {expected})")

        # Concurrently: each check is a full SSH connect+auth+exec, so serially
        # an unreachable node would put its whole 15s connect timeout in front
        # of every node behind it. No return_exceptions: _check swallows its own
        # failures, so a raise here is not a per-node problem and belongs to
        # poll_loop's handler.
        results = await asyncio.gather(
            *(self._check(t, cfg, expected) for t in cfg.ssh.enabled_targets)
        )
        current_drift = {nid for nid, drifted in results if drifted}
        for nid in current_drift - self.last_drift:
            self.log.emit(
                "error",
                f"Drift detected: {nid} K3S_URL mismatch (expected {expected}); run `aureax heal-nodes`",
                source="drift",
            )
        for nid in self.last_drift - current_drift:
            self.log.emit("success", f"Drift cleared: {nid}", source="drift")
        self.last_drift = current_drift

    async def _check(self, target, cfg, expected: str) -> tuple[str, bool]:
        """``(node_id, drifted)`` for one node. Per-node, not per-scan.

        ``get_configured_server_url`` ends in paramiko's connect(), which
        RAISES on an unreachable host rather than returning None. Caught here,
        per node, so one node being down cannot abort the pass before the nodes
        after it are checked — and cannot silence the checker through
        poll_loop's outage flag until it recovers. A node we cannot reach has
        unknown drift, which is not the same as no drift, so its previous
        verdict carries forward.
        """
        from orchestrator.deploy.k3s_agent import (
            DEFAULT_CONNECT_TIMEOUT_SECONDS,
            get_configured_server_url,
        )
        from orchestrator.deploy.ssh import conn_kwargs

        try:
            configured = await asyncio.to_thread(
                get_configured_server_url,
                target.host,
                # The configured connect budget, floored at the helper's own
                # default: this one number bounds both the connect and the
                # remote `sudo cat`, so a fleet slow enough to need a longer
                # connect would otherwise score as undrifted-because-unreachable
                # on every scan.
                timeout=max(
                    DEFAULT_CONNECT_TIMEOUT_SECONDS, cfg.ssh.connect_timeout_seconds
                ),
                **conn_kwargs(target, cfg.ssh.key_store),
            )
        except Exception as e:
            logger.debug("Drift check skipped for %s: %s", target.node_id, e)
            return target.node_id, target.node_id in self.last_drift
        return target.node_id, configured is not None and configured != expected


async def start_all(app, network_config_path: Path) -> list[asyncio.Task]:
    """Spawn the three watchers. Returns the tasks so they can be cancelled
    on shutdown.
    """
    system_log = getattr(app.state, "system_log", None)
    if system_log is None:
        return []
    system_log.emit("info", "Watchers starting: pods, nodes, drift", source="lifecycle")
    return [
        asyncio.create_task(PodWatcher(system_log).run(), name="pod-watcher"),
        asyncio.create_task(NodeWatcher(system_log).run(), name="node-watcher"),
        asyncio.create_task(DriftChecker(system_log, network_config_path).run(), name="drift-checker"),
    ]
