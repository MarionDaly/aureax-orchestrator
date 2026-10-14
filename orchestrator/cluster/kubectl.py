"""Thin wrapper around kubectl commands."""

from __future__ import annotations

import json
import re

from orchestrator.cluster._subprocess import run as _shared_run

# kubectl's default request timeout is 0 — wait forever — so one hung request
# blows whatever deadline the caller promised (Deployer._wait_pod_replaced,
# wait_nodes_registered, PodWatcher.run all count on their loop returning).
# Carried by every call that should answer immediately, reads of whole
# collections included. NOT by `delete`, which waits out the graceful pod
# termination period by design (that alone exceeds this), nor by `apply`.
_REQUEST_TIMEOUT = "--request-timeout=10s"


def _run(args: list[str], check: bool = True):
    """Run a kubectl command and return the completed result.

    Raises ``CommandError`` (a ``CalledProcessError`` carrying kubectl's
    stderr) when check=True and kubectl exits non-zero.
    """
    return _shared_run(["kubectl", *args], check=check)


def wait_api_ready(timeout: int = 120, interval: int = 2) -> bool:
    """Block until the API server answers, or `timeout` elapses.

    A freshly started k3d cluster accepts connections some seconds after the
    container is up. Without this gate, the next step fails with an opaque
    per-resource error instead of a clear "cluster is not reachable".
    """
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        # api_reachable is the single definition of "the API server answers",
        # and its per-attempt bound is what keeps the deadline this function
        # promises: kubectl's own request timeout is 0, i.e. wait forever.
        if api_reachable("5s"):
            return True
        time.sleep(interval)
    return False


def api_reachable(timeout_hint: str = "3s") -> bool:
    """True if the API server answers right now.

    Callers that list resources need to tell "nothing there" apart from "cannot
    ask" — `get_pods` returns ``[]`` for both, and a watcher reading the second
    as the first reports every pod as removed.
    """
    result = _run(["get", "--raw", "/readyz", f"--request-timeout={timeout_hint}"], check=False)
    return result.returncode == 0


# gitVersion as K3s spells it, e.g. "v1.31.5+k3s1" — the exact form
# INSTALL_K3S_VERSION wants. Anything else is treated as "unknown" rather than
# passed through, because a malformed pin makes the k3s installer fail outright
# where an absent one merely installs latest stable.
_K3S_VERSION_RE = re.compile(r"^v\d[\w.\-]*\+k3s\d+$")


def server_version() -> str | None:
    """The API server's own gitVersion, or None when it cannot be read.

    Every agent should be pinned to the version the server already runs:
    get.k3s.io installs latest stable while the k3d server stays on the image
    bundled with the installed k3d release, so nodes joined weeks apart end up
    with kubelets NEWER than the API server — outside the supported skew, and
    surfacing much later as an unexplained NotReady node.

    Read rather than configured, so there is no constant to keep in step. None
    is not fatal anywhere: the caller falls back to an unpinned install and
    says so, because a cluster that cannot answer /version has larger problems
    than version skew.
    """
    result = _run(["get", "--raw", "/version", _REQUEST_TIMEOUT], check=False)
    if result.returncode != 0:
        return None
    try:
        version = json.loads(result.stdout).get("gitVersion", "")
    except (ValueError, AttributeError):
        return None
    return version if _K3S_VERSION_RE.match(version or "") else None


def apply(manifest_path: str, namespace: str | None = None) -> str:
    """Apply a Kubernetes manifest file. Raises ``CommandError`` if it fails."""
    args = ["apply", "-f", manifest_path]
    if namespace:
        args += ["-n", namespace]
    result = _run(args)
    return result.stdout


def get(resource: str, namespace: str | None = None) -> dict:
    """Get Kubernetes resources as parsed JSON.

    {} on ANY kubectl failure — unreachable API, missing namespace, unknown
    resource, RBAC denial. Callers that must tell "nothing there" apart from
    "cannot ask" have to call api_reachable() as well.
    """
    args = ["get", resource, "-o", "json", _REQUEST_TIMEOUT]
    if namespace:
        args += ["-n", namespace]
    result = _run(args, check=False)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def get_nodes() -> list[dict]:
    """Get all cluster nodes."""
    data = get("nodes")
    if not data or "items" not in data:
        return []
    return [
        {
            "name": node["metadata"]["name"],
            "status": next(
                (
                    c["status"]
                    for c in node["status"].get("conditions", [])
                    if c["type"] == "Ready"
                ),
                "Unknown",
            ),
            "roles": ",".join(
                k.split("/")[-1]
                for k in node["metadata"].get("labels", {})
                if k.startswith("node-role.kubernetes.io/")
            ) or "worker",
            "labels": node["metadata"].get("labels", {}),
        }
        for node in data["items"]
    ]


def get_pods(namespace: str = "aureax") -> list[dict]:
    """Get all pods in a namespace."""
    data = get("pods", namespace=namespace)
    if not data or "items" not in data:
        return []
    result = []
    for pod in data["items"]:
        statuses = pod["status"].get("containerStatuses", [])
        # Pending pods have no containerStatuses yet; all([]) is True, so
        # we explicitly require at least one container status to claim ready.
        result.append({
            "name": pod["metadata"]["name"],
            "node": pod["spec"].get("nodeName", ""),
            "status": pod["status"].get("phase", "Unknown"),
            "ready": bool(statuses) and all(cs.get("ready", False) for cs in statuses),
            # Surfaced so PodWatcher can report restart transitions — a
            # crash-looping shard worker keeps phase Running and ready False,
            # which produces no transition at all without this.
            "restarts": sum(cs.get("restartCount", 0) for cs in statuses),
            # No image/imageID key here on purpose. The pre-start build gate
            # used to compare each group member's imageID, and it fired on every
            # start of the real fleet: every node builds `aureax-client:dev`
            # NATIVELY (scripts/build-on-node.sh), so the digests differ by
            # construction while the compiled binaries are identical. The gate
            # now compares the binaries themselves over SSH
            # (orchestrator/cluster/buildcheck.py), and nothing else ever wanted
            # a digest here.
        })
    return result


def label_node(node_name: str, label: str, value: str) -> str:
    """Apply a label to a node. Raises ``CommandError`` if it does not land."""
    result = _run([
        "label", "node", node_name, f"{label}={value}", "--overwrite", _REQUEST_TIMEOUT,
    ])
    return result.stdout


def remove_label(node_name: str, label: str) -> tuple[bool, str]:
    """Remove a label from a node. Returns (ok, detail).

    An absent label counts as success — per-node stop must stay idempotent,
    and kubectl reports that case as `label(s) not found`. Any other failure
    means the label is still gating a workload: per-node stop relies on this
    call, so callers must check `ok` rather than reporting "stopped" for pods
    that are still running.

    An unknown NODE is a failure, not an absent label: `Error from server
    (NotFound): nodes "host-a" not found` contains "not found" too, and taking
    it as success reports "stopped" for a node whose gating label — and every
    pod it gates — is untouched. Hostnames reach this call from SSH, not from
    the cluster, so they can genuinely be wrong. The broad match stays the
    default so a reworded kubectl message fails toward idempotency.
    """
    result = _run(["label", "node", node_name, f"{label}-", _REQUEST_TIMEOUT], check=False)
    detail = (result.stderr or "").strip() or result.stdout.strip()
    if result.returncode != 0:
        node_missing = "NotFound" in detail or re.search(r'nodes? "[^"]+" not found', detail)
        if node_missing or "not found" not in detail:
            return False, detail or f"exit {result.returncode}"
    return True, detail


def delete_pods_on_node(namespace: str, node_name: str) -> tuple[bool, str]:
    """Delete the aureax-edge pod(s) on one node; the controller recreates them
    immediately, remounting /etc/aureax with the freshly pushed config.
    Returns (ok, detail).

    Targeted with --field-selector rather than by name because a DaemonSet
    pod's name is generated — this is how "the pod on that node" is addressed
    without listing pods first.

    The app label is fixed, not a parameter: the only other value in the fleet
    is aureax-shard, and restarting a shard worker tears its group down and
    forces the head to reload (warm, from the worker's tensor cache, but still
    minutes; see ARCHITECTURE.md, "3. Sharded model groups (llama.cpp RPC)")
    — so apply_to_nodes skips workers entirely rather than passing a different
    app.

    This is the ONLY restart primitive the apply path may use. `kubectl
    rollout restart daemonset/...` restarts every pod in the fleet — serially —
    for a change to one node, which on a sharded head means minutes of
    reload per member (tests/test_apply.py pins that it is never invoked).
    """
    result = _run([
        "delete", "pod", "-n", namespace,
        "-l", "app=aureax-edge",
        "--field-selector", f"spec.nodeName={node_name}",
    ], check=False)
    detail = (result.stderr or "").strip() or result.stdout.strip()
    if result.returncode != 0:
        return False, detail or f"exit {result.returncode}"
    # kubectl exits 0 with "No resources found" when nothing matched — that is
    # a real outcome (node not labelled, pod already gone), not a success to
    # report as a restart.
    if "deleted" not in result.stdout:
        return False, detail or "no matching pod on that node"
    return True, detail


def patch(resource: str, name: str, patch_json: str, namespace: str | None = None) -> str:
    """Patch a Kubernetes resource with a JSON merge patch.

    Raises ``CommandError`` if the patch does not land. This is how
    stop/start suspend and resume workloads, so a swallowed failure means the
    operation reports success while the DaemonSet still points at the old
    selector and the pods it was meant to move never move.
    """
    args = ["patch", resource, name, "-p", patch_json, "--type=merge", _REQUEST_TIMEOUT]
    if namespace:
        args += ["-n", namespace]
    result = _run(args)
    return result.stdout


