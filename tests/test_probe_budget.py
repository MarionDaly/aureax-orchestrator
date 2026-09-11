"""Contracts between the shipped k8s/ manifests and the config that feeds them.

The startupProbe budget is the reason this module exists. It is a two-file
arithmetic relationship that was maintained only by a comment: a sharded head
waits for its RPC peers (``node.worker_wait_timeout_seconds``, injected from
the group) and *then* loads the model
(``model.server.health_timeout_seconds``). If the budget in
``k8s/daemonset-edge.yaml`` is shorter than that sum, the kubelet kills the
container mid-load — forever, and with no error that names the cause.

Nothing enforced it before, so lowering ``failureThreshold`` to match one
model, or raising a head's ``health_timeout_seconds`` for another, silently
armed that failure. The manifest is shared by every pod carrying the edge
label, so the check is against the WORST case across all of them — including
the 122B rollback target, which is why the threshold is not tightened to fit
the current model.

The Service invariants at the bottom are here for the same reason: another
cross-file coupling (``externalTrafficPolicy`` and the NodePort literals) that
lives half in ``k8s/`` and half in Python, fails silently when broken, and has
no other reader.
"""

from __future__ import annotations

import pytest
from ruamel.yaml import YAML

from client.config.parser import load_edge_config_raw
from orchestrator.config.loader import load_network_config
from orchestrator.config.schema import (
    DEFAULT_WORKER_WAIT_SECONDS,
    LLAMA_NODE_PORT,
    MCP_NODE_PORT,
    SHARD_WORKER_ROLE,
    LlamaServerConfig,
)

from .conftest import EXAMPLE_NETWORK_CONFIG, PROJECT_ROOT, REAL_NETWORK_CONFIG

# Read off the schema, not copied. This module exists to stop a cross-file
# arithmetic relationship drifting, and it already imports the other half of
# the sum — a literal 120 here would keep computing the budget from a stale
# default the moment the schema's own moves, under-estimating what it enforces.
DEFAULT_HEALTH_TIMEOUT_SECONDS = LlamaServerConfig.model_fields[
    "health_timeout_seconds"
].default

EDGE_DAEMONSET = PROJECT_ROOT / "k8s" / "daemonset-edge.yaml"
CONFIGS_DIR = PROJECT_ROOT / "configs"
K8S_DIR = PROJECT_ROOT / "k8s"

# The budget has to cover every head that is actually DEPLOYED, so both fleet
# definitions are measured: the tracked example (present in every checkout, so
# this is enforced in CI and on a fresh clone) and the operator's own gitignored
# configs/aureax-network.yaml, which is the file a real deploy reads and may
# name heads or timeouts the example does not.
NETWORK_CONFIGS = [
    pytest.param(EXAMPLE_NETWORK_CONFIG, id="example"),
    pytest.param(
        REAL_NETWORK_CONFIG,
        id="operator",
        marks=pytest.mark.skipif(
            not REAL_NETWORK_CONFIG.exists(),
            reason="operator-owned fleet config not present",
        ),
    ),
]


def _startup_probe() -> dict:
    """The edge container's startupProbe, straight from the shipped manifest."""
    manifest = YAML().load(EDGE_DAEMONSET.read_text())
    containers = manifest["spec"]["template"]["spec"]["containers"]
    probes = [c["startupProbe"] for c in containers if "startupProbe" in c]
    assert probes, "daemonset-edge.yaml has no startupProbe — model load is unguarded"
    return probes[0]


def _probe_budget_seconds(probe: dict) -> int:
    """Seconds the kubelet allows before it starts killing the container.

    initialDelaySeconds is counted once, then failureThreshold consecutive
    failures at periodSeconds apart.
    """
    return int(probe.get("initialDelaySeconds", 0)) + int(
        probe["failureThreshold"]
    ) * int(probe["periodSeconds"])


def _as_int(value) -> int | None:
    """The value as an int, or None when it is a template placeholder.

    Parsing rather than string-matching ``${worker_wait_timeout_seconds}``: a
    renamed or quoted placeholder is still not a number, and matching the exact
    spelling would either crash on ``int()`` or wave the new spelling through.
    """
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _edge_probe_startup_costs(network_config) -> list[tuple[str, int, int]]:
    """(node_id, worker_wait, health_timeout) for every pod under the edge probe.

    Not just group heads. ``daemonset-edge.yaml``'s startupProbe applies to
    every pod matching ``aureax.io/role=edge``, and ``label_nodes`` applies that
    label to standalone heads and to the planner as well — so the budget is
    measured against the worst of ALL of them, not only the nodes that appear
    in ``node_groups``.

    worker_wait normally comes from the group: the head's template carries a
    ``${worker_wait_timeout_seconds}`` placeholder that ``build_variables``
    resolves at generation, so reading the template alone yields a string. But
    the supervisor reads ``node.worker_wait_timeout_seconds`` out of the
    RESOLVED config, so a template that hardcodes a NUMBER is what actually
    runs — that number therefore wins here. Measuring only the group value is
    what let a hardcoded head pass this test and still be killed mid-wait.

    A node in no group has no RPC peers, so ``_wait_for_rpc_workers`` returns
    at once and its wait costs nothing.
    """
    cfg = load_network_config(network_config)
    costs: list[tuple[str, int, int]] = []
    for target in cfg.ssh.enabled_targets:
        if not target.config_file:
            continue
        path = CONFIGS_DIR / target.config_file
        assert path.exists(), f"{target.node_id} names a config_file that is not in configs/"
        raw = load_edge_config_raw(path)
        node = raw.get("node") or {}
        if node.get("role") == SHARD_WORKER_ROLE:
            continue  # workers carry daemonset-shard.yaml's probe, not this one
        group = cfg.group_for_node(target.node_id)
        worker_wait = group.worker_wait_timeout_seconds if group else 0
        hardcoded = _as_int(node.get("worker_wait_timeout_seconds"))
        if hardcoded is not None:
            worker_wait = max(worker_wait, hardcoded)
        server = (raw.get("model") or {}).get("server") or {}
        costs.append((
            target.node_id,
            worker_wait,
            int(server.get("health_timeout_seconds", DEFAULT_HEALTH_TIMEOUT_SECONDS)),
        ))
    return costs


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_probe_budget_covers_every_node_carrying_the_edge_probe(network_config) -> None:
    budget = _probe_budget_seconds(_startup_probe())
    costs = _edge_probe_startup_costs(network_config)
    if not costs:
        pytest.skip("no enabled edge targets configured")

    for node_id, worker_wait, health_timeout in costs:
        needed = worker_wait + health_timeout
        assert budget > needed, (
            f"node {node_id!r} needs {worker_wait} + {health_timeout} = {needed}s to "
            f"start, but daemonset-edge.yaml's startupProbe allows only {budget}s. "
            "The kubelet will kill it mid-load. Raise failureThreshold or "
            "lower health_timeout_seconds."
        )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_a_group_head_takes_its_worker_wait_from_the_group(network_config) -> None:
    """A head must keep the placeholder, not a hardcoded number.

    ``worker_wait_timeout_seconds`` is configured on the NodeGroup and reaches
    the node only because ``build_variables`` resolves the placeholder in the
    head's template. Replace it with a literal and the group-level setting
    becomes a silent no-op — the schema says as much — and every other reader
    of that number, this module's budget check included, is then measuring
    something the node does not do.
    """
    cfg = load_network_config(network_config)
    by_id = {t.node_id: t for t in cfg.ssh.targets}
    if not cfg.node_groups:
        pytest.skip("no node_groups configured")

    for group in cfg.node_groups:
        target = by_id.get(group.head)
        assert target is not None, f"group {group.name!r} names an unknown head"
        assert target.config_file, f"group {group.name!r} head has no config_file"
        node = load_edge_config_raw(CONFIGS_DIR / target.config_file).get("node") or {}
        raw = node.get("worker_wait_timeout_seconds")
        assert raw is not None, (
            f"the head of group {group.name!r} has no node.worker_wait_timeout_seconds, "
            f"so it silently waits the {DEFAULT_WORKER_WAIT_SECONDS}s default instead of "
            f"the group's {group.worker_wait_timeout_seconds}s"
        )
        assert _as_int(raw) is None, (
            f"the head of group {group.name!r} hardcodes worker_wait_timeout_seconds="
            f"{raw!r}. That makes the group's {group.worker_wait_timeout_seconds}s a "
            "no-op; keep the ${worker_wait_timeout_seconds} placeholder so the group "
            "stays the single source of truth."
        )


def _longest_configured_worker_wait() -> int:
    """The largest ``worker_wait_timeout_seconds`` any configured group sets.

    The rollback head keeps the ``${worker_wait_timeout_seconds}`` placeholder,
    so the number it actually waits comes from whichever group it is made the
    head of — not from the schema default. Reading the default here would leave
    the rollback unguarded by the one test that exists to guard it: raise both
    groups to 900s and the deployed heads still fit the budget while the revert
    no longer does. Both fleet definitions are scanned for the same reason the
    parametrised tests read both.
    """
    waits = [DEFAULT_WORKER_WAIT_SECONDS]
    for param in NETWORK_CONFIGS:
        path = param.values[0]
        if not path.exists():
            continue
        waits += [g.worker_wait_timeout_seconds for g in load_network_config(path).node_groups]
    return max(waits)


def test_probe_budget_covers_the_rollback_target() -> None:
    """The manifest is shared, so a revert must not walk into a shorter budget.

    configs/aureax-shard-head.yaml is kept unmodified as the 122B rollback
    target. Flipping a head's config_file back to it is a one-line change that
    does not touch k8s/ at all — so the budget has to already cover its much
    longer load, or the rollback crash-loops instead of working.
    """
    rollback = CONFIGS_DIR / "aureax-shard-head.yaml"
    if not rollback.exists():
        pytest.skip("no 122B rollback config in the tree")

    server = (load_edge_config_raw(rollback).get("model") or {}).get("server") or {}
    worker_wait = _longest_configured_worker_wait()
    needed = worker_wait + int(server.get("health_timeout_seconds", DEFAULT_HEALTH_TIMEOUT_SECONDS))
    budget = _probe_budget_seconds(_startup_probe())
    assert budget > needed, (
        f"the rollback head needs {worker_wait} + its health_timeout_seconds = "
        f"{needed}s but the startupProbe allows {budget}s — reverting to it would "
        "crash-loop. Do not tighten failureThreshold to fit only the current model."
    )


def test_worker_probe_is_startup_only() -> None:
    """Shard workers must carry a startupProbe and nothing else.

    rpc-server listens with a backlog of 1 and serves one client, so it does not
    accept a probe connection while loading or generating. A liveness probe
    reads that as death and SIGTERMs the worker mid-load, crash-looping the
    whole group; a readiness probe just pins it at 0/1 while it works fine.
    """
    manifest = YAML().load((PROJECT_ROOT / "k8s" / "daemonset-shard.yaml").read_text())
    for container in manifest["spec"]["template"]["spec"]["containers"]:
        assert "startupProbe" in container, "worker has no startupProbe"
        assert "livenessProbe" not in container, (
            "a livenessProbe on rpc-server SIGTERMs it mid-load — see the "
            "rationale in k8s/daemonset-shard.yaml"
        )
        assert "readinessProbe" not in container, (
            "a readinessProbe pins a working rpc-server at 0/1"
        )


# --- NodePort Services: session affinity and the port literals ---------------

SERVICES = {
    "service-mcp.yaml": MCP_NODE_PORT,
    "service-llama.yaml": LLAMA_NODE_PORT,
}


@pytest.mark.parametrize("manifest_name", sorted(SERVICES))
def test_nodeport_services_keep_traffic_on_the_node_that_was_dialled(manifest_name) -> None:
    """Both Services must keep externalTrafficPolicy: Local.

    Without it kube-proxy load-balances a NodePort across every pod behind the
    selector, so an MCP session's POST and its background GET stream can land
    on different pods and the client sees "Session terminated". The llama
    Service has the same requirement for a different reason: the dashboard and
    the Access tab address a SPECIFIC node's llama-server by its host, and a
    balanced Service silently answers from some other node — which also makes
    a per-node auth probe read the wrong node's auth state.

    Nothing else in the repo reads this field, so removing it is invisible
    until a demo produces intermittent session failures.
    """
    spec = YAML().load((K8S_DIR / manifest_name).read_text())["spec"]

    assert spec.get("type") == "NodePort", (
        f"{manifest_name} is not a NodePort Service — externalTrafficPolicy "
        "only means anything for one, and every client dials <node-ip>:<nodePort>"
    )
    assert spec.get("externalTrafficPolicy") == "Local", (
        f"{manifest_name} lost externalTrafficPolicy: Local — NodePort traffic will "
        "be balanced across pods, breaking MCP session affinity and per-node addressing"
    )


@pytest.mark.parametrize("manifest_name", sorted(SERVICES))
def test_service_nodeports_match_the_python_constants(manifest_name) -> None:
    """The manifests and orchestrator/config/schema.py hold the same literals.

    MCP_NODE_PORT / LLAMA_NODE_PORT were declared once precisely because five
    modules dial these ports; the manifests keep their own copies because this
    prototype ships static YAML. Nothing but this test connects the two halves,
    so changing one side leaves every client dialling a closed port.
    """
    expected = SERVICES[manifest_name]
    spec = YAML().load((K8S_DIR / manifest_name).read_text())["spec"]
    node_ports = [int(p["nodePort"]) for p in spec["ports"] if "nodePort" in p]

    assert node_ports == [expected], (
        f"{manifest_name} publishes {node_ports}, but schema.py declares {expected}. "
        "Both sides have to change together."
    )
