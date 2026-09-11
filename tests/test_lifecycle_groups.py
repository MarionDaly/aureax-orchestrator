"""Lifecycle behaviour for sharded groups.

Every case here is one where the pre-group code would silently do the wrong
thing: report success against a head that never loaded, break a group via a
per-node button, or paint a healthy worker permanently red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator import lifecycle
from orchestrator.config.schema import NetworkConfig
from tests.conftest import (
    NETWORK_BASE,
    FakeKubectl,
    FakeLog,
    make_network_config,
    make_targets,
)

# Captured at import time, before the autouse stub below can replace the module
# attribute: the validation phase's own tests call the real function.
_validate_config_phase = lifecycle._start_phase_validate_config


@pytest.fixture(autouse=True)
def _stub_start_validation(monkeypatch) -> None:
    """Neutralise the config validation `do_start`/`do_start_node` now run first.

    Every fleet in this module is synthetic: its targets name templates
    (``h.yaml``, ``w.yaml``, ``s.yaml``) that do not exist under ``configs/``,
    so the real phase would refuse each of these starts with "config_file not
    found" before any of the label, selector or fan-out logic under test ran.

    The phase itself is not left uncovered: the tests at the end of this module
    call ``_validate_config_phase`` (the real function, captured above)
    directly, and pin its position in ``do_start`` with a recorder of their own.
    """
    monkeypatch.setattr(lifecycle, "_start_phase_validate_config", lambda *_a, **_k: None)


# Two checksums that stand for "the same build" and "another build". They are
# what the gate actually compares: the sha256 of llama-server and rpc-server
# INSIDE each node's image, read over SSH. Not the pods' image digests — every
# node builds `aureax-client:dev` natively, so those differ across a perfectly
# coherent group and the digest version of this gate failed every real start.
_SUMS_A = ("a" * 64, "b" * 64)
_SUMS_B = ("c" * 64, "d" * 64)


def _stub_build_probe(
    monkeypatch, sums: dict[str, tuple[str, str]] | None = None,
    unreadable: tuple[str, ...] = (),
) -> list[str]:
    """Answer the pre-start build gate's SSH probe from a table.

    Without this every test driving a grouped fleet through the label phase
    would open a real SSH connection to 10.77.0.x. Members default to ``_SUMS_A``
    — a coherent group — so a test only has to say what is odd about its fleet.
    Returns the node ids the gate actually asked about.
    """
    from orchestrator.cluster import buildcheck

    probed: list[str] = []

    def probe_group(targets, **_kwargs):
        facts = {}
        for target in targets:
            probed.append(target.node_id)
            if target.node_id in unreadable:
                facts[target.node_id] = {
                    buildcheck.HOST_KEY: target.host, "error": "unreachable over SSH",
                }
                continue
            llama, rpc = (sums or {}).get(target.node_id, _SUMS_A)
            facts[target.node_id] = {
                buildcheck.HOST_KEY: target.host,
                buildcheck.LLAMA_SHA_KEY: llama,
                buildcheck.RPC_SHA_KEY: rpc,
            }
        return facts

    monkeypatch.setattr(buildcheck, "probe_group", probe_group)
    return probed


@pytest.fixture(autouse=True)
def _stub_build_gate(monkeypatch) -> None:
    """Every fleet here is coherent unless its own test says otherwise."""
    _stub_build_probe(monkeypatch)


def _cfg(groups=None) -> NetworkConfig:
    """This module's own fleet: it pins node ids and hosts its assertions read
    back, so it stays a fixture rather than becoming a generic builder."""
    return NetworkConfig.model_validate({
        **NETWORK_BASE,
        "ssh": {"targets": [
            {"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"},
            {"node_id": "w1", "host": "10.77.0.2", "config_file": "w.yaml"},
            {"node_id": "w2", "host": "10.77.0.3", "config_file": "w.yaml"},
            {"node_id": "solo", "host": "10.77.0.9", "config_file": "s.yaml"},
        ]},
        "node_groups": groups if groups is not None else [
            {"name": "g", "head": "h", "workers": ["w1", "w2"]}
        ],
    })


def _roles(monkeypatch, mapping: dict[str, str]) -> None:
    monkeypatch.setattr(
        lifecycle, "node_role",
        lambda target, path: mapping.get(target.node_id, ""),
    )


def test_group_member_ids_fans_out_to_the_whole_group() -> None:
    """Acting on one member is never right: the group serves ONE model."""
    cfg = _cfg()
    members, name = lifecycle._group_member_ids(cfg, "w1")
    assert name == "g"
    assert set(members) == {"h", "w1", "w2"}
    # Head first, so callers can stop head-first / start workers-first.
    assert members[0] == "h"


def test_group_member_ids_leaves_standalone_nodes_alone() -> None:
    cfg = _cfg()
    assert lifecycle._group_member_ids(cfg, "solo") == (["solo"], None)


def test_label_key_is_shard_for_workers(monkeypatch) -> None:
    """Workers are gated by a different label key than edges."""
    cfg = _cfg()
    _roles(monkeypatch, {"w1": "shard_worker", "h": "edge_executor"})
    path = Path("configs/aureax-network.yaml")

    assert lifecycle._label_key_for_node(cfg, "w1", path) == cfg.kubernetes.shard_label
    assert lifecycle._label_key_for_node(cfg, "h", path) == cfg.kubernetes.node_label


def test_split_hostnames_partitions_three_ways(monkeypatch) -> None:
    cfg = _cfg()
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname",
        lambda config, node_id: f"host-{node_id}",
    )
    _roles(monkeypatch, {
        "h": "edge_executor", "w1": "shard_worker",
        "w2": "shard_worker", "solo": "edge_executor",
    })

    edges, workers, planners = lifecycle._split_hostnames_by_role(cfg, Path("configs/x.yaml"))
    assert sorted(edges) == ["host-h", "host-solo"]
    assert sorted(workers) == ["host-w1", "host-w2"]
    assert planners == []


def test_unresolvable_node_is_counted_not_dropped(monkeypatch) -> None:
    """Dropping it shrinks the expected count, so the wait passes trivially.

    That looks like success while a node is silently missing from the fleet.
    """
    cfg = _cfg()
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname",
        lambda config, node_id: None if node_id == "w2" else f"host-{node_id}",
    )
    _roles(monkeypatch, {
        "h": "edge_executor", "w1": "shard_worker",
        "w2": "shard_worker", "solo": "edge_executor",
    })

    edges, workers, _ = lifecycle._split_hostnames_by_role(cfg, Path("configs/x.yaml"))
    assert len(workers) == 2, "an unresolvable worker must still be expected"
    assert len(edges) == 2


def test_disabled_targets_are_excluded(monkeypatch) -> None:
    cfg = NetworkConfig.model_validate({
        **NETWORK_BASE,
        "ssh": {"targets": [
            {"node_id": "a", "host": "10.77.0.1"},
            {"node_id": "gone", "host": "10.77.0.9", "enabled": False},
        ]},
    })
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    _roles(monkeypatch, {})
    edges, workers, planners = lifecycle._split_hostnames_by_role(cfg, Path("configs/x.yaml"))
    assert edges == ["host-a"]
    assert workers == [] and planners == []


def _labels(
    monkeypatch,
    selector: str = "edge",
    patch_outcome: str = "patched",
    remove_result: tuple[bool, str] = (True, ""),
    nodes: list[dict] | None = None,
) -> list[tuple]:
    """Record kubectl label/remove calls instead of making them.

    Also stubs the DaemonSet selector read/patch. Per-node start has to restore
    a selector that a previous whole-stack stop suspended, and
    ``_get_daemonset_node_selector`` shells out to `kubectl` — without these
    stubs every test through this helper would depend on a live cluster.
    ``selector`` is what the DaemonSet currently reports; pass the "stopped"
    value to exercise the restore. ``patch_outcome`` and ``remove_result`` are
    the failure paths: both are ways for a per-node op to report success over
    workloads that are still running.

    ``nodes`` is the cluster listing the double-label pre-flight reads, in
    ``kubectl.get_nodes()``' shape — stubbed for the same reason as the
    selector read, and recorded nowhere: it is the one call here that only
    reads.
    """
    from orchestrator.cluster import kubectl

    calls: list[tuple] = []
    monkeypatch.setattr(kubectl, "get_nodes", lambda: list(nodes or []))
    monkeypatch.setattr(kubectl, "label_node", lambda n, k, v: calls.append(("add", n, k, v)) or "")
    monkeypatch.setattr(
        kubectl, "remove_label",
        lambda n, k: calls.append(("rm", n, k)) or remove_result,
    )
    monkeypatch.setattr(
        lifecycle, "_get_daemonset_node_selector", lambda *_a, **_k: selector
    )
    monkeypatch.setattr(
        lifecycle, "_patch_daemonset_selector",
        lambda _kc, _ns, ds, label, value, log=None: (
            calls.append(("patch", ds, label, value)) or patch_outcome
        ),
    )
    return calls


@pytest.mark.parametrize("op", ["start", "stop"])
def test_partial_group_operation_is_refused(monkeypatch, op) -> None:
    """Acting on the reachable subset is worse than failing.

    The CLI would print "Started 3 node(s)", the head would then wait out its
    peer timeout on the member that was silently skipped, and crash-loop — after
    the operation reported success.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"h": "edge_executor", "w1": "shard_worker", "w2": "shard_worker"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname",
        lambda config, node_id: None if node_id == "w2" else f"host-{node_id}",
    )
    calls = _labels(monkeypatch)
    log = FakeLog()

    fn = lifecycle.do_start_node if op == "start" else lifecycle.do_stop_node
    with pytest.raises(RuntimeError, match="refusing a partial group operation"):
        fn(cfg, "w1", log, Path("configs/x.yaml"))

    assert calls == [], "nothing may be mutated when the group can't be acted on in full"
    assert any("w2" in text for _lvl, text in log.lines)


@pytest.mark.parametrize("op", ["start", "stop"])
def test_whole_group_is_acted_on_when_all_members_resolve(monkeypatch, op) -> None:
    cfg = _cfg()
    _roles(monkeypatch, {"h": "edge_executor", "w1": "shard_worker", "w2": "shard_worker"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch)

    fn = lifecycle.do_start_node if op == "start" else lifecycle.do_stop_node
    fn(cfg, "w1", FakeLog(), Path("configs/x.yaml"))

    touched = [c[1] for c in calls if c[0] in ("add", "rm")]
    assert set(touched) == {"host-h", "host-w1", "host-w2"}, "one member is never enough"

    k8s = cfg.kubernetes
    if op == "start":
        # Workers before the head — the head exits if a peer is unreachable.
        # Asserted over the label WRITES: each is now followed by a strip of the
        # other pass's key, so the head's own two calls are not the only tail.
        assert [c[1] for c in calls if c[0] == "add"][-1] == "host-h"
        by_host = {c[1]: (c[2], c[3]) for c in calls if c[0] == "add"}
        assert by_host["host-w1"] == (k8s.shard_label, k8s.shard_label_value)
        assert by_host["host-h"] == (k8s.node_label, k8s.edge_label_value)
    else:
        # Head first — the reverse makes it abort on a vanished peer.
        assert touched[0] == "host-h"


def test_standalone_node_operation_is_unaffected(monkeypatch) -> None:
    """A node in no group keeps the plain single-node behaviour."""
    cfg = _cfg()
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch)

    lifecycle.do_start_node(cfg, "solo", FakeLog(), Path("configs/x.yaml"))
    assert {c[1] for c in calls} == {"host-solo"}


def test_an_id_that_is_not_a_target_is_named_as_such(monkeypatch) -> None:
    """One "unknown node" message used to cover three different situations."""
    cfg = _cfg()
    _roles(monkeypatch, {})
    monkeypatch.setattr(lifecycle, "_resolve_node_hostname", lambda config, node_id: None)
    _labels(monkeypatch)
    log = FakeLog()

    with pytest.raises(RuntimeError, match=r"not in ssh\.targets"):
        lifecycle.do_start_node(cfg, "nope", log, Path("configs/x.yaml"))
    assert any("nope" in text for _lvl, text in log.lines)


def test_an_unreachable_node_is_not_called_unknown(monkeypatch) -> None:
    """It is in the config; the problem is in the rack, not the YAML."""
    cfg = _cfg()
    _roles(monkeypatch, {})
    monkeypatch.setattr(lifecycle, "_resolve_node_hostname", lambda config, node_id: None)
    _labels(monkeypatch)

    with pytest.raises(RuntimeError, match="unreachable over SSH"):
        lifecycle.do_start_node(cfg, "solo", FakeLog(), Path("configs/x.yaml"))


def test_a_disabled_node_says_disabled(monkeypatch) -> None:
    """`enabled: false` is a flag someone set deliberately — say so.

    The CLI never prints the RuntimeError (cli.py swallows its text), so the
    reason has to reach the per-member log line too.
    """
    cfg = make_network_config(make_targets("a", "off", disabled=("off",)))
    log = FakeLog()

    with pytest.raises(RuntimeError, match="disabled"):
        lifecycle.do_stop_node(cfg, "off", log, Path("configs/x.yaml"))
    assert any("disabled" in text for _lvl, text in log.lines)


def test_wait_pods_filters_by_app(monkeypatch) -> None:
    """Ready worker pods must not satisfy an edge-pod count.

    Without the filter, `do_start` logs success while the head's llama-server
    has never loaded — the demo then runs against a dead endpoint.
    """
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    emitted: list[tuple[str, str]] = []

    pods = [
        {"name": "aureax-shard-aaa", "node": "n1", "ready": True},
        {"name": "aureax-shard-bbb", "node": "n2", "ready": True},
        {"name": "aureax-edge-ccc", "node": "n1", "ready": True},
    ]
    # Two shard pods are Ready on these hosts, but only one edge pod is.
    lifecycle._wait_pods_on_hosts(
        FakeKubectl(pods=pods), "aureax", ["n1", "n2"], 1,
        log=FakeLog(emitted), mgr=None, label="edge", app="aureax-edge",
    )
    assert any("1 edge pod(s) Ready" in t for _, t in emitted)


def test_edge_label_pass_skips_nodes_that_are_not_enabled_targets(monkeypatch) -> None:
    """Only nodes the config asked for may get the edge label.

    The pass iterates every node joined to the cluster, so a deny-list ("not
    control-plane, not deferred") also labelled targets marked
    `enabled: false` — and any other machine sharing the cluster. Each then
    gets an edge pod, which for a disabled target is exactly what disabling
    was supposed to prevent.
    """
    cfg = make_network_config(make_targets("edge-a", "edge-b", disabled=("edge-b",)))

    # edge-b is still joined to the cluster even though it is disabled.
    kube = FakeKubectl(nodes=[
        {"name": "ctrl", "roles": "control-plane", "labels": {}},
        {"name": "edge-a", "roles": "", "labels": {}},
        {"name": "edge-b", "roles": "", "labels": {}},
    ])

    monkeypatch.setattr(
        lifecycle, "_split_hostnames_by_role", lambda _c, _p: (["edge-a"], [], [])
    )
    monkeypatch.setattr(lifecycle, "_get_daemonset_node_selector", lambda *_a, **_k: "edge")
    monkeypatch.setattr(lifecycle, "_wait_pods_on_hosts", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    strips = _stub_label_strips(monkeypatch)

    lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kube)

    assert "edge-a" in kube.labelled_names
    assert [name for name, _key in strips] == ["edge-a"], (
        "the conflicting-label strip must follow the same allow-list as the label pass"
    )
    assert "edge-b" not in kube.labelled_names, "labelled a target marked enabled: false"
    assert "ctrl" not in kube.labelled_names


def test_nodes_status_never_dials_a_workers_rpc_port(monkeypatch, tmp_path) -> None:
    """A worker's state comes from its head, not from dialing :50052.

    rpc-server accepts one client with a listen backlog of 1, so an off-node
    probe goes unaccepted once the head is attached — the dot would flap red
    while the worker does exactly its job, and each probe leaves a queued
    connection behind. daemonset-shard.yaml carries the same rule; see also
    ARCHITECTURE.md, "Invariants and how each is enforced".

    A worker has no NodePort 30800 of its own either, so it reports the group's
    state, read from the head.
    """
    net = tmp_path / "aureax-network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n"
        "    - {node_id: h, host: 10.77.0.1, config_file: h.yaml}\n"
        "    - {node_id: w1, host: 10.77.0.2, config_file: w.yaml}\n"
        "node_groups:\n  - {name: g, head: h, workers: [w1]}\n"
    )
    (tmp_path / "h.yaml").write_text(
        'node: {id: h, role: edge_executor, control_plane: {endpoint: x}}\n'
        "model: {name: M, filename: m.gguf}\n"
    )
    (tmp_path / "w.yaml").write_text(
        'node: {id: w1, role: shard_worker, control_plane: {endpoint: x}}\n'
        "model: {name: M, filename: m.gguf}\n"
    )

    probed: list[tuple[str, int]] = []
    monkeypatch.setattr(
        lifecycle, "tcp_reachable",
        lambda host, port, timeout=0.3: probed.append((host, port)) or True,
    )
    rows = lifecycle._nodes_status(net)
    by_id = {r["node_id"]: r for r in rows}
    assert by_id["h"]["role"] == "edge_executor"
    assert by_id["w1"]["role"] == "shard_worker"
    assert by_id["w1"]["group"] == "g"

    assert ("10.77.0.1", 30800) in probed
    assert not any(port == 50052 for _, port in probed), (
        "no socket may be opened to a worker's RPC port from the orchestrator"
    )
    assert ("10.77.0.2", 30800) not in probed, "a worker has no NodePort of its own"
    # Head and worker share one probe of the head, not two.
    assert probed == [("10.77.0.1", 30800)]
    assert by_id["w1"]["running"] is True


def test_an_unreadable_network_config_is_re_read_once_per_ttl(monkeypatch, tmp_path) -> None:
    """The load-failure path must stamp the cached PATH, not only the timestamp.

    The entry guard requires both a matching path and a fresh timestamp, and
    the path is otherwise stamped only on the success path — so a config that
    has never loaded is re-read and re-warned on every 1.5s dashboard poll,
    which is exactly what the timestamp is there to prevent.
    """
    from orchestrator.config import loader

    missing = tmp_path / "absent.yaml"
    reads = {"n": 0}

    def boom(path):
        reads["n"] += 1
        raise FileNotFoundError(path)

    monkeypatch.setattr(loader, "load_network_config", boom)

    assert lifecycle._nodes_status(missing) == []
    assert lifecycle._nodes_status(missing) == []

    assert reads["n"] == 1, "re-read a config the TTL should have suppressed"


def test_pod_wait_times_out_instead_of_blocking_forever(monkeypatch) -> None:
    """The readiness loop had no deadline and no cancel path from the CLI.

    `_sleep_or_cancel` honours the dashboard's cancel_event, but the CLI passes
    mgr=None — so a pod that never became Ready hung `aureax start` until the
    operator killed it.
    """
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    monkeypatch.setattr(lifecycle, "_log_pod_diagnostics", lambda *a, **k: None)

    log = FakeLog()
    never_ready = [{"name": "aureax-edge-x", "node": "n1", "ready": False}]

    with pytest.raises(RuntimeError, match="timed out"):
        lifecycle._wait_pods_on_hosts(
            FakeKubectl(pods=never_ready), "aureax", ["n1"], 1,
            log=log, mgr=None, label="edge", app="aureax-edge",
            timeout_seconds=10,
        )
    assert any(lvl == "error" and "timed out" in t for lvl, t in log.lines)


def test_pod_wait_still_succeeds_before_the_deadline(monkeypatch) -> None:
    """The deadline must not break the normal path."""
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)

    log = FakeLog()
    ready = [{"name": "aureax-edge-x", "node": "n1", "ready": True}]
    lifecycle._wait_pods_on_hosts(
        FakeKubectl(pods=ready), "aureax", ["n1"], 1,
        log=log, mgr=None, label="edge", app="aureax-edge", timeout_seconds=10,
    )
    assert any(lvl == "success" for lvl, _ in log.lines)


def test_a_planner_that_is_not_a_k8s_node_is_fatal(monkeypatch) -> None:
    """An unresolved planner is a node_id, not a K8s node name.

    _split_hostnames_by_role deliberately carries the node_id as a placeholder
    so the node still counts. Handing that to `kubectl label node` fails with
    "node not found", which reads as a cluster fault; skipping it with a warn
    left the planner unlabelled, unwaited-for, and the start reporting success.
    An edge or worker placeholder is made fatal by its pod wait, so a planner
    must be too.
    """
    cfg = make_network_config(make_targets("edge-a", "planner-01"))
    # The planner is absent from the cluster's node list.
    kube = FakeKubectl(nodes=[{"name": "edge-a", "roles": "", "labels": {}}])
    log = FakeLog()
    # "planner-01" is the node_id placeholder — it is not a K8s node name.
    monkeypatch.setattr(
        lifecycle, "_split_hostnames_by_role", lambda _c, _p: (["edge-a"], [], ["planner-01"])
    )
    monkeypatch.setattr(lifecycle, "_get_daemonset_node_selector", lambda *_a, **_k: "edge")
    monkeypatch.setattr(lifecycle, "_wait_pods_on_hosts", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    _stub_label_strips(monkeypatch)

    with pytest.raises(RuntimeError, match="planner 'planner-01' is not a node in the cluster"):
        lifecycle._start_phase_label_and_wait_pods(cfg, log, None, kube)

    assert "planner-01" not in kube.labelled_names, (
        "passed an unresolved placeholder to kubectl"
    )


def test_per_node_start_restores_a_suspended_daemonset_selector(monkeypatch) -> None:
    """Labelling a node schedules nothing while the DaemonSet is suspended.

    `aureax stop` points the nodeSelector at a value no node carries. A later
    per-node start added the node's label and reported "pods will be scheduled
    by K8s" — but the DaemonSet still matched nothing, so no pod ever appeared.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    # The DaemonSet is currently suspended, as after a whole-stack stop.
    calls = _labels(monkeypatch, selector=cfg.kubernetes.stopped_label_value)

    lifecycle.do_start_node(cfg, "solo", FakeLog(), Path("configs/x.yaml"))

    patches = [c for c in calls if c[0] == "patch"]
    assert patches, "labelled the node but left the DaemonSet suspended"
    assert patches[0][1:] == (
        "aureax-edge", cfg.kubernetes.node_label, cfg.kubernetes.edge_label_value,
    )


def test_per_node_start_does_not_repatch_a_live_selector(monkeypatch) -> None:
    """Idempotent: no patch when the selector is already the live value."""
    cfg = _cfg()
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch, selector=cfg.kubernetes.edge_label_value)

    lifecycle.do_start_node(cfg, "solo", FakeLog(), Path("configs/x.yaml"))

    assert not [c for c in calls if c[0] == "patch"]


# --- role resolution: "" and None mean different things ----------------------

def test_node_role_tells_unset_apart_from_unreadable(tmp_path) -> None:
    """"" routes a node to the edge label; a config we could not read must not.

    A `config_file` typo or a deleted per-node YAML used to read as "no role",
    which demotes a shard worker to edge: per-node stop then removes a label the
    worker never had, reports success, and leaves rpc-server running.
    """
    cfg = make_network_config([
        {"node_id": "a", "host": "10.77.0.1", "config_file": "gone.yaml"},
        {"node_id": "b", "host": "10.77.0.2"},
    ])
    net = tmp_path / "aureax-network.yaml"
    by_id = {t.node_id: t for t in cfg.ssh.targets}

    assert lifecycle.node_role(by_id["a"], net) is None, "a missing file is not 'no role'"
    assert lifecycle.node_role(by_id["b"], net) == "", "no config_file at all IS 'no role'"


def test_node_role_returns_none_for_an_unparseable_config(tmp_path) -> None:
    """A YAML typo in a shared worker template is the likelier trigger."""
    (tmp_path / "a.yaml").write_text("node: {id: a\n")  # unterminated flow mapping
    cfg = make_network_config(make_targets("a"))
    assert lifecycle.node_role(cfg.ssh.targets[0], tmp_path / "aureax-network.yaml") is None


def test_label_key_refuses_to_guess_when_the_config_is_unreadable(tmp_path) -> None:
    cfg = make_network_config(make_targets("a"))
    with pytest.raises(RuntimeError, match="could not read its per-node config"):
        lifecycle._label_key_for_node(cfg, "a", tmp_path / "aureax-network.yaml")


def test_label_key_for_a_group_worker_reads_no_file_at_all(tmp_path) -> None:
    """Group membership already answers it: a worker runs rpc-server."""
    cfg = make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )
    key = lifecycle._label_key_for_node(cfg, "w1", tmp_path / "aureax-network.yaml")
    assert key == cfg.kubernetes.shard_label


def test_split_hostnames_names_every_unreadable_config_at_once(monkeypatch, tmp_path) -> None:
    """One error listing all offenders, not one abort on the first."""
    cfg = make_network_config(make_targets("alpha", "beta"))
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    with pytest.raises(RuntimeError, match="could not read the per-node config") as exc:
        lifecycle._split_hostnames_by_role(cfg, tmp_path / "aureax-network.yaml")
    assert "alpha" in str(exc.value) and "beta" in str(exc.value)


# --- hostname resolution ------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("node1\n", "node1"),
    ("Welcome to node1!\nnode1\n", "node1"),   # rc-file banner ahead of the answer
    ("", None),
    ("   \n", None),
    ("two words\n", None),
])
def test_hostname_from_output_takes_the_last_line(raw, expected) -> None:
    """The whole blob used to be cached as the node's K8s name, where it never
    matches and the pod wait burns its full timeout."""
    assert lifecycle._hostname_from_output(raw) == expected


def test_ssh_target_cache_follows_a_host_remap(monkeypatch) -> None:
    """The map remembers a node_id, not the SSHTarget object it first matched.

    _log_via_ssh reloads the network config on every call precisely to pick up
    edits; a cached object kept dialling the address of whichever load first
    populated the map, so a host remap never reached a running dashboard.
    """
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    before = _cfg()
    assert lifecycle._resolve_ssh_target_for_node(before, "host-solo").host == "10.77.0.9"

    after = NetworkConfig.model_validate({
        **NETWORK_BASE,
        "ssh": {"targets": [{"node_id": "solo", "host": "10.77.0.99", "config_file": "s.yaml"}]},
        "node_groups": [],
    })
    assert lifecycle._resolve_ssh_target_for_node(after, "host-solo").host == "10.77.0.99"


# --- per-node start: labelling and selector restore --------------------------

def test_group_start_stops_at_the_first_label_failure(monkeypatch) -> None:
    """`order` is workers-first so the head never boots without its peers.

    Continuing past a failed worker label and labelling the head anyway
    schedules a llama-server that waits out its peer timeout and crash-loops,
    and the raw CalledProcessError escaped as a traceback (cli.py catches only
    RuntimeError) with the group half-labelled and nothing naming the failure.
    """
    import subprocess

    from orchestrator.cluster import kubectl

    cfg = _cfg()
    _roles(monkeypatch, {"h": "edge_executor", "w1": "shard_worker", "w2": "shard_worker"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch, selector=cfg.kubernetes.stopped_label_value)

    def label(name, key, value):
        calls.append(("add", name, key, value))
        if name == "host-w1":
            raise subprocess.CalledProcessError(1, ["kubectl"], stderr="node not found")

    monkeypatch.setattr(kubectl, "label_node", label)
    log = FakeLog()

    with pytest.raises(RuntimeError, match="could not label w1") as exc:
        lifecycle.do_start_node(cfg, "w1", log, Path("configs/x.yaml"))

    assert "w2" in str(exc.value), "the message must name what was already labelled"
    assert not [c for c in calls if c[0] == "patch"], (
        "un-suspended the DaemonSet for a half-labelled group"
    )
    assert not any(lvl == "success" for lvl, _t in log.lines)


def test_group_start_restores_the_shard_selector_before_the_edge_one(monkeypatch) -> None:
    """Alphabetical order put aureax.io/role (the head) ahead of the workers.

    After a whole-stack stop both selectors point at "stopped", so these
    patches are what actually schedule the pods — the careful workers-first
    label order above buys nothing if the head's DaemonSet resumes first.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"h": "edge_executor", "w1": "shard_worker", "w2": "shard_worker"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch, selector=cfg.kubernetes.stopped_label_value)

    lifecycle.do_start_node(cfg, "w1", FakeLog(), Path("configs/x.yaml"))

    assert [c[1] for c in calls if c[0] == "patch"] == ["aureax-shard", "aureax-edge"]


@pytest.mark.parametrize("outcome,message", [
    ("absent", "not deployed"),
    ("failed", "patch failed"),
])
def test_per_node_start_raises_when_the_selector_restore_does_not_land(
    monkeypatch, outcome, message
) -> None:
    """The label alone schedules nothing while the selector says "stopped"."""
    cfg = _cfg()
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    _labels(monkeypatch, selector=cfg.kubernetes.stopped_label_value, patch_outcome=outcome)
    log = FakeLog()

    with pytest.raises(RuntimeError, match=message) as exc:
        lifecycle.do_start_node(cfg, "solo", log, Path("configs/x.yaml"))

    assert "aureax-edge" in str(exc.value)
    assert not any(lvl == "success" for lvl, _t in log.lines), "reported a start that never happened"


def test_per_node_start_admits_it_resumes_the_whole_fleet(monkeypatch) -> None:
    """Restoring a suspended selector re-schedules every still-labelled node.

    `aureax stop` never removes node labels, so `start --node X` afterwards
    brings the fleet back while the success line claimed one node.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    _labels(monkeypatch, selector=cfg.kubernetes.stopped_label_value)
    log = FakeLog()

    lifecycle.do_start_node(cfg, "solo", log, Path("configs/x.yaml"))

    assert any(lvl == "warn" and "EVERY node" in text for lvl, text in log.lines)
    assert any(lvl == "success" and "every other node" in text for lvl, text in log.lines)


# --- per-node start: label reconciliation ------------------------------------
# The same hazard the fleet start closes, on the path that actually flips a
# role: demoting a head into an RPC slot is the documented 122B rollback, and
# `aureax start --node` is what is reached for afterwards instead of a full
# deploy. These mirror the whole-fleet cases further down.

def test_per_node_start_strips_the_edge_label_from_a_member_it_labels_as_a_worker(
    monkeypatch,
) -> None:
    """A node demoted to shard worker keeps `aureax.io/role=edge` otherwise.

    It then also gets an edge pod: a llama-server with no model file,
    crash-looping against the very group it is supposed to serve.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"h": "edge_executor", "w1": "shard_worker", "w2": "shard_worker"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch)

    lifecycle.do_start_node(cfg, "w1", FakeLog(), Path("configs/x.yaml"))

    assert ("rm", "host-w1", cfg.kubernetes.node_label) in calls
    assert ("rm", "host-w2", cfg.kubernetes.node_label) in calls


def test_per_node_start_strips_the_shard_label_from_a_member_it_labels_as_an_edge(
    monkeypatch,
) -> None:
    """The promotion direction: a former worker now heading its group.

    It would keep `aureax.io/shard` and run rpc-server beside its llama-server,
    the two of them competing for the node's whole RAM.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"h": "edge_executor", "w1": "shard_worker", "w2": "shard_worker"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch)

    lifecycle.do_start_node(cfg, "h", FakeLog(), Path("configs/x.yaml"))

    assert ("rm", "host-h", cfg.kubernetes.shard_label) in calls


def test_a_double_labelled_member_of_a_per_node_start_is_reported(monkeypatch) -> None:
    """The strip fixes it; fixing it silently would leave no record.

    Nothing else in the stack reports that a machine has been running two
    workloads.
    """
    cfg = _cfg()
    k8s = cfg.kubernetes
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch, nodes=[
        {"name": "host-solo", "roles": "",
         "labels": {k8s.node_label: "edge", k8s.shard_label: "worker"}},
    ])
    log = FakeLog()

    lifecycle.do_start_node(cfg, "solo", log, Path("configs/x.yaml"))

    assert any(
        lvl == "error" and "host-solo" in text and k8s.shard_label in text
        for lvl, text in log.lines
    ), f"a double-labelled node was fixed without a word: {log.lines}"
    assert ("rm", "host-solo", k8s.shard_label) in calls


def test_a_double_labelled_node_a_per_node_start_does_not_claim_aborts_it(
    monkeypatch,
) -> None:
    """Fatal here too, because the selector restore is fleet-wide.

    Per-node start does not label the stray, but it does un-suspend the
    DaemonSets — which schedules BOTH pods onto a machine already contending
    with itself for RAM, while the success line names one node.
    """
    cfg = _cfg()
    k8s = cfg.kubernetes
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(monkeypatch, selector=k8s.stopped_label_value, nodes=[
        {"name": "host-solo", "roles": "", "labels": {}},
        {"name": "stray", "roles": "",
         "labels": {k8s.node_label: "edge", k8s.shard_label: "worker"}},
    ])
    log = FakeLog()

    with pytest.raises(RuntimeError, match="in none of the nodes this start claims") as exc:
        lifecycle.do_start_node(cfg, "solo", log, Path("configs/x.yaml"))

    assert f"kubectl label node stray {k8s.shard_label}-" in str(exc.value)
    assert f"kubectl label node stray {k8s.node_label}-" in str(exc.value)
    assert calls == [], "the pre-flight must run before any label write or patch"


def test_a_failed_label_strip_aborts_a_per_node_start(monkeypatch) -> None:
    """Start is the last writer before pods schedule, per-node start included.

    Continuing would restore the DaemonSet selectors over a node still matching
    both of them, and report a start that put two pods on one machine.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    calls = _labels(
        monkeypatch,
        selector=cfg.kubernetes.stopped_label_value,
        remove_result=(False, "connection refused"),
    )
    log = FakeLog()

    with pytest.raises(RuntimeError, match="could not strip") as exc:
        lifecycle.do_start_node(cfg, "solo", log, Path("configs/x.yaml"))

    assert "connection refused" in str(exc.value)
    assert cfg.kubernetes.shard_label in str(exc.value)
    assert not [c for c in calls if c[0] == "patch"], (
        "un-suspended the DaemonSets over a node matching both selectors"
    )
    assert not any(lvl == "success" for lvl, _t in log.lines)


# --- whole-fleet start: phase A0 ---------------------------------------------

class _PhaseKubectl:
    """kubectl stand-in for _start_phase_label_and_wait_pods.

    ``labels`` seeds each node's existing label dict, which is what the
    double-label pre-flight reads. Label REMOVAL is deliberately not a method
    here: the strip runs
    through ``orchestrator.cluster.labels``, which holds the real kubectl
    module rather than the stand-in the phase is handed, so tests install it
    with :func:`_stub_label_strips`.
    """

    def __init__(
        self, names: list[str], *,
        labels: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self._names = names
        self._labels = labels or {}
        self.labelled: list[tuple[str, str, str]] = []

    def get_nodes(self):
        return [
            {"name": n, "roles": "", "labels": dict(self._labels.get(n, {}))}
            for n in self._names
        ]

    def get_pods(self, _namespace):
        # Empty, and no way to seed it: the phase's only pod reader is
        # _wait_pods_on_hosts, which every test here stubs. The build gate asks
        # the nodes over SSH, not the pods — `test_the_gate_compares_the_
        # binaries_not_the_image` pins that this is never called.
        return []

    def label_node(self, name, key, value):
        self.labelled.append((name, key, value))

    def patch(self, *_a, **_k):
        return ""

    @property
    def labelled_names(self) -> list[str]:
        return [name for name, _key, _value in self.labelled]


def _stub_label_strips(
    monkeypatch, result: tuple[bool, str] = (True, "")
) -> list[tuple[str, str]]:
    """Record `(node, label)` strips instead of shelling out to kubectl.

    The start path strips through ``orchestrator.cluster.labels``, which
    imports the real kubectl module — so without this, every test that drives
    the label phase would run `kubectl label node` against whatever cluster the
    developer happens to have. ``result`` is what kubectl reports back; the
    False case is the one that has to abort a start.
    """
    from orchestrator.cluster import kubectl as real_kubectl

    strips: list[tuple[str, str]] = []
    monkeypatch.setattr(
        real_kubectl, "remove_label",
        lambda name, key: (strips.append((name, key)), result)[1],
    )
    return strips


def _phase_env(
    monkeypatch, *, edges: tuple[str, ...], workers: tuple[str, ...] = (),
    planners: tuple[str, ...] = (), order: list[str] | None = None,
    strip: tuple[bool, str] = (True, ""),
) -> list[tuple[str, str]]:
    """Stub everything `_start_phase_label_and_wait_pods` reaches past kubectl.

    Hostnames resolve as ``host-<node_id>`` so a member's node name is
    predictable, and without the stub ``_resolve_node_hostname`` opens an SSH
    connection.
    """
    monkeypatch.setattr(
        lifecycle, "_split_hostnames_by_role",
        lambda _c, _p: (list(edges), list(workers), list(planners)),
    )
    monkeypatch.setattr(lifecycle, "_get_daemonset_node_selector", lambda *_a, **_k: "")
    monkeypatch.setattr(
        lifecycle, "_patch_daemonset_selector",
        lambda _kc, _ns, ds, *_a, **_k: (
            order.append(f"patch:{ds}") if order is not None else None
        ) or "patched",
    )
    monkeypatch.setattr(
        lifecycle, "_wait_pods_on_hosts",
        lambda _kc, _ns, _h, _n, **kw: (
            order.append(f"wait:{kw['app']}") if order is not None else None
        ),
    )
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda _c, node_id: f"host-{node_id}"
    )
    return _stub_label_strips(monkeypatch, result=strip)


def test_shard_workers_are_labelled_and_waited_for_before_the_edges(monkeypatch) -> None:
    """Phase A0 had no coverage at all.

    A head's llama-server exits immediately if an RPC peer is unreachable when
    it loads, so the workers' label, their DaemonSet selector and their pod
    wait all have to precede the edge half.
    """
    cfg = make_network_config(
        make_targets("h", "w1", "w2"),
        [{"name": "g", "head": "h", "workers": ["w1", "w2"]}],
    )
    kc = _PhaseKubectl(
        ["h", "w1", "w2"],
    )
    order: list[str] = []

    monkeypatch.setattr(
        lifecycle, "_split_hostnames_by_role", lambda _c, _p: (["h"], ["w1", "w2"], [])
    )
    monkeypatch.setattr(lifecycle, "_get_daemonset_node_selector", lambda *_a, **_k: "")
    monkeypatch.setattr(
        lifecycle, "_patch_daemonset_selector",
        lambda _kc, _ns, ds, *_a, **_k: order.append(f"patch:{ds}") or "patched",
    )
    monkeypatch.setattr(
        lifecycle, "_wait_pods_on_hosts",
        lambda _kc, _ns, hosts, _n, **kw: order.append(f"wait:{kw['app']}"),
    )
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    # This fleet's K8s node names ARE its node ids.
    monkeypatch.setattr(lifecycle, "_resolve_node_hostname", lambda _c, node_id: node_id)
    _stub_label_strips(monkeypatch)

    lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)

    k8s = cfg.kubernetes
    by_name = {n: (k, v) for n, k, v in kc.labelled}
    assert by_name["w1"] == (k8s.shard_label, k8s.shard_label_value)
    assert by_name["w2"] == (k8s.shard_label, k8s.shard_label_value)
    assert by_name["h"] == (k8s.node_label, k8s.edge_label_value)
    assert order == [
        "patch:aureax-shard", "wait:aureax-shard", "patch:aureax-edge", "wait:aureax-edge",
    ]


def test_an_undeployed_shard_daemonset_raises_before_any_edge_wait(monkeypatch) -> None:
    """Waiting behind a selector that never landed burns the whole timeout."""
    cfg = make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )
    waits: list[str] = []

    monkeypatch.setattr(
        lifecycle, "_split_hostnames_by_role", lambda _c, _p: (["h"], ["w1"], [])
    )
    monkeypatch.setattr(lifecycle, "_get_daemonset_node_selector", lambda *_a, **_k: "")
    monkeypatch.setattr(
        lifecycle, "_patch_daemonset_selector", lambda *_a, **_k: "absent"
    )
    monkeypatch.setattr(
        lifecycle, "_wait_pods_on_hosts",
        lambda _kc, _ns, _h, _n, **kw: waits.append(kw["app"]),
    )
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    _stub_label_strips(monkeypatch)

    with pytest.raises(RuntimeError, match="aureax-shard DaemonSet is not deployed"):
        lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, _PhaseKubectl(["h", "w1"]))
    assert waits == []


def test_an_empty_node_snapshot_is_not_treated_as_an_empty_fleet(monkeypatch) -> None:
    """kubectl.get_nodes() returns [] on ANY API error.

    Left alone that labels nothing, patches the selector anyway and spends the
    whole pod-wait timeout blaming the pods.
    """
    cfg = make_network_config(make_targets("edge-a"))
    monkeypatch.setattr(
        lifecycle, "_split_hostnames_by_role", lambda _c, _p: (["edge-a"], [], [])
    )
    monkeypatch.setattr(lifecycle, "_get_daemonset_node_selector", lambda *_a, **_k: "edge")
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)

    kc = _PhaseKubectl([])
    with pytest.raises(RuntimeError, match="API server is unreachable"):
        lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)
    assert kc.labelled == []


# --- pod-wait diagnostics -----------------------------------------------------

def test_a_crash_is_dumped_once_per_restart_not_once_per_round(monkeypatch) -> None:
    """lastState.terminated persists for the pod's whole life.

    Without dedupe the same 60 stdout lines were re-fetched and re-emitted
    every 30s for the rest of the wait — one SQLite commit per line.
    """
    pod = {
        "metadata": {"uid": "uid-1"},
        "spec": {"nodeName": "n1"},
        "status": {
            "phase": "Running",
            "containerStatuses": [{
                "name": "llama",
                "ready": False,
                "restartCount": 1,
                "state": {"running": {"startedAt": "now"}},
                "lastState": {"terminated": {"reason": "Error", "exitCode": 1}},
            }],
        },
    }
    dumps: list[str] = []
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    monkeypatch.setattr(
        lifecycle, "_diagnostic_snapshot", lambda _ns, _log: ({"aureax-edge-x": pod}, {})
    )
    monkeypatch.setattr(
        lifecycle, "_log_crashed_container_stdout",
        lambda **kw: dumps.append(kw["container"]) or True,
    )

    never_ready = [{"name": "aureax-edge-x", "node": "n1", "ready": False}]
    with pytest.raises(RuntimeError, match="timed out"):
        lifecycle._wait_pods_on_hosts(
            FakeKubectl(pods=never_ready), "aureax", ["n1"], 1,
            log=FakeLog(), mgr=None, label="edge", app="aureax-edge", timeout_seconds=120,
        )

    assert dumps == ["llama"], f"re-dumped the same crash every round: {dumps}"


def test_a_failed_dump_is_retried_next_round(monkeypatch) -> None:
    """Recording the key before the dump would suppress the real crash logs
    for that restart after one transient `kubectl logs` failure."""
    pod = {
        "metadata": {"uid": "uid-1"},
        "spec": {"nodeName": "n1"},
        "status": {
            "phase": "Running",
            "containerStatuses": [{
                "name": "llama",
                "ready": False,
                "restartCount": 2,
                "state": {"running": {"startedAt": "now"}},
                "lastState": {"terminated": {"reason": "Error", "exitCode": 1}},
            }],
        },
    }
    attempts: list[str] = []
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    monkeypatch.setattr(
        lifecycle, "_diagnostic_snapshot", lambda _ns, _log: ({"aureax-edge-x": pod}, {})
    )
    monkeypatch.setattr(
        lifecycle, "_log_crashed_container_stdout",
        lambda **kw: attempts.append(kw["container"]) and False,
    )

    never_ready = [{"name": "aureax-edge-x", "node": "n1", "ready": False}]
    with pytest.raises(RuntimeError, match="timed out"):
        lifecycle._wait_pods_on_hosts(
            FakeKubectl(pods=never_ready), "aureax", ["n1"], 1,
            log=FakeLog(), mgr=None, label="edge", app="aureax-edge", timeout_seconds=90,
        )

    assert len(attempts) > 1, "a failed fetch must not count as dumped"


def test_per_node_stop_refuses_to_report_success_on_a_failed_label_removal(monkeypatch) -> None:
    """The label is what gates the workload.

    kubectl.remove_label already maps "label not found" to ok=True, so a False
    means the label is still there and the pods are still running — reporting
    "Stopped N node(s)" past that is the one lie this path can tell.
    """
    cfg = _cfg()
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    _labels(monkeypatch, remove_result=(False, "connection refused"))
    log = FakeLog()

    with pytest.raises(RuntimeError, match="still running") as exc:
        lifecycle.do_stop_node(cfg, "solo", log, Path("configs/x.yaml"))

    assert "solo" in str(exc.value)
    assert any(lvl == "error" and "solo" in text for lvl, text in log.lines)
    assert not any(lvl == "success" for lvl, _t in log.lines)


# --- whole-fleet start: label reconciliation ---------------------------------
# Both label passes only ever ADDED, so a node whose role flipped kept the label
# it was leaving behind, matched both DaemonSet selectors and ran llama-server
# and rpc-server side by side on one machine. Deploy reconciles that; `aureax
# start` did not, and start is the last writer before pods schedule.

def test_start_strips_the_edge_label_from_a_node_it_labels_as_a_worker(monkeypatch) -> None:
    """A node demoted to shard worker keeps `aureax.io/role=edge` otherwise.

    It then also gets an edge pod: a llama-server with no model file,
    crash-looping against the very group it is supposed to serve.
    """
    cfg = make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )
    kc = _PhaseKubectl(
        ["host-h", "host-w1"],
        labels={"host-w1": {cfg.kubernetes.node_label: "edge"}},
    )
    strips = _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",))

    lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)

    assert ("host-w1", cfg.kubernetes.node_label) in strips


def test_start_strips_the_shard_label_from_a_node_it_labels_as_an_edge(monkeypatch) -> None:
    """The promotion direction: a former worker now heading its group."""
    cfg = make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )
    kc = _PhaseKubectl(
        ["host-h", "host-w1"],
        labels={"host-h": {cfg.kubernetes.shard_label: "worker"}},
    )
    strips = _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",))

    lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)

    assert ("host-h", cfg.kubernetes.shard_label) in strips


def test_start_strips_the_shard_label_from_a_planner(monkeypatch) -> None:
    """The planner is labelled in its own phase, so it needs its own strip."""
    cfg = make_network_config(make_targets("edge-a", "planner-01"))
    kc = _PhaseKubectl(["host-edge-a", "host-planner-01"])
    strips = _phase_env(
        monkeypatch, edges=("host-edge-a",), planners=("host-planner-01",)
    )

    lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)

    assert ("host-planner-01", cfg.kubernetes.shard_label) in strips


def test_a_node_carrying_both_labels_is_reported_at_error_level(monkeypatch) -> None:
    """Nothing else in the stack reports two pods on one node.

    The start fixes it, but silently fixing it would leave the operator with no
    record that a machine had been running two workloads.
    """
    cfg = make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )
    k8s = cfg.kubernetes
    kc = _PhaseKubectl(
        ["host-h", "host-w1"],
        labels={"host-w1": {k8s.node_label: "edge", k8s.shard_label: "worker"}},
    )
    _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",))
    log = FakeLog()

    lifecycle._start_phase_label_and_wait_pods(cfg, log, None, kc)

    assert any(
        lvl == "error" and "host-w1" in text and k8s.shard_label in text
        for lvl, text in log.lines
    ), f"a double-labelled node was fixed without a word: {log.lines}"


def test_a_double_labelled_node_this_start_does_not_claim_aborts_the_start(monkeypatch) -> None:
    """Nothing in this run would fix it, and it is running two pods right now.

    Continuing would schedule the rest of the fleet around a machine already
    contending with itself for RAM, and report success.
    """
    cfg = make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )
    k8s = cfg.kubernetes
    kc = _PhaseKubectl(
        ["host-h", "host-w1", "stray"],
        labels={"stray": {k8s.node_label: "edge", k8s.shard_label: "worker"}},
    )
    strips = _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",))

    with pytest.raises(RuntimeError, match="in none of the nodes this start claims") as exc:
        lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)

    assert f"kubectl label node stray {k8s.shard_label}-" in str(exc.value)
    assert f"kubectl label node stray {k8s.node_label}-" in str(exc.value)
    # `aureax deploy` was offered as the third remedy, and its two label passes
    # walk enabled_targets — which is exactly what a node in neither list is not.
    assert "walk only the targets this fleet file marks enabled" in str(exc.value)
    assert kc.labelled == [] and strips == [], "the pre-flight must run before any label write"


@pytest.mark.parametrize("workers,label_key,daemonset", [
    # The worker pass strips the edge key, so the pod it would leave behind is
    # a llama-server; the edge pass strips the shard key, an rpc-server.
    (("host-w1",), "node_label", "aureax-edge"),
    ((), "shard_label", "aureax-shard"),
])
def test_a_failed_label_strip_aborts_the_start(
    monkeypatch, workers, label_key, daemonset
) -> None:
    """Deploy logs and continues here; start raises.

    The asymmetry is the point: a deploy is normally followed by a start, which
    gets another chance at the node, but start is the last writer before pods
    schedule — a label it failed to remove is a second pod for the whole run.
    """
    cfg = make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )
    kc = _PhaseKubectl(["host-h", "host-w1"])
    _phase_env(
        monkeypatch, edges=("host-h",), workers=workers,
        strip=(False, "connection refused"),
    )
    log = FakeLog()

    with pytest.raises(RuntimeError, match="could not strip") as exc:
        lifecycle._start_phase_label_and_wait_pods(cfg, log, None, kc)

    assert "connection refused" in str(exc.value)
    assert getattr(cfg.kubernetes, label_key) in str(exc.value)
    assert f"{daemonset} DaemonSet" in str(exc.value)
    assert any(lvl == "error" for lvl, _t in log.lines)


# --- whole-fleet start: the group build gate ---------------------------------
# The llama.cpp RPC wire format is version-locked. A head that dials a peer
# built from another commit connects and then stops, with no error on either
# side — 50 minutes of 0/1 Running until the startupProbe budget expires.
#
# The gate asks the NODES for the sha256 of llama-server and rpc-server inside
# their images. It used to compare the pods' image digests instead, and that
# refused every start on the real fleet: each node builds `aureax-client:dev`
# natively, so the digests differ by construction while the binaries do not.

def _grouped_fleet():
    return make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )


def test_a_worker_running_other_binaries_than_the_head_fails_the_start(monkeypatch) -> None:
    cfg = _grouped_fleet()
    kc = _PhaseKubectl(["host-h", "host-w1"])
    order: list[str] = []
    _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",), order=order)
    _stub_build_probe(monkeypatch, sums={"w1": _SUMS_B})

    with pytest.raises(RuntimeError, match="HANGS at the handshake") as exc:
        lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)

    message = str(exc.value)
    assert _SUMS_A[1][:12] in message and _SUMS_B[1][:12] in message, "name both checksums"
    assert "FAIL w1" in message, "name the odd node out"
    assert "./scripts/build-on-node.sh" in message and "./scripts/update-all.sh" in message
    assert kc.labelled == [], "labelled a node in a group that cannot serve"
    assert order == [], "patched a selector on a known-broken group"


def test_identical_binaries_pass_the_gate(monkeypatch) -> None:
    cfg = _grouped_fleet()
    order: list[str] = []
    _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",), order=order)
    log = FakeLog()

    lifecycle._start_phase_label_and_wait_pods(
        cfg, log, None, _PhaseKubectl(["host-h", "host-w1"])
    )

    assert "wait:aureax-edge" in order
    assert any(
        "g: OK   llama-server/rpc-server checksums identical" in text
        for _lvl, text in log.lines
    ), f"the start log has to name the group it cleared: {log.lines}"


def test_a_member_that_could_not_be_probed_warns_and_does_not_block_the_start(monkeypatch) -> None:
    """Fail OPEN on a node we could not ask.

    Refusing a start over an SSH blip is a worse bug than the one this gate
    closes — and the digest gate that failed CLOSED refused every start on a
    fleet whose binaries were identical.
    """
    cfg = _grouped_fleet()
    order: list[str] = []
    _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",), order=order)
    _stub_build_probe(monkeypatch, unreadable=("w1",))
    log = FakeLog()

    lifecycle._start_phase_label_and_wait_pods(
        cfg, log, None, _PhaseKubectl(["host-h", "host-w1"])
    )

    assert any(
        lvl == "warn" and "did not cover this node" in text for lvl, text in log.lines
    ), f"a skipped check has to say so: {log.lines}"
    assert "wait:aureax-edge" in order


def test_the_build_gate_runs_before_anything_is_labelled(monkeypatch) -> None:
    """Position is the whole design.

    Nothing should be labelled or scheduled on a group whose members cannot
    talk to each other, and once the edge wait starts a mismatch costs the full
    startupProbe budget in silence. Unlike the digest gate this replaced, it
    asks the nodes rather than the pods, so it no longer has to wait for the
    edge selector patch to create something to compare.
    """
    cfg = _grouped_fleet()
    order: list[str] = []
    _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",), order=order)
    monkeypatch.setattr(
        lifecycle, "_assert_group_builds_match",
        lambda *_a, **_k: order.append("build-gate"),
    )

    lifecycle._start_phase_label_and_wait_pods(
        cfg, FakeLog(), None, _PhaseKubectl(["host-h", "host-w1"])
    )

    assert order == [
        "build-gate",
        "patch:aureax-shard", "wait:aureax-shard",
        "patch:aureax-edge", "wait:aureax-edge",
    ]


def test_the_gate_compares_the_binaries_not_the_image(monkeypatch) -> None:
    """Every node builds its own `aureax-client:dev`, so two members that
    compiled the same pinned LLAMA_CPP_VERSION still hold different image ids.
    A gate reading a digest fails on every coherent fleet; this one asks each
    node for the bytes the RPC handshake is version-locked across."""
    cfg = _grouped_fleet()
    kc = _PhaseKubectl(["host-h", "host-w1"])
    asked: list[str] = []
    _phase_env(monkeypatch, edges=("host-h",), workers=("host-w1",))
    probed = _stub_build_probe(monkeypatch)
    monkeypatch.setattr(kc, "get_pods", lambda _ns: asked.append("get_pods") or [])

    lifecycle._start_phase_label_and_wait_pods(cfg, FakeLog(), None, kc)

    assert probed == ["h", "w1"], "the gate asks every member of the group"
    assert asked == [], "read a pod listing for a question the pods cannot answer"


def test_the_build_gate_probes_only_enabled_members(monkeypatch) -> None:
    """The gate builds its probe set from ``ssh.enabled_targets``.

    A fleet file cannot express a disabled group member —
    ``NetworkConfig._check_node_groups`` refuses one, because a group cannot
    serve a model with a member excluded from deployment — so this is about
    agreement, not a reachable failure: every other SSH pass here filters the
    same way, and the gate must not become the one that dials a node the
    operator switched off and then pays its whole timeout for the answer.
    """
    cfg = make_network_config(
        make_targets("h", "w1", "w2"),
        [{"name": "g", "head": "h", "workers": ["w1", "w2"]}],
    )
    # Flipped after validation, since the validator is what makes this
    # unreachable from a config file.
    next(t for t in cfg.ssh.targets if t.node_id == "w2").enabled = False
    probed = _stub_build_probe(monkeypatch)

    lifecycle._assert_group_builds_match(cfg, FakeLog())

    assert probed == ["h", "w1"], "SSH-probed a node no pass deploys to"


def test_the_build_gate_is_silent_on_a_fleet_with_no_groups(monkeypatch) -> None:
    """Nothing dials RPC without a group, so there is nothing to be locked to."""
    cfg = make_network_config(make_targets("edge-a"))
    order: list[str] = []
    _phase_env(monkeypatch, edges=("host-edge-a",), order=order)
    probed = _stub_build_probe(monkeypatch)
    log = FakeLog()

    lifecycle._start_phase_label_and_wait_pods(
        cfg, log, None, _PhaseKubectl(["host-edge-a"])
    )

    assert probed == [], "opened an SSH connection for a fleet with no group"
    assert not any("checksum" in text for _lvl, text in log.lines)
    assert "wait:aureax-edge" in order


# --- whole-fleet start: config validation ------------------------------------
# validate_groups/validate_planners run inside `aureax generate` only. `start`
# does not generate, so a fleet file edited since the last generate reached pod
# scheduling unchecked — and every symptom of that is a silent one.

def test_start_validates_the_group_config_before_touching_the_cluster(monkeypatch) -> None:
    """Before the k3d phase, not after: a validation failure must cost nothing."""
    order: list[str] = []
    monkeypatch.setattr(
        lifecycle, "_start_phase_validate_config",
        lambda *_a, **_k: order.append("validate"),
    )
    monkeypatch.setattr(
        lifecycle, "_start_phase_cluster",
        lambda *_a, **_k: order.append("cluster") or False,
    )

    lifecycle.do_start(make_network_config(make_targets("a")), FakeLog())

    assert order == ["validate", "cluster"]


def test_start_refuses_a_fleet_whose_group_validation_fails(monkeypatch, tmp_path) -> None:
    """The same problems that abort a generate, at the same fatality.

    Nothing downstream of start would fix them, and each one has a silent
    runtime symptom: a head serving the whole model alone, a worker labelled
    twice, a planner talking to two hosts that no longer exist.
    """
    from orchestrator.config import templating

    monkeypatch.setattr(
        templating, "validate_groups",
        lambda *_a: ["group 'g': head 'h' has role 'shard_worker' — expected edge_executor"],
    )
    monkeypatch.setattr(
        templating, "validate_planners",
        lambda *_a: ["planner-01: mcp.client.remote_servers is hardcoded"],
    )
    log = FakeLog()

    with pytest.raises(RuntimeError, match="before anything was labelled") as exc:
        _validate_config_phase(_cfg(), log, tmp_path / "aureax-network.yaml")

    assert "expected edge_executor" in str(exc.value)
    assert "hardcoded" in str(exc.value), "planner problems must be fatal too"
    assert sum(1 for lvl, _t in log.lines if lvl == "error") == 2


def test_start_warns_when_node_groups_drifted_from_the_last_generate(monkeypatch, tmp_path) -> None:
    """The nodes hold the configs of the LAST generate, not of this file."""
    from orchestrator.config import group_state, templating

    monkeypatch.setattr(templating, "validate_groups", lambda *_a: [])
    monkeypatch.setattr(templating, "validate_planners", lambda *_a: [])
    monkeypatch.setattr(group_state, "baseline_differs", lambda *_a: True)
    log = FakeLog()

    _validate_config_phase(_cfg(), log, tmp_path / "aureax-network.yaml")

    warns = [text for lvl, text in log.lines if lvl == "warn"]
    assert warns and "aureax generate" in warns[0] and "aureax apply --all" in warns[0]


def test_start_does_not_abort_on_a_stale_baseline(monkeypatch, tmp_path) -> None:
    """Bringing a fleet up after an edit that has not been deployed is legitimate.

    Refusing here would make `aureax start` unusable exactly when the operator
    is trying to get the cluster back before regenerating.
    """
    from orchestrator.config import group_state, templating

    monkeypatch.setattr(templating, "validate_groups", lambda *_a: [])
    monkeypatch.setattr(templating, "validate_planners", lambda *_a: [])
    monkeypatch.setattr(group_state, "baseline_differs", lambda *_a: True)

    # No raise: the assertion is that this call returns at all.
    _validate_config_phase(_cfg(), FakeLog(), tmp_path / "aureax-network.yaml")


def test_per_node_start_validates_the_config_first(monkeypatch) -> None:
    """`start --node` fans out to a whole group, one member at a time.

    A group broken by an edit is worse here than fleet-wide: the members are
    labelled in sequence, so the head can be scheduled against peers the file
    no longer describes.
    """
    calls: list[str] = []
    monkeypatch.setattr(
        lifecycle, "_start_phase_validate_config",
        lambda *_a, **_k: calls.append("validate") or (_ for _ in ()).throw(
            RuntimeError("node_groups validation failed before anything was labelled")
        ),
    )
    _roles(monkeypatch, {"solo": "edge_executor"})
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )
    label_calls = _labels(monkeypatch)

    with pytest.raises(RuntimeError, match="before anything was labelled"):
        lifecycle.do_start_node(_cfg(), "solo", FakeLog(), Path("configs/x.yaml"))

    assert calls == ["validate"]
    assert label_calls == [], "labelled a node before the config was validated"


def test_a_reordered_fleet_file_warns_at_start_through_the_real_baseline(
    monkeypatch, tmp_aureax_home
) -> None:
    """End to end against `group_state`, not a stubbed verdict.

    Pins what the stubbed test above cannot: that the phase asks about THIS
    fleet file (the baseline is keyed by its path) and hands over the config it
    was given, so a reorder recorded by the last generate is actually seen.
    """
    from orchestrator.config import group_state, templating

    monkeypatch.setattr(templating, "validate_groups", lambda *_a: [])
    monkeypatch.setattr(templating, "validate_planners", lambda *_a: [])
    net = tmp_aureax_home / "aureax-network.yaml"
    group_state.write_baseline(net, _cfg())
    log = FakeLog()

    _validate_config_phase(
        _cfg([{"name": "g", "head": "h", "workers": ["w2", "w1"]}]), log, net
    )

    assert any(
        lvl == "warn" and "stale" in text for lvl, text in log.lines
    ), f"a swapped worker order left no trace at start: {log.lines}"
