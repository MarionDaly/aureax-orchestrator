"""`Deployer.apply_to_nodes` — push a config to one node and restart just that pod.

This primitive exists because rotating a key is a three-step dance that fails
silently at every seam: rotate, regenerate, apply. Its shape is three
deliberate asymmetries, each of which is a bug if it drifts:

- configs REGENERATE for the whole fleet (generate_all_edge_configs is the only
  caller of validate_groups, so a partial regeneration could let a group's
  member configs drift out of agreement);
- the PUSH fans out to each requested node's whole GROUP (a group serves one
  model; its members' configs must stay consistent);
- the RESTART set excludes shard workers entirely (rpc-server takes no API key,
  and bouncing a worker invalidates its tensor cache — a ~107s warm head reload
  becomes a ~280s cold one, for a change the worker cannot even use);
- a changed MCP key widens the push BEYOND the group, to every node whose
  template consumes ${mcp_endpoints}: those configs embed their peers' keys, so
  an unpushed consumer keeps dialling with the old one and 401s on every call.

Rotation covers both of a node's keys, llama and MCP, because one restart
applies both and "re-key this node" means the node, not one of its ports.

Nothing here touches SSH or a real cluster: every outward call is stubbed, so
what is under test is the decision-making, not paramiko.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.cluster import kubectl
from orchestrator.config import keystore
from orchestrator.config.schema import HEAD_ROLE, SHARD_WORKER_ROLE
from orchestrator.deploy import deployer as deployer_mod
from orchestrator.deploy.deployer import Deployer
from tests.conftest import make_network_config, make_targets

# head + 2 workers form one group; solo is an ordinary edge node outside it.
ROLES = {
    "head": HEAD_ROLE,
    "w1": SHARD_WORKER_ROLE,
    "w2": SHARD_WORKER_ROLE,
    "solo": HEAD_ROLE,
}
GROUP = {"name": "g", "head": "head", "workers": ["w1", "w2"]}

# Captured at import time, before any fixture replaces it — one test needs the
# real primitive back so it can inspect the argv it builds.
REAL_DELETE_PODS_ON_NODE = kubectl.delete_pods_on_node


class Recorder:
    """Everything apply_to_nodes did to the outside world."""

    def __init__(self) -> None:
        self.pushed: list[list[str]] = []   # node_ids per deploy_configs call
        self.deleted: list[tuple[str, str]] = []  # (namespace, hostname)
        self.generated = 0
        self.log: list[str] = []


@pytest.fixture
def applier(monkeypatch, tmp_path, tmp_aureax_home):
    """A Deployer whose every outward call is recorded instead of performed.

    ``tmp_aureax_home`` is not optional here: apply_to_nodes fingerprints every
    target's MCP key on entry, so without the $HOME redirect these tests would
    read the developer's real key store and their fan-out decisions would
    depend on which keys that machine happens to hold.
    """
    cfg = make_network_config(make_targets("head", "w1", "w2", "solo"), [GROUP])
    # Patched on the Deployer's module: it loads through the (path, mtime)
    # cache, which would stat a path these tests never create.
    monkeypatch.setattr(deployer_mod, "load_network_config_cached", lambda _p: cfg)

    d = Deployer(tmp_path / "configs" / "aureax-network.yaml", "10.77.0.99")
    rec = Recorder()

    def fake_generate(_self):
        rec.generated += 1
        return [tmp_path / f"aureax-edge-{n}.yaml" for n in ROLES]

    def fake_deploy(_self, _paths, targets=None):
        rec.pushed.append([t.node_id for t in targets] if targets is not None else ["*all*"])

    monkeypatch.setattr(Deployer, "generate_configs", fake_generate)
    monkeypatch.setattr(Deployer, "deploy_configs", fake_deploy)
    monkeypatch.setattr(Deployer, "_target_hostname", lambda _s, t: t.node_id)
    monkeypatch.setattr(Deployer, "_target_role", lambda _s, t: ROLES[t.node_id])
    monkeypatch.setattr(kubectl, "get_pods", lambda _ns: [])

    def fake_delete(namespace, node_name, app="aureax-edge"):
        rec.deleted.append((namespace, node_name))
        return True, f'pod "aureax-edge-{node_name}" deleted'

    monkeypatch.setattr(kubectl, "delete_pods_on_node", fake_delete)

    d._rec = rec
    return d


def _apply(d, *args, **kwargs):
    """apply_to_nodes with the recorder wired in as the log sink."""
    return d.apply_to_nodes(*args, log=d._rec.log.append, **kwargs)


# --- the group fan-out --------------------------------------------------------


def test_applying_to_a_worker_pushes_to_the_whole_group(applier) -> None:
    """A group serves ONE model, so its members' configs move together.

    Pushing to the worker alone would leave the head's config describing a
    group state that no longer matches its peers.
    """
    outcomes = _apply(applier, ["w1"])

    assert applier._rec.pushed == [["head", "w1", "w2"]]
    assert set(outcomes) == {"head", "w1", "w2"}
    assert "solo" not in outcomes, "an unrelated node was touched"


def test_only_the_head_is_restarted(applier) -> None:
    """rpc-server takes no API key and holds a tensor cache worth minutes.

    Restarting a worker tears the whole group down and forces a cold reload of
    the head — for a config change the worker has no use for.
    """
    outcomes = _apply(applier, ["w1"])

    assert applier._rec.deleted == [("aureax", "head")]
    assert "restarting" in outcomes["head"]
    for w in ("w1", "w2"):
        assert "restart skipped" in outcomes[w], outcomes[w]
        assert "shard worker" in outcomes[w].lower()


def test_the_skip_is_reported_not_silent(applier) -> None:
    """The operator asked about this node by name; "nothing happened" has to
    come back as a sentence, not an absent dict entry."""
    outcomes = _apply(applier, ["w1", "w2"])
    assert all(outcomes[w].startswith("config pushed;") for w in ("w1", "w2"))


def test_a_standalone_node_is_applied_alone(applier) -> None:
    outcomes = _apply(applier, ["solo"])

    assert applier._rec.pushed == [["solo"]]
    assert applier._rec.deleted == [("aureax", "solo")]
    assert set(outcomes) == {"solo"}


def test_applying_to_every_node_restarts_only_the_non_workers(applier) -> None:
    outcomes = _apply(applier, None)

    assert set(outcomes) == set(ROLES)
    assert [h for _ns, h in applier._rec.deleted] == ["head", "solo"]


def test_configs_are_regenerated_for_the_whole_fleet(applier) -> None:
    """Partial regeneration would skip validate_groups, the only cross-file
    check that a group's configs still agree with each other."""
    _apply(applier, ["solo"])
    assert applier._rec.generated == 1


def test_duplicate_node_ids_are_collapsed(applier) -> None:
    outcomes = _apply(applier, ["head", "head", "w1"])
    assert applier._rec.pushed == [["head", "w1", "w2"]]
    assert set(outcomes) == {"head", "w1", "w2"}


def test_an_unknown_node_id_is_refused_before_anything_happens(applier) -> None:
    """A typo must not silently apply to nothing — or, worse, regenerate and
    push the fleet while reporting an empty result."""
    with pytest.raises(ValueError, match="unknown or disabled node"):
        _apply(applier, ["edge-42"])

    assert applier._rec.generated == 0
    assert applier._rec.pushed == []
    assert applier._rec.deleted == []


# --- --no-restart --------------------------------------------------------------


def test_no_restart_pushes_but_never_deletes_a_pod(applier) -> None:
    outcomes = _apply(applier, ["head"], restart=False)

    assert applier._rec.pushed == [["head", "w1", "w2"]]
    assert applier._rec.deleted == []
    assert all(o == "config pushed" for o in outcomes.values()), outcomes


def test_no_restart_warns_that_the_node_still_runs_the_old_key(applier) -> None:
    """The dangerous half of --no-restart: the config on disk and the process
    enforcing it have diverged, and the in-pod clients follow the config — so
    every /completion and /metrics call 401s until the pod restarts."""
    _apply(applier, ["head"], restart=False)
    assert any("WARNING" in m and "401" in m for m in applier._rec.log)


def test_no_restart_warns_that_the_staleness_reaches_the_peers(applier) -> None:
    """--no-restart reintroduces exactly what the fan-out prevents: the pushed
    consumer config carries the new peer key while its running pod still holds
    the old one. The warning is the only guard left at that point, so it has to
    say that the damage is not confined to the node named on the command line."""
    _apply(applier, ["head"], restart=False)
    warning = next(m for m in applier._rec.log if "WARNING" in m)
    assert "mcp_endpoints" in warning, warning


# --- key rotation --------------------------------------------------------------


def test_rotation_only_touches_nodes_that_already_hold_a_key(
    applier, tmp_aureax_home
) -> None:
    """Rotating a keyless node would create a key that enables nothing.

    Shard workers and auth-off nodes have no ${api_key} in their template, so a
    key minted here would never be read by anything — but `aureax nodes` would
    start reporting auth for them.
    """
    before = keystore.ensure_key("head")

    outcomes = _apply(applier, ["head"], rotate_keys=True)

    assert keystore.read_key("head") != before
    assert keystore.previous_key("head") == before
    # The push fans out to the workers; the rotation must not.
    assert not keystore.has_key("w1")
    assert not keystore.has_key("w2")
    assert any("head" in m and "rotated" in m for m in applier._rec.log)
    assert set(outcomes) == {"head", "w1", "w2"}


def test_rotation_follows_the_requested_set_not_the_group_fan_out(
    applier, tmp_aureax_home
) -> None:
    """Naming a worker rotates nothing — including its head's key.

    The push fans out to the group because member configs must agree; rotation
    deliberately does not, or `apply --node w1 --rotate-key` would silently
    re-key the head an operator never mentioned.
    """
    before = keystore.ensure_key("head")

    _apply(applier, ["w1"], rotate_keys=True)

    assert keystore.read_key("head") == before
    assert not keystore.has_key("w1")
    assert not any("rotated" in m for m in applier._rec.log)


def test_rotation_happens_before_generation(applier, tmp_aureax_home) -> None:
    """Otherwise the pushed config carries the pre-rotation key while the store
    holds the new one — the exact mismatch this command exists to avoid."""
    keystore.ensure_key("solo")
    seen: list[str | None] = []

    def recording_generate(_self):
        seen.append(keystore.read_key("solo"))
        return [Path(f"/tmp/aureax-edge-{n}.yaml") for n in ROLES]

    original = keystore.read_key("solo")
    import types
    applier.generate_configs = types.MethodType(recording_generate, applier)

    _apply(applier, ["solo"], rotate_keys=True)

    assert seen == [keystore.read_key("solo")]
    assert seen[0] != original, "generation ran against the pre-rotation key"


def test_rotation_is_not_performed_unless_asked(applier, tmp_aureax_home) -> None:
    before = keystore.ensure_key("head")
    _apply(applier, ["head"])
    assert keystore.read_key("head") == before


def test_rotation_re_keys_both_of_a_node_s_ports(applier, tmp_aureax_home) -> None:
    """"Re-key this node" means the node, not one of its ports.

    Both keys protect the same pod and the one restart below applies both, so
    rotating them separately would buy a second multi-minute head reload for
    nothing — and would leave a suspected leak half-closed.
    """
    llama_before = keystore.ensure_key("head")
    mcp_before = keystore.ensure_key("head", kind=keystore.KIND_MCP)

    _apply(applier, ["head"], rotate_keys=True)

    assert keystore.read_key("head") != llama_before
    assert keystore.read_key("head", kind=keystore.KIND_MCP) != mcp_before
    # Each kind keeps its OWN .prev, so a harness holding either old key can
    # still be handed the right one while the apply is pending.
    assert keystore.previous_key("head") == llama_before
    assert keystore.previous_key("head", kind=keystore.KIND_MCP) == mcp_before
    log = "\n".join(applier._rec.log)
    assert "head: llama API key rotated" in log, log
    assert "head: MCP API key rotated" in log, log


def test_each_kind_is_rotated_only_where_it_already_exists(applier, tmp_aureax_home) -> None:
    """A node can serve keyed llama and unkeyed MCP, so the two gates are
    independent. Rotate-creating the missing one would enable nothing — no
    template references it — while `aureax nodes` started reporting auth for a
    port that has none."""
    keystore.ensure_key("head")

    _apply(applier, ["head"], rotate_keys=True)

    assert not keystore.has_key("head", kind=keystore.KIND_MCP)
    log = "\n".join(applier._rec.log)
    assert "llama API key rotated" in log
    assert "MCP API key rotated" not in log


# --- the ${mcp_endpoints} fan-out ------------------------------------------------


def _consumes_peer_keys(applier, node_id: str) -> None:
    """Give `node_id` a template that carries the derived peer list.

    A real file rather than a stub, because the deployer decides the fan-out by
    reading templates — a role-based rule would miss the second planner.
    """
    configs = applier.config_path.parent
    configs.mkdir(parents=True, exist_ok=True)
    (configs / f"{node_id}.yaml").write_text(
        'mcp:\n  client:\n    enabled: true\n    remote_servers: "${mcp_endpoints}"\n'
    )


def test_a_rotated_mcp_key_also_pushes_the_peers_that_carry_it(
    applier, tmp_aureax_home
) -> None:
    """A consumer's resolved config embeds every peer's MCP key.

    Left unpushed, its pod keeps dialling with the old key and 401s on every
    tool call — a fleet-wide outage from a single-node rotation, and one with
    no error on the node whose key changed.
    """
    _consumes_peer_keys(applier, "solo")
    keystore.ensure_key("head", kind=keystore.KIND_MCP)

    outcomes = _apply(applier, ["head"], rotate_keys=True)

    assert applier._rec.pushed == [["head", "w1", "w2", "solo"]]
    assert ("aureax", "solo") in applier._rec.deleted, "the consumer was pushed but never restarted"
    assert "solo" in outcomes
    line = next((m for m in applier._rec.log if "MCP key changed" in m), "")
    assert "head" in line and "solo" in line and "401" in line, applier._rec.log


def test_a_first_minted_mcp_key_fans_out_too(applier, tmp_aureax_home, monkeypatch) -> None:
    """Turning MCP auth on for a node mints its key during generate, not during
    rotation. Comparing only the rotation set would miss that half silently —
    and it is the half an operator hits first, right after adding the template
    line."""
    _consumes_peer_keys(applier, "solo")

    def minting_generate(_self):
        applier._rec.generated += 1
        keystore.ensure_key("head", kind=keystore.KIND_MCP)
        return [Path(f"/tmp/aureax-edge-{n}.yaml") for n in ROLES]

    monkeypatch.setattr(Deployer, "generate_configs", minting_generate)

    _apply(applier, ["head"])

    assert applier._rec.pushed == [["head", "w1", "w2", "solo"]]


def test_no_fan_out_when_no_mcp_key_changed(applier, tmp_aureax_home) -> None:
    """The fan-out cold-reloads the consumer's pod, so an ordinary apply must
    not trigger it: every `aureax apply` would otherwise restart the planner."""
    _consumes_peer_keys(applier, "solo")
    keystore.ensure_key("head", kind=keystore.KIND_MCP)

    _apply(applier, ["head"])

    assert applier._rec.pushed == [["head", "w1", "w2"]]
    assert ("aureax", "solo") not in applier._rec.deleted
    assert not any("MCP key changed" in m for m in applier._rec.log)


def test_a_consumer_already_in_the_push_set_is_not_added_twice(
    applier, tmp_aureax_home
) -> None:
    """A node can consume its own peer list — the planner is normally the one
    being applied to. A duplicate entry would push and delete its pod twice."""
    _consumes_peer_keys(applier, "solo")
    keystore.ensure_key("solo", kind=keystore.KIND_MCP)

    _apply(applier, ["solo"], rotate_keys=True)

    assert applier._rec.pushed == [["solo"]]
    assert applier._rec.deleted == [("aureax", "solo")]


def test_the_fan_out_ignores_templates_without_the_peer_list(
    applier, tmp_aureax_home
) -> None:
    """Only ${mcp_endpoints} consumers hold someone else's key. A template that
    merely mentions MCP is not one."""
    configs = applier.config_path.parent
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "solo.yaml").write_text(
        'mcp:\n  server:\n    enabled: true\n    api_key: "${mcp_api_key}"\n'
    )
    keystore.ensure_key("head", kind=keystore.KIND_MCP)

    _apply(applier, ["head"], rotate_keys=True)

    assert applier._rec.pushed == [["head", "w1", "w2"]]


# --- the restart primitive ------------------------------------------------------


def test_restart_never_uses_rollout_restart(applier, monkeypatch) -> None:
    """`kubectl rollout restart daemonset/...` bounces the ENTIRE fleet.

    A DaemonSet rollout is serial, so applying a one-node config change would
    cold-reload every head in the fleet — minutes per member. The apply path
    may only ever delete the one pod on the one node.

    Stubbed at the subprocess boundary rather than at the kubectl helpers, so
    the real delete_pods_on_node runs and its argv is inspected.
    """
    calls: list[list[str]] = []

    class _Result:
        def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
            self.stdout, self.stderr, self.returncode = stdout, stderr, returncode

    def fake_run(args, check=True):
        calls.append(args)
        if args and args[0] == "get":
            return _Result('{"items": []}')
        if args and args[0] == "delete":
            return _Result('pod "aureax-edge-xyz" deleted\n')
        return _Result()

    monkeypatch.setattr(kubectl, "_run", fake_run)
    # Put the REAL primitive back over the fixture's recorder, so the argv it
    # builds is what gets inspected.
    monkeypatch.setattr(kubectl, "delete_pods_on_node", REAL_DELETE_PODS_ON_NODE)

    _apply(applier, ["solo"])

    flat = [" ".join(c) for c in calls]
    assert not any("rollout" in f for f in flat), flat
    delete_calls = [c for c in calls if c and c[0] == "delete"]
    assert len(delete_calls) == 1, flat
    assert "pod" in delete_calls[0]
    assert "--field-selector" in delete_calls[0]
    assert "spec.nodeName=solo" in delete_calls[0]


def test_a_failed_delete_is_reported_as_a_failed_restart(applier, monkeypatch) -> None:
    """"No resources found" exits 0. Reporting that as a restart would tell the
    operator their new key is live when the old pod is still running."""
    monkeypatch.setattr(
        kubectl, "delete_pods_on_node",
        lambda _ns, _host, app="aureax-edge": (False, "no matching pod on that node"),
    )
    outcomes = _apply(applier, ["solo"])
    assert "restart FAILED" in outcomes["solo"]
    assert "no matching pod" in outcomes["solo"]


def test_an_unresolvable_hostname_is_reported_not_skipped(applier, monkeypatch) -> None:
    """The config landed but the pod did not move; saying nothing would leave
    the operator believing the apply completed."""
    monkeypatch.setattr(Deployer, "_target_hostname", lambda _s, _t: "")
    outcomes = _apply(applier, ["solo"])
    assert "restart FAILED" in outcomes["solo"]
    assert applier._rec.deleted == []


# --- waiting for the replacement pod --------------------------------------------


def test_every_pod_is_deleted_before_any_wait_begins(applier, monkeypatch) -> None:
    """--wait is two passes, and the order is the point.

    With the delete and the wait in one loop, node N+1's pod was not deleted
    until node N came back Ready, so `apply --all --wait` reloaded the fleet's
    heads one after another — minutes of cold load each, serially. The cost of
    the fix is that every head reloads at once, which is a real availability
    tradeoff and is why the two passes have to stay visible here.
    """
    events: list[tuple[str, str]] = []

    def fake_delete(namespace, node_name, app="aureax-edge"):
        events.append(("delete", node_name))
        return True, f'pod "aureax-edge-{node_name}" deleted'

    monkeypatch.setattr(kubectl, "delete_pods_on_node", fake_delete)
    # A non-empty pre-delete snapshot per node: an empty one is "unusable", and
    # that path skips the wait entirely.
    monkeypatch.setattr(
        kubectl, "get_pods",
        lambda _ns: [
            {"name": f"aureax-edge-{n}", "node": n, "ready": True, "status": "Running"}
            for n in ROLES
        ],
    )
    monkeypatch.setattr(Deployer, "_apply_wait_budget", lambda _s, _nid: 300)

    def fake_wait(_self, _ns, hostname, _pre, _timeout, _emit):
        events.append(("wait", hostname))
        return True

    monkeypatch.setattr(Deployer, "_wait_pod_replaced", fake_wait)

    outcomes = _apply(applier, None, wait=True)

    # head and solo restart; w1/w2 are shard workers and are never bounced.
    assert events == [
        ("delete", "head"), ("delete", "solo"),
        ("wait", "head"), ("wait", "solo"),
    ]
    assert outcomes["head"] == "applied (pod Ready on new config)"
    assert outcomes["solo"] == "applied (pod Ready on new config)"


def test_wait_ignores_the_pod_it_just_deleted(applier, monkeypatch, fake_clock) -> None:
    """The killed pod stays listed — and `ready` — for a moment after delete.

    A plain readiness wait therefore returned instantly against the pod just
    killed and reported "applied (pod Ready on new config)" for a node still
    running the OLD config. The name-set comparison is the whole point.
    """
    old = {"name": "aureax-edge-old", "node": "solo", "ready": True, "status": "Running"}
    new_pending = {"name": "aureax-edge-new", "node": "solo", "ready": False, "status": "Running"}
    new_ready = {**new_pending, "ready": True}

    sequence = [[old], [old, new_pending], [new_ready]]
    seen: list[int] = []

    def fake_get_pods(_ns):
        seen.append(len(seen))
        return sequence[min(len(seen) - 1, len(sequence) - 1)]

    monkeypatch.setattr(kubectl, "get_pods", fake_get_pods)

    ok = applier._wait_pod_replaced(
        "aureax", "solo", {old["name"]}, 300, applier._rec.log.append
    )

    assert ok is True
    assert len(seen) == 3, "returned before a NEW pod was Ready"


def test_wait_times_out_instead_of_hanging(applier, monkeypatch, fake_clock) -> None:
    """A head that cannot re-attach to its RPC peers never becomes Ready, and
    the outcome has to say so rather than block the CLI forever."""
    old = {"name": "aureax-edge-old", "node": "solo", "ready": True, "status": "Running"}
    monkeypatch.setattr(kubectl, "get_pods", lambda _ns: [old])

    ok = applier._wait_pod_replaced(
        "aureax", "solo", {old["name"]}, 30, applier._rec.log.append
    )

    assert ok is False
    assert any("timed out" in m for m in applier._rec.log)


def test_wait_ignores_pods_on_other_nodes(applier, monkeypatch, fake_clock) -> None:
    """Every DaemonSet pod carries the same app label, so filtering by node is
    what keeps another node's healthy pod from ending the wait."""
    monkeypatch.setattr(
        kubectl, "get_pods",
        lambda _ns: [{"name": "aureax-edge-elsewhere", "node": "head",
                      "ready": True, "status": "Running"}],
    )

    ok = applier._wait_pod_replaced("aureax", "solo", set(), 30, applier._rec.log.append)
    assert ok is False
