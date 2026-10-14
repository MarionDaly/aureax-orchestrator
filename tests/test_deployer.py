"""Deployer labelling, config pairing and TLS distribution.

The edge label pass skipping shard workers is the single most load-bearing group
invariant: a worker that also gets `aureax.io/role=edge` is handed a llama-server
pod with no model file and crash-loops against the very group it serves. It had
no test, so any refactor that loosened the role comparison — treating "" as a
match for role-less configs, say, which `_target_role` returns on ANY read
failure — would break the whole feature silently.

The push is the fleet's only channel into `/etc/aureax`: both DaemonSets mount
it read-only with `type: Directory`, which the kubelet will not create. Since
TLS became mandatory it carries four files per node rather than one, and the
pairing, the modes and the failure-path cleanup are pinned here for each of
them — a mispaired leaf hands one node another node's identity, and a stranded
stage file leaves a private key in a login user's home.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from orchestrator.cluster import k3d, kubectl
from orchestrator.config.schema import HEAD_ROLE, PLANNER_ROLE, SHARD_WORKER_ROLE
from orchestrator.deploy import deployer as deployer_mod
from orchestrator.deploy.deployer import Deployer
from tests.conftest import make_network_config, make_targets

ROLES = {
    "head": HEAD_ROLE,
    "solo": HEAD_ROLE,
    "w1": SHARD_WORKER_ROLE,
    "w2": SHARD_WORKER_ROLE,
    "planner": PLANNER_ROLE,
}
GROUP = {"name": "g", "head": "head", "workers": ["w1", "w2"]}


@pytest.fixture
def tls_material(tmp_path, monkeypatch):
    """A populated fake ``~/.aureax/tls`` for every node in ROLES.

    Every deploy_configs test needs one: the push now refuses a fleet whose
    certificates were never minted, and without the $HOME redirect these tests
    would read — and report on — the developer's real fleet CA.

    Contents are placeholder bytes, not real PEM: deploy_configs only stats the
    files and hands them to upload_files, which is stubbed in every test here.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    store = tmp_path / ".aureax" / "tls"
    store.mkdir(parents=True, exist_ok=True)
    (store / "ca.pem").write_text("--fleet-ca--\n")
    for node_id in ROLES:
        (store / f"{node_id}.pem").write_text(f"--cert {node_id}--\n")
        (store / f"{node_id}.key").write_text(f"--key {node_id}--\n")
    return store


@pytest.fixture
def deployer(monkeypatch, tmp_path, tls_material):
    """A Deployer over a 5-node fleet with all SSH and kubectl calls stubbed."""
    cfg = make_network_config(
        make_targets("head", "w1", "w2", "solo", "planner"), [GROUP]
    )
    # Patched on the Deployer's module: it loads through the (path, mtime)
    # cache, which would stat a path these tests never create.
    monkeypatch.setattr(deployer_mod, "load_network_config_cached", lambda _p: cfg)

    d = Deployer(tmp_path / "configs" / "aureax-network.yaml", "10.77.0.99")

    # hostname == "host-<node_id>", role from the table above.
    monkeypatch.setattr(Deployer, "_target_hostname", lambda _s, t: f"host-{t.node_id}")
    monkeypatch.setattr(Deployer, "_target_role", lambda _s, t: ROLES[t.node_id])

    nodes = [{"name": "k3d-server-0", "roles": "control-plane"}] + [
        {"name": f"host-{n}", "roles": "worker"} for n in ROLES
    ]
    monkeypatch.setattr(kubectl, "get_nodes", lambda: nodes)

    labels: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        kubectl, "label_node",
        lambda name, key, value: labels.append((name, key, value)) or "",
    )
    # Removals go into their own recorder rather than sharing `labels`: the
    # assertions below unpack that list as (name, key, value) 3-tuples, so a
    # tagged removal entry would read back as a node name. Stubbing this at all
    # is load-bearing — left unstubbed, both label passes shell out to the real
    # kubectl binary, which is both a hidden dependency and the reason the
    # label-stripping contract went untested.
    removed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        kubectl, "remove_label",
        lambda name, key: (removed.append((name, key)), (True, ""))[1],
    )
    d._labels = labels
    d._removed = removed
    return d


@pytest.fixture
def applier(deployer, monkeypatch, tmp_path):
    """`deployer` with the apply path's outward calls recorded as well."""
    deployer._deleted: list[tuple[str, str]] = []

    monkeypatch.setattr(
        Deployer, "generate_configs",
        lambda _s: [tmp_path / f"aureax-edge-{n}.yaml" for n in ROLES],
    )
    monkeypatch.setattr(Deployer, "deploy_configs", lambda _s, _paths, targets=None: None)
    monkeypatch.setattr(kubectl, "get_pods", lambda _ns: [])
    monkeypatch.setattr(
        kubectl, "delete_pods_on_node",
        lambda ns, host: (deployer._deleted.append((ns, host)), (True, "deleted"))[1],
    )
    return deployer


def test_edge_label_pass_skips_shard_workers(deployer) -> None:
    """A worker with the edge label gets a llama-server pod with no model."""
    deployer.label_nodes(role=HEAD_ROLE)

    labelled = {name for name, _k, _v in deployer._labels}
    assert labelled == {"host-head", "host-solo"}
    assert "host-w1" not in labelled
    assert "host-w2" not in labelled
    assert "host-planner" not in labelled, "the planner is labelled later, once edges are Ready"
    assert "k3d-server-0" not in labelled


def test_edge_label_pass_uses_the_role_label_key(deployer) -> None:
    deployer.label_nodes(role=HEAD_ROLE)
    keys = {key for _n, key, _v in deployer._labels}
    assert keys == {deployer.config.kubernetes.node_label}


def test_shard_workers_get_the_shard_label_only(deployer) -> None:
    """Workers are gated by a second, orthogonal label key."""
    deployer.label_shard_workers()

    k8s = deployer.config.kubernetes
    assert sorted(deployer._labels) == [
        ("host-w1", k8s.shard_label, k8s.shard_label_value),
        ("host-w2", k8s.shard_label, k8s.shard_label_value),
    ]


def test_edge_label_pass_strips_the_shard_label_it_replaces(deployer) -> None:
    """A node promoted from worker to head must end up with ONE label.

    Both passes used to only ever add, so a promoted node kept
    aureax.io/shard=worker while gaining aureax.io/role=edge — matching both
    DaemonSet selectors and running llama-server and rpc-server side by side on
    one node. Nothing else notices two pods, which is why it has to be stripped
    here rather than detected later.
    """
    deployer.label_nodes(role=HEAD_ROLE)

    shard_label = deployer.config.kubernetes.shard_label
    assert sorted(deployer._removed) == [
        ("host-head", shard_label),
        ("host-solo", shard_label),
    ]
    # Exactly the nodes that were labelled, and only the other pass's key.
    assert {n for n, _k in deployer._removed} == {n for n, _k, _v in deployer._labels}


def test_shard_label_pass_strips_the_edge_label_it_replaces(deployer) -> None:
    """The mirror direction: a head demoted to worker keeps no edge label.

    Left behind, aureax.io/role=edge schedules a llama-server pod with no GGUF
    beside the node's rpc-server — the same two-pods-one-node failure reached
    from the other side.
    """
    deployer.label_shard_workers()

    node_label = deployer.config.kubernetes.node_label
    assert sorted(deployer._removed) == [
        ("host-w1", node_label),
        ("host-w2", node_label),
    ]


def test_a_failed_label_strip_is_reported_not_swallowed(deployer, monkeypatch, caplog) -> None:
    """A node left carrying both labels must be loud, not silent.

    The strip is best-effort — the deploy carries on — so the error log is the
    only signal an operator gets that a node may still match the second
    DaemonSet selector.
    """
    from orchestrator.cluster import kubectl as kc

    monkeypatch.setattr(kc, "remove_label", lambda _n, _k: (False, "node not found"))

    with caplog.at_level("ERROR", logger="aureax.deployer"):
        deployer.label_nodes(role=HEAD_ROLE)

    messages = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert any("host-head" in m and "node not found" in m for m in messages), messages
    assert any(deployer.config.kubernetes.shard_label in m for m in messages), messages


@pytest.mark.parametrize(
    ("returncode", "stderr", "expected_ok"),
    [
        (0, "", True),
        # kubectl's own wording for an already-absent label: per-node stop must
        # stay idempotent.
        (1, 'label "aureax.io/shard" not found.', True),
        # Same substring, opposite meaning: the node is unknown, so the label —
        # and every pod it gates — is untouched. Hostnames reach this call from
        # SSH, not from the cluster, so they can genuinely be wrong.
        (1, 'Error from server (NotFound): nodes "host-a" not found', False),
        (1, "error: You must be logged in to the server (Unauthorized)", False),
    ],
)
def test_remove_label_only_forgives_an_absent_label(
    monkeypatch, returncode, stderr, expected_ok
) -> None:
    """lifecycle takes ok=True as proof the gating label is gone and reports
    "Stopped N node(s)"."""
    class _Result:
        def __init__(self) -> None:
            self.returncode, self.stdout, self.stderr = returncode, "", stderr

    monkeypatch.setattr(kubectl, "_run", lambda _args, check=True: _Result())

    ok, _detail = kubectl.remove_label("host-a", "aureax.io/shard")
    assert ok is expected_ok


def test_shard_workers_skip_unregistered_nodes(deployer, monkeypatch) -> None:
    """Labelling a node the API server doesn't know fails the whole deploy."""
    monkeypatch.setattr(
        kubectl, "get_nodes",
        lambda: [{"name": "host-w1", "roles": "worker"}],  # w2 hasn't joined yet
    )
    deployer.label_shard_workers()
    assert [n for n, _k, _v in deployer._labels] == ["host-w1"]


def test_label_shard_workers_is_inert_without_groups(monkeypatch, tmp_path) -> None:
    cfg = make_network_config(make_targets("a", "b"))
    # Patched on the Deployer's module: it loads through the (path, mtime)
    # cache, which would stat a path these tests never create.
    monkeypatch.setattr(deployer_mod, "load_network_config_cached", lambda _p: cfg)
    d = Deployer(tmp_path / "configs" / "aureax-network.yaml", "10.77.0.99")

    called = []
    monkeypatch.setattr(kubectl, "label_node", lambda *a: called.append(a) or "")
    d.label_shard_workers()
    assert called == []


def test_deploy_configs_pairs_by_node_id_not_position(deployer, monkeypatch) -> None:
    """Position matching silently hands a node another node's config."""
    pushed: list[tuple[str, str]] = []
    monkeypatch.setattr(deployer_mod, "run_command", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda host, items, **k: pushed.extend(
            (host, Path(local).name) for local, _remote, _m, _pm in items
        ),
    )

    # Deliberately shuffled relative to ssh.targets order.
    paths = [
        Path(f"/tmp/aureax-edge-{n}.yaml") for n in ("planner", "w2", "head", "solo", "w1")
    ]
    deployer.deploy_configs(paths)

    # Each node also gets three TLS files; this test is about the config file.
    configs = [(h, n) for h, n in pushed if n.startswith("aureax-edge-")]
    assert dict(configs) == {
        "10.77.0.1": "aureax-edge-head.yaml",
        "10.77.0.2": "aureax-edge-w1.yaml",
        "10.77.0.3": "aureax-edge-w2.yaml",
        "10.77.0.4": "aureax-edge-solo.yaml",
        "10.77.0.5": "aureax-edge-planner.yaml",
    }


def test_deploy_configs_refuses_a_missing_node_config(deployer, monkeypatch) -> None:
    monkeypatch.setattr(deployer_mod, "run_command", lambda *a, **k: (0, "", ""))
    monkeypatch.setattr(deployer_mod, "upload_files", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="no generated config for"):
        deployer.deploy_configs([Path("/tmp/aureax-edge-head.yaml")])


def test_shard_daemonset_is_applied_before_the_edge_one(deployer, monkeypatch, tmp_path) -> None:
    """The edge set carries the head, which exits if a peer is unreachable."""
    deployer.k8s_dir = tmp_path / "k8s"
    deployer.k8s_dir.mkdir()
    for name in ("namespace.yaml", "daemonset-edge.yaml", "daemonset-shard.yaml"):
        (deployer.k8s_dir / name).write_text("{}")

    applied: list[str] = []
    monkeypatch.setattr(kubectl, "apply", lambda p, namespace=None: applied.append(Path(p).name) or "")
    deployer.apply_manifests()

    assert applied.index("daemonset-shard.yaml") < applied.index("daemonset-edge.yaml")


def test_missing_shard_daemonset_is_fatal_with_a_group(deployer, monkeypatch, tmp_path) -> None:
    deployer.k8s_dir = tmp_path / "k8s"
    deployer.k8s_dir.mkdir()
    (deployer.k8s_dir / "namespace.yaml").write_text("{}")

    monkeypatch.setattr(kubectl, "apply", lambda p, namespace=None: "")
    with pytest.raises(FileNotFoundError, match=r"daemonset-shard\.yaml is required"):
        deployer.apply_manifests()


def test_namespace_mismatch_between_config_and_manifests_is_fatal(
    deployer, monkeypatch, tmp_path
) -> None:
    """Deploying into one namespace while stop/start patch another is silent.

    The manifests hardcode metadata.namespace; every runtime operation reads
    config.kubernetes.namespace. When they disagree the stack deploys and
    reports success, then `aureax stop` patches DaemonSets that do not exist
    there and the workload keeps running.
    """
    deployer.k8s_dir = tmp_path / "k8s"
    deployer.k8s_dir.mkdir()
    (deployer.k8s_dir / "namespace.yaml").write_text("metadata:\n  namespace: aureax\n")
    (deployer.k8s_dir / "daemonset-edge.yaml").write_text("metadata:\n  namespace: aureax\n")
    (deployer.k8s_dir / "daemonset-shard.yaml").write_text("metadata:\n  namespace: aureax\n")

    monkeypatch.setattr(kubectl, "apply", lambda p, namespace=None: "")

    # Matching: applies cleanly.
    deployer.config.kubernetes.namespace = "aureax"
    deployer.apply_manifests()

    # Diverged: refused, naming both sides.
    deployer.config.kubernetes.namespace = "aureax-prod"
    with pytest.raises(ValueError, match=r"kubernetes\.namespace is 'aureax-prod'"):
        deployer.apply_manifests()


def test_kind_namespace_manifest_is_actually_validated(deployer, monkeypatch, tmp_path) -> None:
    """namespace.yaml declares its namespace as metadata.name, not `namespace:`.

    The old check only matched `namespace:` keys, so that manifest passed no
    matter what it said — a renamed Namespace object deployed a stack that
    stop/start could not find.
    """
    deployer.k8s_dir = tmp_path / "k8s"
    deployer.k8s_dir.mkdir()
    (deployer.k8s_dir / "namespace.yaml").write_text(
        "apiVersion: v1\nkind: Namespace\nmetadata:\n  name: something-else\n"
    )
    monkeypatch.setattr(kubectl, "apply", lambda p, namespace=None: "")

    deployer.config.kubernetes.namespace = "aureax"
    with pytest.raises(ValueError, match="something-else"):
        deployer.apply_manifests()


def test_deploy_configs_installs_the_config_as_a_root_only_file(deployer, monkeypatch) -> None:
    """The pushed config may carry the node's llama-server API key.

    It is mounted read-only into a container whose process runs as ROOT —
    neither Dockerfile sets a USER and neither DaemonSet sets
    runAsUser/runAsNonRoot (daemonset-edge.yaml's securityContext only drops
    capabilities and privilege escalation, which does not change the uid). So
    root:root 0600 is readable by the supervisor and by nobody else with a
    shell on the node — the default umask would have left it world-readable.
    """
    cmds: list[str] = []
    monkeypatch.setattr(
        deployer_mod, "run_command",
        lambda _h, cmd, **_k: cmds.append(cmd) or (0, "", ""),
    )
    uploads: list[dict] = []
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda _h, items, **k: uploads.extend(
            {"remote": remote, "mode": mode, "parent_mode": parent_mode}
            for _local, remote, mode, parent_mode in items
        ),
    )

    deployer.deploy_configs([Path("/tmp/aureax-edge-solo.yaml")],
                            targets=[t for t in deployer._targets if t.node_id == "solo"])

    assert len(cmds) == 1, cmds
    assert "sudo chown root:root /etc/aureax/aureax-edge.yaml" in cmds[0]
    assert "sudo chmod 0600 /etc/aureax/aureax-edge.yaml" in cmds[0]
    # 0600 from the moment it lands, too — not only after the chmod.
    assert uploads[0]["mode"] == 0o600


# --- TLS material rides the config push ---------------------------------------


def _record_push(deployer, monkeypatch, node_id="solo"):
    """Run one node's push with upload_files/run_command recorded.

    One call carries all four files, so the recorded list is flattened back to
    one entry per file — what these tests are about is which file lands where
    and at which mode, not how many calls it took.
    """
    uploads: list[tuple[Path, str, dict]] = []
    cmds: list[str] = []
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda _h, items, **k: uploads.extend(
            (Path(local), remote, {"mode": mode, "parent_mode": parent_mode})
            for local, remote, mode, parent_mode in items
        ),
    )
    monkeypatch.setattr(
        deployer_mod, "run_command",
        lambda _h, cmd, **_k: cmds.append(cmd) or (0, "", ""),
    )
    deployer.deploy_configs(
        [Path(f"/tmp/aureax-edge-{node_id}.yaml")],
        targets=[t for t in deployer._targets if t.node_id == node_id],
    )
    return uploads, cmds


def test_every_node_gets_its_config_and_its_three_tls_files(
    deployer, monkeypatch, tls_material
) -> None:
    """Four files, one push. TLS is mandatory and there is no plaintext mode,
    so a node that receives its config and no certificates does not fall back —
    the supervisor exits 78 and the node never serves anything. They travel
    together because SSH is the only channel that can reach /etc/aureax at all:
    both DaemonSets mount it read-only with `type: Directory`.
    """
    uploads, _cmds = _record_push(deployer, monkeypatch)

    assert [local.name for local, _r, _k in uploads] == [
        "aureax-edge-solo.yaml", "ca.pem", "solo.pem", "solo.key",
    ]
    # Staged 0600 under a 0700 parent — including the two certificates, which
    # are only widened to 0644 once they are root-owned in /etc/aureax/tls.
    for _local, _remote, kwargs in uploads:
        assert kwargs["mode"] == 0o600
        assert kwargs["parent_mode"] == 0o700


def test_the_four_files_ride_one_ssh_connection(
    deployer, monkeypatch, tls_material
) -> None:
    """One connection and one SFTP channel for the whole set.

    An SSH handshake against an edge node's small CPU costs real time, which is
    why the install below is a single remote command; a per-file upload paid
    four handshakes per node on top of it, for every push.
    """
    per_call: list[int] = []
    monkeypatch.setattr(deployer_mod, "run_command", lambda *_a, **_k: (0, "", ""))
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda _h, items, **_k: per_call.append(len(list(items))),
    )

    deployer.deploy_configs(
        [Path("/tmp/aureax-edge-solo.yaml")],
        targets=[t for t in deployer._targets if t.node_id == "solo"],
    )

    assert per_call == [4]


def test_the_leaf_pushed_is_the_node_s_own_not_a_neighbour_s(
    deployer, monkeypatch, tls_material
) -> None:
    """Mispairing here hands one node another node's private key — which then
    authenticates as that node to every client that trusts the fleet CA. The
    leaf is looked up by the same node_id the config filename encodes, so the
    two cannot drift apart."""
    pushed: dict[str, list[str]] = {}
    monkeypatch.setattr(deployer_mod, "run_command", lambda *_a, **_k: (0, "", ""))
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda host, items, **_k: pushed.setdefault(host, []).extend(
            Path(local).name for local, _r, _m, _pm in items
        ),
    )

    deployer.deploy_configs(
        [Path(f"/tmp/aureax-edge-{t.node_id}.yaml") for t in deployer._targets]
    )

    for target in deployer._targets:
        node_id = target.node_id
        assert pushed[target.host] == [
            f"aureax-edge-{node_id}.yaml", "ca.pem", f"{node_id}.pem", f"{node_id}.key",
        ], node_id


def test_the_install_command_locks_down_the_tls_material(
    deployer, monkeypatch, tls_material
) -> None:
    """One command, and every mode in it is load-bearing.

    node.key 0600 root:root because it is the node's identity and the container
    runs as root; the certificates 0644 because a certificate is not a secret
    and pretending otherwise invites a chmod that breaks a non-root reader; the
    directory 0700 and root-owned because a /etc/aureax/tls left behind owned
    by the login user lets a local account swap the key underneath the pod.
    """
    _uploads, cmds = _record_push(deployer, monkeypatch)

    assert len(cmds) == 1, "a second SSH handshake per file buys nothing"
    cmd = cmds[0]
    assert "sudo mkdir -p /etc/aureax /etc/aureax/tls /opt/aureax/models" in cmd
    assert "sudo chmod 0700 /etc/aureax/tls" in cmd
    for dest in ("/etc/aureax/aureax-edge.yaml", "/etc/aureax/tls/ca.pem",
                 "/etc/aureax/tls/node.pem", "/etc/aureax/tls/node.key"):
        assert f" {dest} " in cmd or cmd.endswith(dest), dest
    chown = next(p for p in cmd.split(" && ") if p.startswith("sudo chown root:root"))
    assert "/etc/aureax/tls/ca.pem" in chown
    assert "/etc/aureax/tls/node.pem" in chown
    assert "/etc/aureax/tls/node.key" in chown
    assert "/etc/aureax/aureax-edge.yaml" in chown
    assert "/etc/aureax/tls " in chown or chown.endswith("/etc/aureax/tls")
    mode_0600 = next(p for p in cmd.split(" && ") if p.startswith("sudo chmod 0600"))
    assert "/etc/aureax/tls/node.key" in mode_0600
    assert "/etc/aureax/aureax-edge.yaml" in mode_0600
    mode_0644 = next(p for p in cmd.split(" && ") if p.startswith("sudo chmod 0644"))
    assert "/etc/aureax/tls/ca.pem" in mode_0644
    assert "/etc/aureax/tls/node.pem" in mode_0644
    assert "node.key" not in mode_0644, "the private key must never be world-readable"
    # Node-side names are fixed (node.pem/node.key), never <node_id>.pem: a
    # renamed node would otherwise strand a file its config no longer names.
    assert "solo.pem" not in cmd and "solo.key" not in cmd


def test_deploy_configs_refuses_a_node_whose_leaf_was_never_minted(
    deployer, monkeypatch, tls_material
) -> None:
    """Refused before ANY node is touched, like the missing-config check.

    Pushing the rest first would leave the fleet half converted, and minting
    here instead would issue a leaf from a CA the other nodes may never have
    been given.
    """
    (tls_material / "w2.key").unlink()
    touched: list[str] = []
    monkeypatch.setattr(deployer_mod, "upload_files", lambda *a, **k: touched.append("up"))
    monkeypatch.setattr(deployer_mod, "run_command", lambda *a, **k: (0, "", ""))

    paths = [Path(f"/tmp/aureax-edge-{t.node_id}.yaml") for t in deployer._targets]
    with pytest.raises(RuntimeError, match=r"aureax generate") as exc:
        deployer.deploy_configs(paths)

    assert "w2" in str(exc.value)
    assert touched == [], "nodes were pushed before the fleet was found incomplete"


def test_deploy_configs_refuses_a_fleet_with_no_ca(deployer, monkeypatch, tls_material) -> None:
    """Without ca.pem no node can verify the orchestrator, so telemetry and
    log shipping fail closed on every node at once."""
    (tls_material / "ca.pem").unlink()
    monkeypatch.setattr(deployer_mod, "upload_files", lambda *a, **k: None)
    monkeypatch.setattr(deployer_mod, "run_command", lambda *a, **k: (0, "", ""))

    with pytest.raises(RuntimeError, match=r"aureax generate") as exc:
        deployer.deploy_configs([Path("/tmp/aureax-edge-solo.yaml")],
                                targets=[t for t in deployer._targets if t.node_id == "solo"])
    assert "fleet CA" in str(exc.value)


def test_deploy_configs_raises_on_nonzero_remote_exit(deployer, monkeypatch) -> None:
    """A failed remote mkdir/mv must abort the deploy, not report success.

    This step exists so a pod is never scheduled onto a node without its
    config; swallowing the exit code produced exactly the ContainerCreating
    hang it guards against.
    """
    monkeypatch.setattr(
        deployer_mod, "run_command",
        lambda _h, cmd, **_k: (
            (1, "", "sudo: a password is required")
            if cmd.startswith("sudo mkdir")
            else (0, "", "")
        ),
    )
    monkeypatch.setattr(deployer_mod, "upload_files", lambda *a, **k: None)

    paths = [Path(f"/tmp/aureax-edge-{t.node_id}.yaml") for t in deployer._targets]
    with pytest.raises(RuntimeError, match="config push"):
        deployer.deploy_configs(paths)


def test_the_config_is_staged_where_only_the_login_user_can_write(deployer, monkeypatch) -> None:
    """The resolved config carries model.server.api_key — and now the node's
    TLS private key rides the same staging directory.

    /tmp is world-writable, so a fixed name there can be pre-created by any
    local account on the node: the put then writes the secret into a file
    somebody else owns, and only the following chmod fails. A unique name
    under a 0700 directory in the login user's home closes both halves.
    """
    uploads: list[dict] = []
    cmds: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda _h, items, **k: uploads.extend(
            {"remote": remote, "mode": mode, "parent_mode": parent_mode}
            for _local, remote, mode, parent_mode in items
        ),
    )
    monkeypatch.setattr(
        deployer_mod, "run_command",
        lambda _h, cmd, **k: cmds.append((cmd, k)) or (0, "", ""),
    )

    solo = [t for t in deployer._targets if t.node_id == "solo"]
    deployer.deploy_configs([Path("/tmp/aureax-edge-solo.yaml")], targets=solo)
    deployer.deploy_configs([Path("/tmp/aureax-edge-solo.yaml")], targets=solo)

    # Four files per push, so the second push's config is uploads[4]: what is
    # under test is that the same file staged twice gets two different names.
    first, second = uploads[0]["remote"], uploads[4]["remote"]
    assert not first.startswith("/tmp/"), first
    assert first != second, "a fixed staging name is guessable and pre-creatable"
    # Within one push, too — a shared name would have each file clobber the last.
    assert len({u["remote"] for u in uploads}) == len(uploads)
    for upload in uploads:
        assert upload["mode"] == 0o600, upload
        assert upload["parent_mode"] == 0o700, upload
    # The install command must move the files this push actually staged.
    assert first in cmds[0][0]
    # And the remote command gets its own budget: sudo can stall 10-30s on a
    # node with incomplete DNS, and the 15s connect budget aborted the deploy
    # with "exited 124" for a node that was merely slow.
    assert cmds[0][1]["timeout"] == 60


def test_one_failed_push_does_not_hide_the_others(deployer, monkeypatch) -> None:
    """Aborting on the first failure left a group's remaining members unpushed
    AND unnamed, so the operator was told about one node while the group sat on
    mixed configs. The stage files must not survive either — two of the four
    are secrets (the resolved config and the node's TLS private key)."""
    staged: list[str] = []
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda _h, items, **_k: staged.extend(remote for _l, remote, _m, _pm in items),
    )
    installs: list[str] = []
    cleanups: list[tuple[str, str]] = []

    def run(host, cmd, **_k):
        if cmd.startswith("rm -f"):
            cleanups.append((host, cmd))
            return 0, "", ""
        installs.append(host)
        return (1, "", "no space left on device") if host == "10.77.0.2" else (0, "", "")

    monkeypatch.setattr(deployer_mod, "run_command", run)

    paths = [Path(f"/tmp/aureax-edge-{t.node_id}.yaml") for t in deployer._targets]
    with pytest.raises(RuntimeError, match="config push failed for w1"):
        deployer.deploy_configs(paths)

    assert len(installs) == len(deployer._targets), "the loop stopped at the first failure"
    assert [h for h, _c in cleanups] == ["10.77.0.2"], "the stage files were left behind"
    # Every file this node staged, not just the config: the upload of any one
    # of the four can be what failed, and the key must never outlive the push.
    w1_staged = [s for s in staged if s in cleanups[0][1]]
    assert len(w1_staged) == 4, (cleanups[0][1], staged)


def test_the_worker_reorder_override_reaches_generation(deployer, monkeypatch) -> None:
    """The flag is the operator's "yes, I meant to reorder that group".

    Generation refuses a reordered `node_groups.workers` list because worker
    order IS the --rpc order: renaming RPC0..n silently re-points every
    tensor_split entry at a different machine. If the flag stops arriving here,
    `aureax deploy --accept-worker-reorder` reports success and generates
    nothing — the override becomes a no-op with no error to notice.
    """
    seen: list[dict] = []
    monkeypatch.setattr(
        deployer_mod, "generate_all_edge_configs",
        lambda *a, **k: seen.append(k) or [],
    )

    deployer.generate_configs()
    assert seen[-1]["accept_worker_reorder"] is False, "the default must stay refusing"

    accepting = Deployer(deployer.config_path, "10.77.0.99", accept_worker_reorder=True)
    accepting.generate_configs()
    assert seen[-1]["accept_worker_reorder"] is True


def test_label_passes_refuse_an_empty_node_list(deployer, monkeypatch) -> None:
    """`[]` is provably an API failure — a live cluster always has the k3d
    server node — and kubectl.get() collapses every non-zero exit to {}. Taken
    at face value both passes become no-ops and the deploy reports success with
    nothing scheduled."""
    monkeypatch.setattr(kubectl, "get_nodes", lambda: [])

    with pytest.raises(RuntimeError, match="API server is unreachable"):
        deployer.label_nodes(role=HEAD_ROLE)
    with pytest.raises(RuntimeError, match="API server is unreachable"):
        deployer.label_shard_workers()
    assert deployer._labels == []


def test_join_nodes_reports_each_failure_and_raises_at_the_end(deployer, monkeypatch) -> None:
    """One unreachable node must not abort the loop — but the deploy must fail.

    Continuing past the failure is what keeps one dead box from blocking the
    rest of the fleet; raising afterwards is what keeps an incompletely joined
    fleet from being labelled and reported as a successful deploy.
    """
    monkeypatch.setattr(deployer_mod.k3d, "get_token", lambda _c: "tok")
    monkeypatch.setattr(deployer_mod.k3d, "get_server_url", lambda _p, _h: "https://10.77.0.99:6443")
    monkeypatch.setattr(deployer_mod.k3d, "probe_host", lambda _c: "10.77.0.1")
    monkeypatch.setattr(Deployer, "bootstrap_key", lambda _s, _t: None)

    monkeypatch.setattr(deployer_mod.kubectl, "server_version", lambda: "v1.32.1+k3s1")

    joined: list[str] = []

    def fake_join(host, server_url, token, **_kw):
        joined.append(host)
        if host == "10.77.0.2":  # w1
            return False, "unreachable: TimeoutError"
        return True, "joined"

    monkeypatch.setattr(deployer_mod, "join_agent", fake_join)

    with pytest.raises(RuntimeError, match="w1"):
        deployer.join_nodes()
    assert len(joined) == len(deployer._targets), "loop stopped at the first failure"


def test_wait_nodes_registered_matches_hostnames_not_a_count(
    deployer, monkeypatch, fake_clock, caplog
) -> None:
    """A cluster that still remembers a decommissioned or renamed node
    satisfies a count comparison while the node we are waiting for is still
    joining — and the labelling pass that follows silently skips it."""
    monkeypatch.setattr(kubectl, "get_nodes", lambda: [
        {"name": "k3d-server-0", "roles": "control-plane"},
        {"name": "host-decommissioned", "roles": "worker"},
        *[{"name": f"host-{n}", "roles": "worker"} for n in ROLES if n != "w2"],
    ])

    with caplog.at_level("INFO", logger="aureax.deployer"):
        assert deployer.wait_nodes_registered(timeout=10, interval=5) is False

    assert any("host-w2" in r.getMessage() for r in caplog.records), caplog.records


def test_wait_nodes_registered_passes_when_every_hostname_is_present(
    deployer, fake_clock
) -> None:
    assert deployer.wait_nodes_registered(timeout=10, interval=5) is True


def test_wait_nodes_registered_ignores_the_control_plane(
    deployer, monkeypatch, fake_clock
) -> None:
    """The k3d server is always there and never an edge node; counting it as
    registered would mask a fleet that has not joined at all."""
    monkeypatch.setattr(
        kubectl, "get_nodes", lambda: [{"name": "k3d-server-0", "roles": "control-plane"}]
    )
    assert deployer.wait_nodes_registered(timeout=10, interval=5) is False


def test_wait_nodes_registered_refuses_to_pass_over_an_unresolved_node(
    deployer, monkeypatch, fake_clock, caplog
) -> None:
    """_target_hostname returns "" on any SSH failure, and dropping those
    targets made "All 3 node(s) registered" true of a 4-node fleet with one
    node that never joined."""
    monkeypatch.setattr(
        Deployer, "_target_hostname",
        lambda _s, t: "" if t.node_id == "w2" else f"host-{t.node_id}",
    )
    monkeypatch.setattr(kubectl, "get_nodes", lambda: [
        {"name": f"host-{n}", "roles": "worker"} for n in ROLES if n != "w2"
    ])

    with caplog.at_level("ERROR", logger="aureax.deployer"):
        assert deployer.wait_nodes_registered(timeout=10, interval=5) is False

    assert any("w2" in r.getMessage() for r in caplog.records), caplog.records


# --- setup_cluster ------------------------------------------------------------


@pytest.fixture
def cluster_calls(monkeypatch):
    """k3d/kubectl stubbed for setup_cluster; returns the recorded calls."""
    calls: list[tuple] = []
    monkeypatch.setattr(k3d, "start_cluster", lambda n: calls.append(("start", n)) or True)
    monkeypatch.setattr(k3d, "create_cluster", lambda n, *_a: calls.append(("create", n)))
    monkeypatch.setattr(k3d, "save_token", lambda n: calls.append(("save_token", n)) or None)
    monkeypatch.setattr(kubectl, "wait_api_ready", lambda timeout=120: True)
    return calls


def test_setup_cluster_starts_an_existing_but_stopped_cluster(
    deployer, monkeypatch, cluster_calls
) -> None:
    """Existing != running. A stopped cluster left every later step talking to
    an unreachable API server, surfacing as a confusing `kubectl label`
    failure rather than "cluster is down"."""
    monkeypatch.setattr(k3d, "cluster_exists", lambda _n: True)
    monkeypatch.setattr(k3d, "cluster_running", lambda _n: False)

    deployer.setup_cluster()

    name = deployer.config.kubernetes.cluster_name
    assert ("start", name) in cluster_calls
    assert not any(c[0] == "create" for c in cluster_calls)


def test_setup_cluster_creates_a_missing_cluster(deployer, monkeypatch, cluster_calls) -> None:
    monkeypatch.setattr(k3d, "cluster_exists", lambda _n: False)
    monkeypatch.setattr(k3d, "cluster_running", lambda _n: False)

    deployer.setup_cluster()

    assert any(c[0] == "create" for c in cluster_calls)
    assert not any(c[0] == "start" for c in cluster_calls)


def test_setup_cluster_refuses_to_continue_against_a_dead_api(
    deployer, monkeypatch, cluster_calls
) -> None:
    """This is the guard that stops full_deploy joining and labelling a fleet
    against an API server that never came up."""
    monkeypatch.setattr(k3d, "cluster_exists", lambda _n: True)
    monkeypatch.setattr(k3d, "cluster_running", lambda _n: True)
    monkeypatch.setattr(kubectl, "wait_api_ready", lambda timeout=120: False)

    with pytest.raises(RuntimeError, match="did not become reachable"):
        deployer.setup_cluster()

    assert not any(c[0] == "save_token" for c in cluster_calls)


def test_full_deploy_pushes_configs_before_it_schedules_anything(
    deployer, monkeypatch
) -> None:
    """The pipeline order is the contract; each step depends on the last.

    generate_configs runs first because it is the only caller of
    validate_groups — an invalid group must be caught before the cluster is
    labelled for it. deploy_configs then has to precede apply_manifests and
    both label passes: the DaemonSets mount /etc/aureax as a hostPath with
    `type: Directory`, which the kubelet refuses to create, so a pod scheduled
    onto a node that never received a config sits in ContainerCreating. And
    workers are labelled before heads because a head's llama-server exits if a
    peer is unreachable at load time.

    Reordering these lines breaks nothing at import time and nothing in any
    other test — it just produces stuck pods or a crash-looping head.
    """
    order: list[str] = []

    def record(name, result=None):
        def stub(_self, *_a, **_k):
            order.append(name)
            return result
        return stub

    monkeypatch.setattr(Deployer, "setup_cluster", record("setup_cluster"))
    # generate_configs' return value is passed straight to deploy_configs.
    monkeypatch.setattr(
        Deployer, "generate_configs",
        record("generate_configs", [Path(f"/tmp/aureax-edge-{n}.yaml") for n in ROLES]),
    )
    monkeypatch.setattr(Deployer, "join_nodes", record("join_nodes"))
    monkeypatch.setattr(Deployer, "wait_nodes_registered", record("wait_nodes_registered", True))
    monkeypatch.setattr(Deployer, "deploy_configs", record("deploy_configs"))
    monkeypatch.setattr(Deployer, "apply_manifests", record("apply_manifests"))
    monkeypatch.setattr(Deployer, "label_shard_workers", record("label_shard_workers"))
    monkeypatch.setattr(Deployer, "label_nodes", record("label_nodes"))

    deployer.full_deploy()

    assert order.index("generate_configs") < order.index("deploy_configs")
    assert order.index("deploy_configs") < order.index("apply_manifests")
    assert order.index("deploy_configs") < order.index("label_shard_workers")
    assert order.index("apply_manifests") < order.index("label_shard_workers")
    assert order.index("label_shard_workers") < order.index("label_nodes")


def test_a_failed_config_push_aborts_before_any_pod_is_deleted(applier, monkeypatch) -> None:
    """Push first, restart second — a node must never be bounced onto a config
    it did not receive. The ordering is the whole guard: per-node
    push-then-restart would leave one node running its OLD config while the
    operator sees only "Apply failed"."""
    def boom(_self, _paths, targets=None):
        raise RuntimeError("push failed")

    monkeypatch.setattr(Deployer, "deploy_configs", boom)

    with pytest.raises(RuntimeError, match="push failed"):
        applier.apply_to_nodes(["head"], log=lambda _m: None)

    assert applier._deleted == []


def test_apply_wait_budget_comes_from_the_node_s_own_config(applier, tmp_aureax_home) -> None:
    """A sharded head needs worker_wait + health_timeout — the same arithmetic
    the startupProbe budget covers. Flattened to 600, every `apply --wait` on a
    head reports a failure for an apply that worked."""
    cfg_dir = tmp_aureax_home / ".aureax" / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "aureax-edge-solo.yaml").write_text(
        "node:\n  worker_wait_timeout_seconds: 300\n"
        "model:\n  server:\n    health_timeout_seconds: 2400\n"
    )

    assert applier._apply_wait_budget("solo") == 2820
    assert applier._apply_wait_budget("head") == 600, "no config file — fall back, do not raise"


def test_apply_wait_reports_a_ready_replacement_pod(
    applier, monkeypatch, fake_clock, tmp_aureax_home
) -> None:
    calls = {"n": 0}
    old = {"name": "old", "node": "host-solo", "ready": True, "status": "Running"}
    new = {"name": "new", "node": "host-solo", "ready": True, "status": "Running"}

    def get_pods(_ns):
        calls["n"] += 1
        return [old] if calls["n"] == 1 else [new]

    monkeypatch.setattr(kubectl, "get_pods", get_pods)

    outcomes = applier.apply_to_nodes(["solo"], wait=True, log=lambda _m: None)
    assert outcomes["solo"].startswith("applied (pod Ready"), outcomes


def test_apply_wait_says_so_when_the_pod_never_becomes_ready(
    applier, monkeypatch, fake_clock, tmp_aureax_home
) -> None:
    """A head that cannot re-attach to its RPC workers never goes Ready; the
    outcome has to say that instead of claiming the new config is live."""
    old = {"name": "old", "node": "host-solo", "ready": True, "status": "Running"}
    monkeypatch.setattr(kubectl, "get_pods", lambda _ns: [old])

    outcomes = applier.apply_to_nodes(["solo"], wait=True, log=lambda _m: None)
    assert "did not become Ready in time" in outcomes["solo"], outcomes


def test_apply_does_not_wait_on_an_unusable_pod_snapshot(
    applier, monkeypatch, fake_clock, tmp_aureax_home
) -> None:
    """kubectl.get_pods returns [] for "API unreachable" as much as for "no
    pods", and the delete that just succeeded proves there WAS one. Waiting
    against an empty snapshot accepts the pod just killed as the replacement."""
    ready = {"name": "old", "node": "host-solo", "ready": True, "status": "Running"}
    calls = {"n": 0}

    def get_pods(_ns):
        calls["n"] += 1
        return [] if calls["n"] == 1 else [ready]  # snapshot lost, then listed again

    monkeypatch.setattr(kubectl, "get_pods", get_pods)

    outcomes = applier.apply_to_nodes(["solo"], wait=True, log=lambda _m: None)
    assert "not waiting" in outcomes["solo"], outcomes
    assert "applied" not in outcomes["solo"]


def test_apply_without_wait_never_lists_pods(applier, monkeypatch) -> None:
    """The snapshot is read at exactly one place, under `wait`. Every dashboard
    Apply takes the default, and each list is a full-namespace get + parse."""
    listed: list[str] = []
    monkeypatch.setattr(kubectl, "get_pods", lambda ns: listed.append(ns) or [])

    applier.apply_to_nodes(["solo"], log=lambda _m: None)
    assert listed == []


def test_full_deploy_labels_only_the_head_role(deployer, monkeypatch) -> None:
    """The edge pass must stay role-filtered.

    An unfiltered call labels shard workers as edges — the single failure the
    rest of this module exists to prevent — and it would also schedule the
    planner before any edge MCP server is up.
    """
    seen: list[dict] = []
    for name in ("setup_cluster", "join_nodes", "deploy_configs",
                 "apply_manifests", "label_shard_workers"):
        monkeypatch.setattr(Deployer, name, lambda _s, *_a, **_k: None)
    monkeypatch.setattr(Deployer, "generate_configs", lambda _s: [])
    monkeypatch.setattr(Deployer, "wait_nodes_registered", lambda _s, *_a, **_k: True)
    monkeypatch.setattr(
        Deployer, "label_nodes", lambda _s, *a, **k: seen.append({"args": a, "kwargs": k})
    )

    deployer.full_deploy()

    assert seen == [{"args": (), "kwargs": {"role": HEAD_ROLE}}]


# --- rpc-cache manifest push ----------------------------------------------------


def _record_fleet_push(deployer, monkeypatch):
    uploads: list[tuple[str, str]] = []
    commands: list[tuple[str, str]] = []
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda host, items, **k: uploads.extend((host, Path(local).name) for local, _r, _m, _pm in items),
    )
    monkeypatch.setattr(
        deployer_mod, "run_command",
        lambda host, cmd, **k: (commands.append((host, cmd)), (0, "", ""))[1],
    )
    return uploads, commands


def test_shard_workers_receive_their_heads_rpc_cache_manifest(deployer, monkeypatch, tmp_path) -> None:
    """A worker holds a share of its HEAD's model, so the manifest is the head
    model's, found by the filename in the head's generated config. Heads and
    standalone nodes hold the GGUF itself and get nothing extra."""
    gen = tmp_path / "gen"
    gen.mkdir()
    paths = {n: gen / f"aureax-edge-{n}.yaml" for n in ("head", "w1", "w2", "solo", "planner")}
    paths["head"].write_text("model:\n  filename: m.gguf\n")
    for n in ("w1", "w2", "solo", "planner"):
        paths[n].write_text("model: {}\n")
    deployer.models_dir = tmp_path / "models"
    deployer.models_dir.mkdir()
    (deployer.models_dir / "m.gguf.rpc-manifest.json").write_text('{"entries": []}')

    uploads, commands = _record_fleet_push(deployer, monkeypatch)
    deployer.deploy_configs(list(paths.values()))

    got_manifest = {h for h, name in uploads if name == "m.gguf.rpc-manifest.json"}
    assert got_manifest == {"10.77.0.2", "10.77.0.3"}
    installs = {h: cmd for h, cmd in commands}
    assert deployer_mod.NODE_RPC_CACHE_MANIFEST in installs["10.77.0.2"]
    assert "chmod 0644 " + deployer_mod.NODE_RPC_CACHE_MANIFEST in installs["10.77.0.3"]
    assert deployer_mod.NODE_RPC_CACHE_MANIFEST not in installs["10.77.0.1"]
    assert deployer_mod.NODE_RPC_CACHE_MANIFEST not in installs["10.77.0.4"]


def test_no_manifest_means_the_push_is_unchanged(deployer, monkeypatch, tmp_path) -> None:
    """A fleet that never ran seed-rpc-cache.py pushes exactly the four files."""
    gen = tmp_path / "gen"
    gen.mkdir()
    paths = [gen / f"aureax-edge-{n}.yaml" for n in ("head", "w1", "w2", "solo", "planner")]
    paths[0].write_text("model:\n  filename: m.gguf\n")
    deployer.models_dir = tmp_path / "models"   # absent on purpose

    uploads, commands = _record_fleet_push(deployer, monkeypatch)
    deployer.deploy_configs(paths)

    per_host = {}
    for h, name in uploads:
        per_host.setdefault(h, []).append(name)
    assert all(len(names) == 4 for names in per_host.values()), per_host
    assert all(deployer_mod.NODE_RPC_CACHE_MANIFEST not in cmd for _h, cmd in commands)


# --- WireGuard rides the same push, and only for enabled groups ---------------


def _fixed_stage_path(prefix: str, suffix: str = "") -> tuple[str, str]:
    """``ssh.stage_path`` with the random part removed, so a whole install
    command can be compared as one string."""
    rel = f".aureax-stage/{prefix}{suffix}"
    return rel, f'"$HOME"/{rel}'


# The install commands a fleet with NO WireGuard ran before the transport
# existed, spelled out byte for byte. Opting a group into WireGuard must not
# change a single character for any node outside it — this is the proof.
_NO_WG_INSTALL = (
    "sudo mkdir -p /etc/aureax /etc/aureax/tls /opt/aureax/models && "
    "sudo chmod 0700 /etc/aureax/tls && "
    'sudo mv "$HOME"/.aureax-stage/aureax-edge.yaml /etc/aureax/aureax-edge.yaml && '
    'sudo mv "$HOME"/.aureax-stage/aureax-ca.pem /etc/aureax/tls/ca.pem && '
    'sudo mv "$HOME"/.aureax-stage/aureax-node.pem /etc/aureax/tls/node.pem && '
    'sudo mv "$HOME"/.aureax-stage/aureax-node.key /etc/aureax/tls/node.key && '
    "sudo chown root:root /etc/aureax/aureax-edge.yaml /etc/aureax/tls "
    "/etc/aureax/tls/ca.pem /etc/aureax/tls/node.pem /etc/aureax/tls/node.key && "
    "sudo chmod 0600 /etc/aureax/aureax-edge.yaml /etc/aureax/tls/node.key && "
    "sudo chmod 0644 /etc/aureax/tls/ca.pem /etc/aureax/tls/node.pem && "
    "sudo chown aureax: /opt/aureax/models"
)


@pytest.mark.parametrize("node_id", ["solo", "head", "w1"])
def test_a_fleet_without_wireguard_runs_the_exact_install_command_it_always_ran(
    deployer, monkeypatch, node_id
) -> None:
    monkeypatch.setattr(deployer_mod, "stage_path", _fixed_stage_path)
    uploads, cmds = _record_push(deployer, monkeypatch, node_id=node_id)

    assert cmds == [_NO_WG_INSTALL]
    assert len(uploads) == 4
    assert not any("wireguard" in remote or "wg0" in remote for _l, remote, _k in uploads)


WG_GROUP = {**GROUP, "wireguard": {"enabled": True}}

_WG_PREFLIGHT = (
    "{ command -v wg-quick >/dev/null 2>&1 || { echo 'wireguard-tools is not installed on this "
    "node (no wg-quick); ./scripts/setup-all.sh installs it' >&2; exit 1; }; } && "
    "{ sudo modprobe wireguard || { echo 'the wireguard kernel module is not available on this "
    "node (modprobe wireguard failed)' >&2; exit 1; }; } && "
)
_WG_INSTALL = (
    "sudo mkdir -p /etc/wireguard && "
    "sudo chown root:root /etc/wireguard && "
    "sudo chmod 0700 /etc/wireguard && "
    'if sudo cmp -s "$HOME"/.aureax-stage/aureax-wg.conf /etc/wireguard/wg0.conf; '
    "then wg_changed=0; else wg_changed=1; fi && "
    'sudo mv "$HOME"/.aureax-stage/aureax-wg.conf /etc/wireguard/wg0.conf && '
    "sudo chown root:root /etc/wireguard/wg0.conf && "
    "sudo chmod 0600 /etc/wireguard/wg0.conf && "
    "sudo systemctl enable --now wg-quick@wg0 && "
    '{ [ "$wg_changed" = 0 ] || sudo systemctl restart wg-quick@wg0; } && '
)
_FINAL_CHOWN = "sudo chown aureax: /opt/aureax/models"


@pytest.fixture
def wg_deployer(deployer):
    """`deployer` with its group on WireGuard and every member's pair minted
    (tls_material has already pointed $HOME at tmp_path)."""
    from orchestrator.config import wgstore

    deployer.config = make_network_config(
        make_targets("head", "w1", "w2", "solo", "planner"), [WG_GROUP]
    )
    for node_id in ("head", "w1", "w2"):
        wgstore.ensure_key(node_id)
    return deployer


@pytest.mark.parametrize("node_id", ["head", "w1"])
def test_a_wireguard_member_gets_its_wg0_conf_in_the_same_push(wg_deployer, monkeypatch, node_id) -> None:
    """Fifth file, same connection, same chain; staged 0600 because it carries
    the node's private key inline, exactly like node.key."""
    from orchestrator.config.templating import render_wg_config

    contents: dict[str, str] = {}
    real_upload = []

    def upload(_host, items, **_k):
        for local, remote, mode, parent_mode in items:
            real_upload.append((Path(local).name, remote, mode, parent_mode))
            if Path(local).name == "wg0.conf":  # the only one of the five on disk here
                contents[remote] = Path(local).read_text()

    monkeypatch.setattr(deployer_mod, "stage_path", _fixed_stage_path)
    monkeypatch.setattr(deployer_mod, "upload_files", upload)
    cmds: list[str] = []
    monkeypatch.setattr(deployer_mod, "run_command", lambda _h, cmd, **_k: cmds.append(cmd) or (0, "", ""))

    wg_deployer.deploy_configs(
        [Path(f"/tmp/aureax-edge-{node_id}.yaml")],
        targets=[t for t in wg_deployer._targets if t.node_id == node_id],
    )

    assert [name for name, *_ in real_upload] == [
        f"aureax-edge-{node_id}.yaml", "ca.pem", f"{node_id}.pem", f"{node_id}.key", "wg0.conf",
    ]
    _name, remote, mode, parent_mode = real_upload[-1]
    assert (mode, parent_mode) == (0o600, 0o700)
    assert remote == ".aureax-stage/aureax-wg.conf"
    assert contents[remote] == render_wg_config(node_id, wg_deployer.config)
    # The whole command, byte for byte: the preflight before anything moves,
    # the unchanged mandatory chain, then the tunnel beside the manifest slot.
    assert cmds == [
        _WG_PREFLIGHT + _NO_WG_INSTALL.removesuffix(_FINAL_CHOWN) + _WG_INSTALL + _FINAL_CHOWN
    ]


@pytest.mark.parametrize("node_id", ["solo", "planner"])
def test_nodes_outside_the_tunnel_push_exactly_what_they_always_did(
    wg_deployer, monkeypatch, node_id
) -> None:
    monkeypatch.setattr(deployer_mod, "stage_path", _fixed_stage_path)
    uploads, cmds = _record_push(wg_deployer, monkeypatch, node_id=node_id)

    assert cmds == [_NO_WG_INSTALL]
    assert len(uploads) == 4


def test_the_local_wg0_conf_does_not_outlive_the_push(wg_deployer, monkeypatch, tmp_path) -> None:
    """The key store already holds the private key; a second long-lived copy
    on the Mac buys nothing. Removed on success and on failure alike."""
    seen: list[Path] = []

    def upload(_host, items, **_k):
        seen.extend(Path(local) for local, *_ in items if Path(local).name == "wg0.conf")

    monkeypatch.setattr(deployer_mod, "upload_files", upload)
    monkeypatch.setattr(deployer_mod, "run_command", lambda h, cmd, **_k: (1, "", "boom") if h == "10.77.0.3" else (0, "", ""))

    paths = [Path(f"/tmp/aureax-edge-{t.node_id}.yaml") for t in wg_deployer._targets]
    with pytest.raises(RuntimeError, match="config push failed for w2"):
        wg_deployer.deploy_configs(paths)

    assert len(seen) == 3, "head, w1 and w2 each get a wg0.conf"
    for path in seen:
        assert not path.exists() and not path.parent.exists(), path


def test_a_failed_wireguard_push_cleans_up_the_staged_conf_too(wg_deployer, monkeypatch) -> None:
    staged: list[str] = []
    monkeypatch.setattr(
        deployer_mod, "upload_files",
        lambda _h, items, **_k: staged.extend(remote for _l, remote, _m, _pm in items),
    )
    cleanups: list[str] = []

    def run(_host, cmd, **_k):
        if cmd.startswith("rm -f"):
            cleanups.append(cmd)
            return 0, "", ""
        return 1, "", "wireguard-tools is not installed on this node"

    monkeypatch.setattr(deployer_mod, "run_command", run)
    with pytest.raises(RuntimeError, match="config push failed for w1"):
        wg_deployer.deploy_configs(
            [Path("/tmp/aureax-edge-w1.yaml")],
            targets=[t for t in wg_deployer._targets if t.node_id == "w1"],
        )

    wg_stage = next(s for s in staged if "aureax-wg" in s)
    assert wg_stage in cleanups[0], "the staged wg0.conf (a private key) was left behind"
    assert len([s for s in staged if s in cleanups[0]]) == 5


def test_a_member_without_a_key_refuses_the_whole_push_before_any_node(wg_deployer, monkeypatch) -> None:
    from orchestrator.config import wgstore

    wgstore.key_paths("w2")[0].unlink()
    touched: list[str] = []
    monkeypatch.setattr(deployer_mod, "upload_files", lambda h, *_a, **_k: touched.append(h))
    monkeypatch.setattr(deployer_mod, "run_command", lambda h, *_a, **_k: touched.append(h) or (0, "", ""))

    paths = [Path(f"/tmp/aureax-edge-{t.node_id}.yaml") for t in wg_deployer._targets]
    with pytest.raises(RuntimeError, match=r"w2 has no WireGuard key.*aureax generate"):
        wg_deployer.deploy_configs(paths)
    assert touched == [], "half a star was pushed"


# --- the install chain's shell logic, executed --------------------------------
# The string assertions above pin WHAT runs; this runs it, with every root path
# moved under tmp_path and sudo/systemctl stubbed, to pin the two things a
# string cannot show: wg-quick is restarted only when the file changed, and a
# node without wireguard-tools is refused before anything is moved.


def _wg_member_command() -> str:
    return _WG_PREFLIGHT + _NO_WG_INSTALL.removesuffix(_FINAL_CHOWN) + _WG_INSTALL + _FINAL_CHOWN


def _run_chain(root: Path, wg_conf: str, *, have_wg_quick: bool = True):
    import subprocess

    home = root / "home"
    stage = home / ".aureax-stage"
    stage.mkdir(parents=True, exist_ok=True)
    for name in ("aureax-edge.yaml", "aureax-ca.pem", "aureax-node.pem", "aureax-node.key"):
        (stage / name).write_text(name)
    (stage / "aureax-wg.conf").write_text(wg_conf)
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = root / "systemctl.log"
    (bin_dir / "sudo").write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  chown|modprobe) exit 0 ;;\n"
        f'  systemctl) shift; echo "$*" >> "{log}"; exit 0 ;;\n'
        "esac\n"
        'exec "$@"\n'
    )
    (bin_dir / "sudo").chmod(0o755)
    wg_quick = bin_dir / "wg-quick"
    if have_wg_quick:
        wg_quick.write_text("#!/bin/sh\nexit 0\n")
        wg_quick.chmod(0o755)
    else:
        wg_quick.unlink(missing_ok=True)
    cmd = _wg_member_command().replace("/etc/", f"{root}/etc/").replace("/opt/", f"{root}/opt/")
    result = subprocess.run(
        ["/bin/sh", "-c", cmd], capture_output=True, text=True, timeout=30,
        env={"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin"},
    )
    calls = log.read_text().splitlines() if log.exists() else []
    log.unlink(missing_ok=True)
    return result, calls


@pytest.mark.skipif(os.name != "posix", reason="runs the chain under /bin/sh")
def test_wg_quick_restarts_only_when_the_installed_file_changed(tmp_path) -> None:
    root = tmp_path / "node"

    first, calls = _run_chain(root, "conf v1\n")
    assert first.returncode == 0, first.stderr
    assert calls == ["enable --now wg-quick@wg0", "restart wg-quick@wg0"], "first install"
    installed = root / "etc" / "wireguard" / "wg0.conf"
    assert installed.read_text() == "conf v1\n"
    assert stat.S_IMODE(installed.stat().st_mode) == 0o600
    assert stat.S_IMODE(installed.parent.stat().st_mode) == 0o700

    again, calls = _run_chain(root, "conf v1\n")
    assert again.returncode == 0, again.stderr
    assert calls == ["enable --now wg-quick@wg0"], "an unchanged tunnel was bounced"

    changed, calls = _run_chain(root, "conf v2\n")
    assert changed.returncode == 0, changed.stderr
    assert calls == ["enable --now wg-quick@wg0", "restart wg-quick@wg0"]
    assert installed.read_text() == "conf v2\n"


@pytest.mark.skipif(os.name != "posix", reason="runs the chain under /bin/sh")
def test_a_node_without_wireguard_tools_is_refused_before_anything_moves(tmp_path) -> None:
    root = tmp_path / "node"
    result, calls = _run_chain(root, "conf\n", have_wg_quick=False)

    assert result.returncode == 1
    assert "wireguard-tools is not installed" in result.stderr
    assert calls == []
    assert not (root / "etc").exists(), "the preflight must run before the first mv"


@pytest.mark.parametrize("script", ["setup-all.sh", "join-node.sh"])
def test_provisioning_installs_what_the_wireguard_preflight_asks_for(script) -> None:
    """The deploy preflight refuses a node without wg-quick by naming
    ./scripts/setup-all.sh, so provisioning has to install it — on every node,
    idempotently, without the dkms recommends — and has to fail clearly on a
    kernel with no wireguard module. Text-level because both run on a node."""
    from tests.conftest import PROJECT_ROOT

    text = (PROJECT_ROOT / "scripts" / script).read_text()
    assert "dpkg -s wireguard-tools" in text, "the install must be skipped when already present"
    assert "apt-get install -y -qq --no-install-recommends wireguard-tools" in text
    assert "modprobe wireguard" in text
    assert "kernel module is not available" in text
