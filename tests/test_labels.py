"""The shared node-label helpers, and the pod image digests the start gate reads.

Both cover the same failure: a node running two pods it should not. A node that
carries `aureax.io/role=edge` and `aureax.io/shard=worker` at once matches both
DaemonSet selectors and runs llama-server beside rpc-server, and a head whose
peers were built from a different commit hangs at the llama.cpp RPC handshake.
Neither says anything in a log — the check is the only signal there is.
"""

from __future__ import annotations

import pytest

from orchestrator.cluster import kubectl, labels
from orchestrator.config.schema import KubernetesConfig

K8S = KubernetesConfig()


def _node(name: str, **node_labels: str) -> dict:
    """A node in the shape `kubectl.get_nodes()` returns (kubectl.py:120-137)."""
    return {"name": name, "status": "True", "roles": "worker", "labels": dict(node_labels)}


# --------------------------------------------------------------------------
# double_labelled_nodes
# --------------------------------------------------------------------------


def test_a_node_carrying_both_label_keys_is_reported() -> None:
    """The whole point: a worker promoted to head keeps its shard label."""
    nodes = [_node("host-a", **{K8S.node_label: "edge", K8S.shard_label: "worker"})]

    assert labels.double_labelled_nodes(nodes, K8S) == ["host-a"]


def test_a_node_carrying_one_label_is_not_reported() -> None:
    """Every correctly labelled node in the fleet carries exactly one of them,
    so a helper that flagged those would fire on every start."""
    nodes = [
        _node("host-edge", **{K8S.node_label: "edge"}),
        _node("host-worker", **{K8S.shard_label: "worker"}),
        _node("k3d-server-0", **{"node-role.kubernetes.io/control-plane": "true"}),
    ]

    assert labels.double_labelled_nodes(nodes, K8S) == []


def test_a_node_with_no_labels_key_at_all_does_not_raise() -> None:
    """`get_nodes()` always sets "labels", but the test doubles across this repo
    build node dicts by hand (tests/test_deployer.py:51-53 has no "labels"), and
    an unrelated fixture growing a KeyError here would be a confusing failure."""
    assert labels.double_labelled_nodes([{"name": "host-a"}, {"name": "b", "labels": None}], K8S) == []


def test_every_double_labelled_node_is_reported_in_listing_order() -> None:
    """The caller prints these beside the node listing it already rendered, so
    the order has to match that listing rather than be re-sorted."""
    nodes = [
        _node("host-c", **{K8S.node_label: "edge", K8S.shard_label: "worker"}),
        _node("host-a", **{K8S.shard_label: "worker"}),
        _node("host-b", **{K8S.node_label: "edge", K8S.shard_label: "worker"}),
    ]

    assert labels.double_labelled_nodes(nodes, K8S) == ["host-c", "host-b"]


def test_a_hand_set_third_value_on_either_key_still_counts() -> None:
    """Matching on the key, not the value: the label passes only ever write
    edge/worker, so anything else got there by hand and the remedy — strip the
    key that does not belong — is identical."""
    nodes = [_node("host-a", **{K8S.node_label: "edge", K8S.shard_label: "leftover"})]

    assert labels.double_labelled_nodes(nodes, K8S) == ["host-a"]


def test_the_label_keys_come_from_the_config_not_from_a_literal() -> None:
    """kubernetes.node_label/shard_label are configurable (schema.py:358, :371);
    a hardcoded key here would silently stop matching a renamed fleet."""
    k8s = KubernetesConfig(node_label="acme.io/role", shard_label="acme.io/shard")
    nodes = [
        _node("renamed", **{"acme.io/role": "edge", "acme.io/shard": "worker"}),
        _node("default-keys", **{"aureax.io/role": "edge", "aureax.io/shard": "worker"}),
    ]

    assert labels.double_labelled_nodes(nodes, k8s) == ["renamed"]


# --------------------------------------------------------------------------
# strip_conflicting_label
# --------------------------------------------------------------------------


def test_a_successful_strip_reports_ok(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_remove(node_name: str, label: str) -> tuple[bool, str]:
        calls.append((node_name, label))
        return True, 'node/host-a unlabeled'

    monkeypatch.setattr(kubectl, "remove_label", fake_remove)

    ok, detail = labels.strip_conflicting_label("host-a", "aureax.io/shard")

    assert (ok, detail) == (True, "node/host-a unlabeled")
    assert calls == [("host-a", "aureax.io/shard")]


def test_a_failed_strip_reports_the_kubectl_detail(monkeypatch) -> None:
    """The detail is the whole diagnosis the caller has — losing it leaves an
    operator with "could not strip" and no reason."""
    monkeypatch.setattr(
        kubectl, "remove_label",
        lambda _n, _l: (False, 'Error from server (NotFound): nodes "host-a" not found'),
    )

    ok, detail = labels.strip_conflicting_label("host-a", "aureax.io/shard")

    assert ok is False
    assert "not found" in detail


def test_an_absent_label_is_a_success(monkeypatch) -> None:
    """Idempotence is load-bearing: both label passes call this on every node
    they claim, and most of those nodes never carried the other label."""
    monkeypatch.setattr(kubectl, "remove_label", lambda _n, _l: (True, 'label "x" not found.'))

    ok, _detail = labels.strip_conflicting_label("host-a", "aureax.io/shard")

    assert ok is True


# --------------------------------------------------------------------------
# kubectl.get_pods() shape
# --------------------------------------------------------------------------
# There is deliberately no image/imageID key here. The pre-start build gate
# used to compare each group member's imageID and fired on every start of the
# real fleet: every node builds `aureax-client:dev` NATIVELY
# (scripts/build-on-node.sh), so the digests differ by construction while the
# compiled binaries are identical. The gate now compares the binaries over SSH
# (orchestrator/cluster/buildcheck.py, tests/test_buildcheck.py), and nothing
# else ever wanted a digest from a pod listing.


def _pods_payload(*container_statuses: list[dict]) -> dict:
    """One pod per argument, each with the given containerStatuses list."""
    return {
        "items": [
            {
                "metadata": {"name": f"aureax-edge-{i}"},
                "spec": {"nodeName": f"host-{i}"},
                "status": {"phase": "Running", "containerStatuses": statuses},
            }
            for i, statuses in enumerate(container_statuses)
        ]
    }


@pytest.fixture
def fake_get(monkeypatch):
    """Replace the module-level `get` that `get_pods` reads through."""

    def install(payload: dict) -> None:
        monkeypatch.setattr(kubectl, "get", lambda _resource, namespace=None: payload)

    return install


def test_no_image_digest_is_surfaced(fake_get) -> None:
    """Pinned, not incidental: a digest here is a false mismatch waiting to be
    compared again. The question it looked like it answered — do these members
    run the same llama.cpp — is answered by buildcheck over SSH."""
    fake_get(_pods_payload([{"ready": True, "imageID": "sha256:aaa"}]))

    pod = kubectl.get_pods()[0]

    assert "images" not in pod and "image" not in pod


def test_the_pod_keys_every_caller_indexes_are_present(fake_get) -> None:
    """`orchestrator/lifecycle.py`, `orchestrator/deploy/deployer.py`,
    `orchestrator/dashboard/watchers.py` and `orchestrator/cli.py` all index
    these by name, so none may be renamed or dropped."""
    fake_get(_pods_payload([{"ready": True, "imageID": "sha256:aaa", "restartCount": 3}]))

    pod = kubectl.get_pods()[0]

    assert pod["name"] == "aureax-edge-0"
    assert pod["node"] == "host-0"
    assert pod["status"] == "Running"
    assert pod["ready"] is True
    assert pod["restarts"] == 3
