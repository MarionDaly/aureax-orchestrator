"""Node-label helpers shared by the deploy pass and the start pass.

Lives in ``orchestrator/cluster/`` rather than beside either caller because
both need it and neither may import the other: ``orchestrator/deploy/deployer.py``
imports ``orchestrator.cluster.kubectl`` at module level while
``orchestrator/lifecycle.py`` imports it lazily inside its functions, and
nothing imports across those two modules. ``cluster`` is the package both
already depend on, so putting the shared code here is the one placement that
cannot introduce an import cycle.

Nothing here logs. The two callers report through different channels — the
deployer's ``logging`` logger, lifecycle's SystemLogger — and a failed strip is
fatal in one and merely loud in the other, so the decision belongs to them.
"""

from __future__ import annotations

from orchestrator.cluster import kubectl


def strip_conflicting_label(node_name: str, label: str) -> tuple[bool, str]:
    """Remove the OTHER pass's label from a node this pass just claimed.

    A node carrying both ``aureax.io/role=edge`` and ``aureax.io/shard=worker``
    matches BOTH DaemonSet selectors and runs llama-server and rpc-server side
    by side on one machine — a head's whole model and a worker's weight shard
    competing for the same RAM. Nothing else notices two pods on one node, so a
    role flip in either direction has to REMOVE the label it is leaving behind
    and not merely add the new one; both passes only ever added.

    Idempotent: ``kubectl.remove_label`` counts an absent label as success, so
    this is a no-op on a node that never held it, and an unknown node is still
    a failure there rather than a silent success.

    Returns ``(ok, detail)`` straight through, unlogged.
    """
    return kubectl.remove_label(node_name, label)


def double_labelled_nodes(nodes: list[dict], k8s) -> list[str]:
    """Names of the nodes in `nodes` carrying BOTH label keys, in listing order.

    Pure over the dicts ``kubectl.get_nodes()`` returns — the ``labels`` key it
    already populates — so every caller that holds a node listing gets this
    check for free, with no second API call on a path that is already several
    kubectl round-trips deep.

    Matches on the KEYS, not on ``edge_label_value``/``shard_label_value``: the
    only writer of either key is ``kubectl.label_node`` in the two label passes,
    which always writes those two values, so key presence is the whole predicate
    for "matches both DaemonSet selectors". A key with some hand-set third value
    is reported too, deliberately — the remedy is the same strip either way.

    `k8s` is a ``KubernetesConfig`` (orchestrator/config/schema.py); the label
    keys are read from it rather than hardcoded because they are configurable
    there.
    """
    hits: list[str] = []
    for node in nodes:
        labels = node.get("labels") or {}
        if k8s.node_label in labels and k8s.shard_label in labels:
            hits.append(node["name"])
    return hits
