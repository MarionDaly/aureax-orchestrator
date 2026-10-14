"""Sharded-group sizing arithmetic that lives in YAML and shell, not in Python.

Changing a group's model or its worker count moves four numbers, and three of
them are outside any module that could type-check them: the worker memory
``requests``/``limits`` in ``k8s/daemonset-shard.yaml``, the readiness ceiling
``EDGE_WAIT`` in ``scripts/update-all.sh``, and the fallback ``EDGE_WAIT_SECS``
in ``scripts/setup-all.sh``. The fourth, ``model.server.health_timeout_seconds``,
is a per-head config value the other three are sized against.

Nothing connected them before this module. Each one fails in a way that does
not name the number that was wrong: a share that outgrows ``limits`` OOM-kills
a worker mid-load, and a ceiling shorter than a real cold load reports "MCP not
responding" for a head that is loading perfectly normally.

``tests/test_probe_budget.py`` already guards the startupProbe half of the same
arithmetic and already parses both fleet files and the edge manifest. This
module imports its parsing rather than re-deriving it, so the two halves cannot
end up reading the same manifest two different ways.

TWO RELATIONS THAT THE COMMENTS SUGGEST BUT THAT ARE DELIBERATELY NOT ENCODED
----------------------------------------------------------------------------
Both fail on values that the manifest and the script call CORRECT, so adding
them would not tighten this module — it would break it against a fleet that is
working. They are written down here so the next person reaches for the reason
before reaching for the assert.

1. The rollback's per-worker share is NOT checked against ``requests``.
   ``71.7 GB -> 66.78 GiB / 7 = 9.54 GiB`` against ``requests.memory: "8Gi"``.
   That is not a bug: ``requests`` is a scheduling reservation and ``limits``
   is the OOM ceiling, and ``k8s/daemonset-shard.yaml:203-205`` claims only the
   11Gi limit for the rollback, never the request. R1 is therefore scoped to
   the groups a fleet file actually configures; R5 is where the rollback's
   memory is genuinely covered.

2. The rollback's give-up point is NOT checked against ``EDGE_WAIT``.
   ``300 + 2400 = 2700 > 1800``. ``scripts/update-all.sh:377-388`` claims only
   that 1800s outlasts the 122B's *cold load* (~1240s), not its give-up point —
   and it is right: 300s of worker wait plus ~1219s of tensor streaming is
   ~1519s in the normal case. R8 is therefore scoped to deployed heads; R9 is
   where the rollback's give-up point is genuinely covered, by setup-all.sh's
   2700s fallback.

The overhead constants below are ADDITIVE, not a multiplier. ``8.8/6.90`` is
1.275x and ``10.0/9.5`` is 1.053x, so no single factor fits both rows: the
difference between them is a KV term that scales with each head's own
``context_size`` (131072 against 4096), not with the model size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from client.config.parser import load_edge_config_raw
from orchestrator.config.loader import load_network_config
from orchestrator.config.schema import (
    DEFAULT_WORKER_WAIT_SECONDS,
    SHARD_WORKER_ROLE,
    ModelRegistryEntry,
)
from tests.conftest import PROJECT_ROOT
from tests.test_probe_budget import (
    DEFAULT_HEALTH_TIMEOUT_SECONDS,
    NETWORK_CONFIGS,
    _as_int,
    _probe_budget_seconds,
    _startup_probe,
)

CONFIGS_DIR = PROJECT_ROOT / "configs"
SHARD_DAEMONSET = PROJECT_ROOT / "k8s" / "daemonset-shard.yaml"
UPDATE_ALL = PROJECT_ROOT / "scripts" / "update-all.sh"
SETUP_ALL = PROJECT_ROOT / "scripts" / "setup-all.sh"
START_SH = PROJECT_ROOT / "docker" / "scripts" / "start.sh"

# The frozen 122B rollback target. Kept unmodified on purpose, so every ceiling
# in this module is a CEILING to preserve rather than a description of today.
ROLLBACK_HEAD_CONFIG = "aureax-shard-head.yaml"

GIB = 2**30
# Decimal GB, because that is the unit `ModelRegistryEntry.size_gb` is declared
# in (orchestrator/config/schema.py:460-462) and the unit model cards quote.
# Reading a registry number as GiB overstates every headroom below by 7%.
GB = 10**9

# k8s/daemonset-shard.yaml:178-179 — "6.90 weights + 1.00 KV + 0.02 recurrent
# + ~0.5 compute + ~0.4 rpc-server/supervisor = ~8.8 GiB". This is the sum of
# the non-weight terms for the PEAK worker (the middle one: an even
# tensor_split is not an even footprint).
#
# NOT INCLUDED: pipeline-parallel input copies. At b11429 a layer-split RPC
# head runs with pipeline parallelism, and ggml_backend_sched keeps
# GGML_SCHED_MAX_COPIES = 4 copies of every cross-split input and every graph
# input per consuming backend — up to ~512 MiB more per worker at 128k context
# (one 512-token mask copy is 128 MiB). This constant was derived with pipeline
# parallelism off and the real growth is unmeasured: re-measure the peak
# worker's RSS on b11429 before trusting it, and change it then, not before.
PEAK_OVERHEAD_GIB = 1.92

# k8s/daemonset-shard.yaml:203-205 — the 122B rollback's 9.5 GiB share becomes
# "~10.0 GiB with overhead". SMALLER than PEAK_OVERHEAD_GIB because
# configs/aureax-shard-head.yaml runs 4096 context, not 131072, so its KV term
# is negligible. Two constants rather than an average: an average passes a case
# that neither manifest number supports.
ROLLBACK_OVERHEAD_GIB = 0.5

# k8s/daemonset-shard.yaml:193-197 — the [n_kv x n_ubatch x n_head_q] F32 score
# tensor llama.cpp reserves when flash attention is OFF, at 131072 / 512 / 16.
FA_OFF_SCORE_TENSOR_GIB = 4.00

# scripts/update-all.sh:402 — "Cold load measures ~17 s/GB".
COLD_LOAD_SECONDS_PER_GB = 17


# --- Parsing helpers ---------------------------------------------------------

_QUANTITY_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<suffix>[EPTGMKk]i?|)\s*$")

_MULTIPLIERS = {
    "": 1,
    "k": 10**3,
    "K": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
    "E": 10**18,
    "Ki": 2**10,
    "Mi": 2**20,
    "Gi": 2**30,
    "Ti": 2**40,
    "Pi": 2**50,
    "Ei": 2**60,
}


def _quantity_gib(value: str) -> float:
    """A K8s memory quantity ("11Gi", "8Gi", "512Mi", "4G") as GiB.

    The suffix is not decoration and the two families are not interchangeable:
    "8G" is 7.45 GiB. Treating one as the other is a 7% error in exactly the
    direction that turns a checked headroom into an OOM kill.
    """
    match = _QUANTITY_RE.match(str(value))
    if not match:
        raise ValueError(f"not a Kubernetes quantity: {value!r}")
    return float(match["num"]) * _MULTIPLIERS[match["suffix"]] / GIB


def _shard_memory() -> tuple[float, float]:
    """(requests, limits) worker memory in GiB, from k8s/daemonset-shard.yaml:219-223."""
    manifest = YAML().load(SHARD_DAEMONSET.read_text())
    containers = manifest["spec"]["template"]["spec"]["containers"]
    sized = [c for c in containers if "resources" in c]
    assert len(sized) == 1, (
        f"daemonset-shard.yaml has {len(sized)} containers with a resources block; "
        "this module assumes the single rpc-server container carries the sizing"
    )
    resources = sized[0]["resources"]
    return (
        _quantity_gib(resources["requests"]["memory"]),
        _quantity_gib(resources["limits"]["memory"]),
    )


def _script_assignments(path: Path, name: str) -> list[int]:
    """Every integer literal assigned to ``name`` in a shell script.

    ALL of them, not the last one: ``EDGE_WAIT`` is assigned twice (the
    with-groups and without-groups branches, scripts/update-all.sh:394-398) and
    ``EDGE_WAIT_SECS`` twice as literals (scripts/setup-all.sh:937, :940). Its
    third assignment (:904) is a command substitution that derives the budget
    from config; the pattern only matches a bare integer, so that one is
    skipped by construction rather than by a special case.
    """
    pattern = re.compile(
        rf"^[ \t]*(?:local[ \t]+|export[ \t]+)?{re.escape(name)}=(\d+)[ \t]*(?:#.*)?$",
        re.MULTILINE,
    )
    values = [int(v) for v in pattern.findall(path.read_text())]
    assert values, (
        f"{path.name} no longer assigns an integer literal to {name} — every ceiling "
        f"check against {name} in this module would then pass vacuously. If the "
        "variable was renamed, rename it here too."
    )
    return values


def _model_for(network_config, filename: str) -> ModelRegistryEntry:
    """The registry entry a head's ``model.filename`` names.

    Keyed by filename, not by name: that is how scripts/parse-config.py's
    ``model_for_node`` branch resolves it, and the head configs say so in their
    own comments (configs/aureax-shard-head-35b.yaml:72-76). Matching on `name`
    here would resolve entries the deployed fleet never reads.
    """
    entries = [m for m in network_config.model_registry.models if m.filename == filename]
    assert entries, (
        f"no model_registry entry has filename {filename!r}. parse-config.py would "
        "emit empty model vars for that node and the failure would surface later, in "
        "download-model.sh — and this module has no size to budget against."
    )
    return entries[0]


@dataclass(frozen=True)
class GroupFacts:
    """Everything the sizing relations need about one head and its workers."""

    name: str
    head_id: str
    head_config: str
    n_workers: int
    model: ModelRegistryEntry
    worker_wait: int
    health_timeout: int
    context_size: int
    flash_attention: object
    # Layers the group keeps on the head (node_groups.head_layers); 0 is the
    # layout every group has run so far. With a positive count the workers
    # share only the remaining layers, and the share below shrinks by the
    # offloaded fraction of the n_layer + 1 placement slots (the extra slot is
    # the output tensor, which always lands on the last worker). The layer
    # count comes from the registry entry's `layers:` hint; without it a
    # head-resident layout cannot be sized here and the test says so.
    head_layers: int = 0
    layers: int | None = None

    @property
    def share_gib(self) -> float:
        """This group's per-worker share of the weights, in GiB."""
        total = self.model.size_gb * GB / GIB
        if self.head_layers > 0:
            assert self.layers, (
                f"group {self.name!r} keeps {self.head_layers} layers on the head but the "
                f"registry entry for {self.model.filename!r} has no `layers:` hint, so the "
                "workers' share cannot be sized — add the model's block_count there"
            )
            slots = self.layers + 1
            total *= (slots - self.head_layers) / slots
        return total / self.n_workers

    @property
    def give_up_seconds(self) -> int:
        """When the head abandons a load: peer wait, then model load."""
        return self.worker_wait + self.health_timeout


def _head_facts(network_config, target, name: str, head_id: str, n_workers: int,
                head_layers: int = 0) -> GroupFacts:
    path = CONFIGS_DIR / target.config_file
    assert path.exists(), f"{head_id} names a config_file that is not in configs/"
    raw = load_edge_config_raw(path)
    model = raw.get("model") or {}
    server = model.get("server") or {}
    node = raw.get("node") or {}

    # Same rule as tests/test_probe_budget.py:113-119: the supervisor reads the
    # RESOLVED config, so a template that hardcodes a number is what actually
    # runs and that number wins over the group's.
    worker_wait = _group_wait(network_config)
    hardcoded = _as_int(node.get("worker_wait_timeout_seconds"))
    if hardcoded is not None:
        worker_wait = max(worker_wait, hardcoded)

    entry = _model_for(network_config, model.get("filename"))
    return GroupFacts(
        name=name,
        head_id=head_id,
        head_config=target.config_file,
        n_workers=n_workers,
        model=entry,
        worker_wait=worker_wait,
        health_timeout=int(server.get("health_timeout_seconds", DEFAULT_HEALTH_TIMEOUT_SECONDS)),
        context_size=int(server.get("context_size", 0)),
        flash_attention=server.get("flash_attention"),
        head_layers=head_layers,
        layers=entry.layers,
    )


def _group_wait(network_config) -> int:
    """The longest ``worker_wait_timeout_seconds`` this fleet file configures.

    Heads keep the ``${worker_wait_timeout_seconds}`` placeholder, so the wait a
    head actually performs comes from whichever group it heads — never from the
    template. Taking the longest configured value (floored at the schema
    default) is what makes these ceilings hold for any head this file could
    point at, including a restored rollback group.
    """
    waits = [g.worker_wait_timeout_seconds for g in network_config.node_groups]
    return max([DEFAULT_WORKER_WAIT_SECONDS, *waits])


def _groups_under_test(network_config: Path) -> list[GroupFacts]:
    """One GroupFacts per ``node_groups`` entry in a fleet file."""
    cfg = load_network_config(network_config)
    by_id = {t.node_id: t for t in cfg.ssh.targets}
    facts = []
    for group in cfg.node_groups:
        target = by_id.get(group.head)
        assert target is not None, f"group {group.name!r} names an unknown head"
        assert target.config_file, f"group {group.name!r} head has no config_file"
        facts.append(
            _head_facts(cfg, target, group.name, group.head, len(group.workers),
                        head_layers=group.head_layers)
        )
    return facts


def _rollback_facts(network_config: Path) -> GroupFacts | None:
    """The frozen 122B rollback head, measured as if its group were restored.

    ``n_workers`` comes from the file's own ``tensor_split`` (7 entries,
    configs/aureax-shard-head.yaml:181), because no fleet file has a 7-worker
    group left to read it from — restoring one is exactly what a revert means,
    and that list is what ``validate_groups`` length-checks it against.

    Takes a fleet file because the registry lookup does: ``size_gb`` lives in
    ``model_registry`` and each fleet file carries its own.
    """
    path = CONFIGS_DIR / ROLLBACK_HEAD_CONFIG
    if not path.exists():
        return None

    cfg = load_network_config(network_config)
    raw = load_edge_config_raw(path)
    model = raw.get("model") or {}
    server = model.get("server") or {}
    split = server.get("tensor_split") or []
    assert split, (
        f"configs/{ROLLBACK_HEAD_CONFIG} has no tensor_split, so the worker count it "
        "ran on can no longer be recovered — the rollback ceilings below would be "
        "measured against a guess"
    )

    node = raw.get("node") or {}
    worker_wait = _group_wait(cfg)
    hardcoded = _as_int(node.get("worker_wait_timeout_seconds"))
    if hardcoded is not None:
        worker_wait = max(worker_wait, hardcoded)

    return GroupFacts(
        name="122B rollback",
        head_id="(rollback target)",
        head_config=ROLLBACK_HEAD_CONFIG,
        n_workers=len(split),
        model=_model_for(cfg, model.get("filename")),
        worker_wait=worker_wait,
        health_timeout=int(server.get("health_timeout_seconds", DEFAULT_HEALTH_TIMEOUT_SECONDS)),
        context_size=int(server.get("context_size", 0)),
        flash_attention=server.get("flash_attention"),
    )


def _require_groups(network_config: Path) -> list[GroupFacts]:
    groups = _groups_under_test(network_config)
    if not groups:
        pytest.skip("no node_groups configured")
    return groups


# --- R1-R5: worker memory ----------------------------------------------------


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_every_configured_group_fits_the_worker_memory_request(network_config) -> None:
    """R1. A worker's weight share must fit what the scheduler reserves for it.

    ``requests`` is what k3s subtracts from a node's allocatable memory when it
    places the pod. Set it below the share and the scheduler will happily place
    a second memory-hungry pod alongside a worker that is about to grow into
    9 GiB, on nodes with no swap.
    """
    requests_gib, _ = _shard_memory()
    for group in _require_groups(network_config):
        assert group.share_gib <= requests_gib, (
            f"group {group.name!r} splits {group.model.filename} ({group.model.size_gb} GB) "
            f"over {group.n_workers} workers = {group.share_gib:.2f} GiB each, but "
            f"k8s/daemonset-shard.yaml requests only {requests_gib:.2f} GiB. Raise "
            "requests.memory or add workers."
        )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_every_configured_group_fits_the_worker_memory_limit(network_config) -> None:
    """R2. Share plus the measured non-weight overhead must stay under the OOM ceiling.

    The overhead is not slack: KV, the recurrent state, the compute buffers and
    the rpc-server/supervisor processes are all resident while the head streams
    tensors in. Exceeding ``limits`` kills the worker during load, which the
    head reports as an unreachable peer.
    """
    _, limits_gib = _shard_memory()
    for group in _require_groups(network_config):
        needed = group.share_gib + PEAK_OVERHEAD_GIB
        assert needed <= limits_gib, (
            f"group {group.name!r} needs {group.share_gib:.2f} GiB of weights + "
            f"{PEAK_OVERHEAD_GIB} GiB of overhead = {needed:.2f} GiB per worker, but "
            f"k8s/daemonset-shard.yaml limits a worker to {limits_gib:.2f} GiB. The "
            "worker OOM-kills mid-load and the head reports it as an unreachable peer."
        )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_the_worker_memory_limit_is_still_the_flash_attention_tripwire(network_config) -> None:
    """R3. The limit must stay BELOW the flash-attention-off peak, on purpose.

    This one asserts the opposite direction from R2 and that is the point.
    ``flash_attention`` over RPC resolves to enabled through llama.cpp's AUTO
    default, so losing the setting is silent — except that without it llama.cpp
    reserves a [n_kv x n_ubatch x n_head_q] F32 score tensor. Raise ``limits``
    past that peak "for headroom" and a worker that has quietly lost flash
    attention fits, eats 4 GiB more than budgeted on a node with no swap, and
    takes the kubelet down with it instead of OOM-killing loudly at startup.
    """
    _, limits_gib = _shard_memory()
    for group in _require_groups(network_config):
        fa_off = group.share_gib + PEAK_OVERHEAD_GIB + FA_OFF_SCORE_TENSOR_GIB
        assert limits_gib < fa_off, (
            f"k8s/daemonset-shard.yaml allows {limits_gib:.2f} GiB per worker, which is "
            f"at or above the {fa_off:.2f} GiB group {group.name!r} would need with "
            f"flash attention off at context_size {group.context_size} "
            f"(configs/{group.head_config} currently sets flash_attention: "
            f"{group.flash_attention!r}). The limit is a tripwire, not headroom — "
            "lowering it back below that figure is what keeps the failure loud."
        )


def test_worker_memory_requests_do_not_exceed_limits() -> None:
    """R4. A request above the limit is rejected by the API server at apply time.

    Cheap to state, and it catches the common half-edit: raising ``requests``
    for a bigger model and forgetting ``limits`` two lines below.
    """
    requests_gib, limits_gib = _shard_memory()
    assert requests_gib <= limits_gib, (
        f"daemonset-shard.yaml requests {requests_gib:.2f} GiB but limits "
        f"{limits_gib:.2f} GiB — the API server rejects the DaemonSet outright."
    )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_the_worker_memory_limit_still_covers_the_122b_rollback_at_seven_workers(
    network_config,
) -> None:
    """R5. The shared manifest must keep covering the frozen rollback target.

    ``limits`` is one number for every worker in the fleet, including whichever
    ones a revert brings back. Tightening it to fit today's 35B would turn the
    one-line ``config_file`` flip back to the 122B into an OOM kill during
    load, discovered only after the revert.
    """
    rollback = _rollback_facts(network_config)
    if rollback is None:
        pytest.skip("no 122B rollback config in the tree")

    _, limits_gib = _shard_memory()
    needed = rollback.share_gib + ROLLBACK_OVERHEAD_GIB
    assert needed <= limits_gib, (
        f"reverting to configs/{rollback.head_config} means {rollback.model.size_gb} GB over "
        f"{rollback.n_workers} workers = {rollback.share_gib:.2f} GiB + "
        f"{ROLLBACK_OVERHEAD_GIB} GiB overhead = {needed:.2f} GiB per worker, above "
        f"daemonset-shard.yaml's {limits_gib:.2f} GiB limit. Do not tighten that limit "
        "to fit the current model."
    )


# --- R6-R9: readiness ceilings ----------------------------------------------


def test_every_edge_wait_ceiling_stays_under_the_edge_startup_probe_budget() -> None:
    """R6. Polling past the kubelet's own budget reports the wrong fault.

    Once the startupProbe budget in k8s/daemonset-edge.yaml expires the kubelet
    begins restarting the container. A script still polling at that point
    prints "MCP not responding" for a pod that is being killed and restarted —
    a different failure, reported as this one, with the real cause invisible.
    """
    budget = _probe_budget_seconds(_startup_probe())
    for path, name in ((UPDATE_ALL, "EDGE_WAIT"), (SETUP_ALL, "EDGE_WAIT_SECS")):
        for value in _script_assignments(path, name):
            assert value < budget, (
                f"scripts/{path.name} polls up to {name}={value}s, but the edge "
                f"startupProbe allows only {budget}s before the kubelet starts killing "
                "the container. Lower the ceiling or raise failureThreshold in "
                "k8s/daemonset-edge.yaml."
            )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_edge_wait_outlasts_a_cold_load_of_every_deployed_and_rollback_model(
    network_config,
) -> None:
    """R7. The health poll has to survive a genuine cold load, not a warm one.

    A fresh node or a wiped tensor cache makes a fully cold load, and a resize
    a partial one (the cache is content-addressed, so a worker misses only the
    tensors it never received); cold load is flat in peer count: the head
    ships the same total bytes either way. A ceiling below that streaming time
    declares a healthy fleet broken every time a node is re-imaged.
    """
    ceiling = max(_script_assignments(UPDATE_ALL, "EDGE_WAIT"))
    groups = list(_require_groups(network_config))
    rollback = _rollback_facts(network_config)
    if rollback is not None:
        groups.append(rollback)

    for group in groups:
        needed = COLD_LOAD_SECONDS_PER_GB * group.model.size_gb
        assert ceiling >= needed, (
            f"a cold load of {group.model.filename} ({group.model.size_gb} GB) takes about "
            f"{needed:.0f}s at {COLD_LOAD_SECONDS_PER_GB} s/GB, but scripts/update-all.sh "
            f"gives up after EDGE_WAIT={ceiling}s. Group {group.name!r} would be called "
            "dead while it is loading normally."
        )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_edge_wait_outlasts_every_deployed_heads_own_give_up_point(network_config) -> None:
    """R8. update-all.sh must outlive the head it is waiting for.

    A head abandons a load at ``worker_wait_timeout_seconds +
    health_timeout_seconds`` and exits, which is a real, diagnosable failure.
    If the script gives up first, that exit is never observed and the operator
    is told the node is merely slow.

    Deployed heads only — see the module docstring for why the rollback's
    give-up point is R9's business and not this test's.
    """
    ceiling = max(_script_assignments(UPDATE_ALL, "EDGE_WAIT"))
    for group in _require_groups(network_config):
        assert ceiling >= group.give_up_seconds, (
            f"the head of group {group.name!r} gives up at {group.worker_wait} + "
            f"{group.health_timeout} = {group.give_up_seconds}s (configs/"
            f"{group.head_config}), but scripts/update-all.sh stops polling at "
            f"EDGE_WAIT={ceiling}s and never sees it fail."
        )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_setup_all_fallback_covers_every_head_including_the_rollback(network_config) -> None:
    """R9. The hardcoded fallback runs when the derivation fails, so it must cover the worst head.

    setup-all.sh normally computes the budget from config; the literal is what
    it falls back to when that computation fails — precisely the situation in
    which nobody knows which head is starting. It therefore has to cover the
    longest give-up point any head in the tree could have, the frozen rollback
    included, while still staying under the kubelet's budget.

    The current values meet this with no slack at all (2700 >= 2700), which is
    why the relation is ``>=``: the fallback was chosen to equal the rollback's
    give-up point, not to exceed it.
    """
    ceiling = max(_script_assignments(SETUP_ALL, "EDGE_WAIT_SECS"))
    groups = _groups_under_test(network_config)
    rollback = _rollback_facts(network_config)
    if rollback is not None:
        groups.append(rollback)
    if not groups:
        pytest.skip("no heads configured and no rollback config in the tree")

    worst = max(groups, key=lambda g: g.give_up_seconds)
    assert ceiling >= worst.give_up_seconds, (
        f"the worst head is {worst.name!r} at {worst.worker_wait} + {worst.health_timeout} = "
        f"{worst.give_up_seconds}s (configs/{worst.head_config}), but scripts/setup-all.sh "
        f"falls back to EDGE_WAIT_SECS={ceiling}s and would report it dead mid-load."
    )

    budget = _probe_budget_seconds(_startup_probe())
    assert ceiling < budget, (
        f"scripts/setup-all.sh's EDGE_WAIT_SECS={ceiling}s fallback runs past the edge "
        f"startupProbe budget of {budget}s, so it would be polling a container the "
        "kubelet has already begun restarting."
    )


# --- R10-R11: the couplings that are not arithmetic --------------------------


@pytest.mark.parametrize("script", [UPDATE_ALL, SETUP_ALL], ids=lambda p: p.name)
def test_both_scripts_derive_the_worker_count_from_the_role_constant(script) -> None:
    """R10. WORKER_COUNT must key off $SHARD_WORKER_ROLE, never a pasted literal.

    Worker count is the divisor in every memory number above and the multiplier
    in setup-all.sh's serial rollout budget. Both scripts get it by counting
    roles, and both export the role from parse-config.py rather than spelling
    it — so a rename in orchestrator/config/schema.py travels. A pasted literal
    would silently count zero workers after that rename: no error, just a
    memory budget divided by nothing and a rollout wait sized for a fleet with
    no shards in it.
    """
    lines = script.read_text().splitlines()

    increments = [ln for ln in lines if "WORKER_COUNT + 1" in ln]
    assert increments, (
        f"scripts/{script.name} no longer increments WORKER_COUNT — if the counting "
        "moved, this check has to move with it"
    )
    for line in increments:
        assert "SHARD_WORKER_ROLE" in line, (
            f"scripts/{script.name} counts a worker on {line.strip()!r}, which does not "
            "test against $SHARD_WORKER_ROLE. parse-config.py exports that variable "
            "precisely so a role rename in orchestrator/config/schema.py reaches the "
            "shell instead of silently counting zero workers."
        )

    # Whole-line comments are exempt: the coupling is documented in prose in
    # both scripts, and naming the role there is how it stays readable.
    code = [ln for ln in lines if not ln.lstrip().startswith("#")]
    stale = [ln.strip() for ln in code if SHARD_WORKER_ROLE in ln]
    assert not stale, (
        f"scripts/{script.name} spells the role literal {SHARD_WORKER_ROLE!r} in code: "
        f"{stale!r}. Use $SHARD_WORKER_ROLE, which parse-config.py exports from "
        "orchestrator/config/schema.py."
    )


@pytest.mark.parametrize("network_config", NETWORK_CONFIGS)
def test_groups_sharing_a_worker_config_file_have_the_same_shape(network_config) -> None:
    """R11. One worker template may serve several groups only while they are identical.

    Templates are shared when targets differ only by placeholder-resolved
    values. A worker template carries the model share's real consequences —
    cache path, thread counts, the tensor_split its head length-checks — so two
    groups sharing one file while running different models or different worker
    counts means one of them is configured for the other's shape, with nothing
    to say so.
    """
    cfg = load_network_config(network_config)
    if not cfg.node_groups:
        pytest.skip("no node_groups configured")
    by_id = {t.node_id: t for t in cfg.ssh.targets}

    sharers: dict[str, list[tuple[str, int, str]]] = {}
    for group in cfg.node_groups:
        head_raw = load_edge_config_raw(CONFIGS_DIR / by_id[group.head].config_file)
        head_model = (head_raw.get("model") or {}).get("filename")
        for worker in group.workers:
            config_file = by_id[worker].config_file
            assert config_file, f"worker {worker!r} in group {group.name!r} has no config_file"
            entry = (group.name, len(group.workers), head_model)
            if entry not in sharers.setdefault(config_file, []):
                sharers[config_file].append(entry)

    for config_file, entries in sharers.items():
        shapes = {(n_workers, model) for _, n_workers, model in entries}
        assert len(shapes) == 1, (
            f"configs/{config_file} is shared by groups with different shapes: "
            f"{sorted(entries)}. Give one of them its own worker config — a shared "
            "template cannot carry two models or two worker counts."
        )


# --- G6: the one literal the image cannot avoid ------------------------------


def test_start_sh_greps_the_shard_worker_role_constant() -> None:
    """docker/scripts/start.sh spells the role literal by hand, so pin it here.

    It cannot source scripts/lib/common.sh (that file is not in the image) and
    it runs before Python, so the literal is unavoidable — the coupling is
    documented at start.sh:39-49. What is avoidable is it going stale: rename
    the role in orchestrator/config/schema.py and every worker pod opens its
    `kubectl logs` with a WARNING about a missing model it is correctly not
    supposed to have.
    """
    text = START_SH.read_text()
    line = next(
        (ln for ln in text.splitlines() if "grep -qE" in ln and "role:" in ln), None
    )
    assert line, (
        "docker/scripts/start.sh no longer greps the node role — the coupling moved "
        "and this check has to move with it"
    )
    assert SHARD_WORKER_ROLE in line, (
        f"docker/scripts/start.sh greps {line.strip()!r}, which does not contain "
        f"SHARD_WORKER_ROLE ({SHARD_WORKER_ROLE!r}) from orchestrator/config/schema.py:29. "
        "Every shard worker pod would open its logs with a WARNING about a missing GGUF "
        "it is correctly not supposed to have."
    )


# --- The parsing this module's arithmetic rests on ---------------------------


@pytest.mark.parametrize(
    ("quantity", "expected_gib"),
    [
        ("11Gi", 11.0),
        ("8Gi", 8.0),
        ("512Mi", 0.5),
        # Decimal suffixes are NOT the binary ones: 4G is 3.725 GiB. A worker
        # limit written "11G" instead of "11Gi" is 0.75 GiB smaller than the
        # author meant, which is most of the margin R2 checks.
        ("4G", 4 * 10**9 / GIB),
        ("1024Mi", 1.0),
        ("2048", 2048 / GIB),
    ],
)
def test_memory_quantity_parsing(quantity, expected_gib) -> None:
    assert _quantity_gib(quantity) == pytest.approx(expected_gib)


def test_an_unparseable_memory_quantity_is_an_error_not_a_zero() -> None:
    """A silent 0 GiB would make every memory relation above pass trivially."""
    with pytest.raises(ValueError, match="not a Kubernetes quantity"):
        _quantity_gib("11 gigabytes")
