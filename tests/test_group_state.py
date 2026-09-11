"""The persisted worker-order baseline (``orchestrator/config/group_state.py``).

Every test here exists because the failure it pins is invisible at runtime: a
reordered ``workers:`` list is valid YAML, generates clean configs, loads,
answers ``/health`` and serves — with ``tensor_split`` addressing different
machines than the operator wrote it for. The baseline file is the only thing
in the system that remembers what the order was last time, so its verdicts,
its keying and its failure modes are all load-bearing.

Fixtures start from the tracked ``configs/aureax-network.example.yaml`` and
mutate deep copies, so the shapes under test are the real fleet's (two groups,
three workers each, a head and a planner) rather than a hand-built config that
might dodge a validator by accident.

``tmp_aureax_home`` is on every test that touches the store: without it these
would read and write the developer's own ``~/.aureax/state``.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from orchestrator.config import group_state
from orchestrator.config.loader import load_network_config
from orchestrator.config.schema import NetworkConfig
from tests.conftest import EXAMPLE_NETWORK_CONFIG

GROUP_A = "qwen35b-a"


def _example() -> NetworkConfig:
    return load_network_config(EXAMPLE_NETWORK_CONFIG)


def _with_workers(cfg: NetworkConfig, workers: list[str], group: str = GROUP_A) -> NetworkConfig:
    """A deep copy of ``cfg`` whose named group has exactly ``workers``."""
    out = cfg.model_copy(deep=True)
    for g in out.node_groups:
        if g.name == group:
            g.workers = workers
    return out


def _with_host(cfg: NetworkConfig, node_id: str, host: str) -> NetworkConfig:
    out = cfg.model_copy(deep=True)
    for t in out.ssh.targets:
        if t.node_id == node_id:
            t.host = host
    return out


def _with_head(cfg: NetworkConfig, head: str, group: str = GROUP_A) -> NetworkConfig:
    out = cfg.model_copy(deep=True)
    for g in out.node_groups:
        if g.name == group:
            g.head = head
    return out


def _with_rpc_port(cfg: NetworkConfig, port: int, group: str = GROUP_A) -> NetworkConfig:
    out = cfg.model_copy(deep=True)
    for g in out.node_groups:
        if g.name == group:
            g.rpc_port = port
    return out


def _without_group(cfg: NetworkConfig, name: str) -> NetworkConfig:
    out = cfg.model_copy(deep=True)
    out.node_groups = [g for g in out.node_groups if g.name != name]
    return out


@pytest.fixture
def net_path(tmp_path: Path) -> Path:
    """A stand-in fleet-file path. The file itself never has to exist — the
    baseline is keyed by the path, and the config object is passed separately."""
    return tmp_path / "aureax-network.yaml"


# --- first run ---------------------------------------------------------------


def test_a_first_run_reports_nothing_and_writes_no_file(tmp_aureax_home, net_path) -> None:
    """A fresh machine must generate. There is nothing to have departed from."""
    cfg = _example()
    assert group_state.order_problems(net_path, cfg) == ([], [])
    assert group_state.baseline_differs(net_path, cfg) is False
    assert not group_state.baseline_path().exists()


def test_order_problems_never_writes_even_after_a_baseline_exists(
    tmp_aureax_home, net_path
) -> None:
    """Recording is the caller's decision, taken only once every config is written."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)
    before = group_state.baseline_path().read_bytes()

    group_state.order_problems(net_path, _with_workers(cfg, ["shard-a3", "shard-a1", "shard-a2"]))

    assert group_state.baseline_path().read_bytes() == before


def test_write_baseline_records_every_group_with_head_port_and_hosts(
    tmp_aureax_home, net_path
) -> None:
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    data = json.loads(group_state.baseline_path().read_text())
    record = data[str(net_path.resolve())]
    assert set(record["groups"]) == {"qwen35b-a", "qwen35b-b"}
    assert record["groups"][GROUP_A] == {
        "head": "shard-head",
        "rpc_port": 50052,
        # The host travels with the id: a worker that keeps its slot and moves
        # to another machine is a peer change the id alone cannot show.
        "workers": [
            ["shard-a1", "192.0.2.11"],
            ["shard-a2", "192.0.2.12"],
            ["shard-a3", "192.0.2.13"],
        ],
    }
    assert record["recorded_at"]


# --- the verdicts ------------------------------------------------------------


def test_a_swap_is_one_fatal_problem(tmp_aureax_home, net_path) -> None:
    """The whole reason the file exists: swapping two entries renames the RPC
    devices, so tensor_split silently addresses other machines."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3"])
    )

    assert warnings == []
    assert len(problems) == 1
    msg = problems[0]
    assert "'qwen35b-a'" in msg
    assert "REORDERED" in msg
    assert "was: shard-a1, shard-a2, shard-a3" in msg
    assert "now: shard-a2, shard-a1, shard-a3" in msg
    assert "RPC0..RPC2" in msg
    assert "tensor_split" in msg
    # The two that swapped are named with both slots; the one that did not move
    # is not, or the operator cannot tell which tensor_split entries moved.
    assert "moved: shard-a1 RPC0 -> RPC1, shard-a2 RPC1 -> RPC0" in msg
    assert "shard-a3 RPC" not in msg
    # The remedy has to be pasteable, and it has to be the flag the CLI and the
    # dashboard's regenerate route actually accept.
    assert "aureax generate --accept-worker-reorder" in msg


def test_a_reorder_in_one_group_does_not_implicate_the_other(tmp_aureax_home, net_path) -> None:
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, _ = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3"])
    )

    assert len(problems) == 1
    assert "qwen35b-b" not in problems[0]


def test_appending_a_worker_is_a_warning_not_a_problem(tmp_aureax_home, net_path) -> None:
    """Resizing a group is the documented operation; it costs a cold load.

    A pure append is the shape the fatal rule has to keep letting through: every
    worker that was already there keeps its slot, so no tensor_split entry moves.
    """
    cfg = _example()
    group_state.write_baseline(net_path, cfg)
    # shard-b1 already resolves to a host; reusing it keeps the fixture honest
    # about what a fourth worker looks like without inventing an ssh target.
    grown = _with_workers(cfg, ["shard-a1", "shard-a2", "shard-a3", "shard-b1"])

    problems, warnings = group_state.order_problems(net_path, grown)

    assert problems == []
    assert len(warnings) == 1
    assert "workers changed from 3 to 4" in warnings[0]
    assert "added shard-b1" in warnings[0]
    assert "COLD load" in warnings[0]


def test_removing_the_LAST_worker_is_a_warning_not_a_problem(tmp_aureax_home, net_path) -> None:
    """Dropping the tail leaves every remaining worker on the slot it had, so
    tensor_split still addresses the machines it was written for (one entry too
    many, which validate_groups catches). Contrast the middle removal below."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a1", "shard-a2"])
    )

    assert problems == []
    assert len(warnings) == 1
    assert "workers changed from 3 to 2" in warnings[0]
    assert "removed shard-a3" in warnings[0]


def test_replacing_a_worker_IN_PLACE_is_a_warning(tmp_aureax_home, net_path) -> None:
    """The third shape in the warning bucket, and the one that reads like
    neither of the other two: the list is the same length, so only the slot's
    OCCUPANT changed. Every retained worker keeps its slot, so no tensor_split
    entry moves — the new machine simply inherits the share the old one had at
    that index, which is what swapping a dead node out means."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a1", "shard-b1", "shard-a3"])
    )

    assert problems == []
    assert len(warnings) == 1
    assert "workers changed from 3 to 3" in warnings[0]
    assert "added shard-b1" in warnings[0] and "removed shard-a2" in warnings[0]
    assert "COLD load" in warnings[0]


def test_promoting_a_worker_to_head_is_a_warning_not_a_reorder(
    tmp_aureax_home, net_path
) -> None:
    """The documented 122B rollback manoeuvre, seen from the baseline: the head
    is demoted into the slot the promoted worker vacated. Same shape as the
    replacement above — RPC0 changes machines, every other slot is untouched —
    so it is a cold load, not a corrupted split."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)
    head = next(g.head for g in cfg.node_groups if g.name == GROUP_A)
    flipped = _with_head(
        _with_workers(cfg, [head, "shard-a2", "shard-a3"]), "shard-a1"
    )

    problems, warnings = group_state.order_problems(net_path, flipped)

    assert problems == []
    assert len(warnings) == 1
    assert f"added {head}" in warnings[0] and "removed shard-a1" in warnings[0]


def test_inserting_a_worker_in_the_MIDDLE_is_fatal(tmp_aureax_home, net_path) -> None:
    """The gap this rule closes. The worker SET changed (shard-b1 is new), so a
    set comparison files this under "appended" — but every worker after the
    insertion point is renamed, exactly as a swap renames them, and
    tensor_split[1] now addresses shard-b1 instead of shard-a2."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a1", "shard-b1", "shard-a2", "shard-a3"])
    )

    assert warnings == []
    assert len(problems) == 1
    msg = problems[0]
    assert "REORDERED" in msg
    assert "was: shard-a1, shard-a2, shard-a3" in msg
    assert "now: shard-a1, shard-b1, shard-a2, shard-a3" in msg
    assert "moved: shard-a2 RPC1 -> RPC2, shard-a3 RPC2 -> RPC3" in msg
    # The device range follows the NEW list, and the remedy is the same flag.
    assert "RPC0..RPC3" in msg
    assert "aureax generate --accept-worker-reorder" in msg
    # A resize warning as well would be two verdicts for one edit; the fatal one
    # is the one that has to be read.
    assert "COLD load" not in msg


def test_removing_a_worker_from_the_MIDDLE_is_fatal(tmp_aureax_home, net_path) -> None:
    """Same shape as the insert: the set changed, and every worker after the hole
    slid down a slot, so tensor_split[1] now addresses shard-a3."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a1", "shard-a3"])
    )

    assert warnings == []
    assert len(problems) == 1
    msg = problems[0]
    assert "REORDERED" in msg
    assert "now: shard-a1, shard-a3" in msg
    assert "moved: shard-a3 RPC2 -> RPC1" in msg
    # Only the worker that survived AND moved is named: shard-a2 is gone, so it
    # has no new slot to report.
    assert "shard-a2 RPC" not in msg


def test_a_reorder_that_also_resizes_is_fatal(tmp_aureax_home, net_path) -> None:
    """Belt and braces on the same rule: appending is only forgiven while the
    workers that stay keep their slots."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3", "shard-b1"])
    )

    assert warnings == []
    assert len(problems) == 1
    assert "moved: shard-a1 RPC0 -> RPC1, shard-a2 RPC1 -> RPC0" in problems[0]


def test_a_workers_host_change_is_a_warning(tmp_aureax_home, net_path) -> None:
    """Same slot, different machine — the --rpc entry moved even though the
    node id did not."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_host(cfg, "shard-a2", "192.0.2.42")
    )

    assert problems == []
    assert len(warnings) == 1
    assert "'shard-a2'" in warnings[0]
    assert "RPC1 slot" in warnings[0]
    assert "192.0.2.12" in warnings[0] and "192.0.2.42" in warnings[0]


def test_an_unchanged_config_reports_nothing(tmp_aureax_home, net_path) -> None:
    cfg = _example()
    group_state.write_baseline(net_path, cfg)
    assert group_state.order_problems(net_path, cfg) == ([], [])


def test_a_group_absent_from_the_baseline_is_silent(tmp_aureax_home, net_path) -> None:
    """A group added since the last generate has no order to have departed from,
    while the group that WAS recorded is still checked in the same pass."""
    cfg = _example()
    only_a = cfg.model_copy(deep=True)
    only_a.node_groups = [g for g in only_a.node_groups if g.name == GROUP_A]
    group_state.write_baseline(net_path, only_a)

    reordered_a = _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3"])
    problems, warnings = group_state.order_problems(net_path, reordered_a)

    assert warnings == []
    assert len(problems) == 1
    assert "'qwen35b-a'" in problems[0]


def test_a_rebaseline_accepts_the_new_order(tmp_aureax_home, net_path) -> None:
    """What ``--accept-worker-reorder`` buys: the caller writes the new order
    and the next run is quiet again."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)
    reordered = _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3"])
    assert group_state.order_problems(net_path, reordered)[0]

    group_state.write_baseline(net_path, reordered)

    assert group_state.order_problems(net_path, reordered) == ([], [])


# --- keying by fleet file ----------------------------------------------------


def test_the_baseline_is_keyed_by_config_path(tmp_aureax_home, tmp_path) -> None:
    """`aureax -c configs/aureax-network.example.yaml generate` runs on every
    fresh clone and in test-all.sh. It must not overwrite the real fleet's
    record and then accuse the next real generate of a reorder."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    cfg = _example()
    reordered = _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3"])

    group_state.write_baseline(a, cfg)
    # b has never been generated from, so its own reordered config is silent...
    assert group_state.order_problems(b, reordered) == ([], [])
    group_state.write_baseline(b, reordered)
    # ...and recording b leaves a's record intact and still enforcing.
    assert group_state.order_problems(a, reordered)[0]
    assert group_state.order_problems(a, cfg) == ([], [])

    data = json.loads(group_state.baseline_path().read_text())
    assert set(data) == {str(a.resolve()), str(b.resolve())}


def test_a_relative_path_hits_the_same_record(tmp_aureax_home, tmp_path, monkeypatch) -> None:
    """resolve(), not absolute(): the same fleet file reached relatively must
    not get a second baseline that never sees the reorder."""
    cfg = _example()
    group_state.write_baseline(tmp_path / "aureax-network.yaml", cfg)

    monkeypatch.chdir(tmp_path)
    problems, _ = group_state.order_problems(
        Path("aureax-network.yaml"), _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3"])
    )

    assert len(problems) == 1


# --- degrading, not raising --------------------------------------------------


def test_an_unreadable_baseline_reports_nothing_and_is_preserved(
    tmp_aureax_home, net_path, caplog
) -> None:
    """Corrupt state must not abort a deploy, and must not be destroyed by the
    read path — order_problems never writes, so the file survives for recovery."""
    path = group_state.baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    with caplog.at_level("WARNING", logger="aureax.config.group_state"):
        problems, warnings = group_state.order_problems(net_path, _example())

    assert (problems, warnings) == ([], [])
    assert group_state.baseline_differs(net_path, _example()) is False
    assert path.read_text() == "{not json"
    assert "could not be read" in caplog.text


def test_a_baseline_that_is_not_an_object_reports_nothing(tmp_aureax_home, net_path) -> None:
    path = group_state.baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]")

    assert group_state.order_problems(net_path, _example()) == ([], [])


def test_a_hand_broken_group_record_reports_nothing(tmp_aureax_home, net_path) -> None:
    """A workers list that is not [[id, host], ...] degrades to "no baseline for
    this group" rather than raising inside a generate."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)
    path = group_state.baseline_path()
    data = json.loads(path.read_text())
    data[str(net_path.resolve())]["groups"][GROUP_A]["workers"] = ["shard-a1", "shard-a2"]
    path.write_text(json.dumps(data))

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a2", "shard-a1", "shard-a3"])
    )

    assert (problems, warnings) == ([], [])


def test_a_duplicated_worker_id_warns_instead_of_raising(tmp_aureax_home, net_path) -> None:
    """The slot comparison is over first occurrences, and the resize branch is
    keyed on list inequality rather than set inequality, precisely so a list that
    names one worker twice cannot reach _host_warnings' zip(strict=True) with
    mismatched lengths and raise inside a generate."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    problems, warnings = group_state.order_problems(
        net_path, _with_workers(cfg, ["shard-a1", "shard-a2", "shard-a3", "shard-a3"])
    )

    assert problems == []
    assert len(warnings) == 1
    assert "workers changed from 3 to 4" in warnings[0]
    assert "listed twice" in warnings[0]


def test_write_baseline_replaces_an_unreadable_file(tmp_aureax_home, net_path, caplog) -> None:
    """Keeping a corrupt file would disable the check permanently. Replacing it
    costs at most one other fleet file's record, which its next generate rewrites."""
    path = group_state.baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    with caplog.at_level("WARNING", logger="aureax.config.group_state"):
        group_state.write_baseline(net_path, _example())

    assert json.loads(path.read_text())[str(net_path.resolve())]["groups"]
    assert "replacing unreadable" in caplog.text


# --- on-disk posture ---------------------------------------------------------


def test_the_baseline_is_0600_in_a_0700_directory(tmp_aureax_home, net_path) -> None:
    """It sits beside ~/.aureax/keys and ~/.aureax/tls and keeps their posture,
    so a widened directory cannot start listing which fleets this Mac drives."""
    group_state.write_baseline(net_path, _example())

    assert stat.S_IMODE(group_state.baseline_path().stat().st_mode) == 0o600
    assert stat.S_IMODE(group_state.state_dir().stat().st_mode) == 0o700


def test_a_pre_existing_wide_state_directory_is_tightened(tmp_aureax_home, net_path) -> None:
    """mkdir's mode only applies when mkdir CREATES the leaf, so a directory
    restored from a backup or made by hand keeps its 0755 without this."""
    d = group_state.state_dir()
    d.mkdir(parents=True, mode=0o755)

    group_state.write_baseline(net_path, _example())

    assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_the_store_follows_home(tmp_aureax_home) -> None:
    """A Path bound at import time would write into the developer's real
    ~/.aureax — every test in this file depends on this staying a function."""
    assert group_state.state_dir() == tmp_aureax_home / ".aureax" / "state"
    assert group_state.baseline_path().name == "node_groups.json"


# --- baseline_differs --------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "mutate", "expected"),
    [
        ("unchanged", lambda c: c, False),
        ("reordered", lambda c: _with_workers(c, ["shard-a2", "shard-a1", "shard-a3"]), True),
        (
            "appended",
            lambda c: _with_workers(c, ["shard-a1", "shard-a2", "shard-a3", "shard-b1"]),
            True,
        ),
        ("removed", lambda c: _with_workers(c, ["shard-a1", "shard-a2"]), True),
        ("host changed", lambda c: _with_host(c, "shard-a2", "192.0.2.42"), True),
        ("head changed", lambda c: _with_head(c, "shard-head-b"), True),
        ("rpc port changed", lambda c: _with_rpc_port(c, 50053), True),
        ("group dropped", lambda c: _without_group(c, "qwen35b-b"), True),
    ],
)
def test_baseline_differs_truth_table(tmp_aureax_home, net_path, label, mutate, expected) -> None:
    """What ``aureax start`` warns on: the nodes are running configs generated
    from a different set of groups than this file now describes. Broader than
    :func:`order_problems` on purpose — a changed head or rpc_port gets no
    verdict of its own there, but it does make the deployed configs stale."""
    cfg = _example()
    group_state.write_baseline(net_path, cfg)

    assert group_state.baseline_differs(net_path, mutate(cfg)) is expected


def test_baseline_differs_is_false_without_a_baseline(tmp_aureax_home, net_path) -> None:
    """Otherwise a machine that has never generated warns on every single start,
    which trains the reader to ignore the line that matters."""
    assert group_state.baseline_differs(net_path, _example()) is False
