"""The last-generated worker order for every node group, remembered on disk.

Worker order inside a ``node_groups`` entry IS the ``--rpc`` order, which is
what names the devices ``RPC0, RPC1, ...`` that ``model.server.devices`` and
``model.server.tensor_split`` index POSITIONALLY. Swapping two entries — or
inserting or removing one in the MIDDLE, which shifts every worker after it —
therefore repoints those splits at different machines, and nothing at runtime
notices: the head loads, ``/health`` answers, the pods stay Ready, and the
group simply serves with its weights on the wrong peers.

The config file alone cannot catch that, because a reordered list is a
perfectly valid list. Catching it needs to know what the order was LAST time
configs were generated, so this module keeps a baseline at
``~/.aureax/state/node_groups.json`` and compares against it.

Two properties worth stating, because both are load-bearing:

* **Keyed by the resolved absolute path of the network config.** Running
  ``aureax -c configs/aureax-network.example.yaml generate`` (which every
  fresh clone and ``scripts/test-all.sh`` does) must not overwrite the real
  fleet's baseline and then accuse the next real generate of a reorder.
* **Nothing here ever raises.** It runs inside ``aureax generate`` and inside
  ``aureax start``; a corrupt or unreadable JSON file must degrade to "no
  baseline, report nothing", never abort a deploy. An unreadable file is
  logged and left exactly where it is — :func:`order_problems` never writes.

Writing is deliberately a separate call (:func:`write_baseline`), so the
caller records an order only after every config was actually written. A
generate that raised halfway must not leave a baseline for an order it never
shipped, or the retry would look like a legitimate second run.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.config.atomic import write_private_file

logger = logging.getLogger("aureax.config.group_state")

# Not a module-level Path: tests redirect $HOME (tests/conftest.py's
# tmp_aureax_home) and a path bound at import time would write into the
# developer's real ~/.aureax. Same reason as keystore.store_dir().
_STATE_DIRNAME = "state"
_BASELINE_FILENAME = "node_groups.json"


def state_dir() -> Path:
    return Path.home() / ".aureax" / _STATE_DIRNAME


def baseline_path() -> Path:
    return state_dir() / _BASELINE_FILENAME


def _read_baseline() -> tuple[dict, bool]:
    """The whole baseline file, plus whether it was READABLE.

    A missing file is readable-and-empty: nothing has been generated from this
    machine yet, which is the normal state on a fresh clone. A file that will
    not parse is neither — it holds an order this process cannot see, so every
    caller must behave as though there were no baseline rather than treat the
    current order as the first one ever recorded.

    The failure is logged rather than swallowed: without a line, a corrupt
    baseline shows up as a reorder check that silently never fires again, with
    nothing anywhere saying why. Mirrors
    ``orchestrator/dashboard/access_service.py::_load_journal``.
    """
    path = baseline_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, True
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        logger.exception("node_groups baseline %s could not be read; reporting it empty", path)
        return {}, False
    if not isinstance(data, dict):
        logger.warning("node_groups baseline %s is not a JSON object; reporting it empty", path)
        return {}, False
    return data, True


def _key(network_config_path: Path) -> str:
    """The baseline's key for one fleet file: its resolved absolute path.

    ``resolve()`` and not ``absolute()``: ``configs/aureax-network.yaml``
    reached through a symlinked checkout or a relative ``--config`` must land
    on the same record, otherwise the same fleet gets two baselines and
    whichever one is younger never sees the reorder.
    """
    return str(Path(network_config_path).resolve())


def _snapshot(network_config) -> dict[str, dict]:
    """The current groups in baseline shape: ``{name: {head, rpc_port, workers}}``.

    The host travels beside the node id because a worker that keeps its slot
    but changes ``host:`` also moves its ``--rpc`` entry to a different
    machine — not a reorder, but not nothing either (see the warning in
    :func:`order_problems`).
    """
    hosts = network_config.host_by_node_id()
    return {
        group.name: {
            "head": group.head,
            "rpc_port": group.rpc_port,
            # .get with a default rather than []: a worker with no ssh.targets
            # entry is already rejected by NetworkConfig._check_node_groups, so
            # this can only be reached by a caller passing something unvalidated
            # — and this module must not be the thing that raises there.
            "workers": [[w, hosts.get(w, "")] for w in group.workers],
        }
        for group in network_config.node_groups
    }


def _recorded_groups(network_config_path: Path) -> dict[str, dict] | None:
    """The recorded groups for one fleet file, or None when there is no usable record.

    None covers all three "say nothing" cases at once — no file, an unreadable
    file, and a record whose shape a hand-edit broke — so every caller has a
    single branch for "there is nothing to compare against".
    """
    data, readable = _read_baseline()
    if not readable:
        return None
    record = data.get(_key(network_config_path))
    if not isinstance(record, dict):
        return None
    groups = record.get("groups")
    if not isinstance(groups, dict):
        logger.warning(
            "node_groups baseline %s has no usable 'groups' record for %s; "
            "treating this as a first generate",
            baseline_path(),
            _key(network_config_path),
        )
        return None
    return groups


def _workers(entry: object) -> list[list[str]] | None:
    """A record's ``workers`` as ``[[node_id, host], ...]``, or None if malformed.

    JSON has no tuples, so a round-tripped pair comes back as a two-item list.
    Anything else is a hand-edit; it degrades to "no baseline for this group"
    rather than raising inside a generate.
    """
    if not isinstance(entry, dict):
        return None
    workers = entry.get("workers")
    if not isinstance(workers, list):
        return None
    pairs: list[list[str]] = []
    for pair in workers:
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            return None
        pairs.append([str(pair[0]), str(pair[1])])
    return pairs


def order_problems(network_config_path: Path, network_config) -> tuple[list[str], list[str]]:
    """``(fatal problems, warnings)`` from comparing the config to its baseline.

    Never writes and never raises. The caller decides what a problem means:
    ``generate`` refuses unless ``--accept-worker-reorder`` was passed, and
    logs each warning at WARNING level (they are returned rather than logged
    here so one message cannot appear twice on one run).

    The verdicts, and why each is the severity it is:

    * No baseline for this fleet file, or a group that is new to it — silent.
      A first generate has nothing to compare against, and refusing there
      would mean a fresh clone could not generate at all.
    * A worker present in BOTH lists changed slot — fatal. This is the case
      that corrupts a ``tensor_split`` silently, and it is wider than a
      same-set reorder: a worker inserted or removed in the MIDDLE renames
      RPC1..RPCn for every worker after it exactly as a swap does, so it lands
      here rather than under "resized". Fatal even when the split is uniform
      (``[1, 1, 1]``, ``configs/aureax-shard-head-35b.yaml``): that file's own
      comment says order-independence holds only while every entry is equal,
      which is one edit away, and any of these invalidates every worker's
      tensor cache regardless. One verdict per group — a middle insert is
      reported as the reorder it is, not as the resize it also is.
    * Every RETAINED worker kept its slot, but the list is not the same —
      warning. Three shapes reach this: an append, a removal from the tail, and
      a same-length REPLACEMENT (``[a1, a2, a3]`` -> ``[a1, a9, a3]``, which is
      also what promoting a worker to head and demoting the old head back into
      its slot looks like from here — the documented 122B rollback). All three
      cost a cold load, not a wrong answer: no retained worker's ``--rpc`` slot
      moved, so every ``tensor_split`` entry still carries the share it was
      written for, and a replaced slot hands that share to the machine that
      took the old one's place.
    * A worker kept its slot but changed host — warning, same reasoning.

    A change of ``head`` or ``rpc_port`` gets no verdict of its own: both are
    checked against the per-node templates by ``templating.validate_groups``,
    and both make :func:`baseline_differs` true, which is what warns at start
    that the nodes are running configs older than this file.
    """
    recorded = _recorded_groups(network_config_path)
    if recorded is None:
        return [], []

    problems: list[str] = []
    warnings: list[str] = []
    for name, current in _snapshot(network_config).items():
        was = _workers(recorded.get(name))
        if was is None:
            # New group, or a record this module could not read back. Either
            # way there is no order to have departed from.
            continue
        now = current["workers"]
        was_ids = [w[0] for w in was]
        now_ids = [w[0] for w in now]

        moved = _shifted_workers(was_ids, now_ids)
        if moved:
            problems.append(_reorder_problem(name, was_ids, now_ids, moved))
            continue
        if was_ids != now_ids:
            # Every retained worker kept its slot: an append, a removal from
            # the tail, or a replacement in place. Compared as LISTS and not as
            # sets, so the only shape that reaches _host_warnings is two
            # identical id lists — a hand-edited baseline with a duplicated id
            # would otherwise arrive at its zip(strict=True) with mismatched
            # lengths and raise inside a generate, which this module must never
            # do.
            warnings.append(_membership_warning(name, was_ids, now_ids))
            continue
        warnings.extend(_host_warnings(name, was, now))

    return problems, warnings


def _shifted_workers(was_ids: list[str], now_ids: list[str]) -> list[tuple[str, int, int]]:
    """Workers present in BOTH lists whose slot moved, as ``(id, was_slot, now_slot)``.

    This, and not "the worker set is equal and the list differs", is the fatal
    test. A worker inserted or removed in the MIDDLE changes the set, so a set
    comparison files it under append/remove — while it renames RPC1..RPCn for
    every worker after it, which is exactly the corruption a swap causes:
    ``[a1, a2, a3]`` -> ``[a1, a4, a2, a3]`` leaves ``tensor_split[1]``
    addressing a4 and every later entry one machine along, silently. Keying on
    retained slots puts those edits with the swap and leaves as warnings the
    shapes where every retained worker keeps its slot: an append, a removal
    from the tail, and — same length, so it reads like neither — a slot whose
    occupant was REPLACED (``[a1, a2, a3]`` -> ``[a1, a9, a3]``, the shape a
    head/worker swap also takes). The replacement is a warning deliberately:
    nothing retained moved, so a9 inherits exactly the ``tensor_split`` share
    a2 had, which is what the operator swapping a dead machine out means.

    First occurrence wins on both sides. A duplicated node id cannot come from
    a validated fleet file, only from a hand-edited baseline, and this module
    reports what it finds rather than raising on it.
    """
    was_slots = _first_slots(was_ids)
    now_slots = _first_slots(now_ids)
    return [
        (worker, slot, now_slots[worker])
        for worker, slot in was_slots.items()
        if worker in now_slots and now_slots[worker] != slot
    ]


def _first_slots(ids: list[str]) -> dict[str, int]:
    """``{node_id: first slot}``, in list order — dicts keep insertion order, so
    the caller reports moved workers oldest-slot-first."""
    slots: dict[str, int] = {}
    for slot, worker in enumerate(ids):
        slots.setdefault(worker, slot)
    return slots


def _reorder_problem(
    name: str, was_ids: list[str], now_ids: list[str], moved: list[tuple[str, int, int]]
) -> str:
    """The refusal. Names the group and every worker whose --rpc slot moved,
    shows was/now, and ends in a pasteable flag."""
    last_device = max(len(now_ids) - 1, 0)
    moved_text = ", ".join(f"{worker} RPC{old} -> RPC{new}" for worker, old, new in moved)
    return (
        f"group {name!r}: the workers list has been REORDERED, not appended to.\n"
        f"    was: {', '.join(was_ids)}\n"
        f"    now: {', '.join(now_ids)}\n"
        f"    moved: {moved_text}\n"
        f"    Worker order IS the --rpc order, which defines the device names "
        f"RPC0..RPC{last_device};\n"
        "    model.server.tensor_split and model.server.devices index them POSITIONALLY,\n"
        "    so those entries now address different machines. Append new workers instead:\n"
        "    only a worker added or removed at the END of the list leaves every other\n"
        "    worker's slot alone.\n"
        "    If this reorder is deliberate, re-run with\n"
        "    `aureax generate --accept-worker-reorder` (also accepted by `aureax deploy`\n"
        "    and `aureax apply`), which records the new order as the new baseline."
    )


def _membership_warning(name: str, was_ids: list[str], now_ids: list[str]) -> str:
    added = [w for w in now_ids if w not in set(was_ids)]
    removed = [w for w in was_ids if w not in set(now_ids)]
    changes = (
        "; ".join(
            part
            for part in (
                f"added {', '.join(added)}" if added else "",
                f"removed {', '.join(removed)}" if removed else "",
            )
            if part
        )
        # Reachable only from a hand-edited baseline: no id came or went, yet
        # the list length changed, so an id appears twice. Say that rather than
        # print empty parentheses.
        or "the same ids, one of them listed twice"
    )
    return (
        f"group {name!r}: workers changed from {len(was_ids)} to {len(now_ids)} "
        f"({changes}) — every worker's tensor cache is invalidated, so the next "
        "start is a COLD load: the head streams every tensor over the LAN again. "
        "Minutes, not seconds. Not an error."
    )


def _host_warnings(name: str, was: list[list[str]], now: list[list[str]]) -> list[str]:
    """One warning per worker that kept its slot and changed address."""
    return [
        (
            f"group {name!r}: worker {new_id!r} kept its RPC{slot} slot but its host "
            f"changed from {old_host} to {new_host} — that --rpc entry now points at a "
            "different machine, so the next start is a COLD load for the whole group "
            "(the new peer has no tensor cache) and the old one is left holding a stale "
            "share of the weights. Not an error."
        )
        for slot, ((_, old_host), (new_id, new_host)) in enumerate(zip(was, now, strict=True))
        if old_host != new_host
    ]


def baseline_differs(network_config_path: Path, network_config) -> bool:
    """Whether this fleet file's groups have changed since the last generate.

    False when there is no baseline. That is deliberate: on a machine that has
    never generated (a fresh clone, a new operator) there is nothing to have
    drifted from, and a warning there would fire on every single start and
    train the reader to ignore it.

    Used by ``aureax start`` to warn that the resolved configs already on the
    nodes were generated from a different set of groups than this file now
    describes — the head will dial the peers it was last generated with.
    """
    recorded = _recorded_groups(network_config_path)
    if recorded is None:
        return False
    return recorded != _snapshot(network_config)


def write_baseline(network_config_path: Path, network_config) -> None:
    """Record the current groups as the order that was last generated.

    Call this only after every config file has been written: a baseline for an
    order that never reached a node would make the retry look like a second,
    legitimate run and hide the reorder for good.

    Other fleet files' records are preserved on a merge. They are NOT preserved
    when the existing file could not be parsed at all — it is replaced, and the
    warning says so. The cost of that is bounded and self-healing: a lost
    record means the next generate against that path records a fresh baseline
    (it reports nothing, it does not falsely refuse), whereas keeping a corrupt
    file would disable the check permanently.
    """
    data, readable = _read_baseline()
    if not readable:
        logger.warning(
            "replacing unreadable node_groups baseline %s; any other fleet file's "
            "recorded worker order is lost and will be recorded again on its next "
            "`aureax generate`",
            baseline_path(),
        )
        data = {}

    data[_key(network_config_path)] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "groups": _snapshot(network_config),
    }
    # 0600 in a 0700 directory: node ids and LAN addresses are already in the
    # fleet file, so this is not a secret, but it sits beside ~/.aureax/keys
    # and ~/.aureax/tls and inherits their posture rather than inventing a
    # second one. The atomic replace is what matters — a half-written baseline
    # reads as a reorder and refuses the next generate.
    write_private_file(baseline_path(), json.dumps(data, indent=2, sort_keys=True) + "\n")
