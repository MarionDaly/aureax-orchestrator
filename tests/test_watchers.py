"""The dashboard's background watchers — what they write into the operator's log.

They had no tests at all. Every line here is a transition the operator reads as
fleet state, and two of the behaviours pinned below (an unreachable API is not
"every pod removed"; undetectable LAN IP is not "the whole fleet drifted") are
fixes for alarms that fired on a healthy fleet.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from orchestrator.dashboard import watchers
from tests.conftest import RecordingLog


def _levels_for(log: RecordingLog, needle: str) -> list[str]:
    return [level for level, text, _ in log.lines if needle in text]


def _gate_after(monkeypatch, target, name, result, calls: int = 3) -> asyncio.Event:
    """Stub ``target.name`` and return an event set once it has been called N times.

    The loops below run at interval 0, so "has it ticked yet?" is a question
    only the code under test can answer — a fixed sleep asserts a scheduling
    outcome instead, and the pause-announced-once assertion then fails on a
    busy machine with nothing wrong. Waiting for the Nth call proves the first
    N-1 ticks completed, which is exactly what "not once per tick" needs.

    The watchers call these helpers through ``asyncio.to_thread``, so the event
    is set on the loop rather than from the worker thread.
    """
    loop = asyncio.get_running_loop()
    event = asyncio.Event()
    seen = []

    def stub(*_a, **_k):
        seen.append(None)
        if len(seen) >= calls:
            loop.call_soon_threadsafe(event.set)
        return result

    monkeypatch.setattr(target, name, stub)
    return event


async def _run_until(task: asyncio.Task, ticked: asyncio.Event) -> None:
    await asyncio.wait_for(ticked.wait(), 5)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# --- PodWatcher ---------------------------------------------------------------


def test_a_new_pod_is_announced_once():
    log = RecordingLog()
    w = watchers.PodWatcher(log)

    w._diff({"edge-abc": ("Running", True, 0)})
    w._diff({"edge-abc": ("Running", True, 0)})

    assert log.texts() == ["Pod appeared: edge-abc (Running, ready)"]


def test_a_restart_bump_is_an_error_even_though_nothing_else_changes():
    """The only signal a crash-looping pod produces: phase stays Running and
    ready stays False, so no other field transitions."""
    log = RecordingLog()
    w = watchers.PodWatcher(log)
    w.last = {"edge-abc": ("Running", False, 3)}

    w._diff({"edge-abc": ("Running", False, 4)})

    assert _levels_for(log, "restarted (3→4)") == ["error"]


def test_a_readiness_loss_is_a_warning_and_a_gain_is_a_success():
    log = RecordingLog()
    w = watchers.PodWatcher(log)
    w.last = {"edge-abc": ("Running", True, 0)}

    w._diff({"edge-abc": ("Running", False, 0)})
    w._diff({"edge-abc": ("Running", True, 0)})

    assert _levels_for(log, "became NotReady") == ["warn"]
    assert _levels_for(log, "became ready") == ["success"]


def test_a_pod_that_disappears_is_reported_removed():
    log = RecordingLog()
    w = watchers.PodWatcher(log)
    w.last = {"edge-abc": ("Running", True, 0)}

    w._diff({})

    assert log.texts() == ["Pod removed: edge-abc"]


async def test_an_unreachable_api_pauses_once_and_removes_nothing(monkeypatch):
    """`get_pods` returns [] both for "namespace empty" and "cannot reach the
    API server", and _diff reads the second as every pod having vanished — an
    unplugged laptop produced a screenful of "Pod removed"."""
    log = RecordingLog()
    w = watchers.PodWatcher(log, interval_seconds=0)
    w.last = {"edge-abc": ("Running", True, 0)}
    ticked = _gate_after(monkeypatch, watchers, "api_reachable_cached", False)

    await _run_until(asyncio.create_task(w.run()), ticked)

    assert [t for t in log.texts() if "removed" in t] == []
    assert [t for t in log.texts() if "paused" in t] == [
        "Pod watcher paused: kubectl API server unreachable"
    ], "the pause must be announced once, not once per tick"
    assert w.last == {"edge-abc": ("Running", True, 0)}, "state kept, not cleared"


# --- NodeWatcher --------------------------------------------------------------


def test_node_transitions_are_rendered_readably():
    log = RecordingLog()
    w = watchers.NodeWatcher(log)

    w._diff({"k3d-server": "True"})
    w._diff({"k3d-server": "False"})
    w._diff({})

    assert log.texts() == [
        "Node appeared: k3d-server (Ready)",
        "Node k3d-server: Ready → NotReady",
        "Node removed: k3d-server",
    ]
    assert [level for level, _, _ in log.lines] == ["success", "warn", "warn"]


# --- DriftChecker -------------------------------------------------------------


async def test_an_undetectable_lan_ip_pauses_instead_of_flagging_every_node(
    monkeypatch, network_config_path
):
    """get_server_url falls back to https://127.0.0.1 when LAN-IP detection
    fails. Comparing every node's real K3S_URL against loopback flagged the
    whole fleet with an error-level "run `aureax heal-nodes`" every 5 minutes;
    both CLI paths guard exactly this.
    """
    from orchestrator.cluster import k3d
    from orchestrator.deploy import k3s_agent

    ticked = _gate_after(monkeypatch, k3d, "get_server_url", "https://127.0.0.1:6443")

    def no_ssh(*a, **k):
        raise AssertionError("no node may be probed once the LAN IP is unknown")

    monkeypatch.setattr(k3s_agent, "get_configured_server_url", no_ssh)

    log = RecordingLog()
    checker = watchers.DriftChecker(
        log, network_config_path, interval_seconds=0, warmup_seconds=0
    )
    await _run_until(asyncio.create_task(checker.run()), ticked)

    assert [t for t in log.texts() if "Drift detected" in t] == []
    paused = [t for t in log.texts() if "paused" in t]
    assert len(paused) == 1, paused
    assert "could not detect the orchestrator LAN IP" in paused[0]


@pytest.mark.parametrize("cls", [watchers.PodWatcher, watchers.NodeWatcher])
def test_watchers_start_quiet(cls):
    """A watcher with nothing to report must emit nothing at all."""
    log = RecordingLog()
    cls(log)._diff({})

    assert log.lines == []
