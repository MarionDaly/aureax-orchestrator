"""Tests for ``LifecycleManager`` state transitions.

These tests inject a fake worker so we don't actually try to start k3d/K8s.
"""

from __future__ import annotations

import asyncio
import signal
import subprocess
from pathlib import Path

import pytest

from orchestrator import lifecycle as lifecycle_module
from orchestrator.lifecycle import (
    BusyError,
    CancelledOp,
    LifecycleManager,
    State,
    SystemLogger,
)


@pytest.fixture
def mgr(tmp_path: Path) -> LifecycleManager:
    fake_cfg = tmp_path / "aureax-network.yaml"
    fake_cfg.write_text("# unused — worker is monkey-patched\n")
    return LifecycleManager(fake_cfg, logger=SystemLogger())


async def test_initial_state_is_idle(mgr):
    assert mgr.state == State.IDLE
    assert not mgr.busy
    assert mgr.run_id is None


async def test_try_start_transitions_to_starting_then_idle(mgr, monkeypatch):
    async def fake_start(_node):
        await asyncio.sleep(0)  # give the loop a chance to mark state

    monkeypatch.setattr(mgr, "_run_start", fake_start)
    run_id = mgr.try_start()
    assert run_id
    # State right after try_start: STARTING + busy
    assert mgr.state == State.STARTING
    assert mgr.busy
    # Wait for the wrapper task to complete
    await mgr._task
    assert mgr.state == State.IDLE
    assert not mgr.busy
    assert mgr.last_error is None


async def test_try_start_while_busy_raises_busy_error(mgr, monkeypatch):
    async def slow_start(_node):
        await asyncio.sleep(0.1)

    monkeypatch.setattr(mgr, "_run_start", slow_start)
    mgr.try_start()
    with pytest.raises(BusyError):
        mgr.try_start()
    await mgr._task


async def test_worker_failure_transitions_to_error(mgr, monkeypatch):
    async def boom(_node):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(mgr, "_run_start", boom)
    mgr.try_start()
    await mgr._task
    assert mgr.state == State.ERROR
    assert mgr.last_error == "simulated failure"
    assert not mgr.busy


async def test_cancelled_op_returns_to_idle_not_error(mgr, monkeypatch):
    async def cancel_inside(_node):
        raise CancelledOp("stop pressed")

    monkeypatch.setattr(mgr, "_run_start", cancel_inside)
    mgr.try_start()
    await mgr._task
    # Cancellation is not failure: state goes back to IDLE.
    assert mgr.state == State.IDLE
    assert "cancelled" in (mgr.last_error or "")


async def test_stop_during_starting_signals_cancel(mgr, monkeypatch):
    cancel_seen = asyncio.Event()

    async def watching_start(_node):
        # Poll for cancel just like the real worker does
        while True:
            if mgr.cancel_event.is_set():
                cancel_seen.set()
                raise CancelledOp("cancelled by user")
            await asyncio.sleep(0.01)

    monkeypatch.setattr(mgr, "_run_start", watching_start)
    mgr.try_start()
    # try_stop while STARTING should cancel rather than queue or raise BusyError
    mgr.try_stop()
    await asyncio.wait_for(cancel_seen.wait(), timeout=1.0)
    await mgr._task
    assert mgr.state == State.IDLE


class _FakeProc:
    """Popen stand-in: alive, and slow to die unless `dies_on_term`."""

    def __init__(self, pid: int = 4242, dies_on_term: bool = False) -> None:
        self.pid = pid
        self._dies_on_term = dies_on_term
        self.waits: list[float] = []

    def poll(self):
        return None  # still running

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self._dies_on_term:
            return 0
        raise subprocess.TimeoutExpired(cmd="setup-all.sh", timeout=timeout)


def _signal_recorder(monkeypatch) -> list[tuple[str, int, int]]:
    """Record killpg/getpgid against orchestrator.lifecycle's `os`."""
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(lifecycle_module.os, "getpgid", lambda pid: pid + 1000)
    monkeypatch.setattr(
        lifecycle_module.os, "killpg",
        lambda pgid, sig: calls.append(("killpg", pgid, sig)),
    )
    return calls


def test_cancel_kills_the_whole_process_group_then_escalates(mgr, monkeypatch):
    """The group kill is what reaches setup-all.sh's ssh/scp/docker children.

    They are spawned in their own session (start_new_session=True), so
    SIGTERMing the bash parent alone leaves a multi-minute image build running
    after the operator pressed Stop.
    """
    calls = _signal_recorder(monkeypatch)
    proc = _FakeProc()
    mgr.subprocess = proc

    mgr.cancel()

    assert mgr.cancel_event.is_set()
    assert calls == [
        ("killpg", 5242, signal.SIGTERM),
        ("killpg", 5242, signal.SIGKILL),
    ]
    assert proc.waits == [5], "SIGKILL escalation needs a bounded wait for SIGTERM"


def test_cancel_does_not_escalate_when_sigterm_worked(mgr, monkeypatch):
    calls = _signal_recorder(monkeypatch)
    mgr.subprocess = _FakeProc(dies_on_term=True)

    mgr.cancel()

    assert calls == [("killpg", 5242, signal.SIGTERM)]


def test_cancel_survives_a_process_that_is_already_gone(mgr, monkeypatch):
    """The pid can exit between poll() and killpg — that is not a failure."""
    def boom(_pid):
        raise ProcessLookupError("no such process")

    monkeypatch.setattr(lifecycle_module.os, "getpgid", boom)
    mgr.subprocess = _FakeProc()

    mgr.cancel()  # must not raise

    assert mgr.cancel_event.is_set()
    assert any(
        "already gone" in line.text for line in mgr.logger.snapshot()
    ), "an already-dead subprocess must still be reported"
