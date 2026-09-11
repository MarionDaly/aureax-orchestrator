"""``_kill_process_on_port`` must target listeners, not everyone on the port.

``aureax stop`` uses this to shut down the dashboard on :7001. `lsof -i :N`
selects any socket whose local OR remote port is N, so without an explicit
LISTEN filter the "stop the dashboard" step also selects every client
connected to it — the operator's own browser included.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys

import pytest

from orchestrator import lifecycle
from orchestrator.lifecycle import _kill_process_on_port
from tests.conftest import RecordingLog

pytestmark = pytest.mark.skipif(shutil.which("lsof") is None, reason="needs lsof")


def test_only_the_listener_is_selected(monkeypatch) -> None:
    """A connected client shares the port but must not be a kill target.

    Driven through `_kill_process_on_port` itself, with lsof left real and only
    the signal stubbed out: asserting on a hand-written lsof invocation would
    keep passing after the `-sTCP:LISTEN` filter was deleted from the code,
    which is the regression this file exists to catch. The client lives in a
    CHILD process on purpose — with both sockets in this one, lsof reports the
    same single pid with or without the filter and the bug is invisible.
    """
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(10)
    port = server.getsockname()[1]

    child = subprocess.Popen(
        [
            sys.executable, "-c",
            "import socket, sys, time\n"
            f"s = socket.create_connection(('127.0.0.1', {port}))\n"
            "sys.stdout.write('connected\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    calls: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(argv, **kwargs):
        calls.append(list(argv))
        return real_run(argv, **kwargs)

    killed: set[int] = set()

    def recording_kill(pid, sig):
        # Never actually signal: the listener here is the pytest process.
        killed.add(pid)

    conn = None
    try:
        conn, _ = server.accept()
        assert child.stdout.readline().strip() == "connected"

        monkeypatch.setattr(lifecycle.subprocess, "run", recording_run)
        monkeypatch.setattr(lifecycle.os, "kill", recording_kill)
        _kill_process_on_port(port, RecordingLog(), label="Dashboard")
        monkeypatch.undo()

        lsof_argv = next((c for c in calls if c and c[0] == "lsof"), None)
        assert lsof_argv is not None, f"lsof was never invoked: {calls}"
        assert "-sTCP:LISTEN" in lsof_argv, f"listener filter dropped: {lsof_argv}"

        assert killed == {os.getpid()}, (
            f"expected only the listening pid to be killed, got {killed} "
            f"(client pid {child.pid})"
        )
    finally:
        monkeypatch.undo()
        child.kill()
        child.wait(timeout=10)
        if child.stdout is not None:
            child.stdout.close()
        if conn is not None:
            conn.close()
        server.close()


def test_reports_not_running_when_nothing_listens() -> None:
    """A free port must report 'not running' rather than killing a client."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # now nothing is listening

    log = RecordingLog()
    _kill_process_on_port(port, log, label="Dashboard")

    assert any("not running" in text for _lvl, text, _src in log.lines), log.lines
    assert not any("Killed" in text for _lvl, text, _src in log.lines), log.lines


def test_a_failed_kill_is_not_reported_as_killed(monkeypatch) -> None:
    """Signalling can fail — EPERM on another user's listener, ESRCH if the pid
    exited between lsof and here. A discarded result meant "Killed dashboard
    process on port 7001" while it was still serving, and `do_stop` then said
    "Stopped."
    """
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "111\n", "")

    def refuse(pid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)
    monkeypatch.setattr(lifecycle.os, "kill", refuse)
    log = RecordingLog()

    _kill_process_on_port(7001, log, label="Dashboard")

    assert any(
        lvl == "warn" and "111" in text and "not permitted" in text.lower()
        for lvl, text, _src in log.lines
    ), log.lines
    assert any(
        lvl == "warn" and "still listening" in text for lvl, text, _src in log.lines
    ), log.lines
    assert not any("Sent SIGTERM" in text for _lvl, text, _src in log.lines), log.lines


def test_lsof_is_bounded_so_a_wedged_mount_cannot_hang_the_stop(monkeypatch) -> None:
    """This is the last step of `aureax stop`; lsof stats every open file."""
    seen: dict = {}

    def fake_run(argv, **kwargs):
        if argv and argv[0] == "lsof":
            seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)
    _kill_process_on_port(7001, RecordingLog(), label="Dashboard")

    assert seen.get("timeout"), "lsof was the only unbounded subprocess in the module"
