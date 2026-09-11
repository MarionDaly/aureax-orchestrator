"""A supervisor only ever kills the child it recorded, and only once its owner is gone.

The edge pods run with `hostPID`, so `client/process_ownership.py::reap` is
choosing a signal target out of the host's whole process table. A stale record,
a recycled PID or a supervisor that is still alive must all resolve to "leave
it alone" — the record's PID start time and boot id are what make that
decision, and the returned PID list is what stops the caller spawning a second
inference process over a live one. See ARCHITECTURE.md, "Process Management
Architecture".
"""

import json
import os
from unittest.mock import Mock

import pytest

from client import process_ownership


def test_unrecorded_processes_are_never_signalled(tmp_path, monkeypatch):
    """Patched at the real gate: `reap` signals through `os.pidfd_open` +
    `signal.pidfd_send_signal`, never `os.kill`, so asserting on `os.kill`
    would hold no matter what the function did."""
    monkeypatch.setattr(process_ownership, "_record_path", lambda *args: tmp_path / "absent.json")
    pidfd, sender = Mock(), Mock()
    monkeypatch.setattr(process_ownership.os, "pidfd_open", pidfd, raising=False)
    monkeypatch.setattr(process_ownership.signal, "pidfd_send_signal", sender, raising=False)
    assert process_ownership.reap("node", "llama-server") == []
    pidfd.assert_not_called()
    sender.assert_not_called()


@pytest.mark.parametrize("same_child,live_owner", [(False, False), (True, True)])
def test_reused_pids_and_live_supervisors_are_never_killed(tmp_path, monkeypatch, same_child, live_owner):
    child = {"pid": 1234, "start": "1", "boot": "test"}
    owner = {"pid": 5678, "start": "2", "boot": "test"}
    record = tmp_path / "child.json"
    record.write_text(json.dumps({"child": child, "owner": owner}))
    monkeypatch.setattr(process_ownership, "_record_path", lambda *args: record)
    monkeypatch.setattr(process_ownership, "_identity", lambda pid: child if pid == 1234 and same_child else owner if live_owner else {})
    pidfd = Mock()
    monkeypatch.setattr(process_ownership.os, "pidfd_open", pidfd, raising=False)
    assert process_ownership.reap("node", "llama-server") == ([1234] if live_owner else [])
    pidfd.assert_not_called()


def test_a_live_child_this_supervisor_owns_is_still_a_conflict(tmp_path, monkeypatch):
    """Ownership by THIS process is not permission to spawn a second child.

    Reporting nothing here told the caller the slot was free while its own live
    child held it, so the supervisor would start a second inference process over
    the first — two llama-servers racing for one port and one model's memory.
    """
    child = {"pid": 1234, "start": "1", "boot": "test"}
    owner = {"pid": os.getpid(), "start": "2", "boot": "test"}
    record = tmp_path / "child.json"
    record.write_text(json.dumps({"child": child, "owner": owner}))
    monkeypatch.setattr(process_ownership, "_record_path", lambda *args: record)
    monkeypatch.setattr(
        process_ownership, "_identity", lambda pid: child if pid == 1234 else owner
    )
    pidfd = Mock()
    monkeypatch.setattr(process_ownership.os, "pidfd_open", pidfd, raising=False)

    assert process_ownership.reap("node", "llama-server") == [1234]
    # Reported as a conflict, never signalled: it is a live child, not an orphan.
    pidfd.assert_not_called()


@pytest.mark.parametrize("raised", [ProcessLookupError, FileNotFoundError])
def test_a_child_that_exits_mid_reap_is_gone_not_surviving(tmp_path, monkeypatch, raised):
    """`pidfd_open` races the recorded child's own exit.

    Both errors mean the process left between the identity check and the open —
    which is the reap's goal. Reporting it as a survivor made the caller refuse
    to spawn over a process that no longer existed, restarting the pod for
    nothing and healing only on the next attempt.
    """
    child = {"pid": 1234, "start": "1", "boot": "test"}
    owner = {"pid": 5678, "start": "2", "boot": "test"}
    record = tmp_path / "child.json"
    record.write_text(json.dumps({"child": child, "owner": owner}))
    monkeypatch.setattr(process_ownership, "_record_path", lambda *args: record)

    def identity(pid):
        if pid == 1234:
            return child
        raise FileNotFoundError(f"/proc/{pid}/stat")  # the owner is long gone

    monkeypatch.setattr(process_ownership, "_identity", identity)
    monkeypatch.setattr(
        process_ownership.os, "pidfd_open", Mock(side_effect=raised), raising=False
    )
    monkeypatch.setattr(process_ownership.signal, "pidfd_send_signal", Mock(), raising=False)

    assert process_ownership.reap("node", "llama-server") == []
