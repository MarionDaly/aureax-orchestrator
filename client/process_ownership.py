"""Track inference children without treating unrelated host processes as orphans."""

import fcntl
import hashlib
import json
import os
import secrets
import signal
from contextlib import contextmanager
from pathlib import Path


def _identity(pid: int) -> dict:
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "start": stat[19], "boot": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def _record_path(node_id: str, binary: str) -> Path:
    name = hashlib.sha256(f"{node_id}:{os.path.basename(binary)}".encode()).hexdigest()[:32]
    return Path("/opt/aureax/run") / f"child-{name}.json"


@contextmanager
def spawn_guard(node_id: str, binary: str):
    """Serialize ownership checks, spawn and recording across pod incarnations."""
    if not Path("/proc/sys/kernel/random/boot_id").exists():
        yield
        return
    path = _record_path(node_id, binary).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(fd, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def remember(node_id: str, binary: str, pid: int) -> None:
    if not Path("/proc/sys/kernel/random/boot_id").exists():
        return  # Host orphan recovery is Linux-only.
    data = {"child": _identity(pid), "owner": _identity(os.getpid())}
    path = _record_path(node_id, binary)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(f".{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def reap(node_id: str, binary: str, timeout: float = 10) -> list[int]:
    """Kill only a recorded child whose owning supervisor has exited.

    PID start time and boot identity prevent PID-reuse mistakes. pidfd pins the
    checked process through signalling; kernels without pidfd fail closed.
    Return surviving/conflicting owned PIDs so the caller cannot spawn over them.
    """
    path = _record_path(node_id, binary)
    verified_pid = None
    try:
        record = json.loads(path.read_text())
        child, owner = record["child"], record["owner"]
        pid = child["pid"]
        if not isinstance(pid, int) or pid <= 1 or _identity(pid) != child:
            return []
        verified_pid = pid
        try:
            if _identity(owner["pid"]) == owner:
                # A conflict whoever the owner is, this process included: the
                # child is alive and unsignalled, so reporting nothing would let
                # the caller spawn a second inference process over one it
                # already owns.
                return [pid]
        except FileNotFoundError:
            pass
        except (OSError, ValueError, IndexError):
            return [pid]
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            return [pid]
        try:
            fd = os.pidfd_open(pid)
        except (ProcessLookupError, FileNotFoundError):
            # The child exited between the identity check above and this open.
            # Gone is the goal, so this is success: reporting it as a survivor
            # makes the caller refuse to spawn over a process that no longer
            # exists, and the pod restarts for nothing.
            return []
        try:
            if _identity(pid) != child:
                return []
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            # poll on pidfd waits for this exact process, not a reused PID.
            import select
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            return [] if poll.poll(int(timeout * 1000)) else [pid]
        except (ProcessLookupError, FileNotFoundError):
            return []  # same race, one step later — the child is already gone
        finally:
            os.close(fd)
    except OSError:
        return [verified_pid] if verified_pid is not None else []
    except (ValueError, KeyError, TypeError, IndexError):
        return []  # Missing/stale/unreadable records never authorize a signal.
