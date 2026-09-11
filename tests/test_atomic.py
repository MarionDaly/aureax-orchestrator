"""The shared private-file writer under orchestrator/config/atomic.py.

Everything here is a permission or a crash-window property, because that is
the entire reason the helper exists: the TLS store and the group-order
baseline both write material that must never be readable by another local
account, and never visible half-written to a concurrent reader (a truncated
PEM surfaces as an opaque OpenSSL parse error, a truncated JSON baseline as a
spurious "worker order changed" refusal).

Every test writes under ``tmp_path``; nothing here touches $HOME.
"""

from __future__ import annotations

import os
import stat
import threading

import pytest

from orchestrator.config.atomic import write_private_file

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _temps(directory) -> list:
    return sorted(directory.glob("*.tmp"))


@pytest.fixture
def keep_umask():
    """Restore the process umask — os.umask is process-global, so a test that
    leaves it at 077 makes every later test's mode assertion pass for the
    wrong reason."""
    old = os.umask(0o022)
    yield
    os.umask(old)


# --- payloads ---------------------------------------------------------------

def test_writes_bytes_and_creates_the_parent(tmp_path) -> None:
    path = tmp_path / "store" / "node.key"

    write_private_file(path, b"\x00\x01binary\xff")

    assert path.read_bytes() == b"\x00\x01binary\xff"
    assert path.parent.is_dir()


def test_str_is_encoded_utf8(tmp_path) -> None:
    """str and bytes callers both exist (JSON state vs. PEM blobs), so the
    encoding has to be pinned rather than left to the platform default."""
    path = tmp_path / "state.json"

    write_private_file(path, '{"node": "edge-éß"}')

    assert path.read_bytes() == '{"node": "edge-éß"}'.encode()


def test_a_non_text_payload_is_refused(tmp_path) -> None:
    with pytest.raises(TypeError):
        write_private_file(tmp_path / "x", {"not": "bytes"})  # type: ignore[arg-type]

    assert not (tmp_path / "x").exists()
    assert _temps(tmp_path) == []


# --- permissions ------------------------------------------------------------

@posix_only
def test_file_is_0600_and_its_directory_0700_by_default(tmp_path) -> None:
    path = tmp_path / "tls" / "node.key"

    write_private_file(path, "secret")

    assert _mode(path) == 0o600
    assert _mode(path.parent) == 0o700


@posix_only
def test_requested_mode_survives_a_hostile_umask(tmp_path, keep_umask) -> None:
    """os.open's mode is masked by the umask exactly as mkdir's is. Under a
    umask of 077 a certificate asked for at 0644 would land 0600 and fail only
    later, on whichever reader is not the owner."""
    os.umask(0o077)
    cert = tmp_path / "tls" / "node.pem"

    write_private_file(cert, "-----BEGIN CERTIFICATE-----\n", mode=0o644, dir_mode=0o700)

    assert _mode(cert) == 0o644
    assert _mode(cert.parent) == 0o700


@posix_only
def test_an_existing_parent_is_chmodded_too(tmp_path) -> None:
    """mkdir's mode is honoured only when it CREATES the leaf, so a directory
    restored from a backup or made by a hand-run `mkdir -p` keeps its 0755 and
    lets any local account list which nodes have key material here."""
    store = tmp_path / "tls"
    store.mkdir(mode=0o755)
    os.chmod(store, 0o755)

    write_private_file(store / "node.key", "secret")

    assert _mode(store) == 0o700


@posix_only
def test_dir_mode_is_honoured(tmp_path) -> None:
    path = tmp_path / "shared" / "f"

    write_private_file(path, "x", mode=0o640, dir_mode=0o750)

    assert _mode(path) == 0o640
    assert _mode(path.parent) == 0o750


# --- atomicity --------------------------------------------------------------

def test_replaces_an_existing_file(tmp_path) -> None:
    path = tmp_path / "node.key"
    write_private_file(path, "old value that is longer")

    write_private_file(path, "new")

    assert path.read_text() == "new"
    assert _temps(tmp_path) == []


@posix_only
def test_replacing_keeps_the_requested_mode_not_the_old_one(tmp_path) -> None:
    path = tmp_path / "node.key"
    write_private_file(path, "first")
    os.chmod(path, 0o666)

    write_private_file(path, "second")

    assert _mode(path) == 0o600


def test_a_failed_replace_leaves_no_temp_and_keeps_the_old_file(tmp_path, monkeypatch) -> None:
    """A crashed writer must leave junk at worst, never a wedge: the temp name
    is per-writer unique, so nothing downstream waits on it — but an
    accumulating pile of them in ~/.aureax is its own bug."""
    path = tmp_path / "node.key"
    write_private_file(path, "original")

    def boom(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        write_private_file(path, "replacement")

    assert path.read_text() == "original"
    assert _temps(tmp_path) == []


def test_a_failed_write_leaves_no_temp(tmp_path, monkeypatch) -> None:
    """Same cleanup, failing before the rename rather than at it — a full disk
    fails here, not at os.replace."""
    path = tmp_path / "node.key"

    def boom(fd):
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", boom)

    with pytest.raises(OSError, match="simulated fsync failure"):
        write_private_file(path, "never lands")

    assert not path.exists()
    assert _temps(tmp_path) == []


# --- durability -------------------------------------------------------------


def _sync_trace(monkeypatch) -> list[str]:
    """The order of fsyncs and renames one write performs."""
    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        return real_fsync(fd)

    def replace(src, dst):
        events.append("replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    return events


@posix_only
def test_the_rename_is_synced_as_well_as_the_bytes(tmp_path, monkeypatch) -> None:
    """Two different losses. The pre-rename fsync keeps a power cut from
    publishing a zero-length key at the real name; the post-rename directory
    fsync keeps it from losing the name itself, which would leave the previous
    key or baseline in place while the caller has been told the write landed.
    """
    events = _sync_trace(monkeypatch)

    write_private_file(tmp_path / "node.key", "secret")

    assert events == ["file", "replace", "dir"]


@posix_only
def test_a_directory_that_cannot_be_synced_still_completes_the_write(
    tmp_path, monkeypatch
) -> None:
    """Best effort on purpose: a directory fsync is not portable to every
    filesystem, and a file that is merely not-yet-durable is recoverable —
    failing here would turn a durability nicety into a failed generate."""
    real_fsync = os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("simulated directory fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)

    path = tmp_path / "node.key"
    write_private_file(path, "secret")

    assert path.read_text() == "secret"
    assert _temps(tmp_path) == []


# --- concurrency ------------------------------------------------------------

def _record_replaces(monkeypatch) -> list[str]:
    """Capture the temp name each write publishes from."""
    seen: list[str] = []
    real = os.replace

    def spy(src, dst):
        seen.append(str(src))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    return seen


def test_temp_names_are_unique_per_write(tmp_path, monkeypatch) -> None:
    seen = _record_replaces(monkeypatch)

    for i in range(25):
        write_private_file(tmp_path / "node.key", f"value {i}")

    assert len(set(seen)) == 25, seen
    assert _temps(tmp_path) == []


def test_two_concurrent_writers_do_not_collide(tmp_path) -> None:
    """Both writers hold an open temp at the same moment.

    A shared temp name would make the second os.open(O_EXCL) raise
    FileExistsError before it ever reached the barrier — the failure this
    forces into the open, since sequential writes never expose it. The
    destination must end up holding one writer's payload whole, never a mix.
    """
    path = tmp_path / "node.key"
    payloads = {"a": "aaaa" * 64, "b": "bb"}
    barrier = threading.Barrier(2, timeout=10)
    real = os.replace
    seen: list[str] = []
    lock = threading.Lock()

    def spy(src, dst):
        with lock:
            seen.append(str(src))
        barrier.wait()
        real(src, dst)

    errors: list[BaseException] = []

    def write(value: str) -> None:
        try:
            write_private_file(path, value)
        except BaseException as e:  # surfaced by the `errors == []` assertion below
            errors.append(e)

    original_replace = os.replace
    os.replace = spy  # type: ignore[assignment]
    try:
        threads = [threading.Thread(target=write, args=(v,)) for v in payloads.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        assert not any(t.is_alive() for t in threads), "a writer blocked at the barrier"
    finally:
        os.replace = original_replace  # type: ignore[assignment]

    assert errors == [], errors
    assert len(set(seen)) == 2, seen
    assert path.read_text() in payloads.values()
    assert _temps(tmp_path) == []
