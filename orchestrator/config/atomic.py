"""Write private state with restrictive creation permissions and an atomic rename.

Each writer uses a unique temporary file, flushes it before replacement, syncs
the parent directory after it, and removes the temporary file on failure. Set
dir_mode=None to preserve an existing source directory.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def write_private_file(
    path: Path,
    data: bytes | str,
    *,
    mode: int = 0o600,
    dir_mode: int | None = 0o700,
) -> None:
    """Atomically write ``data`` to ``path`` at ``mode``, in a ``dir_mode`` parent.

    ``data`` may be ``str`` (encoded UTF-8) or ``bytes`` — the callers are a
    mix: PEM blobs arrive from ``cryptography`` as bytes, JSON state as str.
    """
    path = Path(path)
    payload = data.encode() if isinstance(data, str) else data
    if not isinstance(payload, bytes | bytearray):
        raise TypeError(f"write_private_file needs str or bytes, got {type(data).__name__}")

    if dir_mode is None:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        _ensure_dir(path.parent, dir_mode)

    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as f:
            # os.open's mode argument is masked by the process umask exactly as
            # mkdir's is, so a 0644 certificate written under a umask of 077
            # would land 0600 and only fail later, on whichever reader is not
            # the owner. fchmod on the still-empty file makes the mode exact
            # and can never widen it beyond what the caller asked for.
            os.fchmod(f.fileno(), mode)
            f.write(payload)
            # fsync before the rename, not after: without it the rename can be
            # durable while the bytes it publishes are not, so a power loss
            # leaves a zero-length key or baseline at the real name — the one
            # state os.replace is here to prevent.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def _fsync_dir(directory: Path) -> None:
    """Persist the rename itself, best effort.

    The pre-rename fsync above makes the BYTES durable; the rename is a
    directory metadata operation of its own, so without this a crash can lose
    the new name entirely and leave the previous key or baseline in place. Best
    effort because a directory fsync is not portable to every filesystem and a
    file that is merely not-yet-renamed is a recoverable state — failing the
    write here would turn a durability nicety into an outage.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _ensure_dir(directory: Path, dir_mode: int) -> None:
    """The parent directory, existing and at ``dir_mode``.

    The chmod is not redundant with mkdir's ``mode``: that is honoured only
    when mkdir actually CREATES the leaf and is masked by the umask even then,
    so a ~/.aureax/tls restored from a backup or made by a hand-run ``mkdir -p``
    keeps its 0755 and lets any local account list which nodes have material
    here. Files carry their own mode either way, so what this closes is the
    directory listing.
    """
    directory.mkdir(parents=True, exist_ok=True, mode=dir_mode)
    os.chmod(directory, dir_mode)
