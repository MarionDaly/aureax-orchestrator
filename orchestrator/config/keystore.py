"""Per-node bearer keys, stored outside git on the orchestrator.

Two KINDS of key, one file each per node under ``~/.aureax/keys`` (0600,
directory 0700), holding the key plus a trailing newline:

* ``<node_id>.key``     — llama-server's API key (KIND_LLAMA)
* ``<node_id>.mcp.key`` — the node's MCP server key (KIND_MCP)

Separate files rather than one keyed record, for the same reasons the store is
per-file at all: rotation stays an atomic single-file replace (no
read-modify-write race between two concurrent generates), and rotating one
service cannot disturb the other — `aureax apply --rotate-key` re-keys both
because one pod restart applies both, but a caller that asks for one gets one.

**Both kinds share the ``aureax_sk_`` prefix.** A second prefix would have to
be taught to four independent leak detectors — the two greps in
scripts/test-all.sh, the display masking, and the chat service's
"unresolved placeholder" tell — and the one that got missed would be a silent
hole, which is the opposite of what a distinctive prefix is for. The file name
carries the kind instead.

Keys are created by ``generate_edge_config`` iff a node's template references
``${api_key}`` / ``${mcp_api_key}`` — never as a side effect of validation — so
a wiped node regains its keys from the orchestrator with no manual step (nodes
are re-imaged as routine, so anything needing a hand there is a bug), while
nodes that never asked for auth never get a key file.

Rotation keeps exactly one predecessor per kind (``<node_id>.key.prev``,
``<node_id>.mcp.key.prev``) so the dashboard can offer "copy previous key"
while a rotate has been staged but not yet applied — the window where the
running server still enforces the old key.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from pathlib import Path

from orchestrator.config.atomic import write_private_file

logger = logging.getLogger("aureax.config.keystore")

KEY_PREFIX = "aureax_sk_"

# What key_state() reports. "unreadable" exists because the display helpers all
# collapse to None, and "this node has no key" and "this node has a key I cannot
# read" call for opposite operator actions: run `aureax generate`, or fix the
# file's ownership (a .key left root-owned by a sudo run is the usual cause).
KEY_PRESENT = "present"
KEY_ABSENT = "absent"
KEY_UNREADABLE = "unreadable"

# 32 CSPRNG bytes -> 43 base64url chars (256 bits). The distinctive prefix is
# not cosmetic: it is what makes a leaked key greppable in logs and shipped
# transcripts (test-all.sh asserts shipped logs contain no "aureax_sk_").
_KEY_BYTES = 32

# The two services a node authenticates. The value is the caller-facing name;
# the infix is what separates the two files in one directory. KIND_LLAMA's
# infix is empty so every already-minted llama key keeps the filename it has —
# renaming them would silently 401 a fleet whose configs were generated before
# the rename.
KIND_LLAMA = "llama"
KIND_MCP = "mcp"
_KIND_INFIX = {KIND_LLAMA: "", KIND_MCP: ".mcp"}


def _kind_infix(kind: str) -> str:
    """The filename infix for ``kind``, or ValueError.

    Explicit rather than ``.get(kind, "")``: a typo'd kind that silently fell
    back to the llama file would hand the MCP server llama-server's key, and
    both sides would agree on it — an authentication bug with no error anywhere.
    """
    try:
        return _KIND_INFIX[kind]
    except KeyError:
        raise ValueError(
            f"unknown key kind {kind!r} — expected {KIND_LLAMA!r} or {KIND_MCP!r}"
        ) from None


def _check_node_id(node_id: str) -> str:
    """``node_id`` unchanged, or ValueError. Guards the filename we build.

    Every function here interpolates a node id straight into a path, and not
    every caller arrives via a validated fleet file: the dashboard's key routes
    take the node id from the URL. A "/" or a ".." would write a key outside
    ~/.aureax/keys and read one back the same way, so the check lives at the
    path builder rather than only at the config boundary
    (SSHConfig._check_names_are_path_segments, which covers the fleet file).

    The trailing ".mcp" is the half that boundary does NOT cover — its pattern
    allows dots — and it only became a problem when two kinds started sharing
    one directory: a node genuinely named "edge-01.mcp" would own
    "edge-01.mcp.key", which is edge-01's MCP key file. Two nodes would then be
    handed the same credential, and rotating either would 401 the other.
    """
    if not node_id:
        raise ValueError("node_id must be a non-empty name — it is used as a filename")
    if "/" in node_id or "\\" in node_id or ".." in node_id:
        raise ValueError(
            f"node_id {node_id!r} may not contain '/', '\\' or '..' — it is used "
            "verbatim as a filename under ~/.aureax/keys"
        )
    if node_id.endswith(_KIND_INFIX[KIND_MCP]):
        raise ValueError(
            f"node_id {node_id!r} may not end in '.mcp' — that name is already the "
            f"MCP key file of the node called {node_id[: -len(_KIND_INFIX[KIND_MCP])]!r}"
        )
    return node_id


def store_dir() -> Path:
    """The key directory. A function, not a module constant: tests monkeypatch
    $HOME (tests/conftest.py tmp_aureax_home), and a Path bound at import time
    would write into the developer's real ~/.aureax."""
    return Path.home() / ".aureax" / "keys"


def key_path(node_id: str, kind: str = KIND_LLAMA) -> Path:
    return store_dir() / f"{_check_node_id(node_id)}{_kind_infix(kind)}.key"


def _prev_path(node_id: str, kind: str = KIND_LLAMA) -> Path:
    """The predecessor file beside ``key_path``.

    ``with_name(name + ".prev")``, never ``with_suffix(".prev")`` — with_suffix
    replaces the LAST suffix, so on "edge-01.mcp.key" it would produce
    "edge-01.mcp.prev": a file no reader looks for, and a lost previous key
    during the rotate-to-apply window.
    """
    p = key_path(node_id, kind)
    return p.with_name(p.name + ".prev")


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(_KEY_BYTES)


# Paths already reported as unreadable. `aureax nodes` and the Access tab's 20s
# poll call the display helpers per node, so this warns once per file per
# process instead of once per render.
_warned_unreadable: set[str] = set()


def _read_state(path: Path) -> tuple[str, str | None]:
    """``(state, key)``. Never raises: every caller is a display helper.

    Only FileNotFoundError means "no key". A PermissionError (a .key left
    root-owned by a sudo run), an IsADirectoryError and undecodable bytes are
    all KEY_UNREADABLE: letting any of them escape read_key() takes the CLI's
    Auth column and every field of the dashboard's _auth_block with it.

    Decoding is strict rather than ``errors="replace"``, which would report
    KEY_PRESENT with a U+FFFD-mangled key — the one state KEY_UNREADABLE exists
    to distinguish. A mangled key displays a value the node does not hold and
    fails generation later against the api_key shape check, naming the wrong
    problem.
    """
    try:
        key = path.read_bytes().decode().strip()
    except FileNotFoundError:
        return KEY_ABSENT, None
    except (OSError, UnicodeDecodeError) as e:
        if str(path) not in _warned_unreadable:
            _warned_unreadable.add(str(path))
            logger.warning("key file %s cannot be read (%s) — reporting no key", path, e)
        return KEY_UNREADABLE, None
    return (KEY_PRESENT, key) if key else (KEY_ABSENT, None)


def _read(path: Path) -> str | None:
    return _read_state(path)[1]


def _ensure_store_dir() -> Path:
    """The key directory, existing and 0700.

    The chmod is not redundant with mkdir's ``mode``: that is honoured only when
    mkdir actually CREATES the leaf, so a ~/.aureax/keys restored from a backup
    or made by a hand-run `mkdir -p` keeps its 0755 and advertises which nodes
    hold keys. Files stay 0600 either way (_write_tmp's O_EXCL mode), so what
    this closes is directory listing. Mirrors cli.init and generate_all_edge_configs.
    """
    d = store_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    return d


def _write_tmp(path: Path, key: str) -> Path:
    """Fully written, fsynced 0600 temp file with a per-writer unique name.

    0600 from the first byte — never a window at the umask default, and fchmod
    makes that exact under a restrictive umask. The unique name means there is
    no shared lock file to go stale: a crashed writer leaves only junk, never a
    wedge.

    The fsync is what _create_key_if_absent's os.link and _replace_key's
    os.replace publish: without it a crash can make the new name durable while
    its bytes are not, and a zero-length key file reads as KEY_ABSENT while the
    running server still enforces the old key. Same reasoning, and the same
    order, as orchestrator.config.atomic.write_private_file.
    """
    _ensure_store_dir()
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(key + "\n")
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp


def _create_key_if_absent(path: Path, key: str) -> bool:
    """Atomically publish ``key`` iff no key exists yet. True if we won.

    os.link is the create-if-absent primitive here: unlike os.replace it FAILS
    when the destination exists, so two concurrent creators cannot both
    'succeed' with different keys — the failure mode where one caller ships a
    config carrying a key that is no longer on disk, surfacing only as 401s
    after the pod restarts.
    """
    tmp = _write_tmp(path, key)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _replace_key(path: Path, key: str) -> None:
    """Unconditional replace — rotation semantics, last writer wins.

    Through the shared helper rather than _write_tmp + os.replace: publishing a
    rotated key wants the parent-directory sync too, and there is no reason for
    a second copy of the same dance. _write_tmp stays for _create_key_if_absent,
    whose os.link needs the temporary file to still exist by name.
    """
    write_private_file(path, key + "\n")


def read_key(node_id: str, kind: str = KIND_LLAMA) -> str | None:
    return _read(key_path(node_id, kind))


def previous_key(node_id: str, kind: str = KIND_LLAMA) -> str | None:
    return _read(_prev_path(node_id, kind))


def has_key(node_id: str, kind: str = KIND_LLAMA) -> bool:
    return read_key(node_id, kind) is not None


def key_state(node_id: str, kind: str = KIND_LLAMA) -> str:
    """KEY_PRESENT / KEY_ABSENT / KEY_UNREADABLE for display callers.

    ``has_key`` and the other helpers answer None for both of the last two; this
    is how "key pending — run `aureax generate`" stays distinguishable from a
    key file that exists and cannot be read.
    """
    return _read_state(key_path(node_id, kind))[0]


def ensure_key(node_id: str, kind: str = KIND_LLAMA) -> str:
    """The node's key of this kind, created if absent — every concurrent caller
    converges on the SAME key (`aureax generate` and the dashboard's regenerate
    can overlap, and nothing serialises them)."""
    existing = read_key(node_id, kind)
    if existing:
        return existing
    path = key_path(node_id, kind)
    candidate = generate_key()
    if _create_key_if_absent(path, candidate):
        # Tolerate a stale "<node>.key.tmp" from an older staging layout: this
        # writer never creates that name (_write_tmp's names are per-writer
        # unique), so sweeping it is pure housekeeping and never removes a file
        # another writer is using.
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        return candidate
    # Lost the create race: the winner's key exists. The retry loop covers the
    # file being removed between the failed link and the read by anything
    # outside this process — an operator clearing the store by hand; nothing
    # in the orchestrator removes a key today.
    for _ in range(50):
        key = read_key(node_id, kind)
        if key:
            return key
        time.sleep(0.01)
    raise RuntimeError(f"could not create a {kind} key for {node_id!r} under {store_dir()}")


def rotate_key(node_id: str, kind: str = KIND_LLAMA) -> str:
    """Unconditionally replace the node's key, retaining the old one in .prev.

    Only correct as part of rotate -> regenerate -> apply: a rotation that is
    never applied leaves the running server enforcing the .prev key while every
    regenerated config carries the new one.

    One kind at a time. `aureax apply --rotate-key` re-keys both because a
    single pod restart applies both, but that is the caller's decision to make
    twice, not this function's to make for it.
    """
    old = read_key(node_id, kind)
    if old:
        _replace_key(_prev_path(node_id, kind), old)
    new = generate_key()
    _replace_key(key_path(node_id, kind), new)
    return new


def delete_key(node_id: str, kind: str = KIND_LLAMA) -> bool:
    """Remove this kind of key (and its .prev). Returns whether one existed."""
    existed = key_path(node_id, kind).exists()
    key_path(node_id, kind).unlink(missing_ok=True)
    _prev_path(node_id, kind).unlink(missing_ok=True)
    return existed


def fingerprint(node_id: str, kind: str = KIND_LLAMA) -> str | None:
    """Short non-secret identifier for display: sha256 prefix of the key."""
    key = read_key(node_id, kind)
    if not key:
        return None
    return "sha256:" + hashlib.sha256(key.encode()).hexdigest()[:12]


def last4(node_id: str, kind: str = KIND_LLAMA) -> str | None:
    key = read_key(node_id, kind)
    return key[-4:] if key else None
