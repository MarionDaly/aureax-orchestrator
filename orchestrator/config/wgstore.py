"""Per-node WireGuard keys for the rpc-server tunnel, under ``~/.aureax/wireguard``.

Only members of a node group with ``wireguard.enabled`` ever get a pair, and
only ``aureax generate`` mints one (``generate_all_edge_configs``), the same
rule keystore and tlsstore follow: a status command or a dashboard poll never
creates a credential as a side effect.

Layout (mirrors ``keystore.py`` and ``tlsstore.py``, so one ``ls`` of any of
the three stores reads the same way)::

    ~/.aureax/wireguard/        0700
    ├── <node_id>.key           0600   base64 of the raw 32-byte X25519 private key
    ├── <node_id>.key.prev      0600   the predecessor, kept by rotate_key()
    └── <node_id>.pub           0644   base64 of the raw 32-byte public key

The two encodings are exactly what ``wg genkey`` and ``wg pubkey`` print, so a
file here can be compared against a node's ``wg show`` by eye and pasted into a
hand-written peer section without conversion. The private key is the source of
truth: ``public_key()`` derives from it, and ``.pub`` is a convenience copy kept
in step by ``ensure_key`` and ``rotate_key`` — the same reason ``tls status``
reads certificates rather than trusting a side file.

The private key travels to its node inline in ``/etc/wireguard/wg0.conf``
(wg-quick reads PrivateKey only from the file), which the deployer stages 0600
and installs root:root 0600 — the treatment ``node.key`` gets.

Rotation keeps exactly one predecessor, like keystore: ``.key.prev`` is the key
the node still runs between ``aureax wg regenerate`` and the apply that ships
the new one, which is what an operator comparing ``wg show`` output needs.
"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from orchestrator.config.atomic import write_private_file

# The rule SSHConfig._check_names_are_path_segments applies to a node id,
# imported rather than restated so the fleet-file boundary and this path builder
# cannot drift apart — the same import tlsstore makes.
from orchestrator.config.schema import _PATH_SEGMENT_RE

_KEY_BYTES = 32
_KEY_MODE = 0o600
_PUB_MODE = 0o644
_DIR_MODE = 0o700


def store_dir() -> Path:
    """The WireGuard key directory. A function, not a module constant: tests
    monkeypatch $HOME (tests/conftest.py tmp_aureax_home), and a Path bound at
    import time would write into the developer's real ~/.aureax."""
    return Path.home() / ".aureax" / "wireguard"


def _check_node_id(node_id: str) -> str:
    """``node_id`` unchanged, or ValueError. Guards every filename built here.

    Same rule as keystore._check_node_id — no path separator, no ``..`` — plus
    the single-path-segment pattern tlsstore applies: ``aureax wg regenerate
    --node`` hands an operator string straight to rotate_key, and a "/" there
    would write a private key outside the store.
    """
    if not node_id:
        raise ValueError("node_id must be a non-empty name — it is used as a filename")
    if "/" in node_id or "\\" in node_id or ".." in node_id or not _PATH_SEGMENT_RE.fullmatch(node_id):
        raise ValueError(
            f"node_id {node_id!r} must be a single path segment with no '/', '\\' or '..' — "
            f"it is used verbatim as a filename under {store_dir()}"
        )
    return node_id


def key_paths(node_id: str) -> tuple[Path, Path]:
    """``(private_key_path, public_key_path)`` for one node."""
    _check_node_id(node_id)
    d = store_dir()
    return d / f"{node_id}.key", d / f"{node_id}.pub"


def _prev_path(node_id: str) -> Path:
    """``<node_id>.key.prev`` — with_name, never with_suffix (see keystore._prev_path)."""
    key_path, _ = key_paths(node_id)
    return key_path.with_name(key_path.name + ".prev")


def _ensure_store_dir() -> Path:
    """The store directory, existing and 0700 even if it pre-existed at 0755."""
    d = store_dir()
    d.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    os.chmod(d, _DIR_MODE)
    return d


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _decode_private(text: str, path: Path, node_id: str = "<node>") -> X25519PrivateKey:
    """The key a file holds, or a ValueError naming the file and the fix."""
    try:
        raw = base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError):
        raw = b""
    if len(raw) != _KEY_BYTES:
        raise ValueError(
            f"{path} does not hold a WireGuard private key (expected the base64 of 32 raw "
            f"bytes, as `wg genkey` prints). Replace it with `aureax wg regenerate --node "
            f"{node_id}`."
        )
    return X25519PrivateKey.from_private_bytes(raw)


def generate_private_key() -> str:
    """A fresh private key, base64 — the format `wg genkey` prints."""
    key = X25519PrivateKey.generate()
    return _encode(
        key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )


def derive_public_key(private_key_b64: str) -> str:
    """The public key for a base64 private key — the format `wg pubkey` prints."""
    key = _decode_private(private_key_b64, Path("<private key>"))
    return _encode(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )


def private_key(node_id: str) -> str | None:
    """The node's private key (base64), or None if it has none yet.

    Raises ValueError for a file that exists but is not a key: a silent None
    there would read as "not minted yet", and the next generate would leave the
    corrupt file in place while the deploy refused for a missing key.
    """
    path, _ = key_paths(node_id)
    try:
        text = path.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        text = ""
    _decode_private(text, path, node_id)
    return text.strip()


def public_key(node_id: str) -> str | None:
    """The node's public key (base64), derived from its private key; None if absent."""
    priv = private_key(node_id)
    return None if priv is None else derive_public_key(priv)


def _write_public(node_id: str, pub: str) -> None:
    _, pub_path = key_paths(node_id)
    write_private_file(pub_path, pub + "\n", mode=_PUB_MODE, dir_mode=_DIR_MODE)


def _create_private_if_absent(path: Path, key: str) -> bool:
    """Atomically publish ``key`` iff no key exists yet. True if this caller won.

    os.link fails when the destination exists, so two concurrent generates (the
    CLI and the dashboard's regenerate route) cannot both "succeed" with
    different keys — the case where one of them would render a wg0.conf whose
    key is no longer on disk. Same primitive as keystore._create_key_if_absent.
    """
    _ensure_store_dir()
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _KEY_MODE)
    try:
        with os.fdopen(fd, "w") as f:
            os.fchmod(f.fileno(), _KEY_MODE)
            f.write(key + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
    finally:
        tmp.unlink(missing_ok=True)


def ensure_key(node_id: str) -> str:
    """The node's public key, minting its pair first if it has none.

    Idempotent and convergent: an existing private key is never replaced (that
    is rotate_key's job), and every concurrent caller ends up with the key that
    won the create. The ``.pub`` copy is rewritten whenever it is missing or no
    longer matches the private key.
    """
    key_path, pub_path = key_paths(node_id)
    priv = private_key(node_id)
    if priv is None:
        _create_private_if_absent(key_path, generate_private_key())
        for _ in range(50):
            priv = private_key(node_id)
            if priv is not None:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(f"could not create a WireGuard key for {node_id!r} under {store_dir()}")
    pub = derive_public_key(priv)
    try:
        on_disk = pub_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        on_disk = None
    if on_disk != pub:
        _write_public(node_id, pub)
    return pub


def rotate_key(node_id: str) -> str:
    """Replace the node's pair unconditionally; the old private key goes to ``.key.prev``.

    Returns the new public key. Only correct as part of rotate -> apply ->
    ``aureax stop && aureax start``: the tunnel re-keys when apply restarts
    wg-quick, but apply never restarts a shard worker, and an rpc-server still
    holding the old head's connection refuses the new one.
    """
    key_path, _ = key_paths(node_id)
    try:
        old = private_key(node_id)
    except ValueError:
        old = None  # a corrupt key is exactly what a rotation is for; keep no copy of it
    if old:
        write_private_file(_prev_path(node_id), old + "\n", mode=_KEY_MODE, dir_mode=_DIR_MODE)
    new = generate_private_key()
    write_private_file(key_path, new + "\n", mode=_KEY_MODE, dir_mode=_DIR_MODE)
    pub = derive_public_key(new)
    _write_public(node_id, pub)
    return pub


def previous_private_key(node_id: str) -> str | None:
    """The private key ``rotate_key`` displaced, or None."""
    path = _prev_path(node_id)
    try:
        return path.read_text(encoding="ascii").strip() or None
    except (FileNotFoundError, UnicodeDecodeError):
        return None
