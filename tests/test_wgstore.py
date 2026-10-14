"""The WireGuard key store under ~/.aureax/wireguard.

Each file here ends up inline in a node's /etc/wireguard/wg0.conf, so the
format is load-bearing: base64 of the raw 32-byte key, exactly what `wg genkey`
and `wg pubkey` print, or wg-quick refuses the interface. The modes and the
path guard are the same contract keystore and tlsstore keep.
"""

from __future__ import annotations

import base64
import os
import stat

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from orchestrator.config import wgstore

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")

# RFC 7748 section 6.1 (Alice). WireGuard's keys are Curve25519 keys, so this is
# the vector `wg pubkey` reproduces — an independent check that the encoding
# here is the one wg-quick reads, not merely self-consistent.
RFC7748_PRIVATE = bytes.fromhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
RFC7748_PUBLIC = bytes.fromhex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_store_lives_under_home(tmp_aureax_home) -> None:
    """$HOME is read per call, so the tests' redirect reaches it."""
    assert wgstore.store_dir() == tmp_aureax_home / ".aureax" / "wireguard"
    key, pub = wgstore.key_paths("shard-a1")
    assert key == tmp_aureax_home / ".aureax" / "wireguard" / "shard-a1.key"
    assert pub == tmp_aureax_home / ".aureax" / "wireguard" / "shard-a1.pub"


@posix_only
def test_ensure_key_writes_a_private_and_a_public_file_at_their_modes(tmp_aureax_home) -> None:
    pub = wgstore.ensure_key("shard-a1")
    key_path, pub_path = wgstore.key_paths("shard-a1")

    assert _mode(wgstore.store_dir()) == 0o700
    assert _mode(key_path) == 0o600, "the private key must never be group- or world-readable"
    assert _mode(pub_path) == 0o644, "a public key is not a secret"
    assert pub_path.read_text().strip() == pub


def test_keys_are_base64_of_32_raw_bytes_like_wg_genkey(tmp_aureax_home) -> None:
    """44 characters with one '=' of padding: what `wg genkey` / `wg pubkey` print."""
    pub = wgstore.ensure_key("shard-a1")
    priv = wgstore.private_key("shard-a1")

    for value in (priv, pub):
        assert len(value) == 44 and value.endswith("=")
        assert len(base64.b64decode(value, validate=True)) == 32


def test_the_public_key_is_cryptographys_own_derivation(tmp_aureax_home) -> None:
    pub = wgstore.ensure_key("shard-a1")
    raw = base64.b64decode(wgstore.private_key("shard-a1"))
    expected = X25519PrivateKey.from_private_bytes(raw).public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    assert base64.b64decode(pub) == expected
    assert wgstore.public_key("shard-a1") == pub


def test_derivation_matches_the_rfc_7748_vector() -> None:
    derived = wgstore.derive_public_key(base64.b64encode(RFC7748_PRIVATE).decode())
    assert base64.b64decode(derived) == RFC7748_PUBLIC


def test_ensure_key_is_idempotent_and_repairs_a_missing_public_copy(tmp_aureax_home) -> None:
    first = wgstore.ensure_key("shard-a1")
    priv = wgstore.private_key("shard-a1")
    _, pub_path = wgstore.key_paths("shard-a1")
    pub_path.unlink()

    assert wgstore.ensure_key("shard-a1") == first
    assert wgstore.private_key("shard-a1") == priv, "ensure_key must never replace a key"
    assert pub_path.read_text().strip() == first


def test_absent_keys_read_as_none_and_status_mints_nothing(tmp_aureax_home) -> None:
    assert wgstore.private_key("shard-a1") is None
    assert wgstore.public_key("shard-a1") is None
    assert not wgstore.store_dir().exists(), "a read created the store"


def test_rotation_keeps_exactly_one_predecessor(tmp_aureax_home) -> None:
    first_pub = wgstore.ensure_key("shard-a1")
    first_priv = wgstore.private_key("shard-a1")

    second_pub = wgstore.rotate_key("shard-a1")
    assert second_pub != first_pub
    assert wgstore.public_key("shard-a1") == second_pub
    assert wgstore.previous_private_key("shard-a1") == first_priv
    key_path, pub_path = wgstore.key_paths("shard-a1")
    assert pub_path.read_text().strip() == second_pub

    second_priv = wgstore.private_key("shard-a1")
    wgstore.rotate_key("shard-a1")
    assert wgstore.previous_private_key("shard-a1") == second_priv, ".prev holds only the last one"
    if os.name == "posix":
        assert _mode(key_path.with_name(key_path.name + ".prev")) == 0o600


def test_a_corrupt_private_key_is_named_not_treated_as_absent(tmp_aureax_home) -> None:
    """None would read as "not minted yet" and generate would leave the bad file
    in place while every deploy refused for a missing key."""
    wgstore.ensure_key("shard-a1")
    key_path, _ = wgstore.key_paths("shard-a1")
    key_path.write_text("not-a-key\n")

    with pytest.raises(ValueError, match="aureax wg regenerate --node shard-a1"):
        wgstore.private_key("shard-a1")
    with pytest.raises(ValueError):
        wgstore.ensure_key("shard-a1")
    # The rotation is the repair, and it must work on exactly this state.
    wgstore.rotate_key("shard-a1")
    assert wgstore.public_key("shard-a1") is not None


@pytest.mark.parametrize("node_id", ["", "../evil", "a/b", "a\\b", "..", ".hidden", "a..b"])
def test_a_node_id_that_is_not_one_path_segment_is_refused(tmp_aureax_home, node_id) -> None:
    """`aureax wg regenerate --node` hands an operator string straight here."""
    with pytest.raises(ValueError):
        wgstore.key_paths(node_id)
    with pytest.raises(ValueError):
        wgstore.ensure_key(node_id)
    with pytest.raises(ValueError):
        wgstore.rotate_key(node_id)
    assert not any(tmp_aureax_home.rglob("*.key")), "a refused id still wrote a key"


def test_two_nodes_never_share_a_key(tmp_aureax_home) -> None:
    assert wgstore.ensure_key("shard-a1") != wgstore.ensure_key("shard-a2")
