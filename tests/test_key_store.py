"""The per-node API key store under ~/.aureax/keys.

Two properties carry the whole design. First, permissions: the orchestrator
host is a shared workstation, and a key file that is briefly world-readable
during creation is world-readable long enough to matter — hence an O_EXCL temp
file at 0600 published with os.link (create) or os.replace (rotate), rather
than open()/write() under the ambient umask.
Second, idempotence under concurrency: `aureax generate` resolves ${api_key}
per target and nothing serialises those calls, so two writers for the same node
must converge on ONE key. A node whose config carries a key its key file no
longer holds cannot be authenticated against by anything.

A node now holds two keys — llama-server's and its MCP server's — as two files
in that one directory, so a third property joins them: the kinds must not
touch each other. Rotating one leaves the other's file and its .prev alone, and
the node id that names them is guarded, because "edge-01.mcp" is a legal node
id whose llama key file is another node's MCP key file.

Every test here runs against `tmp_aureax_home`, which redirects $HOME —
keystore.store_dir() is a function, not a module constant, precisely so this
works.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from orchestrator.config import keystore

# 32 CSPRNG bytes -> 43 base64url characters.
KEY_RE = re.compile(r"^aureax_sk_[A-Za-z0-9_-]{43}$")

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_ensure_key_creates_a_key_under_the_tmp_home(tmp_aureax_home) -> None:
    key = keystore.ensure_key("edge-01")

    assert KEY_RE.fullmatch(key), key
    assert keystore.store_dir() == tmp_aureax_home / ".aureax" / "keys"
    assert keystore.key_path("edge-01").exists()
    assert keystore.read_key("edge-01") == key
    assert keystore.has_key("edge-01")


@posix_only
def test_key_file_is_0600_in_a_0700_directory(tmp_aureax_home) -> None:
    """0600 from the first byte — never a window at the umask default."""
    keystore.ensure_key("edge-01")

    assert _mode(keystore.key_path("edge-01")) == 0o600
    assert _mode(keystore.store_dir()) == 0o700


def test_ensure_key_is_idempotent(tmp_aureax_home) -> None:
    """Regeneration happens on every `aureax generate`; it must not re-key.

    A second key would mean the node's freshly generated config no longer
    matches the running llama-server, so every in-pod client 401s until the pod
    restarts.
    """
    first = keystore.ensure_key("edge-01")
    assert keystore.ensure_key("edge-01") == first
    assert keystore.ensure_key("edge-01") == first


def test_key_contains_no_character_llama_cpp_would_mangle(tmp_aureax_home) -> None:
    """llama.cpp splits on commas, treats '#' as a comment and never trims.

    The generator has to stay inside the schema's own accepted alphabet, or
    generation would produce keys its validator rejects.
    """
    for i in range(50):
        key = keystore.ensure_key(f"n{i}")
        assert not any(c in key for c in ",# \t\r\n")
        assert KEY_RE.fullmatch(key)


def test_generate_key_produces_distinct_values() -> None:
    keys = {keystore.generate_key() for _ in range(200)}
    assert len(keys) == 200
    assert all(k.startswith("aureax_sk_") for k in keys)


# --- rotation ----------------------------------------------------------------


def test_rotate_key_replaces_the_key_and_retains_the_predecessor(tmp_aureax_home) -> None:
    """The .prev file covers the window between rotate and apply.

    A rotation that has been staged but not applied leaves the running
    llama-server enforcing the OLD key while every regenerated config already
    carries the new one, so the dashboard has to be able to show the old one.
    """
    old = keystore.ensure_key("edge-01")
    new = keystore.rotate_key("edge-01")

    assert KEY_RE.fullmatch(new)
    assert new != old
    assert keystore.read_key("edge-01") == new
    assert keystore.previous_key("edge-01") == old


@posix_only
def test_rotated_key_files_are_still_0600(tmp_aureax_home) -> None:
    """os.replace preserves the SOURCE file's mode, so the temp file's 0600 is
    what protects the rotated key — a plain write would land at the umask."""
    keystore.ensure_key("edge-01")
    keystore.rotate_key("edge-01")

    assert _mode(keystore.key_path("edge-01")) == 0o600
    assert _mode(keystore.store_dir() / "edge-01.key.prev") == 0o600


@posix_only
def test_a_key_is_fsynced_before_the_name_that_publishes_it(
    tmp_aureax_home, monkeypatch
) -> None:
    """A crash between the write and the publish must not leave a zero-length
    key at the real name: _read_state reports an empty file as KEY_ABSENT while
    the running server still enforces the key it replaced, so the node reads as
    unkeyed and every regenerated config disagrees with it.

    Both publish paths — _create_key_if_absent's os.link and _replace_key's
    write_private_file — sync the bytes first.
    """
    synced: list[int] = []
    real_fsync = os.fsync

    def spy(fd):
        synced.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)

    keystore.ensure_key("edge-01")
    assert synced, "a new key was published without an fsync"

    synced.clear()
    keystore.rotate_key("edge-01")
    assert synced, "a rotated key was published without an fsync"

    assert keystore.key_state("edge-01") == keystore.KEY_PRESENT


def test_rotate_on_a_node_with_no_key_just_creates_one(tmp_aureax_home) -> None:
    key = keystore.rotate_key("fresh")
    assert KEY_RE.fullmatch(key)
    assert keystore.previous_key("fresh") is None


def test_repeated_rotation_keeps_exactly_one_predecessor(tmp_aureax_home) -> None:
    """Exactly one, not a growing history: the store is not an audit log, and
    every retained key is a key that still authenticates somewhere."""
    first = keystore.ensure_key("edge-01")
    second = keystore.rotate_key("edge-01")
    third = keystore.rotate_key("edge-01")

    assert keystore.read_key("edge-01") == third
    assert keystore.previous_key("edge-01") == second
    assert first not in (second, third)
    assert len(list(keystore.store_dir().glob("edge-01.key*"))) == 2


# --- deletion ----------------------------------------------------------------


def test_delete_key_removes_both_files_and_reports_what_it_found(tmp_aureax_home) -> None:
    keystore.ensure_key("edge-01")
    keystore.rotate_key("edge-01")

    assert keystore.delete_key("edge-01") is True
    assert not keystore.key_path("edge-01").exists()
    assert not (keystore.store_dir() / "edge-01.key.prev").exists()
    assert keystore.read_key("edge-01") is None
    assert keystore.has_key("edge-01") is False

    # Deleting again is not an error, and says nothing was there.
    assert keystore.delete_key("edge-01") is False


def test_delete_key_of_an_unknown_node_is_false(tmp_aureax_home) -> None:
    assert keystore.delete_key("never-existed") is False


# --- non-secret displays ------------------------------------------------------


def test_fingerprint_and_last4_never_expose_the_key(tmp_aureax_home) -> None:
    """These are what the CLI's Auth column and the dashboard print.

    The full key must never reach terminal scrollback or shell history, so the
    display helpers have to be derived values, not truncated secrets that grow
    back if someone widens the slice.
    """
    key = keystore.ensure_key("edge-01")

    fp = keystore.fingerprint("edge-01")
    assert fp is not None and fp.startswith("sha256:")
    assert key not in fp
    assert len(fp) == len("sha256:") + 12

    assert keystore.last4("edge-01") == key[-4:]


def test_fingerprint_is_stable_and_changes_on_rotation(tmp_aureax_home) -> None:
    keystore.ensure_key("edge-01")
    before = keystore.fingerprint("edge-01")
    assert keystore.fingerprint("edge-01") == before

    keystore.rotate_key("edge-01")
    assert keystore.fingerprint("edge-01") != before


def test_display_helpers_return_none_without_a_key(tmp_aureax_home) -> None:
    """'No key' must be distinguishable from 'a key I cannot show' — the CLI
    prints "key pending — run `aureax generate`" off exactly this."""
    assert keystore.fingerprint("nobody") is None
    assert keystore.last4("nobody") is None
    assert keystore.read_key("nobody") is None
    assert keystore.previous_key("nobody") is None
    assert keystore.has_key("nobody") is False


# --- concurrency --------------------------------------------------------------


@pytest.mark.parametrize("attempt", range(5))
def test_concurrent_ensure_key_converges_on_one_key(tmp_aureax_home, attempt) -> None:
    """Two writers for the same node must end up holding the SAME key.

    `aureax generate` walks every target and resolves ${api_key} per node, and
    nothing serialises that; two generates at once (the CLI and the dashboard's
    /api/config/regenerate, say) race on the same node. If a create race let
    both writers win, one of them would return a key that is no longer in the
    file — a config shipped with a key the node will never accept, which
    surfaces only as 401s after the pod restarts.

    ensure_key handles this without a lock file: each writer stages its own
    uniquely-named 0600 temp file and publishes it with os.link, which refuses
    to create a name that already exists — so the loser re-reads the winner's
    published key instead of overwriting it with its own.

    Repeated a few times because a race that loses only sometimes still loses.
    """
    node = f"racy-{attempt}"
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = list(pool.map(lambda _i: keystore.ensure_key(node), range(8)))

    assert len(set(keys)) == 1, f"racing writers produced {len(set(keys))} keys"
    assert keys[0] == keystore.read_key(node), "a caller returned a key not on disk"
    assert KEY_RE.fullmatch(keys[0])


def test_a_leftover_temp_file_does_not_wedge_key_creation(tmp_aureax_home) -> None:
    """Leftover temp debris must not block key creation.

    Writers now stage through per-writer unique temp names and publish with an
    os.link create-if-absent, so there is no shared lock file to go stale at
    all; the legacy shared "<node>.key.tmp" name (which an older keystore
    could leave behind after a crash) is swept on a successful create.
    """
    keystore.store_dir().mkdir(parents=True, exist_ok=True)
    stale = keystore.store_dir() / "edge-01.key.tmp"
    stale.write_text("junk\n")

    key = keystore.ensure_key("edge-01")
    assert KEY_RE.fullmatch(key)
    assert key == keystore.read_key("edge-01")
    assert not stale.exists()


@posix_only
def test_an_existing_store_directory_is_tightened_to_0700(tmp_aureax_home) -> None:
    """mkdir's `mode` is honoured only when it CREATES the leaf.

    So a ~/.aureax/keys restored from a backup, made by a hand-run `mkdir -p`,
    or created by a keystore version predating that argument keeps its 0755 and
    lists which nodes hold keys. The files themselves stay 0600 either way.
    """
    store = tmp_aureax_home / ".aureax" / "keys"
    store.mkdir(parents=True)
    os.chmod(store, 0o755)

    keystore.ensure_key("edge-01")

    assert _mode(keystore.store_dir()) == 0o700
    assert _mode(keystore.key_path("edge-01")) == 0o600


def test_a_key_file_that_cannot_be_read_is_not_reported_as_absent(tmp_aureax_home, caplog) -> None:
    """"No key" and "a key I cannot show" call for opposite operator actions:
    run `aureax generate`, or fix a .key left root-owned by a sudo run.

    The display helpers must still not raise — they back the CLI's Auth column
    and every field of the dashboard's Access tab — so they answer None and
    key_state carries the difference. A directory stands in for the unreadable
    file because chmod 000 does not stop root.
    """
    keystore._warned_unreadable.clear()
    keystore.store_dir().mkdir(parents=True, exist_ok=True)
    keystore.key_path("edge-01").mkdir()

    with caplog.at_level(logging.WARNING, logger="aureax.config.keystore"):
        assert keystore.read_key("edge-01") is None
        assert keystore.fingerprint("edge-01") is None
        assert keystore.last4("edge-01") is None
        assert keystore.has_key("edge-01") is False
        assert keystore.key_state("edge-01") == keystore.KEY_UNREADABLE

    assert keystore.key_state("nobody") == keystore.KEY_ABSENT
    assert keystore.key_state("edge-01") != keystore.key_state("nobody")

    # Logged, but once per file: these helpers run per node on every `aureax
    # nodes` row and every 20s Access-tab poll.
    warnings = [r for r in caplog.records if "edge-01.key" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_a_key_file_with_undecodable_bytes_is_unreadable_not_present(
    tmp_aureax_home, caplog
) -> None:
    """Decoding is strict, so corrupt bytes land in KEY_UNREADABLE.

    Replacing them instead would report KEY_PRESENT with a U+FFFD-mangled
    value: fingerprint() and last4() would display a key the node does not
    hold, and generation would substitute it and then fail the api_key shape
    check — an error naming the wrong problem.
    """
    keystore._warned_unreadable.clear()
    keystore.store_dir().mkdir(parents=True, exist_ok=True)
    keystore.key_path("edge-01").write_bytes(b"aureax_sk_\xff\xfe_truncated\n")

    with caplog.at_level(logging.WARNING, logger="aureax.config.keystore"):
        assert keystore.key_state("edge-01") == keystore.KEY_UNREADABLE
        assert keystore.read_key("edge-01") is None
        assert keystore.fingerprint("edge-01") is None


def test_key_state_is_present_for_a_normal_key(tmp_aureax_home) -> None:
    keystore.ensure_key("edge-01")
    assert keystore.key_state("edge-01") == keystore.KEY_PRESENT


# --- key kinds ----------------------------------------------------------------
# A node now holds two credentials: llama-server's key and its MCP server's.
# They are separate files so rotating one cannot disturb the other, and they
# share the aureax_sk_ prefix so the leak greps that already exist keep working
# — the file name, not the value, carries the kind.


def test_the_two_kinds_are_separate_files_with_separate_values(tmp_aureax_home) -> None:
    llama = keystore.ensure_key("edge-01")
    mcp = keystore.ensure_key("edge-01", kind=keystore.KIND_MCP)

    assert keystore.key_path("edge-01") == keystore.store_dir() / "edge-01.key"
    assert keystore.key_path("edge-01", kind=keystore.KIND_MCP) == (
        keystore.store_dir() / "edge-01.mcp.key"
    )
    assert llama != mcp
    assert KEY_RE.fullmatch(mcp), mcp
    assert keystore.read_key("edge-01") == llama
    assert keystore.read_key("edge-01", kind=keystore.KIND_MCP) == mcp


def test_the_default_kind_is_llama_so_existing_callers_are_unchanged(tmp_aureax_home) -> None:
    """Every call site that predates kinds passes a bare node id.

    Renaming the llama file, or making `kind` required, would silently 401 a
    fleet whose configs were generated before the change — the node's config
    still carries the old key while the store answers with a different one.
    """
    key = keystore.ensure_key("edge-01")

    assert keystore.read_key("edge-01") == keystore.read_key("edge-01", kind=keystore.KIND_LLAMA)
    assert keystore.key_path("edge-01") == keystore.key_path("edge-01", kind=keystore.KIND_LLAMA)
    assert keystore.fingerprint("edge-01") == keystore.fingerprint(
        "edge-01", kind=keystore.KIND_LLAMA
    )
    assert keystore.last4("edge-01") == key[-4:]


def test_rotating_one_kind_leaves_the_other_untouched(tmp_aureax_home) -> None:
    """`apply --rotate-key` re-keys both because one pod restart applies both,
    but it does so with two calls — a rotation that also silently re-keyed the
    other service would 401 whichever half was not being re-applied."""
    llama = keystore.ensure_key("edge-01")
    mcp = keystore.ensure_key("edge-01", kind=keystore.KIND_MCP)

    new_mcp = keystore.rotate_key("edge-01", kind=keystore.KIND_MCP)

    assert new_mcp not in (llama, mcp)
    assert keystore.read_key("edge-01") == llama, "llama key moved during an MCP rotation"
    assert keystore.previous_key("edge-01") is None
    assert keystore.previous_key("edge-01", kind=keystore.KIND_MCP) == mcp


def test_the_mcp_predecessor_file_sits_beside_its_key(tmp_aureax_home) -> None:
    """with_name, not with_suffix: with_suffix replaces the LAST suffix, so
    "edge-01.mcp.key" would become "edge-01.mcp.prev" — a name previous_key
    never looks for, losing the old key exactly in the rotate-to-apply window
    the .prev file exists for."""
    keystore.ensure_key("edge-01", kind=keystore.KIND_MCP)
    keystore.rotate_key("edge-01", kind=keystore.KIND_MCP)

    assert (keystore.store_dir() / "edge-01.mcp.key.prev").exists()
    assert not (keystore.store_dir() / "edge-01.mcp.prev").exists()
    assert keystore.previous_key("edge-01", kind=keystore.KIND_MCP) is not None


@posix_only
def test_both_kinds_are_0600_after_creation_and_rotation(tmp_aureax_home) -> None:
    keystore.ensure_key("edge-01")
    keystore.ensure_key("edge-01", kind=keystore.KIND_MCP)
    keystore.rotate_key("edge-01")
    keystore.rotate_key("edge-01", kind=keystore.KIND_MCP)

    store = keystore.store_dir()
    for name in ("edge-01.key", "edge-01.key.prev", "edge-01.mcp.key", "edge-01.mcp.key.prev"):
        assert _mode(store / name) == 0o600, name
    assert _mode(store) == 0o700


def test_state_and_deletion_are_per_kind(tmp_aureax_home) -> None:
    keystore.ensure_key("edge-01")

    assert keystore.key_state("edge-01") == keystore.KEY_PRESENT
    assert keystore.key_state("edge-01", kind=keystore.KIND_MCP) == keystore.KEY_ABSENT
    assert keystore.has_key("edge-01", kind=keystore.KIND_MCP) is False

    keystore.ensure_key("edge-01", kind=keystore.KIND_MCP)
    assert keystore.delete_key("edge-01", kind=keystore.KIND_MCP) is True
    assert keystore.has_key("edge-01", kind=keystore.KIND_MCP) is False
    assert keystore.has_key("edge-01") is True, "deleting the MCP key removed the llama key"


def test_an_unknown_kind_raises_rather_than_falling_back(tmp_aureax_home) -> None:
    """A typo'd kind that fell back to the llama file would hand the MCP server
    llama-server's key — and both ends would agree on it, so nothing anywhere
    would report an error."""
    with pytest.raises(ValueError, match="unknown key kind"):
        keystore.key_path("edge-01", kind="mpc")


# --- node-id guard ------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad", "why"),
    [
        ("a/../b", "a separator escapes ~/.aureax/keys entirely"),
        ("../../etc/passwd", "so does a plain traversal"),
        ("..", "the parent directory itself"),
        ("", "empty would name the directory's own '.key'"),
        ("x.mcp", "collides with node x's MCP key file"),
        ("edge-01.mcp", "the collision that matters on the real fleet"),
    ],
)
def test_node_ids_that_would_escape_or_collide_are_refused(tmp_aureax_home, bad, why) -> None:
    """The dashboard's key routes take the node id straight from the URL, so
    the guard belongs at the path builder, not only at the config boundary.

    ".mcp" is the case the config boundary cannot catch: SSHConfig's path-segment
    pattern allows dots, so "edge-01.mcp" is a legal node id that would own
    edge-01's MCP key file — two nodes holding one credential, and rotating
    either 401s the other.
    """
    with pytest.raises(ValueError):
        keystore.key_path(bad)
    with pytest.raises(ValueError):
        keystore.ensure_key(bad, kind=keystore.KIND_MCP)


def test_ordinary_fleet_node_ids_still_pass(tmp_aureax_home) -> None:
    """The guard must not reject anything the real fleet file uses — dots,
    dashes and digits all appear in shipped node ids."""
    for node_id in ("edge-01", "shard-head-b", "shard-w7", "gpu-01", "node.local-1"):
        assert keystore.key_path(node_id).name.endswith(".key")
        assert KEY_RE.fullmatch(keystore.ensure_key(node_id))
