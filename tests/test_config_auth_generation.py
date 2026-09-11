"""Where a node's API key comes from at generate time, and what a bad one costs.

Auth is enabled by ONE line in a node's template — `api_key: ${api_key}` — and
everything else follows from it: the key file is created iff the template asks
for one, the CLI's Auth column reads the same signal, and the dashboard's
Access tab shows the key for exactly those nodes. Nothing else may create a key
file, because a shard worker that grew one would look auth-enabled while
running rpc-server, which has no key concept at all.

The other half is the schema validator. llama.cpp does not reject a malformed
key: a comma SPLITS the value into several valid keys, '#' starts a comment in
--api-key-file format, and whitespace is kept verbatim. Every one of those
ships a server enforcing a key nobody holds, so the rejection has to happen
here, at generate time, on the Mac.

The same story now runs twice per node: `api_key: ${mcp_api_key}` under
mcp.server turns on the MCP server's bearer check, and ${mcp_endpoints} copies
each peer's key into every planner's remote_servers. All three api_key fields
share one validator and one table of rejected values below, because one minted
string travels all three paths — a value only one of them accepted would
produce a fleet that validates node by node and fails as a whole.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError
from ruamel.yaml import YAML

from orchestrator.config import keystore
from orchestrator.config.schema import (
    MIN_LLAMA_BUILD,
    EdgeConfig,
    LlamaServerConfig,
    MCPServer,
    RemoteServer,
    TLSPaths,
    mcp_client_enabled,
    mcp_server_enabled,
    parse_llama_build,
)
from orchestrator.config.templating import (
    generate_all_edge_configs,
    template_references_api_key,
    template_references_mcp_api_key,
    template_references_mcp_endpoints,
)
from tests.conftest import write_network_yaml

yaml = YAML()

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")

# A standalone edge node. The api_key line is the only thing that varies.
KEYED_YAML = """
node: {id: "${node_id}", role: "edge_executor", control_plane: {endpoint: "https://${orchestrator_ip}:6443"}}
model:
  name: M
  filename: m.gguf
  server: {port: 8080, api_key: "${api_key}"}
mcp: {server: {enabled: true}}
"""

KEYLESS_YAML = KEYED_YAML.replace(', api_key: "${api_key}"', "")


def _write_fleet(tmp_path: Path, files: dict[str, str], targets: list[tuple[str, str]]) -> Path:
    """Write templates + aureax-network.yaml into tmp_path/configs.

    ``files`` is filename -> YAML text; ``targets`` is (node_id, config_file),
    so two targets can deliberately point at the same template.
    """
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    for name, text in files.items():
        (configs / name).write_text(text)

    return write_network_yaml(
        configs,
        [
            {"node_id": n, "host": f"10.77.0.{i + 1}", "config_file": f}
            for i, (n, f) in enumerate(targets)
        ],
    )


def _generated(out_dir: Path, node_id: str) -> dict:
    with open(out_dir / f"aureax-edge-{node_id}.yaml") as f:
        return yaml.load(f)


# --- key resolution ----------------------------------------------------------


def test_template_placeholder_resolves_to_the_nodes_own_key(tmp_aureax_home, tmp_path) -> None:
    """The generated config must carry exactly what the key store holds.

    Any divergence is invisible until the pod restarts: llama-server enforces
    the key from its config, and every in-pod client reads the same file, so a
    config/store mismatch surfaces as the orchestrator 401ing instead.
    """
    net = _write_fleet(tmp_path, {"edge.yaml": KEYED_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    key = keystore.read_key("edge-01")
    assert key is not None and key.startswith("aureax_sk_")
    assert _generated(out, "edge-01")["model"]["server"]["api_key"] == key


def test_regeneration_keeps_the_same_key(tmp_aureax_home, tmp_path) -> None:
    """`aureax generate` runs on every deploy; re-keying there would silently
    invalidate the key the running llama-server is enforcing."""
    net = _write_fleet(tmp_path, {"edge.yaml": KEYED_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")
    first = _generated(out, "edge-01")["model"]["server"]["api_key"]
    generate_all_edge_configs(net, out, "10.77.0.99")

    assert _generated(out, "edge-01")["model"]["server"]["api_key"] == first


def test_two_targets_sharing_one_template_get_different_keys(
    tmp_aureax_home, tmp_path
) -> None:
    """Keys are PER NODE even where the template is shared.

    Both shard heads share configs/aureax-shard-head-35b.yaml, and the six
    workers share one file too — sharing a template is sanctioned for targets
    that differ only by placeholder-resolved values (see ARCHITECTURE.md,
    "2. Configuration: per-node files, not one shared template"). A key
    resolved per FILE rather than per TARGET would hand the whole fleet one
    key, so rotating any node would rotate all of them.
    """
    net = _write_fleet(
        tmp_path,
        {"shared.yaml": KEYED_YAML},
        [("edge-01", "shared.yaml"), ("edge-02", "shared.yaml")],
    )
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    k1 = _generated(out, "edge-01")["model"]["server"]["api_key"]
    k2 = _generated(out, "edge-02")["model"]["server"]["api_key"]
    assert k1 != k2
    assert k1 == keystore.read_key("edge-01")
    assert k2 == keystore.read_key("edge-02")


def test_a_template_without_the_placeholder_creates_no_key(tmp_aureax_home, tmp_path) -> None:
    """Merely generating must never mint a key.

    Shard workers run rpc-server, which has no key concept, and the frozen 122B
    rollback pair is deliberately auth-free. A key file appearing for them would
    make `aureax nodes` report auth that does not exist.
    """
    net = _write_fleet(tmp_path, {"edge.yaml": KEYLESS_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    assert "api_key" not in _generated(out, "edge-01")["model"]["server"]
    assert keystore.read_key("edge-01") is None
    assert not keystore.store_dir().exists() or list(keystore.store_dir().iterdir()) == []


def test_only_the_keyed_node_of_a_mixed_fleet_gets_a_key(tmp_aureax_home, tmp_path) -> None:
    net = _write_fleet(
        tmp_path,
        {"keyed.yaml": KEYED_YAML, "plain.yaml": KEYLESS_YAML},
        [("edge-01", "keyed.yaml"), ("edge-02", "plain.yaml")],
    )

    generate_all_edge_configs(net, tmp_path / "out", "10.77.0.99")

    assert keystore.has_key("edge-01")
    assert not keystore.has_key("edge-02")


# --- generated artefacts are secrets now --------------------------------------


@posix_only
def test_generated_configs_and_their_directory_are_locked_down(
    tmp_aureax_home, tmp_path
) -> None:
    """A resolved config now embeds a bearer token, so ~/.aureax/configs is no
    longer a directory of harmless YAML.

    Tightened unconditionally, including for keyless nodes: doing it only for
    keyed ones would make a file's MODE advertise which nodes have auth.
    """
    net = _write_fleet(
        tmp_path,
        {"keyed.yaml": KEYED_YAML, "plain.yaml": KEYLESS_YAML},
        [("edge-01", "keyed.yaml"), ("edge-02", "plain.yaml")],
    )
    out = tmp_path / "out"

    paths = generate_all_edge_configs(net, out, "10.77.0.99")

    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    for p in paths:
        assert stat.S_IMODE(p.stat().st_mode) == 0o600, p


@posix_only
def test_a_generated_config_is_0600_before_its_first_byte(
    tmp_aureax_home, tmp_path, monkeypatch
) -> None:
    """A plain open() creates at the umask (0644 in practice) and the whole
    api_key is written before the chmod lands. Generation goes through
    atomic.write_private_file for exactly that reason — the same helper the key
    store publishes with — so every file it creates under the output directory
    is 0600 before it holds a byte.
    """
    net = _write_fleet(tmp_path, {"edge.yaml": KEYED_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"

    from orchestrator.config import atomic

    modes: list[int] = []
    real_open = atomic.os.open

    def spy(path, flags, mode=0o777, **kw):
        fd = real_open(path, flags, mode, **kw)
        if flags & os.O_CREAT and str(path).startswith(str(out)):
            modes.append(stat.S_IMODE(os.fstat(fd).st_mode))
        return fd

    monkeypatch.setattr(atomic.os, "open", spy)
    umask = os.umask(0o022)
    try:
        generate_all_edge_configs(net, out, "10.77.0.99")
    finally:
        os.umask(umask)

    assert modes == [0o600], "the key was written into a file created at the umask"


def test_a_dump_that_fails_partway_leaves_the_previous_config_intact(
    tmp_aureax_home, tmp_path, monkeypatch
) -> None:
    """Write to a temp file and rename, never truncate in place.

    A truncated YAML prefix is usually still VALID YAML, so an in-place write
    that dies partway leaves a silently partial config behind — and
    cache.load_node_config_resolved prefers that generated file over the
    template for every read afterwards.
    """
    net = _write_fleet(tmp_path, {"edge.yaml": KEYED_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"
    (good,) = generate_all_edge_configs(net, out, "10.77.0.99")
    before = good.read_text()

    from orchestrator.config import templating as templating_mod

    def die_halfway(data, stream, **kw):
        stream.write("node:\n  id: edge-01\n")
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(templating_mod.yaml, "dump", die_halfway)
    with pytest.raises(RuntimeError):
        generate_all_edge_configs(net, out, "10.77.0.99")

    assert good.read_text() == before
    assert list(out.glob("*.tmp")) == [], "a failed write left its temp file behind"


# --- template_references_api_key ---------------------------------------------


def test_template_references_api_key_reads_the_placeholder(tmp_path) -> None:
    """One helper decides what "auth is on" means, so the CLI's Auth column,
    the dashboard's Access tab and generation itself cannot drift apart."""
    (tmp_path / "keyed.yaml").write_text(KEYED_YAML)
    (tmp_path / "plain.yaml").write_text(KEYLESS_YAML)

    assert template_references_api_key(tmp_path / "keyed.yaml") is True
    assert template_references_api_key(tmp_path / "plain.yaml") is False


def test_template_references_api_key_is_false_for_an_unreadable_file(tmp_path) -> None:
    """Auth-off is the safe answer for a display helper: `aureax nodes` runs
    against configs that may not exist yet and must not raise."""
    assert template_references_api_key(tmp_path / "does-not-exist.yaml") is False

    (tmp_path / "broken.yaml").write_text("model: {server: {api_key: [unclosed\n")
    assert template_references_api_key(tmp_path / "broken.yaml") is False


def test_a_literal_key_in_a_template_is_not_a_placeholder_reference(tmp_path) -> None:
    """Only ${api_key} counts. A hand-pasted literal is a valid (if unmanaged)
    key, and claiming it as store-managed would offer rotation that does
    nothing to the file the node actually reads."""
    (tmp_path / "literal.yaml").write_text(
        KEYED_YAML.replace('"${api_key}"', '"aureax_sk_hand_written_key_value"')
    )
    assert template_references_api_key(tmp_path / "literal.yaml") is False


# --- schema validation --------------------------------------------------------

# Values llama.cpp (and, for the MCP fields, httpx and the bearer middleware)
# accept without complaint while authenticating against something other than
# what was configured. One table, shared by all three api_key fields: the SAME
# minted string is written into model.server.api_key on one node, into
# mcp.server.api_key on another and into every planner's
# mcp.client.remote_servers[].api_key, so a value one validator rejected and
# another waved through would produce a fleet that validates node by node and
# fails as a whole.
MANGLED_KEYS = [
    ("", "empty string is falsy downstream: auth silently disabled"),
    ("short", "under 16 chars"),
    ("aureax_sk_a,aureax_sk_b_padding", "a comma SPLITS it into two valid keys"),
    ("aureax_sk has a space in it", "whitespace is never trimmed by llama.cpp"),
    ("#aureax_sk_commented_out_key", "'#' starts a comment in --api-key-file format"),
    ("aureax_sk_trailing_newline\n", "a trailing newline is kept verbatim"),
    ("aureax_sk_tab\there_padding", "tabs are kept verbatim too"),
    ("aureax_sk_" + "A" * 600, "over the 512-char ceiling"),
]

VALID_KEY = "aureax_sk_" + "A" * 43


def test_a_valid_key_passes() -> None:
    assert LlamaServerConfig(api_key="aureax_sk_" + "A" * 43).api_key.startswith("aureax_sk_")


def test_omitting_the_key_is_how_auth_is_disabled() -> None:
    """None is the auth-off spelling, and must stay valid — the 122B rollback
    pair carries no key at all."""
    assert LlamaServerConfig().api_key is None
    assert LlamaServerConfig(api_key=None).api_key is None


@pytest.mark.parametrize(("bad", "why"), MANGLED_KEYS)
def test_keys_llama_cpp_would_mangle_are_rejected(bad, why) -> None:
    """Each of these produces a server enforcing a key nobody holds — and none
    of them is an error anywhere downstream, which is why they are rejected on
    the Mac at generate time."""
    with pytest.raises(ValidationError, match="api_key"):
        LlamaServerConfig(api_key=bad)


def test_alias_with_a_comma_is_rejected() -> None:
    """b10375 parses --alias as a comma-separated LIST, so "a,b" registers two
    model ids and /v1/models reports something nobody configured."""
    with pytest.raises(ValidationError, match="alias"):
        LlamaServerConfig(alias="a,b")
    assert LlamaServerConfig(alias="qwen3.5-35b").alias == "qwen3.5-35b"


# --- the MCP side of the same rule --------------------------------------------
# `api_key: ${mcp_api_key}` under mcp.server is the one line that turns the
# node's MCP bearer check on, and ${mcp_endpoints} copies each peer's key into
# every planner's remote_servers. Both fields go through the same validator as
# llama's, so the alphabet cannot drift apart between them.


@pytest.mark.parametrize(("bad", "why"), MANGLED_KEYS)
def test_the_mcp_server_key_rejects_every_value_llama_would(bad, why) -> None:
    with pytest.raises(ValidationError, match=re.escape("mcp.server.api_key")):
        MCPServer(api_key=bad)


@pytest.mark.parametrize(("bad", "why"), MANGLED_KEYS)
def test_a_peer_key_in_remote_servers_rejects_every_value_llama_would(bad, why) -> None:
    """A planner's peer list is where a bad key does the most damage: one
    malformed entry is one remote the planner can never call, and the MCP
    server on the other end answers 401 with nothing in ITS log naming the
    cause."""
    with pytest.raises(ValidationError, match="remote_servers"):
        RemoteServer(name="head", url="https://h:30900/mcp", api_key=bad)


def test_omitting_the_mcp_key_is_how_mcp_auth_is_disabled() -> None:
    """None is the auth-off spelling on this side too — the shard-worker
    templates and the frozen 122B rollback pair carry no MCP key at all, and a
    worker runs no MCP server for one to protect."""
    assert MCPServer().api_key is None
    assert MCPServer(api_key=None).api_key is None
    assert RemoteServer(name="head", url="https://h:30900/mcp").api_key is None


def test_one_minted_key_validates_in_all_three_api_key_fields() -> None:
    """The generator writes a node's MCP key into that node's own config AND
    into every planner's peer list, and the keystore mints both kinds from the
    same alphabet — so a key accepted by one field and rejected by another
    would be a fleet that generates and then fails to load."""
    assert LlamaServerConfig(api_key=VALID_KEY).api_key == VALID_KEY
    assert MCPServer(api_key=VALID_KEY).api_key == VALID_KEY
    assert RemoteServer(name="head", url="https://h:30900/mcp", api_key=VALID_KEY).api_key == (
        VALID_KEY
    )

    minted = keystore.generate_key()
    assert LlamaServerConfig(api_key=minted).api_key == minted
    assert MCPServer(api_key=minted).api_key == minted


def test_the_mcp_key_is_not_confused_with_the_enabled_flag() -> None:
    """There is no separate enable flag: mcp.server.enabled controls whether the
    server RUNS, api_key controls whether it checks. A worker with enabled:
    false and no key is the shipped worker shape."""
    assert MCPServer(enabled=False).enabled is False
    assert MCPServer(enabled=False).api_key is None
    assert MCPServer(api_key=VALID_KEY).enabled is True


def test_hardening_defaults_are_the_shipped_fleet_behaviour() -> None:
    """slots OFF is the one new default that CHANGES behaviour; webui stays on."""
    server = LlamaServerConfig()
    assert server.slots is False
    assert server.webui is True
    assert server.cors_origins is None
    assert server.alias is None


# --- llama.cpp build floor ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("b10375", 10375),
        ('llama_build_docker={"build":"b9058"}', 9058),
        ("ghcr.io/ggml-org/llama.cpp:server-b10375", 10375),
        ("image not present", None),
        ("master", None),
        ("", None),
    ],
)
def test_parse_llama_build_extracts_the_build_number(text, expected) -> None:
    """None, never 0, when there is no build number in the text.

    `group doctor` compares against MIN_LLAMA_BUILD; a falsy 0 would report
    "below the floor" and send someone rebuilding images to fix what is
    actually an SSH or probe failure.
    """
    assert parse_llama_build(text) == expected


def test_the_build_floor_is_above_the_rpc_rce_fix() -> None:
    """CVE-2026-34159 was an unauthenticated RCE in rpc-server's tensor-graph
    deserialization; the pinned build must stay at or above the fixed build."""
    assert MIN_LLAMA_BUILD >= 10375
    assert parse_llama_build("b9058") < MIN_LLAMA_BUILD


def test_an_unreadable_template_says_so_before_reporting_auth_off(tmp_path, caplog) -> None:
    """This is the one place that decides whether auth is on, and both display
    consumers call it independently of generation — so a YAML syntax error in a
    head template made `aureax nodes` and the Access tab report "disabled" for a
    node whose auth is on, with nothing logged anywhere.
    """
    (tmp_path / "broken.yaml").write_text("model: {server: {api_key: [unclosed\n")

    with caplog.at_level(logging.WARNING, logger="aureax.config.templating"):
        assert template_references_api_key(tmp_path / "broken.yaml") is False
    assert any("reporting auth as OFF" in r.getMessage() for r in caplog.records)

    # A missing file stays silent: `aureax nodes` legitimately runs before
    # configs exist and asks once per target.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="aureax.config.templating"):
        assert template_references_api_key(tmp_path / "absent.yaml") is False
    assert caplog.records == []


# --- schema surface the auth and TLS work added -------------------------------


def test_the_tls_block_defaults_to_the_paths_the_deployer_installs() -> None:
    """These three strings are a contract with Deployer.deploy_configs, which
    installs ca.pem/node.pem/node.key under /etc/aureax/tls, and with the
    DaemonSets, which mount /etc/aureax read-only. The node-side names are
    fixed rather than <node_id>.pem so that renaming a node cannot strand a
    file the config no longer points at."""
    tls = TLSPaths()

    assert tls.ca_file == "/etc/aureax/tls/ca.pem"
    assert tls.cert_file == "/etc/aureax/tls/node.pem"
    assert tls.key_file == "/etc/aureax/tls/node.key"


def test_every_edge_config_carries_a_tls_block_even_when_none_was_written() -> None:
    """TLS is mandatory fleet-wide, so the model supplies the block rather than
    the template: a config that reached a node without one would be a node that
    cannot build a llama-server command line at all."""
    cfg = EdgeConfig(node={"id": "edge-01"}, model={"name": "m", "filename": "m.gguf"})

    assert cfg.tls == TLSPaths()

    override = EdgeConfig(
        node={"id": "edge-01"},
        model={"name": "m", "filename": "m.gguf"},
        tls={"ca_file": "/tmp/ca.pem"},
    )
    assert override.tls.ca_file == "/tmp/ca.pem"
    assert override.tls.cert_file == "/etc/aureax/tls/node.pem", "a partial block keeps the rest"


@pytest.mark.parametrize(
    ("cfg", "expected", "why"),
    [
        ({}, False, "no mcp block at all"),
        ({"mcp": None}, False, "`mcp:` with nothing under it is valid YAML"),
        ({"mcp": {}}, False, "an mcp block that says nothing about the client"),
        ({"mcp": {"client": None}}, False, "`client:` present but null"),
        ({"mcp": {"client": {}}}, False, "absent means NO — the opposite of the server default"),
        ({"mcp": {"client": {"enabled": False}}}, False, "explicitly off"),
        ({"mcp": {"client": {"enabled": True}}}, True, "the planner shape"),
    ],
)
def test_mcp_client_enabled_truth_table(cfg, expected, why) -> None:
    """"Is this node a planner?" is asked by the planner validator, which
    refuses a hardcoded remote_servers list — so a null block that raised here
    would take `aureax generate` down for the whole fleet, and one that
    defaulted to True would demand the ${mcp_endpoints} placeholder from every
    ordinary edge node."""
    assert mcp_client_enabled(cfg) is expected, why


def test_the_two_mcp_predicates_default_in_opposite_directions() -> None:
    """Every node serves MCP until it opts out (workers do); no node dials
    other MCP servers until it opts in (only the planner does)."""
    assert mcp_server_enabled({"mcp": {"server": {}}}) is True
    assert mcp_client_enabled({"mcp": {"client": {}}}) is False
    assert mcp_server_enabled({}) is True
    assert mcp_client_enabled({}) is False


# --- the MCP key travels the same road, independently -------------------------
# `api_key: ${mcp_api_key}` under mcp.server is the whole switch. The two kinds
# are minted from ONE scan of the template but never together: a head carries
# both lines, the planner neither, and a node with only one must not grow the
# other's file — `aureax nodes` reads key existence per kind and would report
# auth that nothing enforces.

MCP_KEYED_YAML = KEYLESS_YAML.replace(
    "mcp: {server: {enabled: true}}",
    'mcp: {server: {enabled: true, api_key: "${mcp_api_key}"}}',
)
BOTH_KEYED_YAML = KEYED_YAML.replace(
    "mcp: {server: {enabled: true}}",
    'mcp: {server: {enabled: true, api_key: "${mcp_api_key}"}}',
)


def _mcp_key(node_id: str) -> str | None:
    return keystore.read_key(node_id, kind=keystore.KIND_MCP)


def test_the_mcp_placeholder_resolves_to_the_nodes_own_mcp_key(
    tmp_aureax_home, tmp_path
) -> None:
    """Same contract as the llama key: what the config carries is what the
    server enforces, and any divergence surfaces as the dashboard's own Access
    tab 401ing against a node it just configured."""
    net = _write_fleet(tmp_path, {"edge.yaml": MCP_KEYED_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    key = _mcp_key("edge-01")
    assert key is not None and key.startswith("aureax_sk_")
    assert _generated(out, "edge-01")["mcp"]["server"]["api_key"] == key


def test_the_two_kinds_are_separate_keys_in_separate_files(
    tmp_aureax_home, tmp_path
) -> None:
    """One value serving both ports would make a leaked llama key an MCP key
    too — and rotating either would silently rotate the other."""
    net = _write_fleet(tmp_path, {"edge.yaml": BOTH_KEYED_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    generated = _generated(out, "edge-01")
    llama = generated["model"]["server"]["api_key"]
    mcp = generated["mcp"]["server"]["api_key"]

    assert llama != mcp
    assert llama == keystore.read_key("edge-01")
    assert mcp == _mcp_key("edge-01")
    assert keystore.key_path("edge-01").name == "edge-01.key"
    assert keystore.key_path("edge-01", kind=keystore.KIND_MCP).name == "edge-01.mcp.key"


@pytest.mark.parametrize(
    ("template", "wants_llama", "wants_mcp"),
    [
        (KEYED_YAML, True, False),
        (MCP_KEYED_YAML, False, True),
        (BOTH_KEYED_YAML, True, True),
        (KEYLESS_YAML, False, False),
    ],
)
def test_each_kind_is_minted_only_when_its_own_line_is_present(
    tmp_aureax_home, tmp_path, template, wants_llama, wants_mcp
) -> None:
    """The planner carries neither line (it serves no MCP and its llama port is
    open); the shard workers carry neither and validate_groups refuses both. A
    mint driven by "the template asks for A" would give them B as well."""
    net = _write_fleet(tmp_path, {"n.yaml": template}, [("edge-01", "n.yaml")])

    generate_all_edge_configs(net, tmp_path / "out", "10.77.0.99")

    assert keystore.has_key("edge-01") is wants_llama
    assert keystore.has_key("edge-01", kind=keystore.KIND_MCP) is wants_mcp


def test_regeneration_keeps_the_same_mcp_key(tmp_aureax_home, tmp_path) -> None:
    """`aureax generate` runs on every deploy and every apply. Re-keying there
    would invalidate the key every planner in the fleet is already holding."""
    net = _write_fleet(tmp_path, {"edge.yaml": MCP_KEYED_YAML}, [("edge-01", "edge.yaml")])
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")
    first = _generated(out, "edge-01")["mcp"]["server"]["api_key"]
    generate_all_edge_configs(net, out, "10.77.0.99")

    assert _generated(out, "edge-01")["mcp"]["server"]["api_key"] == first


def test_two_targets_sharing_one_template_get_different_mcp_keys(
    tmp_aureax_home, tmp_path
) -> None:
    """Both shard heads share configs/aureax-shard-head-35b.yaml. A key resolved
    per FILE would hand the fleet one MCP key, so rotating either head would
    rotate both — and the planner's config would still carry only one of them."""
    net = _write_fleet(
        tmp_path,
        {"shared.yaml": MCP_KEYED_YAML},
        [("edge-01", "shared.yaml"), ("edge-02", "shared.yaml")],
    )
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    assert (
        _generated(out, "edge-01")["mcp"]["server"]["api_key"]
        != _generated(out, "edge-02")["mcp"]["server"]["api_key"]
    )


# --- generation order must not decide what a planner ships --------------------

PLANNER_YAML = """
node: {id: "${node_id}", role: "cognitive_planner"}
model: {name: M, filename: m.gguf}
mcp:
  server: {enabled: false}
  client: {enabled: true, remote_servers: "${mcp_endpoints}"}
"""


def test_the_planner_gets_every_peer_key_even_when_generated_first(
    tmp_aureax_home, tmp_path
) -> None:
    """The PLANNER IS LISTED FIRST here, deliberately.

    generate_all_edge_configs walks enabled_targets in fleet-file order, and
    ${mcp_endpoints} reads each peer's key with read_key. Without a pre-mint
    pass the planner is resolved before the head's key exists, so the very first
    `aureax generate` on a fleet ships a planner config missing the key of every
    node that happens to sort after it — and the planner then 401s on every call
    with nothing anywhere naming the cause.
    """
    net = _write_fleet(
        tmp_path,
        {"planner.yaml": PLANNER_YAML, "head.yaml": MCP_KEYED_YAML},
        [("planner-01", "planner.yaml"), ("head-01", "head.yaml")],
    )
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    remotes = _generated(out, "planner-01")["mcp"]["client"]["remote_servers"]
    assert [r["name"] for r in remotes] == ["head-01"]
    assert remotes[0]["api_key"] == _mcp_key("head-01")
    assert remotes[0]["url"] == "https://10.77.0.2:30900/mcp"


def test_an_auth_off_peer_contributes_no_key_to_the_planner(
    tmp_aureax_home, tmp_path
) -> None:
    """Absent, not empty: RemoteServer rejects "" because it is falsy
    downstream, so an empty value would read as "this peer is open" while the
    peer enforced a key."""
    net = _write_fleet(
        tmp_path,
        {"planner.yaml": PLANNER_YAML, "head.yaml": KEYLESS_YAML},
        [("planner-01", "planner.yaml"), ("head-01", "head.yaml")],
    )
    out = tmp_path / "out"

    generate_all_edge_configs(net, out, "10.77.0.99")

    (remote,) = _generated(out, "planner-01")["mcp"]["client"]["remote_servers"]
    assert "api_key" not in remote


def test_template_references_mcp_api_key_reads_only_its_own_placeholder(tmp_path) -> None:
    """Two switches, two readers. A helper that answered "auth is on" for either
    line would light the CLI's mcp Auth column for every llama-keyed node."""
    (tmp_path / "llama.yaml").write_text(KEYED_YAML)
    (tmp_path / "mcp.yaml").write_text(MCP_KEYED_YAML)

    assert template_references_api_key(tmp_path / "llama.yaml") is True
    assert template_references_mcp_api_key(tmp_path / "llama.yaml") is False
    assert template_references_api_key(tmp_path / "mcp.yaml") is False
    assert template_references_mcp_api_key(tmp_path / "mcp.yaml") is True


def test_template_references_mcp_endpoints_finds_the_consumers(tmp_path) -> None:
    """The deployer's fan-out set: a rotated peer key must also reach every node
    whose resolved config carries that peer's key, or the planner keeps dialling
    with the old one."""
    (tmp_path / "planner.yaml").write_text(PLANNER_YAML)
    (tmp_path / "head.yaml").write_text(MCP_KEYED_YAML)

    assert template_references_mcp_endpoints(tmp_path / "planner.yaml") is True
    assert template_references_mcp_endpoints(tmp_path / "head.yaml") is False
    assert template_references_mcp_endpoints(tmp_path / "absent.yaml") is False
