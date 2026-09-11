"""Split-GGUF handling and derived MCP endpoints.

Both exist so that a list nobody maintains cannot go stale. A model too large
to publish as one file arrives as `name-00001-of-00003.gguf`, and every part is
derived from the name of part one rather than configured.

The planner's remote_servers is derived the same way: it resolves to every
enabled target whose own config serves MCP, so shard workers and other planners
are excluded automatically and no hand-written list of hosts can drift away
from the fleet it names.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from orchestrator.config.schema import ModelRegistryEntry, gguf_part_names
from orchestrator.config.templating import (
    PLACEHOLDER_DUMMIES,
    build_variables,
    mcp_endpoints,
)
from tests.conftest import make_network_config, make_targets

SPLIT = "Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf"


# --- part derivation ---------------------------------------------------------

def test_single_file_model_yields_itself() -> None:
    """Callers need no special case for the common one-file model."""
    assert gguf_part_names("Qwen3.6-27B-Q4_K_M.gguf") == ["Qwen3.6-27B-Q4_K_M.gguf"]


def test_split_model_expands_to_every_part() -> None:
    assert gguf_part_names(SPLIT) == [
        "Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf",
        "Qwen3.5-122B-A10B-UD-Q3_K_XL-00002-of-00003.gguf",
        "Qwen3.5-122B-A10B-UD-Q3_K_XL-00003-of-00003.gguf",
    ]


def test_part_count_comes_from_the_name_not_config() -> None:
    """The `-of-NNNNN` suffix is the source of truth — nothing to keep in sync."""
    assert len(gguf_part_names("m-00001-of-00007.gguf")) == 7


def _entry(**kw) -> ModelRegistryEntry:
    base = {
        "name": "M", "filename": SPLIT, "size_gb": 57.0,
        "quantization": "UD-Q3_K_XL", "target_roles": ["edge_executor"],
    }
    return ModelRegistryEntry(**{**base, **kw})


def test_registry_entry_must_name_part_one() -> None:
    """Pointing llama.cpp at part 2 fails deep in the loader, long after deploy."""
    with pytest.raises(ValidationError, match="must name part 1"):
        _entry(filename="Qwen3.5-122B-A10B-UD-Q3_K_XL-00002-of-00003.gguf")


def test_part_urls_preserve_the_repo_subdirectory() -> None:
    """Split quants live under a per-quant directory; the local name does not.

    Getting this wrong is a 404 on every part — which is exactly what happened
    when the path was assumed to be flat.
    """
    e = _entry(
        huggingface_repo="unsloth/Qwen3.5-122B-A10B-GGUF",
        huggingface_file="UD-Q3_K_XL/" + SPLIT,
    )
    urls = e.part_urls()
    assert len(urls) == 3
    assert urls[0].endswith(
        "/resolve/main/UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00001-of-00003.gguf"
    )
    assert urls[2].endswith("-00003-of-00003.gguf")
    # Local names stay flat — that is what lands in models/.
    assert e.part_filenames[0] == SPLIT


def test_part_urls_empty_without_a_repo() -> None:
    """A locally-staged model has nothing to download."""
    assert _entry().part_urls() == []


def test_registry_entry_rejects_a_huggingface_file_naming_a_later_part() -> None:
    """`filename` and `huggingface_file` are checked the same way: llama.cpp is
    handed part 1 and discovers the rest itself."""
    with pytest.raises(ValidationError, match="must name part 1"):
        _entry(
            huggingface_repo="unsloth/Qwen3.5-122B-A10B-GGUF",
            huggingface_file="UD-Q3_K_XL/Qwen3.5-122B-A10B-UD-Q3_K_XL-00002-of-00003.gguf",
        )


def test_registry_entry_rejects_a_local_remote_part_count_mismatch() -> None:
    """download-model.sh zips part_filenames and part_urls POSITIONALLY, so the
    shorter list truncates the download — a model missing tensors, which
    surfaces much later as a loader failure on the node."""
    with pytest.raises(ValidationError, match="declares 3 part"):
        _entry(huggingface_repo="r", huggingface_file="m-00001-of-00002.gguf")


# --- integrity: declared digests and a pinned revision ------------------------
# Weights are multi-gigabyte files fetched from a mutable ref and handed to
# llama.cpp's GGUF deserializer on every head. A declared digest is what turns
# "the right number of bytes arrived" into "the right bytes arrived", so the
# shape of that declaration is checked where the registry is parsed rather than
# on nine nodes at once after the download.

DIGEST = "a" * 64


def test_an_entry_declares_no_digest_by_default() -> None:
    """Every registry entry that predates the field keeps validating, and keeps
    the size-only checks it has always had."""
    e = _entry(filename="m.gguf")
    assert e.sha256 is None
    assert e.sha256_parts == []
    assert e.revision is None


def test_a_single_file_model_takes_one_digest() -> None:
    assert _entry(filename="m.gguf", sha256=DIGEST).sha256 == DIGEST


def test_a_split_model_takes_one_digest_per_part() -> None:
    """Each part is fetched separately and can be truncated or substituted on
    its own, so one digest for the set would verify nothing."""
    e = _entry(sha256_parts=[DIGEST, "b" * 64, "c" * 64])
    assert len(e.sha256_parts) == len(e.part_filenames) == 3


@pytest.mark.parametrize("digests", [[DIGEST], [DIGEST] * 2, [DIGEST] * 4])
def test_a_part_digest_list_of_the_wrong_length_is_rejected(digests) -> None:
    """The digests are paired with part_filenames by POSITION, exactly as
    part_urls is — a short list verifies the wrong file, a long one verifies a
    file that is never fetched."""
    with pytest.raises(ValidationError, match="declares 3 part"):
        _entry(sha256_parts=digests)


@pytest.mark.parametrize("bad", [
    "A" * 64,      # uppercase: shasum prints lowercase, and the compare is a string compare
    "a" * 63,      # truncated
    "a" * 65,      # over-long
    "g" * 64,      # not hex
    "sha256:" + "a" * 64,
])
def test_a_malformed_digest_is_rejected(bad) -> None:
    with pytest.raises(ValidationError, match="is not a sha256 digest"):
        _entry(filename="m.gguf", sha256=bad)


def test_the_two_digest_fields_are_mutually_exclusive() -> None:
    """One names a whole file, the other one part each — accepting both leaves
    it undecided which one a verifier should honour."""
    with pytest.raises(ValidationError, match="not both"):
        _entry(filename="m.gguf", sha256=DIGEST, sha256_parts=[DIGEST])


def test_a_whole_file_digest_on_a_split_model_is_rejected() -> None:
    """Nothing ever computes a digest over the concatenation, so this one would
    be declared, shipped and silently never checked."""
    with pytest.raises(ValidationError, match="declares 3 parts"):
        _entry(sha256=DIGEST)


def test_part_urls_default_to_the_mutable_main_ref() -> None:
    e = _entry(huggingface_repo="r", huggingface_file=SPLIT)
    assert all("/resolve/main/" in u for u in e.part_urls())


def test_a_revision_pins_every_part_url() -> None:
    """`main` moves. Two provisions of one fleet can otherwise fetch different
    bytes from the same URL, which a declared digest then reports as a
    mismatch rather than as the upstream change it is."""
    commit = "0123456789abcdef0123456789abcdef01234567"
    e = _entry(huggingface_repo="r", huggingface_file=SPLIT, revision=commit)
    urls = e.part_urls()
    assert len(urls) == 3
    assert all(f"/resolve/{commit}/" in u for u in urls)
    assert "/resolve/main/" not in " ".join(urls)


def test_a_split_declaring_zero_parts_is_rejected() -> None:
    """`-of-00000` makes gguf_part_names() return nothing at all, so the entry
    validated with an empty part list and download-model.sh then reported a
    missing registry entry — sending the operator to the one correct place."""
    with pytest.raises(ValidationError, match="declares 0 parts"):
        _entry(filename="m-00001-of-00000.gguf")


# --- derived MCP endpoints ---------------------------------------------------

HEAD_YAML = 'node: {id: h, role: edge_executor, control_plane: {endpoint: x}}\nmodel: {name: M, filename: m.gguf}\nmcp: {server: {enabled: true}}\n'
WORKER_YAML = 'node: {id: w, role: shard_worker, control_plane: {endpoint: x}}\nmodel: {name: M, filename: m.gguf}\nmcp: {server: {enabled: false}}\n'
PLANNER_YAML = 'node: {id: p, role: cognitive_planner, control_plane: {endpoint: x}}\nmodel: {name: M, filename: m.gguf}\nmcp: {server: {enabled: false}, client: {enabled: true}}\n'


@pytest.fixture
def fleet(tmp_path):
    """head + 2 shard workers + planner, with per-node configs on disk."""
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    for w in ("w1.yaml", "w2.yaml"):
        (tmp_path / w).write_text(WORKER_YAML)
    (tmp_path / "p.yaml").write_text(PLANNER_YAML)
    cfg = make_network_config(
        make_targets("h", "w1", "w2", "p"),
        [{"name": "g", "head": "h", "workers": ["w1", "w2"]}],
    )
    return cfg, tmp_path


def test_only_mcp_serving_nodes_are_offered(fleet) -> None:
    """Shard workers run rpc-server and the planner is a client — neither serves."""
    cfg, d = fleet
    eps = mcp_endpoints(cfg, d)
    assert [e["name"] for e in eps] == ["h"]
    assert eps[0]["url"] == "https://10.77.0.1:30900/mcp"
    assert eps[0]["transport"] == "streamable-http"


def test_every_endpoint_is_https(fleet) -> None:
    """There is no plaintext MCP listener to fall back to.

    The node serves :9000/:30900 from its own leaf under /etc/aureax/tls, so a
    planner handed an http:// URL does not degrade — it fails to connect, once
    per retry, forever.
    """
    cfg, d = fleet
    assert all(e["url"].startswith("https://") for e in mcp_endpoints(cfg, d))


def test_an_ipv6_peer_host_is_bracketed(tmp_path) -> None:
    """The peer host goes straight into a URL authority, same rule as
    ${orchestrator_url}: unbracketed, an IPv6 literal's own colons read as the
    port separator and no HTTP client parses the endpoint."""
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    cfg = make_network_config(
        [{"node_id": "h", "host": "fd00::1", "config_file": "h.yaml"}]
    )
    assert mcp_endpoints(cfg, tmp_path)[0]["url"] == "https://[fd00::1]:30900/mcp"


def test_a_node_never_dials_itself(fleet) -> None:
    cfg, d = fleet
    assert mcp_endpoints(cfg, d, exclude_node_id="h") == []


# --- peer MCP keys ------------------------------------------------------------
# A planner's remote_servers is the one file that concentrates every peer's MCP
# bearer key. The key travels with the endpoint because there is nowhere else
# for it to come from: the planner dials nodes it has never authenticated to.

MCP_KEYED_HEAD_YAML = HEAD_YAML.replace(
    "mcp: {server: {enabled: true}}",
    'mcp: {server: {enabled: true, api_key: "${mcp_api_key}"}}',
)


def test_a_keyed_peer_contributes_its_key(tmp_aureax_home, tmp_path) -> None:
    """The key comes from the store, keyed by the PEER's node id."""
    from orchestrator.config import keystore

    (tmp_path / "h.yaml").write_text(MCP_KEYED_HEAD_YAML)
    cfg = make_network_config(make_targets("h"))
    minted = keystore.ensure_key("h", kind=keystore.KIND_MCP)

    assert mcp_endpoints(cfg, tmp_path)[0]["api_key"] == minted


def test_an_unkeyed_peer_contributes_no_api_key_field(tmp_aureax_home, tmp_path) -> None:
    """Absent, not empty. RemoteServer rejects "" (it is falsy downstream, so it
    would read as "this peer is open" while the peer enforces a key)."""
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    cfg = make_network_config(make_targets("h"))

    assert "api_key" not in mcp_endpoints(cfg, tmp_path)[0]


def test_the_gate_is_the_peers_template_not_a_leftover_key_file(
    tmp_aureax_home, tmp_path
) -> None:
    """Turning a node's MCP auth off leaves its .mcp.key on the Mac.

    Gating on the file would keep shipping that credential into every planner's
    config for a door that is now open — and the planner would send a bearer
    header the peer no longer checks, so nothing would ever surface the drift.
    """
    from orchestrator.config import keystore

    (tmp_path / "h.yaml").write_text(HEAD_YAML)  # auth-off template
    keystore.ensure_key("h", kind=keystore.KIND_MCP)  # stale key from before
    cfg = make_network_config(make_targets("h"))

    assert "api_key" not in mcp_endpoints(cfg, tmp_path)[0]


def test_mcp_endpoints_never_mints_a_key(tmp_aureax_home, tmp_path) -> None:
    """This runs inside build_variables, which the Access tab reaches through
    desired_config_hash on a 20 s browser poll. read_key, never ensure_key: a
    page refresh must not create fleet credentials — the same rule
    generate_edge_config observes for the node's own keys."""
    from orchestrator.config import keystore

    (tmp_path / "h.yaml").write_text(MCP_KEYED_HEAD_YAML)
    cfg = make_network_config(make_targets("h"))

    entries = mcp_endpoints(cfg, tmp_path)

    assert "api_key" not in entries[0], "a key was minted from the polled path"
    assert not keystore.key_path("h", kind=keystore.KIND_MCP).exists()


def test_disabled_targets_are_not_offered(tmp_path) -> None:
    """A disabled node is deliberately not deployed; dialing it hangs forever."""
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "h2.yaml").write_text(HEAD_YAML)
    cfg = make_network_config(make_targets("h", "h2", disabled=("h2",)))
    assert [e["name"] for e in mcp_endpoints(cfg, tmp_path)] == ["h"]


def test_a_null_mcp_server_block_is_skipped_not_crashed(tmp_path) -> None:
    """`mcp:` with an empty `server:` parses to None, not {}.

    ``dict.get("server", {})`` returns that None rather than its default, so
    reading ``.enabled`` off it raised AttributeError — and since this runs
    inside `aureax generate`, one malformed-but-parseable per-node config took
    the whole fleet's generation down with a traceback.
    """
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "h2.yaml").write_text(
        "node: {id: h2, role: edge_executor, control_plane: {endpoint: x}}\n"
        "model: {name: M, filename: m.gguf}\n"
        "mcp:\n  server:\n"
    )
    cfg = make_network_config(make_targets("h", "h2"))

    # An absent `enabled` defaults to true, so h2 is still offered — the point
    # is that it is decided, not raised.
    assert [e["name"] for e in mcp_endpoints(cfg, tmp_path)] == ["h", "h2"]


def test_planner_build_variables_resolve_to_the_head(fleet) -> None:
    """End to end: what ${mcp_endpoints} actually injects into the planner."""
    cfg, d = fleet
    planner = next(t for t in cfg.ssh.targets if t.node_id == "p")
    v = build_variables(planner, cfg, "10.77.0.99", configs_dir=d)
    assert v["mcp_endpoints"] == [
        {"name": "h", "url": "https://10.77.0.1:30900/mcp", "transport": "streamable-http"}
    ]


def test_mcp_endpoints_key_exists_without_a_configs_dir(fleet) -> None:
    """configs_dir is omitted here because this test asserts on the KEY SET.

    PLACEHOLDER_DUMMIES is compared against exactly this set, and a missing name
    there makes the dashboard reject configs that deploy fine.
    """
    cfg, _ = fleet
    v = build_variables(cfg.ssh.targets[0], cfg, "10.77.0.99")
    assert v["mcp_endpoints"] == []
    assert set(v) == set(PLACEHOLDER_DUMMIES)


def test_an_unreadable_per_node_config_is_logged_not_silently_dropped(tmp_path, caplog) -> None:
    """This function exists because hand-listed endpoints went stale silently.

    Swallowing a parse error reintroduces that silence from the other side: the
    planner ships without an endpoint and nothing anywhere says why. The catch
    stays broad — `mcp: "yes"` parses fine and raises AttributeError here, and
    letting that escape takes the whole fleet's generation down.
    """
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "h2.yaml").write_text(
        "node: {id: h2, role: edge_executor, control_plane: {endpoint: x}}\n"
        "model: {name: M, filename: m.gguf}\n"
        'mcp: "yes"\n'
    )
    cfg = make_network_config(make_targets("h", "h2"))

    with caplog.at_level(logging.WARNING, logger="aureax.config.templating"):
        assert [e["name"] for e in mcp_endpoints(cfg, tmp_path)] == ["h"]

    assert any("h2.yaml" in r.getMessage() for r in caplog.records)
