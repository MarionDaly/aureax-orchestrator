"""Access-tab service logic: badge truth tables, URL construction, and the
no-plaintext-keys invariant — all pure, no HTTP and no fleet."""

from __future__ import annotations

import json
import os
import ssl
import stat
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from ruamel.yaml import YAML

from orchestrator.config import keystore
from orchestrator.config.schema import LLAMA_NODE_PORT, MCP_NODE_PORT
from orchestrator.dashboard import access_service
from tests.conftest import write_network_yaml

yaml = YAML()

MCP = keystore.KIND_MCP


# --- classify_staged / classify: the badge truth tables ----------------------

@pytest.mark.parametrize(
    ("auth_desired", "has_key", "desired_hash", "journal", "expected"),
    [
        # Auth wanted but no key minted yet: pending, whatever the journal says.
        (True, False, None, None, "pending"),
        (True, False, None, {"applied_hash": "aaaa"}, "pending"),
        # Desired hash uncomputable (no orchestrator IP): honest unknown.
        (True, True, None, {"applied_hash": "aaaa"}, "unknown"),
        (False, False, None, None, "unknown"),
        # Never journalled: never_applied, not green.
        (True, True, "aaaa", None, "never_applied"),
        (False, False, "aaaa", None, "never_applied"),
        # Journal disagrees with desired: pending.
        (True, True, "aaaa", {"applied_hash": "bbbb"}, "pending"),
        # Journal agrees: in_sync (final verdict still needs the live probe).
        (True, True, "aaaa", {"applied_hash": "aaaa"}, "in_sync"),
        (False, False, "aaaa", {"applied_hash": "aaaa"}, "in_sync"),
        # Hashes agree but the pod never restarted: the file is on the node and
        # the process still runs the previous alias/context/key. Not green.
        (True, True, "aaaa", {"applied_hash": "aaaa", "restarted": False}, "pending"),
        (False, False, "aaaa", {"applied_hash": "aaaa", "restarted": False}, "pending"),
        (True, True, "aaaa", {"applied_hash": "aaaa", "restarted": True}, "in_sync"),
    ],
)
def test_classify_staged_truth_table(auth_desired, has_key, desired_hash, journal, expected):
    state, _msg = access_service.classify_staged(auth_desired, has_key, desired_hash, journal)
    assert state == expected


@pytest.mark.parametrize(
    ("auth_desired", "staged", "live", "expected"),
    [
        (False, "unsupported", None, "unsupported"),
        # Staged problems outrank live readings.
        (True, "pending", {"auth_live": "secured", "key_matches": True}, "pending"),
        (True, "never_applied", {"auth_live": "secured"}, "never_applied"),
        (True, "unknown", {"auth_live": "secured"}, "unknown"),
        # in_sync + unverifiable live: NEVER green.
        (True, "in_sync", None, "unknown"),
        (True, "in_sync", {"auth_live": "unreachable"}, "unknown"),
        (True, "in_sync", {"auth_live": "unknown"}, "unknown"),
        # in_sync + live agrees: applied.
        (True, "in_sync", {"auth_live": "secured", "key_matches": True}, "applied"),
        (False, "in_sync", {"auth_live": "open"}, "applied"),
        # Drift, both directions.
        (True, "in_sync", {"auth_live": "open"}, "drift"),
        (True, "in_sync", {"auth_live": "secured", "key_matches": False}, "drift"),
        (False, "in_sync", {"auth_live": "secured"}, "drift"),
        # key_matches unverified (no local key to try): stays applied, since
        # the server IS enforcing a key and nothing contradicts ours.
        (True, "in_sync", {"auth_live": "secured", "key_matches": None}, "applied"),
    ],
)
def test_classify_final_truth_table(auth_desired, staged, live, expected):
    badge, _msg = access_service.classify(auth_desired, staged, live)
    assert badge == expected


def test_drift_and_unknown_carry_actionable_messages():
    _, msg = access_service.classify(True, "in_sync", {"auth_live": "open"})
    assert "Apply" in msg
    _, msg = access_service.classify(True, "in_sync", {"auth_live": "secured", "key_matches": False})
    assert "Rotate" in msg
    _, msg = access_service.classify(True, "in_sync", None)
    assert "verify" in msg.lower()


# --- node_access_list: URLs, roles, and the no-plaintext invariant -----------

def _write_fleet(tmp_path: Path) -> Path:
    """Network config + templates: a 1+1 group (head+worker), a standalone
    edge node with llama auth enabled, a second one with MCP auth enabled, and
    a planner that derives its peer list. Shapes mirror the real configs/ files
    minimally."""
    configs = tmp_path / "configs"
    configs.mkdir()

    def edge_template(name: str, role: str, extra_server: dict | None = None,
                      mcp_enabled: bool = True, mcp_server: dict | None = None,
                      mcp_client: dict | None = None) -> None:
        mcp: dict = {"server": {"enabled": mcp_enabled, **(mcp_server or {})}}
        if mcp_client is not None:
            mcp["client"] = mcp_client
        data = {
            "node": {"id": "${node_id}", "role": role},
            "model": {
                "name": "TestModel",
                "filename": "m.gguf",
                "server": {"host": "0.0.0.0", "port": 8080, **(extra_server or {})},
            },
            "mcp": mcp,
            # Real templates embed the orchestrator IP here — it is what makes
            # desired_config_hash track the Mac's address.
            "telemetry": {"remote": {
                "endpoint": "https://${orchestrator_ip}:7001/api/v1/telemetry",
            }},
            # Real templates carry this section; the generation-time
            # config_hash is stamped into it, and desired_config_hash reads it.
            "deployment": {},
        }
        with open(configs / name, "w") as f:
            yaml.dump(data, f)

    edge_template("edge.yaml", "edge_executor",
                  {"api_key": "${api_key}", "alias": "test-model"})
    # MCP auth on, llama auth off: the two are independent template lines, and
    # a fleet where they always agree would not notice them being conflated.
    edge_template("mcp-keyed.yaml", "edge_executor", {"alias": "test-model"},
                  mcp_server={"api_key": "${mcp_api_key}"})
    edge_template("head.yaml", "edge_executor", {"rpc_servers": "${rpc_servers}",
                                                 "gpu_layers": 99})
    worker = {
        "node": {"id": "${node_id}", "role": "shard_worker"},
        "model": {"name": "TestModel", "filename": "m.gguf", "rpc_server": {"port": 50052}},
        "mcp": {"server": {"enabled": False}},
    }
    with open(configs / "worker.yaml", "w") as f:
        yaml.dump(worker, f)
    edge_template("planner.yaml", "cognitive_planner", None, mcp_enabled=False,
                  mcp_client={"enabled": True, "remote_servers": "${mcp_endpoints}"})

    return write_network_yaml(
        configs,
        [
            {"node_id": "edge-01", "host": "10.77.0.1", "config_file": "edge.yaml"},
            {"node_id": "head-01", "host": "10.77.0.2", "config_file": "head.yaml"},
            {"node_id": "worker-01", "host": "10.77.0.3", "config_file": "worker.yaml"},
            {"node_id": "planner-01", "host": "10.77.0.4", "config_file": "planner.yaml"},
            {"node_id": "mcp-01", "host": "10.77.0.5", "config_file": "mcp-keyed.yaml"},
        ],
        groups=[{"name": "g1", "head": "head-01", "workers": ["worker-01"]}],
    )


def test_node_access_list_urls_and_roles(tmp_aureax_home, tmp_path):
    entries = {e["node_id"]: e
               for e in access_service.node_access_list(_write_fleet(tmp_path), None)}

    edge = entries["edge-01"]
    assert edge["api_base_url"] == "https://10.77.0.1:30800/v1"
    assert edge["chat_ui_url"] == "https://10.77.0.1:30800/"
    assert edge["mcp_url"] == "https://10.77.0.1:30900/mcp"
    assert edge["model_id"] == "test-model"  # alias wins over model.name
    assert edge["auth"]["supported"] and edge["auth"]["desired_enabled"]

    worker = entries["worker-01"]
    assert worker["serves_llama"] is False
    # Null URLs make it structurally impossible to copy a worker address.
    assert worker["api_base_url"] is None and worker["chat_ui_url"] is None
    assert worker["auth"]["supported"] is False
    assert worker["auth"]["state"] == "unsupported"
    assert worker["served_by"]["node_id"] == "head-01"
    assert worker["served_by"]["api_base_url"] == "https://10.77.0.2:30800/v1"

    planner = entries["planner-01"]
    assert planner["serves_llama"] is True
    assert planner["api_base_url"] == "https://10.77.0.4:30800/v1"
    assert planner["mcp_url"] is None  # its :30900 serves nothing

    head = entries["head-01"]
    assert head["group"] == "g1" and "worker-01" in head["group_members"]
    assert head["auth"]["desired_enabled"] is False  # no ${api_key} in head.yaml


def test_every_advertised_url_is_https(tmp_aureax_home, tmp_path):
    """No plaintext spelling anywhere in the payload. Every listener in the
    fleet is TLS-only, so an http:// address here is a URL that cannot connect
    at all — and a copy button is where a wrong scheme survives longest."""
    entries = access_service.node_access_list(_write_fleet(tmp_path), None)
    dumped = json.dumps(entries)
    assert "http://" not in dumped, dumped
    assert "https://10.77.0.5:30900/mcp" in dumped


def test_an_ipv6_host_is_bracketed_in_every_advertised_url(tmp_aureax_home, tmp_path):
    """A bare IPv6 literal in a URL authority is not a URL: its own colons read
    as the port separator, so httpx refuses it outright. Every copy button, env
    snippet and probe in this module builds its address the same way, so the
    bracketing has to happen where the host meets the scheme."""
    configs = _write_fleet(tmp_path).parent
    path = write_network_yaml(
        configs,
        [{"node_id": "edge-v6", "host": "fd00::1", "config_file": "edge.yaml"}],
        filename="ipv6-network.yaml",
    )

    entry = access_service.node_access_list(path, None)[0]

    assert entry["api_base_url"] == f"https://[fd00::1]:{LLAMA_NODE_PORT}/v1"
    assert entry["chat_ui_url"] == f"https://[fd00::1]:{LLAMA_NODE_PORT}/"
    assert entry["mcp_url"] == f"https://[fd00::1]:{MCP_NODE_PORT}/mcp"
    assert httpx.URL(entry["api_base_url"]).port == LLAMA_NODE_PORT
    assert httpx.URL(entry["mcp_url"]).port == MCP_NODE_PORT
    with pytest.raises(httpx.InvalidURL):
        httpx.URL(f"https://fd00::1:{LLAMA_NODE_PORT}/v1")


def test_the_mcp_auth_sub_block_is_per_node_and_per_service(tmp_aureax_home, tmp_path):
    """Each node has TWO keyed services. Reporting only llama-server's leaves a
    staged-but-unapplied MCP key invisible on the card that exists to show it."""
    entries = {e["node_id"]: e
               for e in access_service.node_access_list(_write_fleet(tmp_path), None)}

    edge = entries["edge-01"]["auth"]
    # Serves MCP, but its template carries no ${mcp_api_key} line.
    assert edge["desired_enabled"] is True
    assert edge["mcp"]["supported"] is True
    assert edge["mcp"]["desired_enabled"] is False

    keyed = entries["mcp-01"]["auth"]
    assert keyed["desired_enabled"] is False, "llama auth leaked from the MCP line"
    assert keyed["mcp"]["desired_enabled"] is True
    assert keyed["mcp"]["state"] == "pending"  # auth on, no key minted yet
    assert set(keyed["mcp"]) == set(k for k in keyed if k != "mcp"), (
        "the MCP sub-block must carry the same field names as the llama block"
    )

    # A planner serves llama and no MCP; a worker serves neither.
    assert entries["planner-01"]["auth"]["mcp"]["supported"] is False
    assert entries["planner-01"]["auth"]["supported"] is True
    assert entries["worker-01"]["auth"]["mcp"]["supported"] is False


def test_the_mcp_sub_block_reports_an_unreadable_key_file(tmp_aureax_home, tmp_path):
    """Same rule as the llama key, against the MCP file: "no key" and "a key I
    cannot read" call for opposite actions, and the message must name the file
    whose ownership is wrong."""
    path = _write_fleet(tmp_path)
    keystore.key_path("mcp-01", kind=MCP).mkdir(parents=True)

    entries = {e["node_id"]: e for e in access_service.node_access_list(path, None)}
    mcp = entries["mcp-01"]["auth"]["mcp"]
    assert mcp["key_state"] == keystore.KEY_UNREADABLE
    assert mcp["state"] == "unknown"
    assert "mcp-01.mcp.key" in mcp["message"]


def test_worker_served_by_respects_heads_webui_toggle(tmp_aureax_home, tmp_path):
    """A worker card must not advertise a chat UI its head has disabled —
    served_by is built from the head's ENTRY, not a re-derived URL."""
    path = _write_fleet(tmp_path)
    head_yaml = path.parent / "head.yaml"
    data = yaml.load(head_yaml.read_text())
    data["model"]["server"]["webui"] = False
    with open(head_yaml, "w") as f:
        yaml.dump(data, f)

    entries = {e["node_id"]: e for e in access_service.node_access_list(path, None)}
    assert entries["head-01"]["chat_ui_url"] is None
    served = entries["worker-01"]["served_by"]
    assert served["api_base_url"] == "https://10.77.0.2:30800/v1"
    assert served["chat_ui_url"] is None


def test_node_access_list_never_returns_plaintext_keys(tmp_aureax_home, tmp_path):
    """The security invariant of the whole payload design: this is a polled
    endpoint on an unauthenticated LAN bind.

    Both kinds, because both are minted into the same directory and reported
    through the same block shape — the MCP key reaches inference through
    ``run_inference``, so leaking it is not the lesser failure.
    """
    path = _write_fleet(tmp_path)
    key = keystore.ensure_key("edge-01")
    mcp_key = keystore.ensure_key("mcp-01", kind=MCP)
    assert key.startswith("aureax_sk_") and mcp_key.startswith("aureax_sk_")
    dumped = json.dumps(access_service.node_access_list(path, None), ensure_ascii=False)
    for secret in (key, mcp_key):
        assert secret not in dumped
        # The random part of the key must never appear anywhere, masked or not.
        assert secret.removeprefix("aureax_sk_")[:-4] not in dumped
    assert "aureax_sk_" not in dumped.replace("aureax_sk_…", "")  # only the mask form allowed


def test_env_snippet_shapes(tmp_aureax_home, tmp_path):
    entries = {e["node_id"]: e
               for e in access_service.node_access_list(_write_fleet(tmp_path), None)}
    edge_lines = "\n".join(entries["edge-01"]["env_snippet"])
    assert "OPENAI_BASE_URL=https://10.77.0.1:30800/v1" in edge_lines
    assert "<copy from the key field" in edge_lines  # keyed node: no literal key
    head_lines = "\n".join(entries["head-01"]["env_snippet"])
    assert "sk-no-key-required" in head_lines  # open node: placeholder, not empty
    # Workers point at the head.
    worker_lines = "\n".join(entries["worker-01"]["env_snippet"])
    assert "https://10.77.0.2:30800/v1" in worker_lines


def test_mcp_snippet_shapes(tmp_aureax_home, tmp_path):
    """The harness reads AUREAX_MCP_API_KEY, so that is the variable the block
    has to name — and it is a separate line from OPENAI_API_KEY because nothing
    that speaks the OpenAI-compatible endpoint reads it."""
    entries = {e["node_id"]: e
               for e in access_service.node_access_list(_write_fleet(tmp_path), None)}

    keyed = "\n".join(entries["mcp-01"]["mcp_snippet"])
    assert "AUREAX_MCP_URL=https://10.77.0.5:30900/mcp" in keyed
    assert "AUREAX_MCP_API_KEY=<copy from the MCP key field above>" in keyed

    # MCP auth off: a commented line, never an empty assignment — an exported
    # empty value is sent as "Bearer " and reads as a wrong key, not no key.
    open_node = "\n".join(entries["edge-01"]["mcp_snippet"])
    assert open_node.splitlines()[1].startswith("#")
    assert "AUREAX_MCP_API_KEY=" not in open_node

    # A node serving no MCP advertises no MCP variables at all.
    assert entries["planner-01"]["mcp_snippet"] == []
    assert entries["worker-01"]["mcp_snippet"] == []


def test_desired_hash_side_effect_free_and_per_target(tmp_aureax_home, tmp_path):
    """desired_config_hash must not mint keys (a browser poll must not create
    fleet credentials), and must reflect per-target resolution."""
    path = _write_fleet(tmp_path)
    from orchestrator.config.loader import load_network_config

    cfg = load_network_config(str(path))
    edge = next(t for t in cfg.ssh.enabled_targets if t.node_id == "edge-01")

    # No key yet -> None, and STILL no key afterwards.
    assert access_service.desired_config_hash(path.parent, edge, cfg, "10.77.0.99") is None
    assert not keystore.has_key("edge-01")

    keystore.ensure_key("edge-01")
    h1 = access_service.desired_config_hash(path.parent, edge, cfg, "10.77.0.99")
    assert h1
    # The hash tracks the orchestrator IP: a Mac that changed networks makes
    # every node honestly read "pending apply".
    h2 = access_service.desired_config_hash(path.parent, edge, cfg, "10.77.0.98")
    assert h2 and h2 != h1


def test_a_missing_mcp_key_makes_the_hash_uncomputable_without_minting_one(
    tmp_aureax_home, tmp_path
):
    """Same rule as the llama key, one template line over: generation would
    mint it and a browser poll may not, so the honest answer is None until
    `aureax generate` (or Apply) has run."""
    path = _write_fleet(tmp_path)
    from orchestrator.config.loader import load_network_config

    cfg = load_network_config(str(path))
    keyed = next(t for t in cfg.ssh.enabled_targets if t.node_id == "mcp-01")

    assert access_service.desired_config_hash(path.parent, keyed, cfg, "10.77.0.99") is None
    assert not keystore.has_key("mcp-01", kind=MCP), "a polled route minted an MCP key"
    assert not keystore.key_path("mcp-01", kind=MCP).exists()

    keystore.ensure_key("mcp-01", kind=MCP)
    assert access_service.desired_config_hash(path.parent, keyed, cfg, "10.77.0.99")


def test_a_peers_missing_mcp_key_leaves_the_planner_computable(tmp_aureax_home, tmp_path):
    """The guard covers the node's OWN key only.

    A planner's ${mcp_endpoints} carries every peer's key, so a peer keyed a
    moment ago genuinely leaves the planner pending an apply — and "pending" is
    the honest badge for that. Returning None instead would report the one node
    whose config really is out of date as merely uncomputable.
    """
    path = _write_fleet(tmp_path)
    from orchestrator.config.loader import load_network_config

    cfg = load_network_config(str(path))
    planner = next(t for t in cfg.ssh.enabled_targets if t.node_id == "planner-01")

    assert not keystore.has_key("mcp-01", kind=MCP)
    assert access_service.desired_config_hash(path.parent, planner, cfg, "10.77.0.99")
    assert not keystore.has_key("mcp-01", kind=MCP), "generated a peer's key on a poll"


def test_journal_entries_record_both_key_fingerprints(tmp_aureax_home, tmp_path):
    """One apply installs both keys, so a record naming only the llama one
    cannot say afterwards which MCP key the running pod is enforcing."""
    keystore.ensure_key("edge-01")
    keystore.ensure_key("edge-01", kind=MCP)

    entry = access_service.journal_entries({"edge-01": "config pushed"}, "r1", "t")["edge-01"]

    assert entry["key_fingerprint"] == keystore.fingerprint("edge-01")
    assert entry["mcp_key_fingerprint"] == keystore.fingerprint("edge-01", kind=MCP)
    assert entry["key_fingerprint"] != entry["mcp_key_fingerprint"]


def test_journal_round_trip(tmp_aureax_home, tmp_path):
    access_service.record_applied({"n1": {"applied_hash": "abcd", "applied_at": "t"}})
    access_service.record_applied({"n2": {"applied_hash": "efgh"}})
    j = access_service.read_journal()
    assert j["n1"]["applied_hash"] == "abcd" and j["n2"]["applied_hash"] == "efgh"
    # Corrupt journal degrades to empty, not a crash on a polled route.
    access_service.journal_path().write_text("{not json")
    assert access_service.read_journal() == {}


def test_mask_key():
    assert access_service.mask_key("aureax_sk_ABCDEFGH9fQ2") == "aureax_sk_…9fQ2"
    assert access_service.mask_key(None) is None
    assert access_service.mask_key("") is None


def test_desired_hash_is_uncomputable_without_an_orchestrator_ip(tmp_aureax_home, tmp_path):
    """Detection returns "" (not None) when it fails, and generating against an
    empty ${orchestrator_ip} produces a hash no apply can ever have journalled
    — every node would read "Saved but not applied" for ever."""
    path = _write_fleet(tmp_path)
    from orchestrator.config.loader import load_network_config

    cfg = load_network_config(str(path))
    head = next(t for t in cfg.ssh.enabled_targets if t.node_id == "head-01")
    assert access_service.desired_config_hash(path.parent, head, cfg, "") is None
    assert access_service.desired_config_hash(path.parent, head, cfg, None) is None
    assert access_service.desired_config_hash(path.parent, head, cfg, "10.77.0.99")


def test_an_applied_node_reaches_in_sync(tmp_aureax_home, tmp_path):
    """The whole point of the journal: generate -> record -> badge goes green.

    Nothing connected generated_config_hash to desired_config_hash before, so a
    None from either (a renamed output file, a missing deployment stamp) left
    every node reading "never applied" after an apply that worked.
    """
    from orchestrator.config.loader import load_network_config
    from orchestrator.config.templating import generate_all_edge_configs

    path = _write_fleet(tmp_path)
    # validate_groups refuses a head that does not hop the group's setting
    # through to its own node block; generation runs for the whole fleet.
    head_yaml = path.parent / "head.yaml"
    head = yaml.load(head_yaml.read_text())
    head["node"]["worker_wait_timeout_seconds"] = "${worker_wait_timeout_seconds}"
    with open(head_yaml, "w") as f:
        yaml.dump(head, f)

    cfg = load_network_config(str(path))
    edge = next(t for t in cfg.ssh.enabled_targets if t.node_id == "edge-01")
    keystore.ensure_key("edge-01")

    generate_all_edge_configs(path, Path.home() / ".aureax" / "configs", "10.77.0.99")

    desired = access_service.desired_config_hash(path.parent, edge, cfg, "10.77.0.99")
    assert desired
    assert access_service.generated_config_hash("edge-01") == desired
    assert access_service.generated_config_hash("never-generated") is None

    entries = access_service.journal_entries(
        {"edge-01": "config pushed; pod restarting"}, "run1", "t"
    )
    access_service.record_applied(entries)
    state, _msg = access_service.classify_staged(
        True, True, desired, access_service.read_journal()["edge-01"]
    )
    assert state == "in_sync"


def test_journal_entries_records_only_what_was_pushed(tmp_aureax_home, tmp_path):
    """The substring predicate is the contract with Deployer's outcome strings:
    a node whose push failed must keep reading as not-applied."""
    entries = access_service.journal_entries(
        {
            "a": "config pushed",
            "b": "config pushed; restart FAILED (could not resolve hostname over SSH)",
            "c": "applied (pod Ready on new config)",
            "d": "unknown — apply aborted: RuntimeError",
        },
        "run7",
        "2026-01-01T00:00:00+00:00",
    )
    assert set(entries) == {"a", "b", "c"}, "a node that was never pushed was journalled"
    assert entries["c"]["run_id"] == "run7"
    assert entries["c"]["applied_at"] == "2026-01-01T00:00:00+00:00"


def test_an_unreadable_key_file_is_not_reported_as_no_key(tmp_aureax_home, tmp_path):
    """"No key" and "a key I cannot read" call for opposite actions: generate
    one, or fix the file's ownership. Collapsing them told the operator to
    Apply, which cannot fix it."""
    path = _write_fleet(tmp_path)
    # A directory where the key file belongs: an OSError on read, exactly like
    # the .key left root-owned by a sudo run that this reports.
    keystore.key_path("edge-01").mkdir(parents=True)

    entries = {e["node_id"]: e for e in access_service.node_access_list(path, None)}
    auth = entries["edge-01"]["auth"]
    assert auth["key_state"] == keystore.KEY_UNREADABLE
    assert auth["has_key"] is False
    assert auth["state"] == "unknown"
    assert "cannot be read" in auth["message"]


def test_a_workers_snippet_says_a_key_is_needed_when_its_head_is_keyed(
    tmp_aureax_home, tmp_path
):
    """A worker points at its head's /v1, so the HEAD's auth decides whether a
    key is needed — the worker's own auth block is "unsupported"."""
    path = _write_fleet(tmp_path)
    head_yaml = path.parent / "head.yaml"
    data = yaml.load(head_yaml.read_text())
    data["model"]["server"]["api_key"] = "${api_key}"
    with open(head_yaml, "w") as f:
        yaml.dump(data, f)

    entries = {e["node_id"]: e for e in access_service.node_access_list(path, None)}
    worker_lines = "\n".join(entries["worker-01"]["env_snippet"])
    assert "https://10.77.0.2:30800/v1" in worker_lines
    assert "sk-no-key-required" not in worker_lines, (
        "told the operator the head's keyed endpoint is open — every call 401s"
    )
    assert "head-01" in worker_lines


# --- the live probe: status -> auth_live is the whole verdict ----------------

class _ScriptedClient:
    """Stands in for httpx.AsyncClient. Answers each GET/POST/DELETE from one
    script; an Exception in the script is raised instead of returned.

    A scripted entry is either a status code or ``(status, headers)`` — the MCP
    probe reads ``mcp-session-id`` off the response to clean up after itself.
    """

    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str, dict]] = []

    def _answer(self, method, url, headers):
        self.calls.append((method, url, dict(headers or {})))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, resp_headers = outcome if isinstance(outcome, tuple) else (outcome, {})
        return SimpleNamespace(status_code=status, headers=resp_headers)

    async def get(self, url, headers=None):
        return self._answer("GET", url, headers)

    async def post(self, url, headers=None, content=None):
        self.bodies = [*getattr(self, "bodies", []), content]
        return self._answer("POST", url, headers)

    async def delete(self, url, headers=None):
        return self._answer("DELETE", url, headers)


@pytest.fixture
def reachable(monkeypatch):
    """Skip the TCP gate: these tests are about the HTTP verdict."""
    monkeypatch.setattr(access_service, "tcp_reachable", lambda *_a, **_kw: True)


@pytest.fixture
def fleet_ca(monkeypatch):
    """Stand in for the fleet CA context.

    Every probe verifies against it, and there is none under a tmp $HOME — a
    real one would need `aureax generate`. The context object itself is never
    exercised here: these tests script the client.
    """
    ctx = ssl.create_default_context()
    monkeypatch.setattr(access_service, "tls_context", lambda: ctx)
    return ctx


def test_the_auth_probe_path_is_behind_the_key(tmp_aureax_home):
    """/props is chosen because it is NOT auth-exempt on the pinned build. Point
    this at an exempt path and every keyed node probes 200 — "drift" for a fleet
    that is perfectly configured, with nothing failing."""
    assert access_service._AUTH_PROBE_PATH not in {
        "/health", "/v1/health", "/models", "/v1/models", "/",
    }


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "secured"), (200, "open"), (204, "open"), (503, "unknown"), (500, "unknown")],
)
async def test_probe_maps_status_to_auth_live(tmp_aureax_home, reachable, status, expected):
    client = _ScriptedClient(status)
    result = await access_service.probe_one("10.77.0.1", "edge-01", client)

    assert result["auth_live"] == expected
    assert result["reachable"] is True
    assert result["http_status"] == status
    assert client.calls[0][1] == (
        f"https://10.77.0.1:{LLAMA_NODE_PORT}{access_service._AUTH_PROBE_PATH}"
    )


@pytest.mark.parametrize(
    ("host", "authority"),
    [("10.77.0.1", "10.77.0.1"), ("edge-01.lan", "edge-01.lan"), ("fd00::1", "[fd00::1]")],
)
async def test_both_probes_bracket_an_ipv6_host(tmp_aureax_home, reachable, host, authority):
    """tcp_reachable handles IPv6, so an unbracketed probe URL passes the gate
    and then dies in httpx with InvalidURL — which the generic handler files as
    auth_live "unknown". Every node on an IPv6 fleet would sit on the unknown
    badge forever, and drift (a pod still serving unauthenticated) could never
    be seen at all."""
    llama = _ScriptedClient(401)
    await access_service.probe_one(host, "edge-01", llama)
    assert llama.calls[0][1] == (
        f"https://{authority}:{LLAMA_NODE_PORT}{access_service._AUTH_PROBE_PATH}"
    )

    mcp = _ScriptedClient(401)
    await access_service.probe_mcp_one(host, "edge-01", mcp)
    assert mcp.calls[0][1] == f"https://{authority}:{MCP_NODE_PORT}/mcp"


async def test_a_loading_model_is_unknown_not_open(tmp_aureax_home, reachable):
    """503 comes from the readiness middleware, which runs BEFORE auth — it
    proves nothing. Reading it as "open" makes every sharded head report drift
    for its whole multi-minute load."""
    result = await access_service.probe_one("10.77.0.1", "edge-01", _ScriptedClient(503))
    assert result["auth_live"] == "unknown"
    assert "503" in result["detail"]


async def test_a_transport_failure_is_unknown_never_open(tmp_aureax_home, reachable):
    result = await access_service.probe_one(
        "10.77.0.1", "edge-01", _ScriptedClient(httpx.ConnectError("boom"))
    )
    assert result["auth_live"] == "unknown"
    assert result["detail"] == "ConnectError"
    assert result["key_matches"] is None


async def test_an_unreachable_node_never_reaches_http(tmp_aureax_home, monkeypatch):
    monkeypatch.setattr(access_service, "tcp_reachable", lambda *_a, **_kw: False)
    client = _ScriptedClient(200)
    result = await access_service.probe_one("10.77.0.1", "edge-01", client)
    assert result == {
        "node_id": "edge-01", "reachable": False, "auth_live": "unreachable",
        "key_matches": None, "http_status": None, "detail": "",
    }
    assert client.calls == [], "probed a node the TCP gate said was down"


@pytest.mark.parametrize(("authed_status", "expected"), [(200, True), (401, False)])
async def test_key_matches_is_decided_by_the_authenticated_retry(
    tmp_aureax_home, reachable, authed_status, expected
):
    key = keystore.ensure_key("edge-01")
    client = _ScriptedClient(401, authed_status)

    result = await access_service.probe_one("10.77.0.1", "edge-01", client)

    assert result["auth_live"] == "secured"
    assert result["key_matches"] is expected
    assert client.calls[1][2]["Authorization"] == f"Bearer {key}"


async def test_a_failed_key_check_does_not_undo_a_proven_secured(tmp_aureax_home, reachable):
    """The 401 already proved the node enforces a key. Losing that verdict
    because the second request timed out drops the badge on a healthy node."""
    keystore.ensure_key("edge-01")
    client = _ScriptedClient(401, httpx.ReadTimeout("slow"))

    result = await access_service.probe_one("10.77.0.1", "edge-01", client)

    assert result["auth_live"] == "secured"
    assert result["key_matches"] is None
    assert "key check failed" in result["detail"]


async def test_probe_all_isolates_one_bad_node(tmp_aureax_home, fleet_ca, monkeypatch):
    """A host string that is not a valid authority raises InvalidURL, which is
    not an HTTPError — ungathered it 500s /api/access/live and every node loses
    its live column."""
    async def fake_probe_one(host, node_id, client=None):
        if node_id == "bad":
            raise httpx.InvalidURL("not an authority")
        return {"node_id": node_id, "auth_live": "open"}

    monkeypatch.setattr(access_service, "probe_one", fake_probe_one)

    out = await access_service.probe_all([
        {"node_id": "good", "host": "10.77.0.1", "serves_llama": True},
        {"node_id": "bad", "host": "fd00::5", "serves_llama": True},
    ])

    assert out["good"]["auth_live"] == "open"
    assert out["bad"]["auth_live"] == "unknown"
    assert out["bad"]["detail"] == "InvalidURL"


# --- TLS: a rejected certificate is its own, fixable state -------------------


def _tls_wrapped() -> Exception:
    """An httpx.ConnectError with the SSL failure TWO links down.

    That is the shape httpx actually delivers: httpcore wraps the verification
    error and httpx wraps that, so a check against the raised type alone never
    sees the ssl.SSLError at all.
    """
    verify_error = ssl.SSLCertVerificationError("unable to get local issuer certificate")
    transport_error = ConnectionError("httpcore: connect failed")
    transport_error.__cause__ = verify_error
    outer = httpx.ConnectError("connection failed")
    outer.__cause__ = transport_error
    return outer


async def test_a_rejected_certificate_names_the_fix(tmp_aureax_home, reachable):
    """Without this branch the failure lands in the generic handler AFTER the
    TCP gate passed, so the card reads "reachable, ConnectError" and names
    nothing — while the actual cause is a node still holding a certificate from
    a previous CA, which one apply fixes."""
    result = await access_service.probe_one(
        "10.77.0.1", "edge-01", _ScriptedClient(_tls_wrapped())
    )

    assert result["auth_live"] == "unknown"
    assert result["detail"] == "TLS: certificate not trusted — run aureax apply"


async def test_a_plain_connect_error_is_not_reported_as_a_tls_problem(
    tmp_aureax_home, reachable
):
    """The TLS branch keys off an ssl.SSLError in the cause chain, not off the
    exception type httpx raises — every transport failure is a ConnectError."""
    result = await access_service.probe_one(
        "10.77.0.1", "edge-01", _ScriptedClient(httpx.ConnectError("refused"))
    )
    assert result["detail"] == "ConnectError"


async def test_probes_report_unknown_when_this_mac_holds_no_fleet_ca(
    tmp_aureax_home, reachable
):
    """No CA means no probe — never a probe without verification. A probe that
    skipped verification would report "secured" for a node presenting any
    certificate at all, which is exactly the false green this module exists to
    prevent."""
    entries = [{"node_id": "edge-01", "host": "10.77.0.1", "serves_llama": True,
                "mcp_url": "https://10.77.0.1:30900/mcp"}]

    for out in (await access_service.probe_all(entries),
                await access_service.probe_mcp_all(entries)):
        assert out["edge-01"]["auth_live"] == "unknown"
        assert "aureax generate" in out["edge-01"]["detail"]

    # The single-node path answers the same way rather than raising into a
    # polled route.
    one = await access_service.probe_one("10.77.0.1", "edge-01")
    assert one["auth_live"] == "unknown" and "aureax generate" in one["detail"]


# --- the MCP probe -----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "secured"), (200, "open"), (400, "open"), (404, "unknown"), (502, "unknown")],
)
async def test_mcp_probe_maps_status_to_auth_live(
    tmp_aureax_home, reachable, status, expected
):
    """400 is "open", not a failure: the deliberately invalid body reaches the
    MCP transport's own validation only when nothing rejected the missing
    Authorization header first."""
    client = _ScriptedClient(status)
    result = await access_service.probe_mcp_one("10.77.0.1", "edge-01", client)

    assert result["auth_live"] == expected
    assert result["http_status"] == status
    method, url, headers = client.calls[0]
    assert method == "POST"
    assert url == f"https://10.77.0.1:{MCP_NODE_PORT}/mcp"
    # Both content types, or the SDK's transport rejects the request before it
    # ever reaches the bearer middleware — and the probe would read "open" on a
    # secured node.
    assert headers["Accept"] == "application/json, text/event-stream"
    assert "Authorization" not in headers, "the unauthenticated leg must send no key"
    assert client.bodies[0] == b"{}"


async def test_the_mcp_probe_never_sends_a_real_initialize(tmp_aureax_home, reachable):
    """StreamableHTTPSessionManager registers a server-side session for ANY
    POST without an Mcp-Session-Id, before the body is parsed. A real
    initialize from the 20 s poll would leak one session per node per poll for
    the life of the pod."""
    assert access_service._MCP_PROBE_BODY == b"{}"
    assert b"initialize" not in access_service._MCP_PROBE_BODY


async def test_the_mcp_probe_deletes_the_session_it_created(tmp_aureax_home, reachable):
    """The cleanup half of the same finding: the 400 response carries the
    session id the POST registered, so the probe can hand it back."""
    key = keystore.ensure_key("edge-01", kind=MCP)
    client = _ScriptedClient(401, (400, {"mcp-session-id": "sess-1"}), 200)

    result = await access_service.probe_mcp_one("10.77.0.1", "edge-01", client)

    assert result["auth_live"] == "secured"
    assert result["key_matches"] is True
    method, url, headers = client.calls[-1]
    assert method == "DELETE"
    assert url == f"https://10.77.0.1:{MCP_NODE_PORT}/mcp"
    assert headers["mcp-session-id"] == "sess-1"
    assert headers["Authorization"] == f"Bearer {key}"


async def test_an_open_mcp_port_has_its_session_cleaned_up_too(tmp_aureax_home, reachable):
    """The unauthenticated leg registers a session on an OPEN node exactly as
    the authed one does on a secured node — polling it every 20 s and never
    cleaning up is the same leak."""
    client = _ScriptedClient((400, {"mcp-session-id": "sess-open"}), 200)

    result = await access_service.probe_mcp_one("10.77.0.1", "edge-01", client)

    assert result["auth_live"] == "open"
    assert [c[0] for c in client.calls] == ["POST", "DELETE"]
    assert client.calls[-1][2]["mcp-session-id"] == "sess-open"
    assert "Authorization" not in client.calls[-1][2]


async def test_a_response_with_no_session_id_triggers_no_delete(tmp_aureax_home, reachable):
    client = _ScriptedClient(200)
    await access_service.probe_mcp_one("10.77.0.1", "edge-01", client)
    assert [c[0] for c in client.calls] == ["POST"]


@pytest.mark.parametrize(("authed_status", "expected"), [(200, True), (401, False)])
async def test_the_mcp_key_check_uses_the_mcp_key(
    tmp_aureax_home, reachable, authed_status, expected
):
    """The node's MCP key, not its llama key: they are separate files and
    separate rotations, and presenting the wrong one reads as "the node is
    enforcing a key we do not hold"."""
    keystore.ensure_key("edge-01")  # llama key, must not be used here
    mcp_key = keystore.ensure_key("edge-01", kind=MCP)
    client = _ScriptedClient(401, authed_status)

    result = await access_service.probe_mcp_one("10.77.0.1", "edge-01", client)

    assert result["key_matches"] is expected
    assert client.calls[1][2]["Authorization"] == f"Bearer {mcp_key}"


async def test_a_failed_mcp_key_check_keeps_the_proven_secured_verdict(
    tmp_aureax_home, reachable
):
    keystore.ensure_key("edge-01", kind=MCP)
    client = _ScriptedClient(401, httpx.ReadTimeout("slow"))

    result = await access_service.probe_mcp_one("10.77.0.1", "edge-01", client)

    assert result["auth_live"] == "secured"
    assert result["key_matches"] is None
    assert "key check failed" in result["detail"]


async def test_an_unreachable_mcp_port_never_reaches_http(tmp_aureax_home, monkeypatch):
    """The TCP gate keeps a node that is down from costing the full HTTP
    timeout once per poll, on the MCP port as on the llama one."""
    seen: list[int] = []

    def gate(_host, port, _timeout):
        seen.append(port)
        return False

    monkeypatch.setattr(access_service, "tcp_reachable", gate)
    client = _ScriptedClient(200)

    result = await access_service.probe_mcp_one("10.77.0.1", "edge-01", client)

    assert result["reachable"] is False and result["auth_live"] == "unreachable"
    assert seen == [MCP_NODE_PORT]
    assert client.calls == []


async def test_a_rejected_certificate_on_the_mcp_port_names_the_fix(
    tmp_aureax_home, reachable
):
    result = await access_service.probe_mcp_one(
        "10.77.0.1", "edge-01", _ScriptedClient(_tls_wrapped())
    )
    assert result["detail"] == "TLS: certificate not trusted — run aureax apply"


async def test_probe_mcp_all_covers_exactly_the_mcp_serving_nodes(
    tmp_aureax_home, fleet_ca, monkeypatch
):
    """Gated on mcp_url, which node_access_list already nulls for every node
    whose own config turns the MCP server off — so the two cannot disagree."""
    async def fake(host, node_id, client=None):
        return {"node_id": node_id, "auth_live": "secured"}

    monkeypatch.setattr(access_service, "probe_mcp_one", fake)

    out = await access_service.probe_mcp_all([
        {"node_id": "edge-01", "host": "10.77.0.1", "mcp_url": "https://10.77.0.1:30900/mcp"},
        {"node_id": "planner-01", "host": "10.77.0.4", "mcp_url": None},
        {"node_id": "worker-01", "host": "10.77.0.3", "mcp_url": None},
    ])

    assert set(out) == {"edge-01"}


def test_a_failed_restart_is_journalled_and_keeps_the_badge_off_green(tmp_aureax_home):
    """A push whose pod restart failed puts the new config on the node while the
    running process still serves the previous one. Green there would claim a
    live alias, context size or key the pod does not have."""
    entries = access_service.journal_entries(
        {
            "a": "config pushed; pod restarting",
            "b": "config pushed; restart FAILED (could not resolve hostname over SSH)",
        },
        "run9",
        "2026-01-01T00:00:00+00:00",
    )

    assert entries["a"]["restarted"] is True
    assert entries["b"]["restarted"] is False

    ok_state, _ = access_service.classify_staged(False, False, "aaaa", {
        **entries["a"], "applied_hash": "aaaa",
    })
    bad_state, bad_msg = access_service.classify_staged(False, False, "aaaa", {
        **entries["b"], "applied_hash": "aaaa",
    })
    assert ok_state == "in_sync"
    assert bad_state == "pending", "a node whose pod never restarted read as in sync"
    assert "never restarted" in bad_msg


def test_an_unreadable_journal_is_preserved_instead_of_overwritten(tmp_aureax_home):
    """read_journal degrades to {} so a polled route keeps answering, which
    means the writer is the only thing standing between a truncated file and
    every other node's record being replaced with nothing."""
    access_service.record_applied({"n1": {"applied_hash": "aaaa"}})
    access_service.journal_path().write_text("{not json")

    access_service.record_applied({"n2": {"applied_hash": "bbbb"}})

    aside = access_service.journal_path().with_suffix(".json.corrupt")
    assert aside.read_text() == "{not json", "the unreadable journal was destroyed"
    assert access_service.read_journal() == {"n2": {"applied_hash": "bbbb"}}


def test_the_journal_is_written_atomically_and_privately(tmp_aureax_home, monkeypatch):
    """The journal is the APPLIED column: lose it and every card drops to
    "Never applied from here" for a fleet that is in sync.

    So the write goes through the tree's atomic helper rather than a hand-rolled
    rename — fsync before the replace, a unique temporary that is unlinked when
    the write fails, and a 0700 directory that is chmodded even when it already
    existed (mkdir's mode applies only when it creates the leaf).
    """
    path = access_service.journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o755)

    access_service.record_applied({"n1": {"applied_hash": "aaaa"}})

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [f.name for f in path.parent.iterdir()] == [path.name]

    real_replace = os.replace

    def failing_replace(src, dst, *a, **kw):
        if str(dst).endswith("apply-state.json"):
            raise OSError("disk full")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", failing_replace)
    try:
        with pytest.raises(OSError, match="disk full"):
            access_service.record_applied({"n2": {"applied_hash": "bbbb"}})
    finally:
        # Restored by hand, not with monkeypatch.undo(): the same monkeypatch
        # instance carries tmp_aureax_home's $HOME redirect, and undoing that
        # would point the reads below at the real store.
        monkeypatch.setattr(os, "replace", real_replace)

    assert access_service.read_journal() == {"n1": {"applied_hash": "aaaa"}}, (
        "a failed write published a truncated journal"
    )
    assert [f.name for f in path.parent.iterdir()] == [path.name], "left a temporary behind"


def test_an_unmovable_unreadable_journal_is_not_overwritten(tmp_aureax_home, monkeypatch):
    """If the file cannot even be moved aside, the run goes unrecorded rather
    than replacing records this process never saw."""
    access_service.journal_path().parent.mkdir(parents=True, exist_ok=True)
    access_service.journal_path().write_text("{not json")

    def refuse(src, dst):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(access_service.os, "replace", refuse)

    access_service.record_applied({"n2": {"applied_hash": "bbbb"}})

    assert access_service.journal_path().read_text() == "{not json"
