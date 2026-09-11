"""The dashboard's Access-tab HTTP surface (plus the Config tab's regenerate
override, which shares this router's app): where a plaintext key may appear,
and the audit line that has to appear with it.

``access_api``'s module docstring calls these "security invariants (pinned by
tests)", but nothing pinned the routes — only ``access_service``, one layer
down. That mattered because the dashboard binds 0.0.0.0:7001 with no
authentication of its own: the accountability for handing out a fleet
credential is that exactly FOUR routes do it (the current and previous key of
each of the node's two keyed services), on an explicit click, and that each
writes a warn line naming the node and the caller. Dropping the ``emit`` call
was a silent change.

Everything here is served through the app rather than by calling the handlers,
because half of what is being pinned is which route the key comes out of.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from ruamel.yaml import YAML

from orchestrator.config import keystore
from orchestrator.dashboard import access_api, access_service
from orchestrator.dashboard.app import create_app
from tests.conftest import PROJECT_ROOT

yaml = YAML()

MCP = keystore.KIND_MCP


def _write_fleet(tmp_path) -> str:
    """A three-node fleet: llama auth staged on, nothing staged, MCP auth on.

    Deliberately not the shipped configs/aureax-network.yaml — keystore writes
    under $HOME and these tests mint keys, so the fleet has to be one whose
    node ids are this test's own.
    """
    configs = tmp_path / "configs"
    configs.mkdir()

    def template(name: str, server: dict, mcp_server: dict | None = None) -> None:
        data = {
            "node": {"id": "${node_id}", "role": "edge_executor"},
            "model": {
                "name": "TestModel",
                "filename": "m.gguf",
                "server": {"host": "0.0.0.0", "port": 8080, **server},
            },
            "mcp": {"server": {"enabled": True, **(mcp_server or {})}},
            "telemetry": {"remote": {
                "endpoint": "https://${orchestrator_ip}:7001/api/v1/telemetry",
            }},
            "deployment": {},
        }
        with open(configs / name, "w") as f:
            yaml.dump(data, f)

    template("keyed.yaml", {"api_key": "${api_key}", "alias": "test-model"})
    template("open.yaml", {"alias": "test-model"})
    # MCP auth on with llama auth off: the two template lines are independent,
    # and a fixture where they always agree cannot catch them being conflated.
    template("mcp-keyed.yaml", {"alias": "test-model"},
             mcp_server={"api_key": "${mcp_api_key}"})

    network = {
        "network": {"name": "t", "version": "0.1.0"},
        "model_registry": {"models": []},
        "ssh": {"targets": [
            {"node_id": "edge-01", "host": "10.77.0.1", "config_file": "keyed.yaml"},
            {"node_id": "edge-02", "host": "10.77.0.2", "config_file": "open.yaml"},
            {"node_id": "edge-03", "host": "10.77.0.3", "config_file": "mcp-keyed.yaml"},
        ]},
    }
    path = configs / "aureax-network.yaml"
    with open(path, "w") as f:
        yaml.dump(network, f)
    return str(path)


@pytest.fixture
def client(tmp_aureax_home, tmp_path, monkeypatch):
    """A TestClient over the real app, with every outward call stubbed.

    ``_orchestrator_ip`` shells out to detect the Mac's LAN address, and it
    memoizes across tests for a minute — pinning it keeps both the config-hash
    and this fixture deterministic.
    """
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    app = create_app(network_config_path=_write_fleet(tmp_path))
    return TestClient(app)


def _audit_lines(client_) -> list:
    return [
        line for line in client_.app.state.system_log.snapshot(since=0)
        if line.source == "access"
    ]


# --- the one route that returns a plaintext key ------------------------------


def test_revealing_a_key_returns_it_and_writes_an_audit_line(client) -> None:
    """This is the ONLY route allowed to hand out a plaintext key.

    The dashboard has no authentication of its own, so the warn line naming the
    node and the client address is the entire accountability trail for a
    credential leaving the orchestrator.
    """
    key = keystore.ensure_key("edge-01")

    resp = client.get("/api/access/edge-01/key")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"node_id": "edge-01", "api_key": key}

    audit = _audit_lines(client)
    assert len(audit) == 1, audit
    assert audit[0].level == "warn", f"a key reveal must not be an info line: {audit[0]}"
    assert "edge-01" in audit[0].text
    assert audit[0].source == "access"


def test_revealing_a_previous_key_is_audited_the_same_way(client) -> None:
    """The .prev key is a live credential too — llama-server is still
    enforcing it until the rotate is applied."""
    keystore.ensure_key("edge-01")
    keystore.rotate_key("edge-01")
    previous = keystore.previous_key("edge-01")
    assert previous

    resp = client.get("/api/access/edge-01/key/previous")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"node_id": "edge-01", "api_key": previous}
    assert resp.json()["api_key"] != keystore.read_key("edge-01")

    audit = _audit_lines(client)
    assert [line.level for line in audit] == ["warn"]
    assert "edge-01" in audit[0].text
    assert "PREVIOUS" in audit[0].text, "the audit line must say which key was revealed"


@pytest.mark.parametrize(
    "path",
    [
        "/api/access/edge-02/key", "/api/access/edge-02/key/previous",
        "/api/access/edge-02/mcp-key", "/api/access/edge-02/mcp-key/previous",
    ],
)
def test_a_node_with_no_key_is_a_404_and_not_an_empty_string(client, path) -> None:
    """An open node must read as "no key exists", not as a node whose key is ""
    — an empty 200 is what a caller would paste into a bearer header."""
    resp = client.get(path)

    assert resp.status_code == 404, resp.text
    assert "edge-02" in resp.json()["detail"]


# --- the MCP key: a second credential, on the same terms ---------------------


@pytest.mark.parametrize("previous", [False, True])
def test_revealing_an_mcp_key_returns_it_and_writes_both_audit_records(
    client, caplog, previous,
) -> None:
    """Holding the MCP key reaches inference through ``run_inference``, so it
    is exactly as sensitive as the llama key and gets the identical two-record
    audit: the durable ``emit`` (suppressed on failure) AND a logger line, so
    a credential never leaves the orchestrator with no trace anywhere.
    """
    keystore.ensure_key("edge-03", kind=MCP)
    if previous:
        keystore.rotate_key("edge-03", kind=MCP)
    expected = (
        keystore.previous_key("edge-03", kind=MCP)
        if previous
        else keystore.read_key("edge-03", kind=MCP)
    )
    suffix = "/previous" if previous else ""

    with caplog.at_level(logging.WARNING, logger="aureax.dashboard.access"):
        resp = client.get(f"/api/access/edge-03/mcp-key{suffix}")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"node_id": "edge-03", "api_key": expected}

    audit = _audit_lines(client)
    assert [line.level for line in audit] == ["warn"]
    assert "edge-03" in audit[0].text
    assert "MCP API key" in audit[0].text, "the audit line must say WHICH key was revealed"
    assert ("PREVIOUS" in audit[0].text) is previous
    # The second record. emit's write is suppressed on failure, so the logger
    # line is the half that survives a broken system log.
    assert any("MCP API key" in r.getMessage() and "edge-03" in r.getMessage()
               for r in caplog.records), caplog.records
    assert not any(expected in r.getMessage() for r in caplog.records), (
        "the audit line carried the key itself"
    )


def test_the_two_key_kinds_come_out_of_different_routes(client) -> None:
    """Separate files, separate rotations — a route that served either would
    hand out whichever key happened to exist."""
    llama = keystore.ensure_key("edge-01")
    mcp = keystore.ensure_key("edge-01", kind=MCP)
    assert llama != mcp

    assert client.get("/api/access/edge-01/key").json()["api_key"] == llama
    assert client.get("/api/access/edge-01/mcp-key").json()["api_key"] == mcp


def test_an_unknown_node_is_rejected_on_the_mcp_routes_too(client) -> None:
    assert client.get("/api/access/not-a-node/mcp-key").status_code == 404
    assert _audit_lines(client) == [], "a rejected request must not log a reveal"


def test_an_unknown_node_is_rejected_before_the_keystore_is_touched(client) -> None:
    """The node id reaches keystore.read_key, which builds a path from it."""
    resp = client.get("/api/access/not-a-node/key")

    assert resp.status_code == 404, resp.text
    assert _audit_lines(client) == [], "a rejected request must not log a reveal"


# --- every other route stays key-free ----------------------------------------


def test_the_polled_routes_never_carry_a_plaintext_key(client, monkeypatch) -> None:
    """/api/access/nodes and /api/access/live are polled by an open LAN bind.

    A key in either payload hands every credential on the fleet to anyone who
    can reach the dashboard port, without so much as a log line. Both kinds,
    including the nested ``auth.mcp`` sub-block the MCP half is reported in.
    """
    key = keystore.ensure_key("edge-01")
    mcp_key = keystore.ensure_key("edge-03", kind=MCP)

    async def fake_probe_all(entries):
        # The real one opens a TCP socket per node; the fleet here is fictional.
        return {
            e["node_id"]: {
                "node_id": e["node_id"], "reachable": True, "auth_live": "secured",
                "key_matches": True, "http_status": 401, "detail": "",
            }
            for e in entries
        }

    monkeypatch.setattr(access_service, "probe_all", fake_probe_all)
    monkeypatch.setattr(access_service, "probe_mcp_all", fake_probe_all)

    for path in ("/api/access/nodes", "/api/access/live"):
        resp = client.get(path)
        assert resp.status_code == 200, (path, resp.text)
        body = json.dumps(resp.json(), ensure_ascii=False)
        for secret in (key, mcp_key):
            assert secret not in body, f"{path} leaked a plaintext key"
            # Not even the random half, masked or otherwise.
            assert secret.removeprefix("aureax_sk_")[:-4] not in body, (
                f"{path} leaked key material"
            )

    assert _audit_lines(client) == [], "polling must not fill the audit log"


def test_the_live_route_reports_each_port_separately(client, monkeypatch) -> None:
    """A staged-but-unapplied MCP key is invisible to the llama probe, so the
    two verdicts cannot share one badge — the tab would read a node green while
    its :30900 is still open to the LAN."""
    async def llama_probe(entries):
        return {e["node_id"]: {"node_id": e["node_id"], "reachable": True,
                               "auth_live": "secured", "key_matches": True,
                               "http_status": 401, "detail": ""}
                for e in entries}

    async def mcp_probe(entries):
        return {e["node_id"]: {"node_id": e["node_id"], "reachable": True,
                               "auth_live": "open", "key_matches": None,
                               "http_status": 400, "detail": ""}
                for e in entries}

    monkeypatch.setattr(access_service, "probe_all", llama_probe)
    monkeypatch.setattr(access_service, "probe_mcp_all", mcp_probe)

    node = client.get("/api/access/live").json()["nodes"]["edge-03"]

    assert node["auth_live"] == "secured"
    assert node["mcp"]["auth_live"] == "open"
    assert node["mcp"]["badge"], "the MCP half carries its own badge"
    assert node["mcp"]["node_id"] == "edge-03"


# --- detection: the IP the badges and the apply must agree on ----------------


def test_the_empty_ip_cache_never_reads_as_fresh(monkeypatch) -> None:
    """monotonic() counts from boot, so a 0.0 sentinel reads as a fresh reading
    for the first 60s of uptime: detection never runs, every badge says
    "unknown" and every Apply 400s on an IP nothing has tried to find. Asserted
    against the sentinel itself — comparing it to the live clock is satisfied by
    the host's uptime rather than by the value."""
    from orchestrator import lifecycle

    assert access_api._IP_CACHE_EMPTY[0] == float("-inf")

    calls = []
    monkeypatch.setattr(
        lifecycle, "detect_mac_lan_ip", lambda cfg=None: calls.append(cfg) or "10.1.2.3",
    )
    monkeypatch.setattr(access_api, "_ip_cache", access_api._IP_CACHE_EMPTY)

    assert access_api._orchestrator_ip() == "10.1.2.3"
    assert len(calls) == 1


def test_the_read_routes_detect_against_the_fleet_not_the_default_route(
    client, monkeypatch,
) -> None:
    """The detected IP is baked into every generated config, so it feeds
    desired_config_hash. A read route detecting on the DEFAULT route while the
    apply detects on a node route makes every badge read "Saved but not
    applied" for a fleet that is in sync."""
    seen = []
    monkeypatch.setattr(
        access_api, "_orchestrator_ip", lambda cfg=None: seen.append(cfg) or "10.77.0.99",
    )

    assert client.get("/api/access/nodes").status_code == 200

    assert seen, "the read route never detected an orchestrator IP"
    assert seen[0] is not None, "detected on the default route, without the fleet"


# --- staging routes refuse what would stage nothing --------------------------


def test_rotate_is_refused_for_a_node_whose_template_has_no_api_key(client) -> None:
    """A key for an auth-OFF node enables nothing, while the card grows a
    Reveal/Copy/Rotate block implying auth the node does not enforce. The CLI
    path refuses the same thing (test_apply.py)."""
    resp = client.post("/api/access/edge-02/auth/rotate")

    assert resp.status_code == 409, resp.text
    assert "auth is off" in resp.json()["detail"]
    assert not keystore.has_key("edge-02"), "minted a key for a node with auth off"


def test_rotate_works_once_auth_is_staged_in_the_template(client) -> None:
    resp = client.post("/api/access/edge-01/auth/rotate")

    assert resp.status_code == 200, resp.text
    assert resp.json()["action"] == "generated"
    assert resp.json()["kind"] == "llama", "an unqualified rotate must mean the llama key"
    assert keystore.has_key("edge-01")
    assert not keystore.has_key("edge-01", kind=MCP), "rotated a kind nobody asked for"


def test_rotate_is_gated_per_kind_on_that_kinds_template_line(client) -> None:
    """The gate has to read the line the caller is rotating: gating the MCP
    rotate on the llama placeholder 409s a node whose MCP auth is on, and
    serves one whose is off."""
    # edge-03: MCP auth on, llama auth off.
    assert client.post("/api/access/edge-03/auth/rotate",
                       json={"kind": "llama"}).status_code == 409
    resp = client.post("/api/access/edge-03/auth/rotate", json={"kind": "mcp"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["action"] == "generated"
    assert keystore.has_key("edge-03", kind=MCP)
    assert not keystore.has_key("edge-03")

    # edge-01: the mirror image.
    assert client.post("/api/access/edge-01/auth/rotate",
                       json={"kind": "mcp"}).status_code == 409


def test_rotating_one_kind_leaves_the_other_key_untouched(client) -> None:
    """Separate files, separate rotations. A rotate that moved both would
    invalidate a harness holding the key the operator did not touch."""
    keystore.ensure_key("edge-03", kind=MCP)
    keystore.ensure_key("edge-03")  # a stale llama key from an earlier config
    llama_before = keystore.read_key("edge-03")
    mcp_before = keystore.read_key("edge-03", kind=MCP)

    resp = client.post("/api/access/edge-03/auth/rotate", json={"kind": "mcp"})

    assert resp.json()["action"] == "rotated"
    assert keystore.read_key("edge-03", kind=MCP) != mcp_before
    assert keystore.previous_key("edge-03", kind=MCP) == mcp_before
    assert keystore.read_key("edge-03") == llama_before


def test_an_unknown_kind_is_a_400_on_both_staging_routes(client) -> None:
    """Silently falling back to llama would rotate or toggle the wrong
    service's credential and report success."""
    for resp in (
        client.post("/api/access/edge-01/auth/rotate", json={"kind": "rpc"}),
        client.put("/api/access/edge-01/auth", json={"enabled": True, "kind": "rpc"}),
    ):
        assert resp.status_code == 400, resp.text
        assert "kind" in resp.json()["detail"]


# --- the auth toggle, per kind ----------------------------------------------


def test_the_toggle_defaults_to_the_llama_key_and_edits_model_server(client, tmp_path) -> None:
    """The default keeps every caller written against the single-key route
    working unchanged — the llama key is the one those callers meant."""
    template = Path(_template_dir(client)) / "open.yaml"

    assert client.put("/api/access/edge-02/auth", json={"enabled": True}).json()["kind"] == "llama"

    data = yaml.load(template.read_text())
    assert data["model"]["server"]["api_key"] == "${api_key}"
    assert "api_key" not in data["mcp"]["server"]


@pytest.mark.parametrize("enabled", [True, False])
def test_the_mcp_toggle_edits_mcp_server_and_nothing_else(client, enabled) -> None:
    """kind="mcp" writes ${mcp_api_key} under mcp.server. Writing it under
    model.server would hand llama-server the MCP key and leave :30900 open."""
    template = Path(_template_dir(client)) / "mcp-keyed.yaml"

    resp = client.put("/api/access/edge-03/auth", json={"enabled": enabled, "kind": "mcp"})

    assert resp.status_code == 200, resp.text
    assert resp.json()["kind"] == "mcp"
    data = yaml.load(template.read_text())
    assert data["mcp"]["server"].get("api_key") == ("${mcp_api_key}" if enabled else None)
    assert "api_key" not in data["model"]["server"], "the MCP toggle touched llama's section"


def test_the_toggle_leaves_no_temporary_file_in_the_configs_directory(
    client, tmp_path, monkeypatch,
) -> None:
    """configs/ is git-tracked and nothing in .gitignore matches a stray
    `<template>.tmp`, so a half-written temporary left there is untracked but
    not ignored — a later `git add -A` commits it. The atomic writer uses a
    unique name and unlinks it when the write fails."""
    configs = Path(_template_dir(client))
    before = sorted(p.name for p in configs.iterdir())

    real_replace = os.replace

    def failing_replace(src, dst, *a, **kw):
        # Only the publish of this template: everything else in the request has
        # to keep working, or the test would pass for the wrong reason.
        if str(dst).endswith("open.yaml"):
            raise OSError("disk full")
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="disk full"):
        client.put("/api/access/edge-02/auth", json={"enabled": True})

    assert sorted(f.name for f in configs.iterdir()) == before, "left a temporary behind"


def test_the_toggle_keeps_the_template_world_readable(client) -> None:
    """Templates are git-tracked source at 0644, not private state: writing one
    at the keystore's 0600 would leave a mode the next checkout fights."""
    template = Path(_template_dir(client)) / "open.yaml"

    assert client.put("/api/access/edge-02/auth", json={"enabled": True}).status_code == 200

    assert stat.S_IMODE(template.stat().st_mode) == 0o644


def test_a_template_with_no_mcp_section_is_a_409_not_a_new_section(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """Mirrors the model.server rule: this route edits an existing section, it
    does not invent one. Inventing `mcp.server` would stage a key for a server
    the node never starts."""
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path = _fleet_with(tmp_path, [
        {"node_id": "bare", "host": "10.77.0.9", "config_file": "bare.yaml"},
    ])
    bare = Path(path).parent / "bare.yaml"
    with open(bare, "w") as f:
        yaml.dump({
            "node": {"id": "${node_id}", "role": "edge_executor"},
            "model": {"name": "TestModel", "filename": "m.gguf",
                      "server": {"host": "0.0.0.0", "port": 8080}},
        }, f)
    before = bare.read_text()

    resp = _client_for(path).put("/api/access/bare/auth", json={"enabled": True, "kind": "mcp"})

    assert resp.status_code == 409, resp.text
    assert "mcp.server" in resp.json()["detail"]
    assert bare.read_text() == before


# A shipped head template, used verbatim: the toggle's whole risk is what a
# ruamel round-trip does to a REAL file — 190 lines of block sequences, quoted
# placeholders, explicit nulls and the comment blocks that are this repo's
# documentation. The fixture templates above are too plain to notice any of it.
SHIPPED_HEAD_TEMPLATE = PROJECT_ROOT / "configs" / "aureax-shard-head-35b.yaml"

RESERVED_COMMENT = "    # RESERVED — documentation only."


def _fleet_with_shipped_head(tmp_path) -> tuple[str, Path, str]:
    path = _fleet_with(tmp_path, [
        {"node_id": "head-01", "host": "10.77.0.7", "config_file": "head35.yaml"},
    ])
    template = Path(path).parent / "head35.yaml"
    template.write_text(SHIPPED_HEAD_TEMPLATE.read_text())
    return path, template, template.read_text()


@pytest.mark.parametrize(
    ("kind", "line"),
    [("mcp", '    api_key: "${mcp_api_key}"'), ("llama", '    api_key: "${api_key}"')],
)
def test_disabling_auth_removes_that_line_and_touches_nothing_else(
    tmp_aureax_home, tmp_path, monkeypatch, kind, line,
) -> None:
    """A default-configured ruamel YAML() rewrote ~190 lines of this file on
    every click: block sequences dedented two columns, `deployed_at: null`
    emptied, long lines folded at 80 — and the five-line RESERVED comment block
    deleted outright, because ruamel files a comment against the key it follows
    and that key was the one being popped. None of it broke the config, which is
    why nothing noticed; the comments are the documentation here.
    """
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path, template, before = _fleet_with_shipped_head(tmp_path)

    resp = _client_for(path).put(
        "/api/access/head-01/auth", json={"enabled": False, "kind": kind},
    )

    assert resp.status_code == 200, resp.text
    assert template.read_text() == before.replace(line + "\n", "", 1)
    assert RESERVED_COMMENT in template.read_text()


def test_re_enabling_restores_the_line_and_leaves_the_rest_byte_identical(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """Off then on: every other line comes back exactly as it was, and the
    restored line carries its quotes. It is NOT byte-identical — the comment
    block that followed the key is re-attached to the preceding key on the next
    read, so the line lands below it rather than above (see
    access_api._insert_api_key). That is the whole remaining delta.
    """
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path, template, before = _fleet_with_shipped_head(tmp_path)
    client_ = _client_for(path)

    for enabled in (False, True):
        resp = client_.put(
            "/api/access/head-01/auth", json={"enabled": enabled, "kind": "mcp"},
        )
        assert resp.status_code == 200, resp.text

    after = template.read_text()
    assert sorted(after.splitlines()) == sorted(before.splitlines())
    assert '    api_key: "${mcp_api_key}"' in after
    data = yaml.load(after)
    assert data["mcp"]["server"]["api_key"] == "${mcp_api_key}"
    # The other kind's line is a separate placeholder and was never touched.
    assert data["model"]["server"]["api_key"] == "${api_key}"
    # And it settles: a second cycle is a no-op against the first one's output.
    for enabled in (False, True):
        client_.put("/api/access/head-01/auth", json={"enabled": enabled, "kind": "mcp"})
    assert template.read_text() == after


def _template_dir(client_) -> str:
    return str(Path(client_.app.state.lifecycle.network_config_path).parent)


def _fleet_with(tmp_path, extra_targets: list[dict]) -> str:
    path = _write_fleet(tmp_path)
    data = yaml.load(Path(path).read_text())
    data["ssh"]["targets"].extend(extra_targets)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _client_for(path: str) -> TestClient:
    return TestClient(create_app(network_config_path=path))


@pytest.mark.parametrize("kind", ["llama", "mcp"])
def test_auth_cannot_be_staged_into_a_shard_worker_template(
    tmp_aureax_home, tmp_path, monkeypatch, kind,
) -> None:
    """A worker runs rpc-server and no MCP server, so NEITHER key means
    anything there — and one worker template serves every worker in the fleet,
    so writing a placeholder stages auth for none of them and dirties all."""
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path = _fleet_with(tmp_path, [
        {"node_id": "w1", "host": "10.77.0.4", "config_file": "worker.yaml"},
    ])
    worker_yaml = Path(path).parent / "worker.yaml"
    with open(worker_yaml, "w") as f:
        yaml.dump({
            "node": {"id": "${node_id}", "role": "shard_worker"},
            "model": {"name": "TestModel", "filename": "m.gguf",
                      "rpc_server": {"port": 50052}},
            "mcp": {"server": {"enabled": False}},
        }, f)
    before = worker_yaml.read_text()

    resp = _client_for(path).put("/api/access/w1/auth", json={"enabled": True, "kind": kind})

    assert resp.status_code == 400, resp.text
    assert "shard-worker" in resp.json()["detail"]
    assert worker_yaml.read_text() == before, "rewrote a worker template anyway"


def test_a_config_file_pointing_outside_configs_is_refused(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """Defence in depth for the template writer. The schema now rejects a
    traversing ``config_file`` when the network config loads (so the fleet is
    unusable rather than exploitable), but this route joins the value onto
    configs/ and REWRITES the result, so it pins containment itself."""
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path = Path(_write_fleet(tmp_path))
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            lifecycle=SimpleNamespace(network_config_path=str(path)),
        )),
    )

    assert access_api._template_path(request, "keyed.yaml") == path.parent / "keyed.yaml"
    for escape in ("../outside.yaml", "/etc/passwd", "sub/../../outside.yaml"):
        with pytest.raises(HTTPException) as excinfo:
            access_api._template_path(request, escape)
        assert excinfo.value.status_code == 400, escape


def test_a_fleet_whose_config_file_escapes_configs_does_not_load(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path = _fleet_with(tmp_path, [
        {"node_id": "evil", "host": "10.77.0.9", "config_file": "../outside.yaml"},
    ])
    outside = tmp_path / "outside.yaml"
    outside.write_text("untouched: true\n")

    resp = _client_for(path).put("/api/access/evil/auth", json={"enabled": True})

    assert resp.status_code == 503, resp.text
    assert outside.read_text() == "untouched: true\n"


# --- apply: what gets started, and what a failed run reports -----------------


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({"node_ids": []}, "an empty selection restarted the whole fleet"),
        ({}, "an omitted selection restarted the whole fleet"),
    ],
)
def test_apply_never_widens_an_empty_selection_to_the_fleet(client, body, why) -> None:
    """A fleet-wide apply restarts every head, each a multi-minute model reload.
    The CLI requires --all for it; the dashboard requires {"all": true}."""
    resp = client.post("/api/access/apply", json=body)

    assert resp.status_code == 400, why
    assert client.get("/api/access/apply/state").json()["running"] is False


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({"node_ids": ["nope"]}, "an unknown id"),
        ({"node_ids": ["edge-01", "nope"]}, "a mixed known/unknown list"),
    ],
)
def test_apply_refuses_an_unknown_node_id_before_it_answers_202(client, body, why) -> None:
    """The same 404 the single-node route gives. Deployer.apply_to_nodes does
    reject an unknown id, but only after this route has answered 202: the caller
    reads success, the abort surfaces on the state poll, and the failure handler
    then reports the REAL nodes in the list as possibly half-written when
    nothing was pushed at all."""
    resp = client.post("/api/access/apply", json=body)

    assert resp.status_code == 404, f"{why} was accepted: {resp.text}"
    assert client.get("/api/access/apply/state").json()["running"] is False


class _FakeDeployer:
    """Records the apply instead of SSHing to a fictional fleet."""

    outcomes: ClassVar[dict] = {}
    error: ClassVar[Exception | None] = None
    calls: ClassVar[list] = []

    def __init__(self, config_path, ip) -> None:
        self.config_path, self.ip = config_path, ip

    def apply_to_nodes(self, node_ids, restart=True, log=None):
        type(self).calls.append((list(node_ids or []), restart))
        if type(self).error is not None:
            raise type(self).error
        return dict(type(self).outcomes)


@pytest.fixture
def fake_deployer(monkeypatch):
    from orchestrator.deploy import deployer as deployer_mod

    _FakeDeployer.outcomes, _FakeDeployer.error, _FakeDeployer.calls = {}, None, []
    monkeypatch.setattr(deployer_mod, "Deployer", _FakeDeployer)
    return _FakeDeployer


def _wait_for_apply(client_, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client_.get("/api/access/apply/state").json()
        if not state["running"]:
            return state
        time.sleep(0.02)
    raise AssertionError("the apply never finished")


def test_a_successful_apply_journals_what_was_pushed(client, fake_deployer) -> None:
    fake_deployer.outcomes = {"edge-01": "config pushed; pod restarting"}

    resp = client.post("/api/access/edge-01/apply")

    assert resp.status_code == 202, resp.text
    state = _wait_for_apply(client)
    assert state["nodes"] == fake_deployer.outcomes
    assert state["error"] is None
    assert fake_deployer.calls == [(["edge-01"], True)]
    assert "edge-01" in access_service.read_journal()


def test_an_aborted_apply_names_the_nodes_it_may_have_touched(client, fake_deployer) -> None:
    """deploy_configs raises on the FIRST node that fails, so nodes earlier in
    the push already hold the new config with an unrestarted pod. An empty
    result tells the operator nothing about which half of the fleet was hit."""
    fake_deployer.error = RuntimeError("config push failed for edge-01")

    assert client.post("/api/access/apply", json={"all": True}).status_code == 202

    state = _wait_for_apply(client)
    assert "RuntimeError" in state["error"]
    assert set(state["nodes"]) == {"edge-01", "edge-02", "edge-03"}, state["nodes"]
    assert all("apply aborted" in v for v in state["nodes"].values())
    assert access_service.read_journal() == {}, "journalled a run that never pushed"


# --- the push set an aborted run reports ------------------------------------


def test_the_push_set_includes_every_consumer_of_the_derived_peer_list(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """A template carrying ${mcp_endpoints} gets every peer's MCP key baked
    into its resolved config, so apply_to_nodes pushes to it whenever an MCP
    key changed. This list runs BEFORE the run and cannot know whether a
    fingerprint will drift — and it exists to name what "may have been
    touched", so under-reporting is the failure it must not have.
    """
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path = _fleet_with(tmp_path, [
        {"node_id": "planner-01", "host": "10.77.0.9", "config_file": "planner.yaml"},
    ])
    planner = Path(path).parent / "planner.yaml"
    with open(planner, "w") as f:
        yaml.dump({
            "node": {"id": "${node_id}", "role": "cognitive_planner"},
            "model": {"name": "TestModel", "filename": "m.gguf",
                      "server": {"host": "0.0.0.0", "port": 8080}},
            "mcp": {"server": {"enabled": False},
                    "client": {"enabled": True, "remote_servers": "${mcp_endpoints}"}},
        }, f)

    touched = access_api._apply_push_set(Path(path), ["edge-01"])

    assert touched[0] == "edge-01", "the requested node must stay first"
    assert "planner-01" in touched, "the derived consumer was not named"
    assert "edge-02" not in touched, "an unrelated node was named"


def test_the_push_set_never_names_a_node_twice(tmp_aureax_home, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(access_api, "_orchestrator_ip", lambda cfg=None: "10.77.0.99")
    path = _write_fleet(tmp_path)
    touched = access_api._apply_push_set(Path(path), None)
    assert touched == sorted(set(touched), key=touched.index)
    assert set(touched) == {"edge-01", "edge-02", "edge-03"}


# --- config regeneration: the worker-reorder override ------------------------


def test_the_reorder_override_reaches_the_generator_itself(client, monkeypatch) -> None:
    """The last hop, which the route-level tests cannot see.

    tests/test_dashboard_app.py pins the route → ``_generate_serialised`` half
    by stubbing that helper, so a helper that accepted the flag and dropped it
    on the floor would stay green there: only generate_all_edge_configs can
    honour it, because the refusal happens inside generation.
    """
    from orchestrator.config import templating

    seen: list[bool] = []

    def fake_generate(_cfg, _out, _ip, *, accept_worker_reorder=False):
        seen.append(accept_worker_reorder)
        return []

    monkeypatch.setattr(templating, "generate_all_edge_configs", fake_generate)

    assert client.post("/api/config/regenerate",
                       json={"orchestrator_ip": "10.77.0.99"}).status_code == 200
    assert client.post("/api/config/regenerate",
                       json={"orchestrator_ip": "10.77.0.99",
                             "accept_worker_reorder": True}).status_code == 200

    assert seen == [False, True], "the override defaulted on, or never arrived"


# --- generated-config redaction: both key kinds, every occurrence ------------

_PLANNER_CONFIG = """\
node:
  id: planner-01
model:
  server:
    api_key: aureax_sk_LLAMAKEYVALUE0001
mcp:
  server:
    enabled: false
    api_key: aureax_sk_MCPKEYVALUE0002
  client:
    enabled: true
    remote_servers:
      - name: edge-01
        url: https://10.77.0.1:30900/mcp
        api_key: aureax_sk_PEERKEY0001
      - name: edge-02
        url: https://10.77.0.2:30900/mcp
        api_key: aureax_sk_PEERKEY0002
      - name: edge-03
        url: https://10.77.0.3:30900/mcp
        api_key: aureax_sk_PEERKEY0003
"""


def test_generated_config_redaction_covers_both_kinds_and_every_peer(client) -> None:
    """One regex covers all of them because it matches the KEY NAME, not the
    path — free by accident, which is exactly why it needs pinning.

    A planner's resolved config is the one file that concentrates fleet-wide
    credentials: its own two keys plus one per peer. This route serves that
    file to anyone who can reach the unauthenticated dashboard port.
    """
    from orchestrator.config.loader import generated_config_dir
    from orchestrator.dashboard.config_api import REDACTION_MARKER

    generated = generated_config_dir()
    generated.mkdir(parents=True, exist_ok=True)
    (generated / "aureax-edge-planner-01.yaml").write_text(_PLANNER_CONFIG)

    resp = client.get("/api/config/aureax-edge-planner-01.yaml")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["redacted"] is True
    assert "aureax_sk_" not in body["content"], body["content"]
    # Five values: llama, this node's MCP key, and one per peer. A marker count
    # below that means a key line survived somewhere in the file.
    assert body["content"].count(REDACTION_MARKER) == 5
    # The keys ARE the only thing redacted — the peer URLs stay readable, since
    # the file is served to be read.
    assert "https://10.77.0.3:30900/mcp" in body["content"]


def test_a_finished_apply_releases_the_lifecycle_interlock(client, fake_deployer) -> None:
    """lifecycle_api._apply_in_flight reads runner.running for its 409 guard, so
    a latched True 409s every later apply AND every start/stop."""
    fake_deployer.error = ImportError("no paramiko")

    assert client.post("/api/access/edge-01/apply").status_code == 202
    _wait_for_apply(client)

    runner = client.app.state.access_apply
    assert runner.running is False
    assert runner.snapshot()["running"] is False
    assert client.post("/api/access/edge-01/apply").status_code == 202


def test_the_apply_flag_is_claimed_before_the_first_await(client, fake_deployer, monkeypatch) -> None:
    """Two POSTs must not both see running=False.

    _start_apply awaits detection and the push-set before it can dispatch, and
    every await yields the loop — so the claim has to happen in the same
    uninterrupted stretch as the guard, or two applies push and restart the
    same nodes at once.
    """
    seen: list[bool] = []

    def _detected(config_path):
        seen.append(client.app.state.access_apply.running)
        return "10.77.0.99"

    monkeypatch.setattr(access_api, "_detected_ip", _detected)

    assert client.post("/api/access/edge-01/apply").status_code == 202

    assert seen == [True], "the apply flag was still False across the first await"
    # fake_deployer keeps the dispatched run off the network (the example
    # fleet's addresses are unroutable, so a real Deployer would sit in SSH
    # connect timeouts); waiting for it keeps the executor thread from
    # outliving the client.
    _wait_for_apply(client)


def test_an_undetectable_ip_releases_the_apply_flag(client, monkeypatch) -> None:
    """The 400 path leaves nothing to run the executor's finally, so it has to
    release the claim itself — a latched True 409s every later apply and every
    lifecycle start/stop."""
    monkeypatch.setattr(access_api, "_detected_ip", lambda _p: None)

    assert client.post("/api/access/edge-01/apply").status_code == 400

    assert client.app.state.access_apply.running is False
    assert client.get("/api/access/apply/state").json()["running"] is False
