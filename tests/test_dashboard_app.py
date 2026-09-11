"""Smoke tests for the FastAPI dashboard application factory, plus the API
behaviours an operator reads as fleet state.

Handlers are called directly rather than over a test client: they take
``Request`` only to reach ``app.state``, and the questions here are about what
they return, not how they are routed.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import shutil
import socket
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from orchestrator.dashboard import api as api_mod
from orchestrator.dashboard import app as app_mod
from orchestrator.dashboard.app import create_app


def dash_client(network_config_path) -> TestClient:
    """A test client for an app built against ``network_config_path``."""
    app = create_app(network_config_path=str(network_config_path))
    # No context manager anywhere below: skipping the lifespan keeps the
    # watchers (and their kubectl polling) out of a unit test. The one test
    # that is ABOUT the lifespan enters it explicitly.
    return TestClient(app)


def test_create_app_builds(tmp_aureax_home, network_config_path):
    app = create_app(network_config_path=str(network_config_path))
    assert app.title == "Aureax Dashboard"


def test_expected_routes_are_mounted(tmp_aureax_home, network_config_path):
    """Spot-check one path per router.

    Read from the generated OpenAPI schema rather than by walking
    ``app.routes``. That attribute is FastAPI's internal bookkeeping and its
    shape is not stable across versions — 0.141 replaced the flattened route
    copies with an ``_IncludedRouter`` marker, so route-list introspection
    reported an app with no API at all while every endpoint still served 200.
    The schema is public API and describes what is actually reachable.
    """
    app = create_app(network_config_path=str(network_config_path))
    paths = set(app.openapi()["paths"])

    assert "/api/nodes" in paths                          # api.py
    assert "/api/lifecycle/state" in paths                # lifecycle_api.py
    assert "/api/config" in paths                         # config_api.py
    assert "/api/chat/targets" in paths                   # chat_api.py


def test_app_state_is_populated(tmp_aureax_home, network_config_path):
    app = create_app(network_config_path=str(network_config_path))
    assert app.state.lifecycle is not None
    assert app.state.chat is not None
    assert app.state.system_log is not None
    assert app.state.metrics_storage is not None
    assert app.state.metrics_queries is not None


def test_a_corrupt_store_disables_its_own_tab_only(tmp_aureax_home, tmp_path):
    """Each store runs its DDL in its constructor, so an unreadable file there
    aborts `aureax dashboard` outright unless it is caught — taking lifecycle,
    config and access down with one truncated database."""
    junk = tmp_path / "junk"
    junk.mkdir()
    for name in ("metrics.db", "system.db", "sessions.db"):
        (junk / name).write_bytes(b"not a sqlite database")

    app = create_app(
        metrics_db_path=str(junk / "metrics.db"),
        network_config_path=_network_yaml(tmp_path),
        chat_db_path=str(junk / "sessions.db"),
        log_db_path=str(junk / "system.db"),
    )

    assert app.state.metrics_storage is None
    assert app.state.metrics_queries is None, "a lazy connection would 500 /api/nodes"
    assert app.state.chat is None, "a service over a missing store is not a service"
    assert app.state.log_store is None
    # The rest of the dashboard is still up.
    assert app.state.lifecycle is not None
    assert app.state.system_log is not None
    client = TestClient(app)
    assert client.get("/api/nodes").json() == []
    assert client.post("/api/v1/telemetry", content="").status_code == 503


# --- "cannot ask" is not "nothing there" -------------------------------------


def test_cluster_reports_an_error_when_the_api_server_is_unreachable(monkeypatch):
    """An unreachable K3s API must not render as a healthy, empty cluster.

    `kubectl.get` runs with check=False and returns {} for any failure, so the
    handler's own except branch never fires for an outage — it fell through to
    {"status": "ok", "nodes": []} and the Overview tab showed "K3s Nodes (0)"
    while the cluster was simply unreachable.
    """
    monkeypatch.setattr(api_mod.kubectl, "api_reachable", lambda *a, **k: False)
    monkeypatch.setattr(api_mod.kubectl, "get", lambda *a, **k: {})

    assert api_mod.cluster_status()["status"] == "error"
    assert api_mod.pod_status()["status"] == "error"


def test_cluster_still_reports_ok_when_the_cluster_is_genuinely_empty(monkeypatch):
    """The other half of the distinction: reachable, nothing deployed."""
    monkeypatch.setattr(api_mod.kubectl, "api_reachable", lambda *a, **k: True)
    monkeypatch.setattr(api_mod.kubectl, "get", lambda *a, **k: {"items": []})

    assert api_mod.cluster_status() == {"status": "ok", "nodes": []}
    assert api_mod.pod_status() == {"status": "ok", "pods": []}


# --- telemetry ingest ---------------------------------------------------------


def test_telemetry_ingest_skips_a_bare_scalar_line_and_keeps_the_rest(tmp_path):
    """One weird line must not lose the batch it arrived in.

    `json.loads("5")` is a valid parse returning an int, and the field check
    then raised "TypeError: argument of type 'int' is not iterable" — which
    escaped the receiver and 500'd the whole push, so the shipper retried the
    same doomed batch forever.
    """
    from orchestrator.telemetry.receiver import ingest
    from orchestrator.telemetry.storage import MetricsStorage

    storage = MetricsStorage(str(tmp_path / "m.db"))
    now = datetime.now(UTC).isoformat()
    body = "\n".join([
        json.dumps({"node_id": "n1", "metric_name": "cpu", "metric_value": 1.0, "timestamp": now}),
        "5",
        "null",
        "true",
        json.dumps({"node_id": "n1", "metric_name": "mem", "metric_value": 2.0, "timestamp": now}),
    ])

    count, records = ingest(body, storage)
    assert count == 2
    assert [r["metric_name"] for r in records] == ["cpu", "mem"]


# --- generated configs are secrets now ---------------------------------------


def test_reading_a_generated_config_redacts_the_api_key(
    tmp_aureax_home, network_config_path, monkeypatch, tmp_path
):
    """A generated config carries the node's resolved llama-server key.

    This route is served on an unauthenticated 0.0.0.0 bind, so without
    redaction "GET /api/config/aureax-edge-<id>.yaml" hands every key on the
    fleet to anyone who can reach the dashboard port. The marker is visible
    rather than silent so an operator can tell a redacted view from a config
    that genuinely has no key.
    """
    from orchestrator.config.loader import generated_config_path

    path = generated_config_path("edge-01")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "model:\n"
        "  server:\n"
        "    port: 8080\n"
        "    api_key: aureax_sk_realkey12345678\n"
    )

    resp = dash_client(network_config_path).get("/api/config/aureax-edge-edge-01.yaml")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "generated"
    assert body["redacted"] is True
    assert "<redacted" in body["content"]
    # Nowhere in the response — not the content, not a path, not an echo.
    assert "aureax_sk_realkey" not in resp.text


def test_reading_a_template_is_not_redacted(tmp_aureax_home, network_config_path):
    """Templates carry the literal "${api_key}", which is not a secret.

    Redacting them would break the editor: the placeholder line is how auth is
    turned on for a node, so an operator has to be able to see and save it.
    """
    from pathlib import Path

    from orchestrator.dashboard import config_api

    resp = dash_client(network_config_path).get("/api/config/aureax-edge.yaml")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["kind"] == "edge-template"
    assert body["redacted"] is False
    # Served byte-identical, placeholder included.
    assert body["content"] == Path(config_api.CONFIGS_DIR / "aureax-edge.yaml").read_text()


def test_lifecycle_nodes_lists_every_enabled_target(tmp_aureax_home, network_config_path):
    """/api/lifecycle/nodes had no automated coverage, so a signature drift in
    the `node_role` helper it borrows 500'd the live endpoint while the whole
    suite stayed green. Exercised through the request path on purpose — direct
    handler calls would not have caught it either.
    """
    from orchestrator.config.loader import load_network_config

    resp = dash_client(network_config_path).get("/api/lifecycle/nodes")

    assert resp.status_code == 200
    cfg = load_network_config(str(network_config_path))
    assert {n["node_id"] for n in resp.json()} == {
        t.node_id for t in cfg.ssh.enabled_targets
    }


# --- routing and node ingest --------------------------------------------------
# The HTTPS listener is bound to the orchestrator's LAN address because edge
# nodes push telemetry and logs to it, and the dashboard is deliberately
# unauthenticated (schema.py: server.dashboard.host).


def _network_yaml(tmp_path, extra: str = "") -> str:
    """A minimal, valid network config on disk.

    One target on loopback: `LifecycleManager.status()` TCP-probes every target,
    so pointing the fleet at the real hosts costs 0.3s of connect timeout per
    node in tests that only care about a status code.
    """
    path = tmp_path / "aureax-network.yaml"
    path.write_text(
        "network:\n  name: t\n  version: '0.1.0'\n"
        "model_registry:\n  models: []\n"
        "ssh:\n  targets:\n    - node_id: edge-01\n      host: 127.0.0.1\n"
        + extra
    )
    return str(path)


def test_a_control_route_answers_with_no_headers_at_all(tmp_aureax_home, tmp_path):
    """No token, no cookie, no login: opening https://<host>:7001/ is the whole
    access story, and a control route has to answer a bare request. TLS
    authenticates the SERVER to the fleet; it authorizes no caller."""
    app = create_app(network_config_path=_network_yaml(tmp_path))

    resp = TestClient(app).get("/api/nodes", headers={})

    assert resp.status_code == 200, resp.text


def test_node_ingest_works(tmp_aureax_home, tmp_path):
    """The two routes the fleet pushes to — the entire reason the dashboard
    binds 0.0.0.0."""
    app = create_app(network_config_path=_network_yaml(tmp_path))
    client = TestClient(app)
    now = datetime.now(UTC).isoformat()

    metric = client.post(
        "/api/v1/telemetry",
        content=json.dumps(
            {"node_id": "edge-01", "metric_name": "cpu", "metric_value": 1.0, "timestamp": now}
        ),
    )
    log = client.post(
        "/api/v1/logs",
        content=json.dumps({"node_id": "edge-01", "level": "info", "text": "hi", "timestamp": now}),
    )

    assert metric.status_code == 200, metric.text
    assert metric.json() == {"ingested": 1}
    assert log.status_code == 200, log.text


def test_an_oversized_ingest_body_is_rejected_before_it_is_buffered(tmp_aureax_home, tmp_path):
    """request.body() had no cap on the routes the fleet posts to."""
    from orchestrator.dashboard import _http_utils

    app = create_app(network_config_path=_network_yaml(tmp_path))
    body = "x" * (_http_utils.MAX_BODY_BYTES + 1)

    resp = TestClient(app).post("/api/v1/logs", content=body)

    assert resp.status_code == 413, resp.text


def test_an_oversized_control_body_is_rejected_before_it_is_parsed(tmp_aureax_home, tmp_path):
    """read_json is the body reader for every state-changing route here.

    The cap has to sit in the read itself: Content-Length is client-supplied
    and absent for a chunked body, so nothing above this bounds the buffer.
    """
    from orchestrator.dashboard import _http_utils

    app = create_app(network_config_path=_network_yaml(tmp_path))
    body = "x" * (_http_utils.MAX_JSON_BYTES + 1)

    resp = TestClient(app).post("/api/lifecycle/start", content=body)

    assert resp.status_code == 413, resp.text


# --- /api/nodes freshness -----------------------------------------------------


def test_last_seen_is_the_newest_sample_not_the_first_metric_alphabetically(tmp_path):
    """last_seen is a MAX over the server-stamped arrival time of each row.

    get_latest_all is ordered by metric_name, so row 0 is whichever metric
    sorts first alphabetically rather than the most recent sample — hence the
    MAX. It is taken over ``received_at`` and never over the node-supplied
    ``timestamp``, which is copied verbatim from the pushed JSONL: a node whose
    clock runs ahead would otherwise pin the dashboard's "online" dot (a 120s
    freshness check on this field) green long after it stopped reporting.
    """
    from orchestrator.telemetry.queries import MetricsQueries
    from orchestrator.telemetry.storage import MetricsStorage

    db = str(tmp_path / "m.db")
    storage = MetricsStorage(db)
    future = (datetime.now(UTC) + timedelta(days=365)).isoformat()
    # Two pushes, so the arrival stamps differ. "cpu" sorts first by
    # metric_name, arrives first, and claims a clock a year ahead.
    storage.insert([
        {"node_id": "edge-01", "metric_name": "cpu", "metric_value": 1.0, "timestamp": future},
    ])
    storage.insert([
        {"node_id": "edge-01", "metric_name": "mem", "metric_value": 2.0,
         "timestamp": datetime.now(UTC).isoformat()},
    ])
    with sqlite3.connect(db) as conn:
        arrivals = [r[0] for r in conn.execute("SELECT received_at FROM metrics ORDER BY id")]

    nodes = api_mod._collect_nodes(MetricsQueries(db))

    assert [n["node_id"] for n in nodes] == ["edge-01"]
    assert nodes[0]["last_seen"] == max(arrivals)
    assert nodes[0]["last_seen"] != future, "a node's own clock decided last_seen"


def test_collect_nodes_groups_the_fleet_from_one_query(tmp_path, monkeypatch):
    """One connection and one query for every node, not one per node.

    Every open dashboard tab polls this route, so it must group the whole fleet
    from a single query. Each node must still get only its own metrics, and its
    own arrival time.
    """
    from orchestrator.telemetry.queries import MetricsQueries
    from orchestrator.telemetry.storage import MetricsStorage

    db = str(tmp_path / "m.db")
    storage = MetricsStorage(db)
    # Both inside the retention window: insert() drives the hourly prune, and a
    # 2020 stamp would be deleted before this could read it back.
    now = datetime.now(UTC)
    old = (now - timedelta(hours=1)).isoformat()
    new = now.isoformat()
    storage.insert([
        {"node_id": "edge-02", "metric_name": "cpu", "metric_value": 9.0, "timestamp": old},
        {"node_id": "edge-01", "metric_name": "cpu", "metric_value": 1.0, "timestamp": old},
        {"node_id": "edge-01", "metric_name": "mem", "metric_value": 2.0, "timestamp": new},
    ])
    with sqlite3.connect(db) as conn:
        arrival = conn.execute("SELECT MAX(received_at) FROM metrics").fetchone()[0]

    queries = MetricsQueries(db)
    opened = []
    real_conn = MetricsQueries._conn
    monkeypatch.setattr(
        MetricsQueries, "_conn",
        lambda self: (opened.append(1), real_conn(self))[1],
    )

    nodes = api_mod._collect_nodes(queries)

    assert len(opened) == 1, f"{len(opened)} connections for a 2-node fleet"
    assert [n["node_id"] for n in nodes] == ["edge-01", "edge-02"]
    assert nodes[0]["metrics"] == {"cpu": 1.0, "mem": 2.0}
    assert nodes[1]["metrics"] == {"cpu": 9.0}, "another node's metrics leaked in"
    # One push, so every row shares one server-stamped arrival time; the nodes'
    # own clocks (old/new above) do not decide this field.
    assert nodes[0]["last_seen"] == arrival
    assert nodes[1]["last_seen"] == arrival


# --- lifespan: what shutdown actually releases --------------------------------


def test_the_lifespan_cancels_the_watchers_and_closes_chat(
    tmp_aureax_home, tmp_path, monkeypatch
):
    """Every other test skips the lifespan, so nothing exercised shutdown.

    ChatService holds an MCP pool per chat target, each with a live
    streamable-http session; close() had no caller at all until this hook, and
    the watcher tasks poll kubectl forever if they are not cancelled.
    """
    import asyncio

    from orchestrator.dashboard import app as app_mod

    started: list[asyncio.Task] = []

    async def fake_start_watchers(app, path):
        started.append(asyncio.create_task(asyncio.sleep(3600)))
        return started

    monkeypatch.setattr(app_mod, "start_watchers", fake_start_watchers)

    app = create_app(network_config_path=_network_yaml(tmp_path))
    closed: list[int] = []

    class StubChat:
        async def close(self):
            closed.append(1)

    app.state.chat = StubChat()

    with TestClient(app):
        pass

    assert started and started[0].cancelled()
    assert closed == [1], "ChatService.close() must be awaited exactly once"


def test_configured_telemetry_retention_reaches_the_store(tmp_aureax_home, tmp_path):
    """This wiring is the only thing giving server.telemetry.storage
    .retention_days an effect; a silent failure reverts the store to its
    30-day class default with nothing to diagnose from."""
    path = _network_yaml(
        tmp_path, "server:\n  telemetry:\n    storage:\n      retention_days: 3\n"
    )

    app = create_app(network_config_path=path)

    assert app.state.metrics_storage.retention_days == 3


# --- lifecycle start/stop vs. a config apply ----------------------------------


@pytest.mark.parametrize("path", ["/api/lifecycle/start", "/api/lifecycle/stop"])
def test_start_and_stop_are_409_while_a_config_apply_runs(
    tmp_aureax_home, tmp_path, monkeypatch, path
):
    """A start/stop racing an apply would relabel nodes mid-push."""
    app = create_app(network_config_path=_network_yaml(tmp_path))

    def never(*a, **k):
        raise AssertionError("the lifecycle must not be touched during an apply")

    monkeypatch.setattr(app.state.lifecycle, "try_start", never)
    monkeypatch.setattr(app.state.lifecycle, "try_stop", never)
    app.state.access_apply.running = True

    resp = TestClient(app).post(path, json={})

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"] == "a config apply is running"


def test_start_is_accepted_when_no_apply_is_running(tmp_aureax_home, tmp_path, monkeypatch):
    app = create_app(network_config_path=_network_yaml(tmp_path))
    monkeypatch.setattr(app.state.lifecycle, "try_start", lambda node=None: "run-7")

    resp = TestClient(app).post("/api/lifecycle/start", json={})

    assert resp.status_code == 202, resp.text
    assert resp.json()["run_id"] == "run-7"


@pytest.mark.parametrize("path", ["/api/lifecycle/start", "/api/lifecycle/stop"])
@pytest.mark.parametrize("node", [5, {"a": 1}, ["edge-01"], True])
def test_a_non_string_node_is_a_400_and_never_a_lifecycle_error(
    tmp_aureax_home, tmp_path, monkeypatch, path, node
):
    """`try_start`/`try_stop` take ``str | None``. Anything else truthy used to
    reach group resolution and fail deep inside the run, where the manager
    records ERROR and a last_error the operator then sees on the dashboard —
    for what was only a malformed request."""
    app = create_app(network_config_path=_network_yaml(tmp_path))

    def never(*a, **k):
        raise AssertionError("a malformed body must not start a run")

    monkeypatch.setattr(app.state.lifecycle, "try_start", never)
    monkeypatch.setattr(app.state.lifecycle, "try_stop", never)

    resp = TestClient(app).post(path, json={"node": node})

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "'node' must be a string"
    assert app.state.lifecycle.state.value != "error"


# --- config read/write: the only gate on these files --------------------------


@pytest.fixture
def config_client(tmp_aureax_home, tmp_path, monkeypatch):
    """A client whose CONFIGS_DIR is a tmp copy, never the repo's.

    A real PUT into configs/ would rewrite the frozen rollback pair, which has
    to stay byte-for-byte what it was (see ARCHITECTURE.md, "2. Configuration:
    per-node files, not one shared template").
    """
    from orchestrator.dashboard import config_api

    configs = (tmp_path / "configs").resolve()
    configs.mkdir()
    network = configs / "aureax-network.yaml"
    network.write_text(
        "network:\n  name: t\n  version: '0.1.0'\n"
        "model_registry:\n  models: []\n"
        "ssh:\n"
        "  targets:\n"
        "    - node_id: edge-01\n      host: 10.77.0.1\n      password: \"test-password-1\"\n"
        "    - node_id: edge-02\n      host: 10.77.0.2\n      password: \"other-pw\"\n"
        "    - node_id: edge-03\n      host: 10.77.0.3\n      password_env: AUREAX_SSH_PASSWORD\n"
    )
    monkeypatch.setattr(config_api, "CONFIGS_DIR", configs)
    app = create_app(network_config_path=str(network))
    client = TestClient(app)
    client.configs_dir = configs
    client.marker = config_api.REDACTION_MARKER
    return client


@pytest.mark.parametrize(
    "name", ["..%2F..%2Fetc%2Fpasswd", "aureax-edge-x.yaml/../../x", "/etc/passwd"]
)
def test_a_traversing_config_name_never_reaches_a_file(config_client, name):
    """_resolve is the only gate on which files this router reads and writes.

    404 or 405: a name carrying a path separator does not match
    /api/config/{name} at all and falls through to the static mount, which
    serves GET and HEAD only. Neither answer touches configs/.
    """
    before = sorted(p.name for p in config_client.configs_dir.iterdir())

    assert config_client.get(f"/api/config/{name}").status_code in (404, 405)
    assert config_client.put(
        f"/api/config/{name}", json={"content": "x: 1\n"}
    ).status_code in (404, 405)
    assert sorted(p.name for p in config_client.configs_dir.iterdir()) == before


def test_saving_a_generated_config_is_403(tmp_aureax_home, network_config_path, monkeypatch, tmp_path):
    """Generated configs are output, not source — and they carry a live key."""
    from orchestrator.config.loader import generated_config_path

    target = generated_config_path("edge-01")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("model:\n  server:\n    port: 8080\n")

    resp = dash_client(network_config_path).put(
        "/api/config/aureax-edge-edge-01.yaml", json={"content": "model: {}\n"}
    )

    assert resp.status_code == 403, resp.text
    assert target.read_text() == "model:\n  server:\n    port: 8080\n"


def test_the_shipped_example_is_readable_but_never_saved_in_place(config_client):
    """The example is listed so a fresh clone has something to open and copy.
    It is git-tracked, so a PUT on it from this unauthenticated route would
    rewrite the repository: it reads like a source file and saves like a
    generated one."""
    example = config_client.configs_dir / "aureax-network.example.yaml"
    original = (config_client.configs_dir / "aureax-network.yaml").read_text()
    example.write_text(original)

    listed = {e["name"]: e for e in config_client.get("/api/config").json()}
    assert listed["aureax-network.example.yaml"]["editable"] is False
    assert listed["aureax-network.yaml"]["editable"] is True
    assert config_client.get("/api/config/aureax-network.example.yaml").status_code == 200

    resp = config_client.put(
        "/api/config/aureax-network.example.yaml", json={"content": "network:\n  name: x\n"}
    )

    assert resp.status_code == 403, resp.text
    assert example.read_text() == original


def test_an_invalid_config_body_is_rejected_and_the_file_is_untouched(config_client):
    path = config_client.configs_dir / "aureax-network.yaml"
    before = path.read_text()

    resp = config_client.put("/api/config/aureax-network.yaml", json={"content": "ssh: [\n"})

    assert resp.status_code == 400, resp.text
    assert resp.json()["ok"] is False
    assert resp.json()["errors"]
    assert path.read_text() == before


def test_a_valid_save_round_trips_byte_for_byte(config_client):
    content = (
        "network:\n  name: t2\n  version: '0.2.0'\n"
        "model_registry:\n  models: []\n"
        "ssh:\n  targets:\n    - node_id: edge-09\n      host: 10.77.0.9\n"
    )

    resp = config_client.put("/api/config/aureax-network.yaml", json={"content": content})

    assert resp.status_code == 200, resp.text
    assert config_client.get("/api/config/aureax-network.yaml").json()["content"] == content


def test_saving_back_a_redacted_view_unedited_leaves_the_file_byte_for_byte(config_client):
    """The round trip that carries a literal password goes through ruamel, and
    ruamel's dump defaults are not this project's layout: dash-offset sequences
    get re-indented and long lines folded at 80 columns. An operator who opened
    a fleet file and saved it without editing anything must get their own file
    back, not a reformatted one."""
    path = config_client.configs_dir / "aureax-network.yaml"
    before = path.read_text()

    redacted = config_client.get("/api/config/aureax-network.yaml").json()
    assert redacted["redacted"] is True, "no secret in the file, so nothing is re-dumped"
    resp = config_client.put("/api/config/aureax-network.yaml",
                             json={"content": redacted["content"]})

    assert resp.status_code == 200, resp.text
    assert path.read_text() == before


# --- aureax-network.yaml carries the fleet's SSH passwords --------------------


def test_reading_the_network_config_redacts_ssh_passwords(config_client):
    """Same leak class as the generated api_key, one file over: this route
    used to hand every LAN host working SSH credentials for the whole fleet."""
    body = config_client.get("/api/config/aureax-network.yaml").json()

    assert body["redacted"] is True
    assert "test-password-1" not in body["content"]
    assert "other-pw" not in body["content"]
    assert body["content"].count("<redacted") == 2
    # password_env names an env var, not a secret.
    assert "AUREAX_SSH_PASSWORD" in body["content"]


def test_saving_back_a_redacted_view_restores_the_real_passwords(config_client):
    """The editor round-trips what GET returned, so without the restore a save
    writes the redaction marker over the credentials the next deploy uses."""
    path = config_client.configs_dir / "aureax-network.yaml"
    redacted = config_client.get("/api/config/aureax-network.yaml").json()["content"]
    edited = redacted.replace("name: t\n", "name: renamed\n")

    resp = config_client.put("/api/config/aureax-network.yaml", json={"content": edited})

    assert resp.status_code == 200, resp.text
    on_disk = path.read_text()
    assert 'password: "test-password-1"' in on_disk
    assert 'password: "other-pw"' in on_disk
    assert "<redacted" not in on_disk
    assert "name: renamed" in on_disk


def test_a_password_typed_over_the_marker_is_kept(config_client):
    path = config_client.configs_dir / "aureax-network.yaml"
    redacted = config_client.get("/api/config/aureax-network.yaml").json()["content"]
    edited = redacted.replace(
        f"password: {config_client.marker}", "password: brand-new", 1
    )

    resp = config_client.put("/api/config/aureax-network.yaml", json={"content": edited})

    assert resp.status_code == 200, resp.text
    on_disk = path.read_text()
    assert "password: brand-new" in on_disk
    assert 'password: "other-pw"' in on_disk, "the second, still-redacted line must be restored"


def test_adding_a_node_with_an_explicit_password_preserves_redacted_nodes(config_client):
    from io import StringIO

    from ruamel.yaml import YAML
    path = config_client.configs_dir / "aureax-network.yaml"
    yaml = YAML()
    doc = yaml.load(config_client.get("/api/config/aureax-network.yaml").json()["content"])
    doc["ssh"]["targets"].insert(0, {"node_id": "edge-04", "host": "10.77.0.4", "password": "added"})
    stream = StringIO()
    yaml.dump(doc, stream)
    resp = config_client.put("/api/config/aureax-network.yaml", json={"content": stream.getvalue()})
    assert resp.status_code == 200, resp.text
    nodes = {n["node_id"]: n for n in yaml.load(path.read_text())["ssh"]["targets"]}
    assert nodes["edge-01"]["password"] == "test-password-1"
    assert nodes["edge-04"]["password"] == "added"



# --- regenerate takes an address or a hostname, not any string ----------------


@pytest.mark.parametrize(
    "value", [1, None, "", "10.77.0.1; rm -rf /", "192.168.1..42", "http://10.77.0.1", "10.77.0.1:7001"]
)
def test_regenerate_rejects_an_unusable_orchestrator_ip(config_client, value):
    """It is written verbatim into every node's telemetry endpoint, so it is
    where the whole fleet ships to after the next Apply."""
    resp = config_client.post("/api/config/regenerate", json={"orchestrator_ip": value})

    assert resp.status_code == 400, resp.text


@pytest.mark.parametrize("value", ["10.77.0.99", "fd00::1", "orchestrator.local"])
def test_regenerate_applies_the_same_rule_as_the_cli(config_client, value, monkeypatch):
    """This endpoint used to call ipaddress.ip_address() directly, which is
    stricter than --orchestrator-ip's rule — so a fleet whose orchestrator is
    named by an mDNS hostname regenerated from `aureax generate` and 400'd from
    the dashboard's Apply button. Both now go through
    loader.validate_orchestrator_host.
    """
    from orchestrator.dashboard import config_api

    seen: list[str] = []
    monkeypatch.setattr(
        config_api, "_generate_serialised",
        lambda _cfg, ip, _accept=False: seen.append(ip) or [],
    )

    resp = config_client.post("/api/config/regenerate", json={"orchestrator_ip": value})

    assert resp.status_code == 200, resp.text
    assert seen == [value], "the validator rejected a value the CLI accepts"


# --- regenerate and a reordered worker list -----------------------------------


@pytest.mark.parametrize(
    ("body", "want"),
    [({}, False), ({"accept_worker_reorder": False}, False),
     ({"accept_worker_reorder": True}, True)],
)
def test_the_reorder_override_reaches_generation_and_is_off_by_default(
    config_client, monkeypatch, body, want,
):
    """Worker order IS the --rpc order, and tensor_split/devices index RPC0..n
    positionally — so swapping two workers silently re-addresses every entry.
    Generation refuses one unless the caller opts in, and the Apply button must
    not opt in on the operator's behalf."""
    from orchestrator.dashboard import config_api

    seen: list[bool] = []
    monkeypatch.setattr(
        config_api, "_generate_serialised",
        lambda _cfg, _ip, accept_worker_reorder=False: (
            seen.append(accept_worker_reorder) or []
        ),
    )

    resp = config_client.post(
        "/api/config/regenerate", json={"orchestrator_ip": "10.77.0.99", **body},
    )

    assert resp.status_code == 200, resp.text
    assert seen == [want]


@pytest.mark.parametrize("value", ["true", 1, None, []])
def test_a_non_boolean_reorder_override_is_a_400(config_client, value):
    """A truthy string would turn the override on for anyone who sent
    "false" — the one value an operator would expect to mean off."""
    resp = config_client.post(
        "/api/config/regenerate",
        json={"orchestrator_ip": "10.77.0.99", "accept_worker_reorder": value},
    )

    assert resp.status_code == 400, resp.text
    assert "accept_worker_reorder" in resp.json()["detail"]


def test_a_refused_reorder_surfaces_the_message_that_names_the_flag(
    config_client, monkeypatch,
):
    """There is deliberately no checkbox: the 400 carries generation's own
    refusal, which shows the before/after order and names the CLI flag, so the
    operator reads what changed rather than clicking past it."""
    from orchestrator.dashboard import config_api

    def _refuse(_cfg, _ip, accept_worker_reorder=False):
        raise ValueError(
            "group qwen35b-a: worker order changed ... re-run with "
            "`aureax generate --accept-worker-reorder`"
        )

    monkeypatch.setattr(config_api, "_generate_serialised", _refuse)

    resp = config_client.post(
        "/api/config/regenerate", json={"orchestrator_ip": "10.77.0.99"},
    )

    assert resp.status_code == 400, resp.text
    assert "--accept-worker-reorder" in resp.json()["detail"]


def test_a_tls_store_fault_is_a_400_carrying_its_remedy(config_client, monkeypatch):
    """Generation mints the CA and every leaf, and the TLS store reports both
    of its unrecoverable states as RuntimeError — a half-present CA pair, and a
    leaf it could not produce. Neither used to be in the except tuple, so the
    one message that names the fix (`aureax tls regenerate --ca`) went to the
    server log and the operator saw "Regenerate failed: 500".
    """
    from orchestrator.dashboard import config_api

    def _half_present_ca(_cfg, _ip, accept_worker_reorder=False):
        raise RuntimeError(
            "fleet CA is half present (ca.pem without ca.key) — "
            "run `aureax tls regenerate --ca`"
        )

    monkeypatch.setattr(config_api, "_generate_serialised", _half_present_ca)

    resp = config_client.post(
        "/api/config/regenerate", json={"orchestrator_ip": "10.77.0.99"},
    )

    assert resp.status_code == 400, resp.text
    assert "aureax tls regenerate --ca" in resp.json()["detail"]


# --- response headers ---------------------------------------------------------


def test_security_headers_are_set_on_api_and_static_responses(tmp_aureax_home, tmp_path):
    """Set from middleware, not a <meta> tag: a meta CSP covers neither the
    JSON routes (GET /api/access/{node_id}/key among them) nor frame-ancestors.

    Every section builds HTML by concatenation with escapeHtml as the only
    barrier, so 'self' is what keeps one missed interpolation from executing.
    """
    app = create_app(network_config_path=_network_yaml(tmp_path))
    client = TestClient(app)

    for resp in (client.get("/api/nodes"), client.get("/")):
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-security-policy"].startswith("default-src 'self'")
        assert "frame-ancestors 'none'" in resp.headers["content-security-policy"]
        assert resp.headers["x-frame-options"] == "DENY"
        assert resp.headers["x-content-type-options"] == "nosniff"


def test_the_static_page_loads_without_a_token(tmp_aureax_home, tmp_path):
    """The browser has to reach the page before it can present anything."""
    app = create_app(network_config_path=_network_yaml(tmp_path))

    resp = TestClient(app).get("/")

    assert resp.status_code == 200
    assert "Aureax" in resp.text


def test_hsts_is_sent_and_pins_nothing_beyond_this_host(tmp_aureax_home, tmp_path):
    """The dashboard is served over TLS, so a downgrade to http:// should be
    refused by the browser rather than merely unusual.

    includeSubDomains and preload are deliberately absent, and their absence is
    the assertion: this host is normally an IP literal (https://<lan-ip>:7001),
    for which browsers ignore HSTS entirely, and when it IS named — `mac.local`
    — includeSubDomains would pin every sibling .local name to HTTPS and break
    unrelated LAN services the operator runs. preload would do the same
    permanently, from a list this deployment can never be removed from.
    """
    app = create_app(network_config_path=_network_yaml(tmp_path))
    client = TestClient(app)

    for resp in (client.get("/api/nodes"), client.get("/")):
        assert resp.status_code == 200, resp.text
        value = resp.headers["strict-transport-security"]
        assert value == "max-age=31536000", value
        assert "includesubdomains" not in value.lower()
        assert "preload" not in value.lower()


def test_a_streamed_response_keeps_every_chunk_and_one_set_of_headers():
    """The SSE routes (lifecycle logs, chat) stream for the life of a browser
    tab, so the header middleware must touch response.start only.

    Asserted against a scripted ASGI app rather than a real SSE route because
    TestClient runs the application to completion before returning — an
    endless stream would hang the suite, which is exactly why this middleware
    is pure ASGI and not BaseHTTPMiddleware in the first place.
    """
    chunks = [b"event: log\n", b"data: 1\n\n", b"data: 2\n\n"]

    async def streaming_app(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        for i, chunk in enumerate(chunks):
            await send({"type": "http.response.body", "body": chunk,
                        "more_body": i < len(chunks) - 1})

    sent: list = []

    async def record(message):
        sent.append(message)

    asyncio.run(app_mod.SecurityHeadersMiddleware(streaming_app)(
        {"type": "http"}, None, record,
    ))

    start, *body = sent
    assert [m["body"] for m in body] == chunks, "the middleware buffered or dropped a frame"
    assert [m.get("more_body", False) for m in body] == [True, True, False]
    names = [k for k, _ in start["headers"]]
    assert names.count(b"strict-transport-security") == 1
    for header in app_mod._SECURITY_HEADERS:
        assert header in start["headers"]


def test_a_non_http_scope_passes_through_untouched():
    """Lifespan and websocket scopes carry no response.start to decorate, and
    wrapping ``send`` for them would put this middleware in the path of the
    startup/shutdown handshake."""
    seen: list = []

    async def inner(scope, receive, send):
        seen.append((scope, receive, send))

    async def sentinel_send(_message):  # pragma: no cover - never called
        raise AssertionError("send was called for a lifespan scope")

    scope = {"type": "lifespan"}
    asyncio.run(app_mod.SecurityHeadersMiddleware(inner)(scope, "recv", sentinel_send))

    assert seen == [(scope, "recv", sentinel_send)], (
        "a non-http scope was rewrapped instead of passed straight through"
    )


# --- TLS: the dashboard has no plaintext mode ---------------------------------


def test_run_cannot_be_called_without_a_certificate_and_key():
    """ssl_certfile/ssl_keyfile are keyword-only with NO defaults, which is the
    whole guarantee: TLS is mandatory fleet-wide, so a caller must not be able
    to start this listener in plaintext by forgetting an argument. Every node
    ships telemetry and logs to https://<orchestrator>:7001.
    """
    with pytest.raises(TypeError):
        app_mod.run(port=0)

    with pytest.raises(TypeError):
        app_mod.run(port=0, ssl_certfile="/tmp/o.pem")

    with pytest.raises(TypeError):
        app_mod.run(port=0, ssl_keyfile="/tmp/o.key")


def test_run_hands_uvicorn_the_certificate_and_key(tmp_aureax_home, tmp_path, monkeypatch):
    """uvicorn is what terminates TLS: the ASGI app below it never sees a
    socket, so these two kwargs arriving intact is the only thing that makes
    the fleet's port HTTPS."""
    servers = _fake_uvicorn(monkeypatch)
    _fake_binder(monkeypatch)

    app_mod.run(
        port=7001,
        metrics_db_path=str(tmp_path / "metrics.db"),
        network_config_path=_network_yaml(tmp_path),
        chat_db_path=str(tmp_path / "chat.db"),
        log_db_path=str(tmp_path / "log.db"),
        host="192.0.2.7",
        ssl_certfile=str(tmp_path / "orchestrator.pem"),
        ssl_keyfile=str(tmp_path / "orchestrator.key"),
    )

    seen = servers[0].config.kwargs
    assert seen["ssl_certfile"] == str(tmp_path / "orchestrator.pem")
    assert seen["ssl_keyfile"] == str(tmp_path / "orchestrator.key")
    # The bind that carries fleet ingest is this ONE address — the one the
    # certificate names and every node's ${orchestrator_url} points at — not a
    # wildcard, which would leave the loopback listener nothing to bind.
    assert seen["host"] == "192.0.2.7"
    assert seen["port"] == 7001
    assert servers[0].config.app is not None


def test_the_sse_routes_and_the_static_mount_survive_the_tls_change(
    tmp_aureax_home, tmp_path,
):
    """Nothing about TLS reaches the ASGI layer, but the headers middleware sits
    in front of both, so the mount and the streaming routes are asserted here
    rather than assumed.

    Routes are read from the OpenAPI schema, not ``app.routes``, for the reason
    test_expected_routes_are_mounted records: the route list is FastAPI's own
    bookkeeping and has already changed shape once under this app. The static
    mount is not in that schema, so it is asserted by fetching a file.
    """
    app = create_app(network_config_path=_network_yaml(tmp_path))

    paths = set(app.openapi()["paths"])
    assert "/api/lifecycle/logs" in paths
    assert "/api/chat/sessions/{session_id}/stream" in paths

    resp = TestClient(app).get("/utils.js")
    assert resp.status_code == 200, resp.text
    assert resp.headers["strict-transport-security"] == "max-age=31536000"


# --- shutdown vs. an in-flight lifecycle op -----------------------------------


class _FakeLifecycle:
    """Enough of LifecycleManager for the drain: a state, a busy flag, a task.

    ``wait`` mirrors the real one (shielded, bounded) rather than being a stub:
    the drain calls it, and the point of the method existing at all is that
    app.py no longer reaches for ``_task``.
    """

    def __init__(self, state, coro) -> None:
        import asyncio

        self.state = state
        self.busy = True
        self.cancel_calls = 0
        self._task = asyncio.create_task(coro)

    def cancel(self) -> None:
        self.cancel_calls += 1

    async def wait(self, timeout=None) -> None:
        import asyncio

        if timeout is None:
            await asyncio.shield(self._task)
            return
        await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)


async def test_shutdown_cancels_an_in_flight_start():
    """The worker sits in asyncio.to_thread on the loop's DEFAULT executor and
    can wait an hour, so loop close (which ends in shutdown_default_executor)
    hung on Ctrl-C during a Start."""
    import asyncio

    from orchestrator.dashboard import app as app_mod
    from orchestrator.lifecycle import State

    lc = _FakeLifecycle(State.STARTING, asyncio.sleep(0))

    await app_mod._drain_lifecycle(lc)

    assert lc.cancel_calls == 1


async def test_shutdown_waits_out_a_stop_instead_of_cancelling_it():
    """Cancelling mid-STOP makes _wait_pods_terminated raise past
    k3d.stop_cluster, leaving the selectors patched and the control plane up —
    a stop that runs to completion today would abort half-done."""
    import asyncio

    from orchestrator.dashboard import app as app_mod
    from orchestrator.lifecycle import State

    lc = _FakeLifecycle(State.STOPPING, asyncio.sleep(0))

    await app_mod._drain_lifecycle(lc)

    assert lc.cancel_calls == 0


async def test_shutdown_gives_up_on_a_worker_that_never_returns(monkeypatch):
    """The wait is bounded: a worker can be inside a blocking kubectl or SSH
    call with no cancel point, and an unbounded await reproduces the hang."""
    import asyncio

    from orchestrator.dashboard import app as app_mod
    from orchestrator.lifecycle import State

    monkeypatch.setattr(app_mod, "_LIFECYCLE_DRAIN_TIMEOUT", 0.05)
    lc = _FakeLifecycle(State.STOPPING, asyncio.sleep(3600))

    await asyncio.wait_for(app_mod._drain_lifecycle(lc), timeout=2.0)

    lc._task.cancel()


async def test_shutdown_ignores_an_idle_lifecycle():
    import asyncio

    from orchestrator.dashboard import app as app_mod
    from orchestrator.lifecycle import State

    lc = _FakeLifecycle(State.STARTING, asyncio.sleep(0))
    lc.busy = False

    await app_mod._drain_lifecycle(lc)
    await app_mod._drain_lifecycle(None)

    assert lc.cancel_calls == 0
    lc._task.cancel()


async def test_the_drain_joins_through_the_public_wait(monkeypatch):
    """app.py used to reach for ``lc._task``, making the manager's private
    attribute part of another module's contract."""
    import asyncio

    from orchestrator.dashboard import app as app_mod
    from orchestrator.lifecycle import State

    lc = _FakeLifecycle(State.STOPPING, asyncio.sleep(0))
    seen: list[float | None] = []

    async def wait(timeout=None):
        seen.append(timeout)

    lc.wait = wait

    await app_mod._drain_lifecycle(lc)

    assert seen == [app_mod._LIFECYCLE_DRAIN_TIMEOUT]
    lc._task.cancel()


async def test_lifecycle_wait_does_not_cancel_the_run_it_gave_up_on():
    """The real manager's wait() is shielded: a stop cancelled mid-flight
    leaves the DaemonSet selectors patched and the control plane up, so the
    bounded drain must give up on the worker without killing it."""
    import asyncio

    from orchestrator.lifecycle import LifecycleManager

    mgr = LifecycleManager(network_config_path=None)

    # Nothing has ever run: wait() must return rather than raise.
    await mgr.wait(0.01)

    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(3600)

    mgr._task = asyncio.create_task(slow())
    await started.wait()

    with pytest.raises(asyncio.TimeoutError):
        await mgr.wait(0.05)
    assert not mgr._task.done(), "the shield must keep the worker alive"

    mgr._task.cancel()


def test_a_non_utf8_json_body_is_a_400_not_a_500(tmp_aureax_home, tmp_path):
    """UnicodeDecodeError and JSONDecodeError are unrelated ValueError
    subclasses, so raw bytes 500'd out of the helper written to stop exactly
    that: a client's malformed request reported as a server fault."""
    app = create_app(network_config_path=_network_yaml(tmp_path))

    resp = TestClient(app).post("/api/lifecycle/start", content=b"\xff\xfe\x00")

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "invalid JSON body"


# --- two listeners, one process ----------------------------------------------
# `aureax dashboard` serves the SAME app twice: HTTPS on the orchestrator's LAN
# address, which is what every node's ${orchestrator_url} names, and plain HTTP
# on loopback so a browser on the Mac itself needs no certificate. Both are one
# process on one port, which is what keeps `aureax stop`'s kill-by-port honest.


class _RecordedConfig:
    """Stand-in for ``uvicorn.Config`` that only remembers its arguments."""

    def __init__(self, app, **kwargs) -> None:
        self.app = app
        self.kwargs = kwargs


class _RecordedServer:
    """Stand-in for ``uvicorn.Server``: serves nothing, records the sockets."""

    def __init__(self, config) -> None:
        self.config = config
        self.should_exit = False
        self.served: list = []

    async def serve(self, sockets=None) -> None:
        self.served = list(sockets or [])


def _fake_uvicorn(monkeypatch) -> list[_RecordedServer]:
    """Capture the servers ``run()`` builds instead of starting any."""
    import uvicorn

    servers: list[_RecordedServer] = []

    def make_server(config):
        server = _RecordedServer(config)
        servers.append(server)
        return server

    monkeypatch.setattr(uvicorn, "Config", _RecordedConfig)
    monkeypatch.setattr(uvicorn, "Server", make_server)
    return servers


class _FakeSocket:
    """A stand-in for a bound listening socket, closeable like the real one."""

    def __init__(self, host: str, port: int) -> None:
        self.address = (host, port)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fake_binder(monkeypatch) -> list[tuple[str, int]]:
    """Record every requested bind without touching the network.

    ``_listen_socket`` is exercised for real by the tests below it; here the
    question is only which addresses ``run()`` asks for, and a real bind would
    need a routable LAN address this machine may not have.
    """
    requested: list[tuple[str, int]] = []

    def fake(host: str, port: int):
        requested.append((host, port))
        return _FakeSocket(host, port)

    monkeypatch.setattr(app_mod, "_listen_socket", fake)
    return requested


def test_run_serves_the_fleet_over_https_and_this_machine_over_plain_loopback(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """One process, one port, two listeners.

    The HTTPS one is the fleet's: it is bound to the single address the
    certificate was minted for, not a wildcard, because a wildcard socket owns
    loopback too and would leave the second listener nothing to bind. The plain
    one is loopback-only, so `http://localhost:7001` opens from the
    orchestrator with no certificate and still never crosses the LAN.
    """
    servers = _fake_uvicorn(monkeypatch)
    requested = _fake_binder(monkeypatch)

    app_mod.run(
        port=7001,
        network_config_path=_network_yaml(tmp_path),
        host="192.0.2.7",
        ssl_certfile="/tls/orchestrator.pem",
        ssl_keyfile="/tls/orchestrator.key",
    )

    assert requested == [("192.0.2.7", 7001), ("127.0.0.1", 7001), ("::1", 7001)]
    assert len(servers) == 2, "the fleet listener and the operator's listener"
    https, loopback = servers
    assert https.config.kwargs["host"] == "192.0.2.7"
    assert https.config.kwargs["ssl_certfile"] == "/tls/orchestrator.pem"
    assert https.config.kwargs["ssl_keyfile"] == "/tls/orchestrator.key"
    assert [sock.address for sock in https.served] == [("192.0.2.7", 7001)]

    assert loopback.config.kwargs["host"] == "127.0.0.1"
    assert loopback.config.kwargs.get("ssl_certfile") is None, "loopback is plaintext"
    assert loopback.config.kwargs.get("ssl_keyfile") is None
    assert [sock.address for sock in loopback.served] == [("127.0.0.1", 7001), ("::1", 7001)]
    # Closed on the way out, so an in-process caller (this test) is not left
    # holding the port after run() returns.
    assert all(sock.closed for sock in https.served + loopback.served)

    # Same port for both, so `aureax stop` still finds one listener-holding pid
    # on server.dashboard.port (lifecycle._kill_process_on_port).
    assert https.config.kwargs["port"] == loopback.config.kwargs["port"] == 7001


def test_run_lets_the_fleet_reach_the_dashboard_by_the_address_it_serves(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """`host` is the address the orchestrator leaf was minted for AND the one
    stamped into every node's telemetry endpoint, so it has to be a name the
    browser boundary answers to. A hostname is legal there, and a node dialing
    one sends it as its Host header — matching neither loopback nor the
    socket's own IP literal."""
    servers = _fake_uvicorn(monkeypatch)
    _fake_binder(monkeypatch)

    app_mod.run(
        port=7001,
        network_config_path=_network_yaml(tmp_path),
        host="mac-orchestrator.local",
        ssl_certfile="/tls/orchestrator.pem",
        ssl_keyfile="/tls/orchestrator.key",
    )

    client = TestClient(servers[0].config.app, base_url="https://127.0.0.1:7001")
    assert client.get("/api/config",
                      headers={"host": "mac-orchestrator.local:7001"}).status_code == 200
    assert client.get("/api/config",
                      headers={"host": "attacker.example:7001"}).status_code == 403


def test_only_the_https_listener_gets_the_apps_lifespan(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """A second lifespan would start every background watcher twice."""
    servers = _fake_uvicorn(monkeypatch)
    _fake_binder(monkeypatch)

    app_mod.run(
        port=7001,
        network_config_path=_network_yaml(tmp_path),
        host="192.0.2.7",
        ssl_certfile="/tls/orchestrator.pem",
        ssl_keyfile="/tls/orchestrator.key",
    )

    https, loopback = servers
    assert https.config.app is not loopback.config.app
    assert isinstance(loopback.config.app, app_mod.LifespanOnlyShim)
    # The same app object underneath: one LifecycleManager, one ChatService,
    # one set of stores, whichever door a request came through.
    assert loopback.config.app.app is https.config.app


def test_loopback_http_false_serves_the_fleet_listener_alone(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    servers = _fake_uvicorn(monkeypatch)
    requested = _fake_binder(monkeypatch)

    app_mod.run(
        port=7001,
        network_config_path=_network_yaml(tmp_path),
        host="192.0.2.7",
        ssl_certfile="/tls/orchestrator.pem",
        ssl_keyfile="/tls/orchestrator.key",
        loopback_http=False,
    )

    assert requested == [("192.0.2.7", 7001)]
    assert len(servers) == 1
    assert servers[0].config.kwargs["ssl_certfile"] == "/tls/orchestrator.pem"


def test_a_loopback_dashboard_host_serves_https_only(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """server.dashboard.host = 127.0.0.1 is the SSH-tunnel setup the schema
    describes: HTTPS already owns the port on loopback, so a plaintext listener
    beside it would be an EADDRINUSE against this very process."""
    servers = _fake_uvicorn(monkeypatch)
    requested = _fake_binder(monkeypatch)

    app_mod.run(
        port=7001,
        network_config_path=_network_yaml(tmp_path),
        host="127.0.0.1",
        ssl_certfile="/tls/orchestrator.pem",
        ssl_keyfile="/tls/orchestrator.key",
    )

    assert requested == [("127.0.0.1", 7001)]
    assert len(servers) == 1


def test_a_wildcard_host_is_refused_rather_than_silently_dropping_a_listener(
    tmp_aureax_home, tmp_path, monkeypatch,
) -> None:
    """0.0.0.0 owns 127.0.0.1:<port> as well, so the two listeners cannot both
    exist — and the resolution ("bind the orchestrator's own address") belongs
    in the message, not in a listener that quietly did not start."""
    _fake_uvicorn(monkeypatch)
    requested = _fake_binder(monkeypatch)

    with pytest.raises(ValueError, match="wildcard"):
        app_mod.run(
            port=7001,
            network_config_path=_network_yaml(tmp_path),
            host="0.0.0.0",
            ssl_certfile="/tls/orchestrator.pem",
            ssl_keyfile="/tls/orchestrator.key",
        )

    assert requested == [], "nothing may be bound before the contradiction is reported"


# --- the shim that keeps the lifespan single ---------------------------------


def test_the_shim_completes_a_lifespan_without_ever_calling_the_app() -> None:
    """It has to answer startup and shutdown itself: uvicorn waits for both
    replies, and forwarding them would run create_app's lifespan a second
    time — a second copy of every watcher, polling kubectl in parallel."""
    calls: list[dict] = []

    async def app(scope, receive, send):
        calls.append(scope)

    shim = app_mod.LifespanOnlyShim(app)
    sent: list[dict] = []
    inbox = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]

    async def receive():
        return inbox.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(shim({"type": "lifespan", "state": {}}, receive, send))

    assert calls == [], "the app's own lifespan must not run on this listener"
    assert [m["type"] for m in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


def test_the_shim_forwards_an_http_scope_untouched() -> None:
    """Everything that is not a lifespan is the real dashboard: same routes,
    same middleware, same app.state, whichever listener took the connection."""
    seen: list[tuple] = []

    async def app(scope, receive, send):
        seen.append((scope, receive, send))

    shim = app_mod.LifespanOnlyShim(app)
    scope = {"type": "http", "path": "/api/cluster"}

    async def receive():
        return {}

    async def send(_message):
        return None

    asyncio.run(shim(scope, receive, send))

    assert seen == [(scope, receive, send)]


# --- binding: a busy port names the process holding it -----------------------


def test_a_busy_port_is_reported_with_the_pid_that_holds_it(
    tmp_aureax_home, tmp_path,
) -> None:
    """The operator's report was `[Errno 48] address already in use` and
    nothing else — uvicorn logs the OSError and exits 1, which does not say
    that the thing on the port is usually their own earlier dashboard."""
    if shutil.which("lsof") is None:
        pytest.skip("needs lsof to name the holder")
    holder = socket.socket()
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]

    try:
        with pytest.raises(OSError) as excinfo:
            # host=127.0.0.1 so this test needs no LAN address; the fleet
            # listener binds it and finds `holder` already there.
            app_mod.run(
                port=port,
                network_config_path=_network_yaml(tmp_path),
                host="127.0.0.1",
                ssl_certfile="/tls/orchestrator.pem",
                ssl_keyfile="/tls/orchestrator.key",
            )
    finally:
        holder.close()

    message = str(excinfo.value)
    assert excinfo.value.errno == errno.EADDRINUSE
    assert "address already in use" in message
    assert str(os.getpid()) in message, f"the holding pid has to be named: {message}"
    assert "--port" in message, "and a way out that does not require finding it"


def test_the_loopback_pair_binds_both_families_when_the_host_has_them() -> None:
    """Firefox resolves localhost to ::1 first. A refused ::1 does fall back,
    but binding both is what makes http://localhost open with no round trip
    spent being refused — and a host without IPv6 must still get 127.0.0.1."""
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    socks = app_mod._loopback_sockets(port, logging.getLogger("test"))
    try:
        bound = [sock.getsockname()[0] for sock in socks]
    finally:
        for sock in socks:
            sock.close()

    assert bound[0] == "127.0.0.1", "the v4 loopback is not optional"
    if _has_ipv6_loopback():
        assert bound == ["127.0.0.1", "::1"]
    else:
        assert bound == ["127.0.0.1"], "no IPv6 here is not a failed dashboard"


def _has_ipv6_loopback() -> bool:
    """Whether ::1 can be bound on this machine at all."""
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


# --- /api/nodes/{node_id}: one node's whole story ----------------------------
# Four sources joined on the node id — config, kubectl, metrics, logs — with no
# fleet I/O of its own. The tests below pin the join, the per-section status
# fields that keep one dead source from blanking the page, and the promise that
# opening the view costs the Overview tab no extra kubectl.


def _detail_yaml(tmp_path, targets: str, node_config: str = "") -> str:
    """A fleet file plus the per-node config its target points at.

    The detail route reads BOTH — identity is the target's fields joined with
    the node's own config — so a fixture with only one of them exercises half
    the handler.
    """
    (tmp_path / "edge-01.yaml").write_text(
        "node:\n  id: edge-01\n  role: edge_executor\n" + node_config
    )
    path = tmp_path / "aureax-network.yaml"
    path.write_text(
        "network:\n  name: t\n  version: '0.1.0'\n"
        "model_registry:\n  models: []\n"
        "ssh:\n  targets:\n" + targets
    )
    return str(path)


_ONE_TARGET = (
    "    - node_id: edge-01\n      host: 10.77.0.1\n"
    "      config_file: edge-01.yaml\n      display_name: Edge One\n      tags: [bench]\n"
)


def _fake_kubectl(monkeypatch, nodes: list, pods: list, *, reachable: bool = True):
    """Serve fixed `kubectl get` payloads; return the resource-name call log."""
    calls: list[str] = []

    def get(resource, *_a, **_k):
        calls.append(resource)
        return {"items": nodes if resource == "nodes" else pods}

    monkeypatch.setattr(api_mod.kubectl, "api_reachable", lambda *a, **k: reachable)
    monkeypatch.setattr(api_mod.kubectl, "get", get)
    return calls


def _k8s_node(name: str, ip: str, **status) -> dict:
    return {
        "metadata": {"name": name},
        "status": {
            "addresses": [{"type": "InternalIP", "address": ip}],
            "conditions": [{"type": "Ready", "status": "True"}],
            "nodeInfo": {"kernelVersion": "6.6.51", "osImage": "Debian 12",
                         "architecture": "arm64", "kubeletVersion": "v1.31.4+k3s1",
                         "containerRuntimeVersion": "containerd://2.0.0"},
            "capacity": {"cpu": "4", "memory": "16269312Ki"},
            **status,
        },
    }


def _k8s_pod(name: str, node_name: str, host_ip: str) -> dict:
    return {
        "metadata": {"name": name},
        "spec": {"nodeName": node_name},
        "status": {
            "phase": "Running",
            "hostIP": host_ip,
            "startTime": "2026-09-10T08:00:00Z",
            "containerStatuses": [{
                "name": "aureax-client",
                "ready": True,
                "restartCount": 1,
                "image": "aureax-client:dev",
                "imageID": "sha256:deadbeef",
                "state": {"running": {"startedAt": "2026-09-10T09:30:00Z"}},
                "lastState": {"terminated": {
                    "reason": "OOMKilled", "exitCode": 137,
                    "finishedAt": "2026-09-10T09:29:00Z",
                }},
            }],
        },
    }


@pytest.mark.parametrize("node_id", ["nope", "edge-02"])
def test_a_node_this_fleet_does_not_deploy_to_is_a_404(
    tmp_aureax_home, tmp_path, node_id,
):
    """Unknown and disabled answer alike: a disabled target has no pods, no
    keys and no certificate, so a per-node view of it would be about nothing.
    The 404 is also what bounds the detail cache's key space to real ids."""
    path = _detail_yaml(
        tmp_path,
        _ONE_TARGET + "    - node_id: edge-02\n      host: 10.77.0.2\n      enabled: false\n",
    )

    resp = TestClient(create_app(network_config_path=path)).get(f"/api/nodes/{node_id}")

    assert resp.status_code == 404, resp.text
    assert node_id in resp.json()["detail"]


def test_an_unreadable_fleet_file_is_a_503_not_a_500(tmp_aureax_home, tmp_path):
    """The fleet file is re-read per request (mtime-keyed), so it can go bad
    under a running dashboard — mid-edit is exactly when this view is open."""
    path = _detail_yaml(tmp_path, _ONE_TARGET)
    client = TestClient(create_app(network_config_path=path))
    assert client.get("/api/nodes/edge-01").status_code == 200

    (tmp_path / "aureax-network.yaml").write_text("ssh: [this is not a fleet]\n")

    resp = client.get("/api/nodes/edge-01")

    assert resp.status_code == 503, resp.text
    assert "unreadable" in resp.json()["detail"]


def test_an_unreachable_api_server_is_not_a_node_with_no_pods(
    tmp_aureax_home, tmp_path, monkeypatch,
):
    """`kubectl.get` returns {} for any failure, so "we could not ask" and
    "nothing is scheduled here" arrive identically — and the second, rendered
    on a detail page, reads as this node having been dropped from the fleet."""
    _fake_kubectl(monkeypatch, [], [], reachable=False)
    path = _detail_yaml(tmp_path, _ONE_TARGET)

    k8s = TestClient(create_app(network_config_path=path)).get("/api/nodes/edge-01").json()["k8s"]

    assert k8s["status"] == "error"
    assert k8s["message"] == api_mod._UNREACHABLE["message"]
    assert (k8s["node"], k8s["pods"], k8s["matched_by"]) == (None, [], None)


def test_a_reachable_cluster_that_does_not_know_this_node_says_so(
    tmp_aureax_home, tmp_path, monkeypatch,
):
    """The other half: asked and answered, this host is on no node and no pod.
    `matched_by` is None rather than a guess, so the view can say which of the
    three matches identified the node instead of implying one did."""
    _fake_kubectl(monkeypatch, [_k8s_node("other", "10.77.0.9")], [])
    path = _detail_yaml(tmp_path, _ONE_TARGET)

    k8s = TestClient(create_app(network_config_path=path)).get("/api/nodes/edge-01").json()["k8s"]

    assert k8s["status"] == "ok"
    assert (k8s["matched_by"], k8s["node"], k8s["pods"]) == (None, None, [])


def test_an_internal_ip_match_keeps_what_the_two_grids_throw_away(
    tmp_aureax_home, tmp_path, monkeypatch,
):
    """The whole point of the route: the cached kubectl payloads already carry
    the box's identity and the pod's timing, and /api/cluster and /api/pods
    discard both. `restarts: 1` with a Running pod is indistinguishable from a
    healthy one without the exit code and the time it happened.
    """
    _fake_kubectl(
        monkeypatch,
        [_k8s_node("aureax-edge-01", "10.77.0.1",
                   conditions=[{"type": "Ready", "status": "True"},
                               {"type": "MemoryPressure", "status": "True"},
                               {"type": "DiskPressure", "status": "False"}])],
        [_k8s_pod("aureax-client-abc", "aureax-edge-01", "10.77.0.1")],
    )
    path = _detail_yaml(tmp_path, _ONE_TARGET)

    resp = TestClient(create_app(network_config_path=path)).get("/api/nodes/edge-01")
    k8s = resp.json()["k8s"]
    container = k8s["pods"][0]["containers"][0]

    assert k8s["matched_by"] == "internal-ip"
    assert k8s["node"]["kernel_version"] == "6.6.51"
    assert k8s["node"]["status"] == "Ready"
    assert k8s["node"]["pressure"] == ["MemoryPressure"], "only conditions that are True"
    assert k8s["pods"][0]["start_time"] == "2026-09-10T08:00:00Z"
    assert k8s["pods"][0]["host_ip"] == "10.77.0.1"
    assert container["started_at"] == "2026-09-10T09:30:00Z", "the CURRENT process"
    # The reason and the exit code are already in last_error; the time they
    # happened is the one thing that string cannot carry.
    assert container["last_error"] == "OOMKilled (exit 137)"
    assert container["last_finished_at"] == "2026-09-10T09:29:00Z"
    # Every node builds aureax-client:dev natively, so image and imageID differ
    # between nodes by construction; kubectl.get_pods dropped them for that
    # reason and reintroducing them here would resurrect the false positive.
    assert "imageID" not in resp.text and '"image"' not in resp.text


def test_a_node_missing_from_the_node_listing_is_still_found_by_its_pods(
    tmp_aureax_home, tmp_path, monkeypatch,
):
    """A node can drop out of `kubectl get nodes` while its pods are still
    reported. The pod's hostIP places them; it says nothing about the node
    object, so that stays null rather than being invented from the pod."""
    _fake_kubectl(monkeypatch, [], [_k8s_pod("aureax-client-abc", "aureax-edge-01", "10.77.0.1")])
    path = _detail_yaml(tmp_path, _ONE_TARGET)

    k8s = TestClient(create_app(network_config_path=path)).get("/api/nodes/edge-01").json()["k8s"]

    assert k8s["matched_by"] == "host-ip"
    assert k8s["node"] is None
    assert [p["name"] for p in k8s["pods"]] == ["aureax-client-abc"]


def test_identity_still_renders_when_neither_store_is_open(tmp_path, monkeypatch):
    """A corrupt metrics or log database disables its own card only — the same
    rule create_app already follows for the tabs. Identity is pure config, so
    it is the one section that answers on a dashboard with no storage at all.
    """
    _fake_kubectl(monkeypatch, [], [])
    junk = tmp_path / "junk"
    junk.mkdir()
    for name in ("metrics.db", "system.db"):
        (junk / name).write_bytes(b"not a sqlite database")
    path = _detail_yaml(
        tmp_path, _ONE_TARGET,
        "model:\n  name: Qwen3.5-4B\n  filename: q4.gguf\n  quantization: Q4_K_M\n"
        "  server:\n    context_size: ${context_size}\n    gpu_layers: 99\n",
    )
    app = create_app(
        network_config_path=path,
        metrics_db_path=str(junk / "metrics.db"),
        log_db_path=str(junk / "system.db"),
        chat_db_path=str(tmp_path / "c.db"),
    )

    body = TestClient(app).get("/api/nodes/edge-01").json()

    assert body["metrics"]["status"] == "unavailable"
    assert body["logs"]["status"] == "unavailable"
    assert body["identity"]["display_name"] == "Edge One"
    assert body["identity"]["tags"] == ["bench"]
    assert body["identity"]["role"] == "edge_executor"
    assert body["identity"]["model"]["name"] == "Qwen3.5-4B"
    assert body["identity"]["model"]["gpu_layers"] == 99
    # A fleet that has never run `aureax generate` falls back to the SOURCE
    # template, where per-node values are still placeholders. "${context_size}"
    # rendered as this node's context reads as a fact.
    assert body["identity"]["model"]["context_size"] is None


def test_a_store_that_fails_mid_read_blanks_its_card_and_says_why(
    tmp_aureax_home, tmp_path, monkeypatch, caplog,
):
    """An OPEN store can still fail: locked mid-write, truncated by a power cut,
    or on a disk that filled. That is a different path from the unopened store
    above and it used to 500 the whole page — losing identity and the K8s
    placement, which is the pair the view is opened for. The WARNING is the only
    trace of it, since the card itself says nothing about the cause."""
    _fake_kubectl(monkeypatch, [], [])
    path = _detail_yaml(tmp_path, _ONE_TARGET)
    app = create_app(
        network_config_path=path,
        metrics_db_path=str(tmp_path / "m.db"),
        log_db_path=str(tmp_path / "s.db"),
        chat_db_path=str(tmp_path / "c.db"),
    )

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(app.state.log_store, "recent", boom)

    with caplog.at_level(logging.WARNING, logger="aureax.dashboard"):
        resp = TestClient(app).get("/api/nodes/edge-01")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["logs"] == {"status": "unavailable", "entries": []}
    assert body["identity"]["display_name"] == "Edge One", "the page survived"
    assert body["metrics"]["status"] == "ok"
    assert "edge-01" in caplog.text and "database is locked" in caplog.text


def test_detail_last_seen_is_the_arrival_time_the_overview_card_shows(
    tmp_aureax_home, tmp_path, monkeypatch,
):
    """Both views' freshness dot is a 120 s check on this field, so they must
    read it from the same place: the fleet listing /api/nodes serves, over
    server-stamped `received_at`. A node's own clock decides neither."""
    _fake_kubectl(monkeypatch, [], [])
    path = _detail_yaml(tmp_path, _ONE_TARGET)
    app = create_app(
        network_config_path=path,
        metrics_db_path=str(tmp_path / "m.db"),
        log_db_path=str(tmp_path / "s.db"),
        chat_db_path=str(tmp_path / "c.db"),
    )
    client = TestClient(app)
    future = (datetime.now(UTC) + timedelta(days=365)).isoformat()
    client.post("/api/v1/telemetry", content=json.dumps(
        {"node_id": "edge-01", "metric_name": "cpu_percent",
         "metric_value": 12.5, "timestamp": future}
    ))

    metrics = client.get("/api/nodes/edge-01").json()["metrics"]

    assert metrics["latest"] == {"cpu_percent": 12.5}
    assert metrics["last_seen"] == client.get("/api/nodes").json()[0]["last_seen"]
    assert metrics["last_seen"] != future, "a node's own clock decided last_seen"
    # The series is filtered and ordered on `timestamp`, the node's OWN clock,
    # because that is the only thing that orders samples against each other.
    # The two fields therefore read different clocks on purpose: a node whose
    # clock is wrong shows a series that visibly disagrees with the dot beside
    # it, rather than a plausible line drawn at the wrong times.
    assert metrics["history"] == {"cpu_percent": [[future, 12.5]]}


def test_opening_the_detail_view_costs_the_grids_no_extra_kubectl(
    tmp_aureax_home, tmp_path, monkeypatch,
):
    """It reuses the "nodes" and "pods" cache entries the Overview grids fill
    rather than keying its own. Two shells out per TTL for the whole tab, not
    two more per open detail view."""
    calls = _fake_kubectl(
        monkeypatch,
        [_k8s_node("aureax-edge-01", "10.77.0.1")],
        [_k8s_pod("aureax-client-abc", "aureax-edge-01", "10.77.0.1")],
    )
    path = _detail_yaml(tmp_path, _ONE_TARGET)
    client = TestClient(create_app(network_config_path=path))

    client.get("/api/cluster")
    client.get("/api/pods")
    client.get("/api/nodes/edge-01")
    client.get("/api/nodes/edge-01")

    assert sorted(calls) == ["nodes", "pods"]
