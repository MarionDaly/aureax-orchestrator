"""The helpers several modules were each carrying their own copy of.

Every function here replaced two or more near-identical implementations, so a
regression is a regression in every caller at once — and in most cases the
copies had already drifted, which is what these tests pin.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import EXAMPLE_NETWORK_CONFIG, PROJECT_ROOT, RecordingLog

# --- raw-config readers (orchestrator/config/schema.py) ----------------------
# Copied into the edge image, so the supervisor, the deployer, templating and
# the dashboard all read node.role and mcp.server.enabled through these.


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({}, ""),
        ({"node": None}, ""),
        ({"node": {}}, ""),
        ({"node": {"role": None}}, ""),
        ({"node": {"role": "shard_worker"}}, "shard_worker"),
    ],
)
def test_node_role_reads_a_null_section_as_no_role(cfg, expected) -> None:
    """`node:` with nothing under it is valid YAML and parses to None.

    ``.get("node", {})`` returns that None rather than the default, so reading
    ``.role`` off it raised AttributeError inside `aureax generate` and took
    the whole fleet's generation down. Every caller wants "" here.
    """
    from orchestrator.config.schema import node_role

    assert node_role(cfg) == expected


@pytest.mark.parametrize(
    "cfg, expected",
    [
        ({}, True),
        ({"mcp": None}, True),
        ({"mcp": {"server": None}}, True),
        ({"mcp": {"server": {}}}, True),
        ({"mcp": {"server": {"enabled": False}}}, False),
    ],
)
def test_mcp_server_enabled_defaults_to_yes_and_survives_a_null_section(cfg, expected) -> None:
    """Absent means yes — only a shard worker turns this off explicitly."""
    from orchestrator.config.schema import mcp_server_enabled

    assert mcp_server_enabled(cfg) is expected


def test_the_in_pod_port_defaults_are_the_schema_field_defaults() -> None:
    """One number, not four. The client re-declared both in parser.py and
    entrypoint.py, where the schema's own defaults could not reach them."""
    from orchestrator.config.schema import (
        DEFAULT_LLAMA_PORT,
        DEFAULT_RPC_PORT,
        LlamaServerConfig,
        RpcServerConfig,
    )

    assert LlamaServerConfig().port == DEFAULT_LLAMA_PORT
    assert RpcServerConfig().port == DEFAULT_RPC_PORT


def test_resolve_rpc_port_reads_a_null_port_as_the_default() -> None:
    """Paired with resolve_rpc_host for the reason that one exists: the
    supervisor's listen probe must dial exactly what rpc-server bound, and a
    bare ``port:`` (None) used to reach argv as the string "None"."""
    from client.config.parser import resolve_rpc_port
    from orchestrator.config.schema import DEFAULT_RPC_PORT

    assert resolve_rpc_port({}) == DEFAULT_RPC_PORT
    assert resolve_rpc_port({"model": {"rpc_server": None}}) == DEFAULT_RPC_PORT
    assert resolve_rpc_port({"model": {"rpc_server": {"port": None}}}) == DEFAULT_RPC_PORT
    assert resolve_rpc_port({"model": {"rpc_server": {"port": 50099}}}) == 50099


# --- generated-config paths (orchestrator/config/loader.py) ------------------


def test_generated_config_path_encodes_the_node_id(tmp_aureax_home) -> None:
    """``aureax-edge-<node_id>.yaml`` is what Deployer.deploy_configs pairs
    paths to targets by, so six modules built this name by hand."""
    from orchestrator.config.loader import generated_config_dir, generated_config_path

    assert generated_config_dir() == tmp_aureax_home / ".aureax" / "configs"
    assert generated_config_path("edge-01") == generated_config_dir() / "aureax-edge-edge-01.yaml"


def test_generated_config_path_honours_an_explicit_directory(tmp_path, tmp_aureax_home) -> None:
    """Generation writes into the output directory it was given; the NAME is
    still built in one place."""
    from orchestrator.config.loader import generated_config_path

    assert generated_config_path("head", tmp_path) == tmp_path / "aureax-edge-head.yaml"


def test_generated_config_dir_is_resolved_per_call(tmp_path, monkeypatch) -> None:
    """A module-level constant froze whatever $HOME was at import time, which
    is why two dashboard modules had to be monkeypatched to be testable."""
    from orchestrator.config.loader import generated_config_dir

    monkeypatch.setenv("HOME", str(tmp_path / "a"))
    first = generated_config_dir()
    monkeypatch.setenv("HOME", str(tmp_path / "b"))
    assert generated_config_dir() != first


# --- the per-node config lookup both dashboard services share ----------------


def _write_yaml(path, data) -> None:
    from ruamel.yaml import YAML

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        YAML().dump(data, f)


def test_load_node_config_resolved_prefers_the_generated_copy(tmp_path, tmp_aureax_home) -> None:
    """The template still holds unresolved ``${...}`` placeholders — one of
    them a list the planner's pool iterates — so the generated file wins."""
    from types import SimpleNamespace

    from orchestrator.config.cache import load_node_config_resolved
    from orchestrator.config.loader import generated_config_path

    configs = tmp_path / "configs"
    _write_yaml(configs / "edge.yaml", {"node": {"id": "${node_id}", "role": "edge_executor"},
                                        "model": {"server": {"api_key": "${api_key}"}}})
    _write_yaml(generated_config_path("edge-01"),
                {"node": {"id": "edge-01", "role": "edge_executor"},
                 "model": {"server": {"api_key": "aureax_sk_resolved00000"}}})

    target = SimpleNamespace(node_id="edge-01", config_file="edge.yaml")
    cfg = load_node_config_resolved(configs, target)
    assert cfg["model"]["server"]["api_key"] == "aureax_sk_resolved00000"


def test_load_node_config_resolved_falls_back_to_the_template(tmp_path, tmp_aureax_home) -> None:
    """A fleet that has never run `aureax generate` must still resolve roles —
    those are literals and survive the fallback."""
    from types import SimpleNamespace

    from orchestrator.config.cache import load_node_config_resolved

    configs = tmp_path / "configs"
    _write_yaml(configs / "worker.yaml", {"node": {"id": "${node_id}", "role": "shard_worker"}})

    target = SimpleNamespace(node_id="w1", config_file="worker.yaml")
    assert load_node_config_resolved(configs, target)["node"]["role"] == "shard_worker"


def test_load_node_config_resolved_returns_empty_when_nothing_is_readable(
    tmp_path, tmp_aureax_home
) -> None:
    """Both callers render a node from this; a raise would 500 a polled route."""
    from types import SimpleNamespace

    from orchestrator.config.cache import load_node_config_resolved

    target = SimpleNamespace(node_id="ghost", config_file="missing.yaml")
    assert load_node_config_resolved(tmp_path, target) == {}


# --- the router helpers in orchestrator/dashboard/_http_utils.py -------------


class _Req:
    """Just enough Request for the two app.state readers."""

    def __init__(self, **state) -> None:
        from types import SimpleNamespace

        self.app = SimpleNamespace(state=SimpleNamespace(**state))


def test_emit_tags_the_line_with_its_router_and_never_raises() -> None:
    """A log write is a side effect of a route that has already done its work,
    so a SQLite hiccup must not turn a successful save into a 500."""
    from orchestrator.dashboard._http_utils import emit

    lines = []

    class _Log:
        def emit(self, level, text, source):
            lines.append((level, text, source))

    emit(_Req(system_log=_Log()), "info", "saved", source="config")
    assert lines == [("info", "saved", "config")]

    class _Boom:
        def emit(self, *_a, **_kw):
            raise RuntimeError("disk full")

    emit(_Req(system_log=_Boom()), "info", "saved", source="access")
    emit(_Req(), "info", "no log configured at all", source="access")


def test_network_config_path_503s_without_a_manager_unless_given_a_fallback(tmp_path) -> None:
    """Two error contracts, one helper: the routes that go on to touch the
    fleet must 503, while the Config tab lists and validates files whether or
    not the lifecycle manager came up."""
    from fastapi import HTTPException

    from orchestrator.dashboard._http_utils import network_config_path

    with pytest.raises(HTTPException) as exc:
        network_config_path(_Req(lifecycle=None))
    assert exc.value.status_code == 503

    fallback = tmp_path / "configs" / "aureax-network.yaml"
    assert network_config_path(_Req(lifecycle=None), fallback=fallback) == fallback.resolve()

    from types import SimpleNamespace

    mgr = SimpleNamespace(network_config_path=str(tmp_path / "other.yaml"))
    assert network_config_path(_Req(lifecycle=mgr)) == (tmp_path / "other.yaml").resolve()


# --- the watcher poll frame (orchestrator/dashboard/watchers.py) -------------


async def _drive(coro_fn, ticks: float = 0.05):
    task = asyncio.create_task(coro_fn())
    await asyncio.sleep(ticks)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_poll_loop_announces_an_outage_once_and_again_after_recovery() -> None:
    """A watcher polls every 10 s and an unreachable cluster is a normal state
    to sit in for minutes — the unfiltered version filled the log console."""
    from orchestrator.dashboard.watchers import poll_loop

    log = RecordingLog()
    calls = {"n": 0}

    async def tick():
        calls["n"] += 1
        # fail, fail, succeed, then fail again
        if calls["n"] in (1, 2, 4, 5):
            raise RuntimeError("kubectl API server unreachable")

    await _drive(lambda: poll_loop(log, "Pod watcher", "pods", 0, tick))

    paused = [t for _, t, _ in log.lines if "paused" in t]
    assert paused[:2] == [
        "Pod watcher paused: kubectl API server unreachable",
        "Pod watcher paused: kubectl API server unreachable",
    ], "one line per outage, and a recovered watcher must be able to warn again"
    assert {s for _, _, s in log.lines} == {"pods"}


async def test_poll_loop_keeps_running_after_a_tick_raises() -> None:
    """A watcher that dies stops watching silently for the life of the
    process."""
    from orchestrator.dashboard.watchers import poll_loop

    calls = {"n": 0}

    async def tick():
        calls["n"] += 1
        raise RuntimeError("nope")

    await _drive(lambda: poll_loop(RecordingLog(), "W", "w", 0, tick))
    assert calls["n"] > 1


# --- the shell helpers in scripts/lib/common.sh ------------------------------

COMMON_SH = PROJECT_ROOT / "scripts" / "lib" / "common.sh"

needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _sh(script: str) -> subprocess.CompletedProcess:
    """Source common.sh under `set -uo pipefail`, then run `script`.

    Same harness as tests/test_common_sh.py: `set -u` is part of the contract
    (every consumer runs under `set -euo pipefail`) and `-e` is left off so a
    probe can observe a non-zero return instead of dying on it.
    """
    env = dict(os.environ)
    env.pop("MAC_IP", None)
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{COMMON_SH}"\n{textwrap.dedent(script)}'],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


@needs_bash
def test_project_tar_excludes_the_fleets_credentials_and_the_models() -> None:
    """configs/aureax-network.yaml holds the SSH password for EVERY target, and
    models/ holds tens of GB no Dockerfile COPYs.

    ``models``, not ``models/``: tar treats the trailing slash literally, so
    the slashed spelling update-all.sh used matched nothing.
    """
    proc = _sh("""
        tar() { echo "TAR: $*"; }
        project_tar /tmp/out.tar.gz
    """)
    assert proc.returncode == 0, proc.stderr
    assert "--exclude=configs" in proc.stdout
    assert "--exclude=models " in proc.stdout + " ", "trailing slash would ship the GGUFs"
    assert "--exclude=.venv" in proc.stdout
    assert "--exclude=.git" in proc.stdout


@needs_bash
def test_sync_project_tree_builds_its_fallback_archive_through_project_tar() -> None:
    """One exclude list. update-all.sh builds the archive once per run and
    hands it over; sync_project_tree builds it when nobody has."""
    proc = _sh("""
        tar() { echo "TAR: $*"; }
        scp() { :; }
        ssh() { :; }
        sync_project_tree beta root "/tmp/my key" /tmp/remote
    """)
    assert proc.returncode == 0, proc.stderr
    assert "--exclude=configs" in proc.stdout


@needs_bash
def test_detect_node_arch_never_executes_what_the_node_printed(tmp_path) -> None:
    """The facts text is whatever the NODE's shell chose to print, and every
    shard worker exposes an unauthenticated rpc-server on the LAN — so `eval`
    here is a path from one compromised node to the orchestrator. Parsed with
    sed instead; pick_dockerfile fails closed on the resulting junk.
    """
    marker = tmp_path / "pwned"
    proc = _sh(f"""
        ssh() {{ printf 'ARCH=aarch64\\nHAS_NVIDIA=no; touch {marker}\\n'; }}
        detect_node_arch host user /tmp/key
        echo "arch=[$NODE_ARCH] nvidia=[$NODE_HAS_NVIDIA]"
    """)
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists(), "the node's output was executed on the orchestrator"
    assert "arch=[aarch64]" in proc.stdout
    assert "nvidia=[no; touch" in proc.stdout, "the value is kept verbatim, just inert"


def _fake_ca(tmp_path) -> str:
    """A readable file for $AUREAX_CA, so curl_tls actually reaches curl.

    mcp_health_wait dials https through curl_tls, which refuses — 77, curl
    never invoked — when the fleet CA is not a readable file (common.sh's
    curl_tls). A health-wait test that leaves AUREAX_CA alone therefore
    exercises that gate instead of the loop it means to pin, and would also
    read differently on a machine that happens to have a real
    ~/.aureax/tls/ca.pem. The CONTENT is never parsed: curl is shadowed in
    every one of these. The CA gate itself is pinned in tests/test_common_sh.py.
    """
    ca = tmp_path / "ca.pem"
    ca.write_text("not a real certificate")
    return str(ca)


@needs_bash
@pytest.mark.parametrize("budget", ["0", "-30"])
def test_mcp_health_wait_probes_exactly_once_for_a_zero_budget(tmp_path, budget) -> None:
    """setup-all.sh and test-all.sh report health, they do not wait for it —
    and a caller whose shared deadline has already expired passes a budget of
    0 or less but must still get one probe.

    The curl stub records to a file: the helper sends curl's own output to
    /dev/null, which is the point (the status is the answer). 22 is curl's
    "HTTP error" status, i.e. a node that answered something other than 2xx —
    a real failure, not the CA gate, which never reaches curl at all.
    """
    rec = tmp_path / "curl.log"
    proc = _sh(f"""
        curl() {{ echo "CURL: $*" >> "{rec}"; return 22; }}
        sleep() {{ echo "SLEPT"; }}
        AUREAX_CA="{_fake_ca(tmp_path)}"
        mcp_health_wait 10.77.0.1 30900 {budget} || echo "rc=$?"
    """)
    assert rec.read_text().count("CURL:") == 1, rec.read_text()
    assert "SLEPT" not in proc.stdout
    assert "rc=1" in proc.stdout


@needs_bash
def test_mcp_health_wait_dials_the_port_it_is_given(tmp_path) -> None:
    """The port is a parameter, filled from $MCP_NODE_PORT — which
    parse-config.py exports from schema.py — rather than a fourth literal.

    The scheme rides along in the same assertion: the probe is https with the
    fleet CA in front of it, so a plaintext regression here fails as loudly as
    a wrong port would. Recorded one argument per line because the position of
    --cacert relative to the URL is part of what is being pinned.
    """
    rec = tmp_path / "curl.log"
    ca = _fake_ca(tmp_path)
    proc = _sh(f"""
        curl() {{ printf '%s\\n' "$@" >> "{rec}"; return 0; }}
        AUREAX_CA="{ca}"
        mcp_health_wait 10.77.0.1 31234 0 && echo "healthy"
    """)
    assert proc.returncode == 0, proc.stderr
    argv = rec.read_text().splitlines()
    url = "https://10.77.0.1:31234/health"
    assert url in argv, argv
    assert argv.index("--cacert") < argv.index(url), argv
    assert argv[argv.index("--cacert") + 1] == ca, argv
    assert not [a for a in argv if a.startswith("http://")], argv
    assert "healthy" in proc.stdout


@needs_bash
def test_mcp_health_wait_stops_as_soon_as_health_answers(tmp_path) -> None:
    """The budget is a ceiling, not a target: a fleet-wide wait must cost only
    as long as the slowest node actually takes."""
    proc = _sh(f"""
        ATTEMPT=0
        curl() {{ ATTEMPT=$((ATTEMPT + 1)); [ "$ATTEMPT" -ge 2 ]; }}
        sleep() {{ :; }}
        AUREAX_CA="{_fake_ca(tmp_path)}"
        mcp_health_wait 10.77.0.1 30900 60 && echo "attempts=$ATTEMPT"
    """)
    assert proc.returncode == 0, proc.stderr
    assert "attempts=2" in proc.stdout


@needs_bash
def test_parse_config_exports_the_nodeports_from_the_schema() -> None:
    """Six shell sites had 30900/30800 typed out while every Python caller read
    them from schema.py, so a NodePort change left the shell probing the old
    one."""
    # Named explicitly: parse-config.py defaults to configs/aureax-network.yaml,
    # which is operator-owned and gitignored, so a fresh clone has no such file
    # and this test's subject — the NodePort constants — is not about it.
    proc = subprocess.run(
        [sys.executable, "scripts/parse-config.py", "all", str(EXAMPLE_NETWORK_CONFIG)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr

    from orchestrator.config.schema import LLAMA_NODE_PORT, MCP_NODE_PORT

    assert f"MCP_NODE_PORT={MCP_NODE_PORT}" in proc.stdout
    assert f"LLAMA_NODE_PORT={LLAMA_NODE_PORT}" in proc.stdout


# --- the one /v1/chat/completions request body -------------------------------
# client/inference/chat_client.py::build_chat_payload. The dashboard's
# ChatLlamaClient and the edge's LlamaClient both dial that endpoint; each
# building its own body is how the two drift apart on a field only one of them
# exercises.


class _ChatResponse:
    """Enough of an httpx.Response for both clients: chat() reads .text."""

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


def _capture_post(client, monkeypatch, body: str) -> list:
    sent: list = []

    def _post(url, json=None, **_kw):
        sent.append((url, json))
        return _ChatResponse(body)

    monkeypatch.setattr(client._client, "post", _post)
    return sent


def test_both_llama_clients_send_the_same_chat_completions_body(monkeypatch) -> None:
    """One builder, so a field is never set on one client and forgotten on the
    other — and neither can quietly go back to a raw prompt, which is what
    broke inference on one test bench's Gemma 4 node."""
    from client.inference.chat_client import ChatLlamaClient
    from client.inference.llama_client import LlamaClient

    body = json.dumps({"choices": [{"message": {"content": "ok"}}]})
    messages = [{"role": "user", "content": "hi"}]

    dashboard = ChatLlamaClient("https://10.77.0.1:30800")
    dashboard_sent = _capture_post(dashboard, monkeypatch, body)
    edge = LlamaClient("https://127.0.0.1:8080")
    edge_sent = _capture_post(edge, monkeypatch, body)

    dashboard.chat(messages, temperature=0.3, max_tokens=64)
    edge.chat(messages, temperature=0.3, max_tokens=64)

    dashboard_url, dashboard_body = dashboard_sent[0]
    edge_url, edge_body = edge_sent[0]

    assert dashboard_url == edge_url == "/v1/chat/completions"
    # Every field the two share is byte-identical, chat_template_kwargs aside:
    # that ONE key is the deliberate divergence below, and pinning the rest is
    # what stops a field being set on one client and forgotten on the other.
    assert dashboard_body == {k: v for k, v in edge_body.items() if k != "chat_template_kwargs"}
    assert dashboard_body["stream"] is False
    # The edge client disables thinking because a tool call must spend its
    # budget on an answer (the live evidence is in
    # client/inference/llama_client.py); dashboard chat keeps thinking on, so
    # it must send no chat_template_kwargs at all.
    assert "chat_template_kwargs" not in dashboard_body
    assert edge_body["chat_template_kwargs"] == {"enable_thinking": False}
    dashboard.close()
    edge.close()


def test_tool_schemas_ride_only_when_there_are_tools(monkeypatch) -> None:
    """llama.cpp rejects a request carrying `tools` unless llama-server was
    started with --jinja, so an empty list must not become an empty `tools`
    key that only the dashboard's targets survive."""
    from client.inference.chat_client import build_chat_payload

    bare = build_chat_payload([], temperature=0.1, max_tokens=8, tools=[])
    armed = build_chat_payload([], temperature=0.1, max_tokens=8, tools=[{"type": "function"}])

    assert "tools" not in bare and "tool_choice" not in bare
    assert armed["tools"] == [{"type": "function"}]
    assert armed["tool_choice"] == "auto"


def test_chat_template_kwargs_ride_only_when_given() -> None:
    """The key reaches the GGUF's own Jinja template, so it must not appear on
    a request that never asked for it — the dashboard's chat path passes
    nothing here and has to keep sending a body without the field."""
    from client.inference.chat_client import build_chat_payload

    omitted = build_chat_payload([], temperature=0.1, max_tokens=8)
    explicit_none = build_chat_payload([], temperature=0.1, max_tokens=8, chat_template_kwargs=None)
    empty = build_chat_payload([], temperature=0.1, max_tokens=8, chat_template_kwargs={})
    given = build_chat_payload(
        [], temperature=0.1, max_tokens=8, chat_template_kwargs={"enable_thinking": False}
    )

    assert "chat_template_kwargs" not in omitted
    assert "chat_template_kwargs" not in explicit_none
    assert "chat_template_kwargs" not in empty
    assert given["chat_template_kwargs"] == {"enable_thinking": False}
