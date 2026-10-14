"""Guards for the ``aureax`` Click CLI.

Covers the subcommand table and --help rendering, the 0700/0600 modes ``init``
lays down under ~/.aureax, which nodes ``bootstrap-keys`` skips, ``heal-nodes``'
loopback guard, the four states of ``nodes``' two Auth columns, every flag
combination ``apply`` accepts or rejects, the ``tls`` group, the dashboard's
HTTPS wiring, and every verdict ``group doctor`` exits non-zero on.

Doesn't exercise SSH/k3d — those need real infrastructure; the doctor tests
monkeypatch ``orchestrator.deploy.ssh.run_command`` with canned probe output.
"""

from __future__ import annotations

import os
import re
import stat

import click
import pytest
from click.testing import CliRunner

from orchestrator.cli import cli
from tests.conftest import EXAMPLE_NETWORK_CONFIG

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")
# uid 0 bypasses the read bit, so a 0o000 file is still readable and a test that
# makes one unreadable has nothing to assert. Containerised CI runs as root by
# default, which is where this bites.
not_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses file modes"
)

# Passed to every invocation below whose subject is NOT the config file. The
# root group callback loads the fleet definition before dispatching to any
# subcommand, and its default — configs/aureax-network.yaml — is operator-owned
# and gitignored, so it is simply absent from a fresh clone. Relying on it made
# tests about ~/.aureax's layout or the subcommand table fail for a reason that
# has nothing to do with what they assert.
_CFG = ["-c", str(EXAMPLE_NETWORK_CONFIG)]

_FLEET_HEADER = (
    "network: {name: t, version: '0.1.0'}\n"
    "model_registry: {models: []}\n"
    "ssh:\n"
    "  targets:\n"
)


def test_help_renders():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Aureax" in result.output


def test_all_subcommands_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    expected = [
        "init", "nodes", "status", "generate", "deploy", "apply", "stop", "start",
        "heal-nodes", "dashboard", "group", "bootstrap-keys", "tls", "wg",
    ]
    for cmd in expected:
        assert cmd in result.output, f"missing subcommand: {cmd}"


def test_group_subcommands_registered():
    runner = CliRunner()
    result = runner.invoke(cli, [*_CFG, "group", "--help"])
    assert result.exit_code == 0
    for cmd in ("list", "doctor"):
        assert cmd in result.output, f"missing group subcommand: {cmd}"


def test_bootstrap_keys_skips_key_only_targets(monkeypatch, tmp_path):
    """A target with no password is already key-only — don't touch it.

    setup-all.sh runs this before its SSH preflight, so it has to be safe on a
    fleet that never had a password at all.
    """
    # Patched on the module, not the command: bootstrap_keys imports the helper
    # inside the function body, so the lookup happens at call time.
    import orchestrator.deploy.ssh as ssh_mod

    calls = []
    monkeypatch.setattr(
        ssh_mod, "ensure_key_installed",
        lambda host, pub, **kw: calls.append(host) or (True, "installed"),
    )

    net = tmp_path / "aureax-network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n"
        "  bootstrap_public_key: '~/.ssh/id_ed25519.pub'\n"
        "  targets:\n"
        "    - {node_id: keyonly, host: 10.77.0.1}\n"
        "    - {node_id: haspw,   host: 10.77.0.2, password: pw}\n"
        "    - {node_id: off,     host: 10.77.0.3, password: pw, enabled: false}\n"
    )

    result = CliRunner().invoke(cli, ["-c", str(net), "bootstrap-keys"])
    assert result.exit_code == 0, result.output
    assert calls == ["10.77.0.2"], "only the password target is bootstrapped"
    assert "keyonly" in result.output and "skipped" in result.output


def test_bootstrap_keys_exits_nonzero_when_a_password_node_fails(monkeypatch, tmp_path):
    """That node will fail the bash preflight moments later — say so now."""
    import orchestrator.deploy.ssh as ssh_mod

    monkeypatch.setattr(
        ssh_mod, "ensure_key_installed",
        lambda host, pub, **kw: (False, "auth failed"),
    )

    net = tmp_path / "aureax-network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n"
        "  bootstrap_public_key: '~/.ssh/id_ed25519.pub'\n"
        "  targets:\n"
        "    - {node_id: haspw, host: 10.77.0.2, password: pw}\n"
    )

    result = CliRunner().invoke(cli, ["-c", str(net), "bootstrap-keys"])
    assert result.exit_code == 1
    assert "auth failed" in result.output


def test_group_list_shows_head_and_workers(tmp_path):
    """Against a config this test owns, not whatever is in configs/.

    Invoking without -c read the developer's real aureax-network.yaml, so the
    test asserted on the shipped fleet's shape and hard-failed on any config
    with no node_groups — a configuration the suite itself treats as valid
    (test-all.sh section 6 skips for exactly that case).
    """
    net = tmp_path / "network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n"
        "  targets:\n"
        "    - {node_id: h1, host: 10.77.0.30}\n"
        "    - {node_id: w1, host: 10.77.0.31}\n"
        "    - {node_id: w2, host: 10.77.0.32}\n"
        "node_groups:\n"
        "  - {name: g1, head: h1, workers: [w1, w2], rpc_port: 50052}\n"
    )

    result = CliRunner().invoke(cli, ["-c", str(net), "group", "list"])
    assert result.exit_code == 0, result.output
    # Addressing a group means addressing its head, so both endpoints must be
    # shown against the head's host.
    assert "head:" in result.output
    assert "worker:" in result.output
    assert "10.77.0.31:50052" in result.output
    # https, not http: both listeners on the head are TLS-only, so a plaintext
    # URL copied out of this listing fails at the first byte with no fallback.
    assert "https://10.77.0.30:30900/mcp" in result.output
    assert "https://10.77.0.30:30800" in result.output
    assert "http://" not in result.output
    # A group is addressed through its head, so the head's auth state is the
    # group's — printed here rather than making the operator cross-reference.
    # Both keys, because both URLs above are printed and they are protected
    # separately.
    assert "auth:    llama:" in result.output
    assert "mcp:" in result.output


def test_group_list_brackets_an_ipv6_head(tmp_path):
    """Both endpoint lines are meant to be copied into a curl or a browser, so
    an IPv6 head must be bracketed: a bare literal's own colons read as the port
    separator. The fleet CA's name constraints admit fc00::/7, so this is a
    supported shape. The worker line is an address:port pair, not a URL, and
    keeps the plain spelling `--rpc` wants.
    """
    net = tmp_path / "network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n"
        "  targets:\n"
        "    - {node_id: h1, host: 'fd00::10'}\n"
        "    - {node_id: w1, host: 'fd00::11'}\n"
        "node_groups:\n"
        "  - {name: g1, head: h1, workers: [w1], rpc_port: 50052}\n"
    )

    result = CliRunner().invoke(cli, ["-c", str(net), "group", "list"])
    assert result.exit_code == 0, result.output
    assert "https://[fd00::10]:30900/mcp" in result.output
    assert "https://[fd00::10]:30800" in result.output
    assert "https://fd00::10" not in result.output


def test_group_list_on_a_fleet_without_groups():
    """A groupless fleet is valid and must report so, not error."""
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as d:
        net = _P(d) / "network.yaml"
        net.write_text(
            "network: {name: t, version: '0.1.0'}\n"
            "model_registry: {models: []}\n"
            "ssh:\n"
            "  targets:\n"
            "    - {node_id: e1, host: 10.77.0.30}\n"
        )
        result = CliRunner().invoke(cli, ["-c", str(net), "group", "list"])
        assert result.exit_code == 0, result.output
        assert "No node groups configured" in result.output


def test_nodes_lists_configured_targets(tmp_path):
    """Against a config this test owns, and asserting on the targets themselves.

    `Network:` and `Nodes (N):` are printed before the per-target loop, so a
    test that asserts only on those labels passes for a fleet with no targets
    at all — and, invoked without -c, read whatever config happened to be in
    the developer's cwd (same defect as test_group_list_shows_head_and_workers).
    """
    net = tmp_path / "network.yaml"
    net.write_text(
        "network: {name: fixture-net, version: '9.9.9'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n"
        "  targets:\n"
        "    - {node_id: alpha, host: 10.77.0.30, tags: [head]}\n"
        "    - {node_id: bravo, host: 10.77.0.31, user: someone, port: 2222, enabled: false}\n"
    )

    result = CliRunner().invoke(cli, ["-c", str(net), "nodes"])
    assert result.exit_code == 0, result.output
    assert "Network: fixture-net v9.9.9" in result.output
    assert "Nodes (2):" in result.output
    assert "alpha" in result.output
    assert "aureax@10.77.0.30:22" in result.output
    assert "head" in result.output
    # A disabled target is still listed, and is marked as such.
    assert "bravo" in result.output
    assert "someone@10.77.0.31:2222" in result.output
    assert "(disabled)" in result.output
    # Whether a node's llama-server and MCP server enforce a key is fleet state
    # an operator cannot get anywhere else without reading the generated YAML
    # by hand — and the two ports carry separate keys, so both are listed.
    assert "Auth:    llama:" in result.output
    assert "         mcp:  " in result.output


def test_init_creates_state_dir(tmp_aureax_home):
    runner = CliRunner()
    result = runner.invoke(cli, [*_CFG, "init"])
    assert result.exit_code == 0
    # keys/ holds the per-node llama-server and MCP API keys, tls/ the fleet CA
    # and every leaf, state/ the node_groups baseline the reorder check reads;
    # without them the first `aureax generate` on a fresh Mac is the thing that
    # creates each store.
    expected_subdirs = ["ssh-keys", "telemetry", "configs", "keys", "tls", "state"]
    for sd in expected_subdirs:
        assert (tmp_aureax_home / ".aureax" / sd).is_dir(), f"missing: {sd}"
    # models/ and registry/ are NOT created: nothing in the repo reads either
    # (GGUFs live in the repo's models/ on the Mac and /opt/aureax/models on
    # nodes), and reset-all.sh's comment says so.
    for unused in ("models", "registry"):
        assert not (tmp_aureax_home / ".aureax" / unused).exists(), f"unused dir: {unused}"


@posix_only
def test_init_locks_down_every_secret_directory(tmp_aureax_home):
    """ssh-keys/ holds the private keys the whole fleet authenticates with.

    tls/ holds ca.key, the single secret every node's trust in the orchestrator
    hangs on: anyone who can read it can mint a certificate for any node in the
    fleet. state/ carries no secret and is here for one rule over the tree
    rather than a per-directory one somebody has to look up.

    mkdir honours the umask (0755) and ignores its own mode when the directory
    already exists, so only an explicit chmod keeps these out of reach of other
    local users — and asserting only `is_dir()` let either chmod be dropped
    without a single failure.
    """
    assert CliRunner().invoke(cli, [*_CFG, "init"]).exit_code == 0
    for sd in ("ssh-keys", "keys", "configs", "tls", "state"):
        mode = stat.S_IMODE((tmp_aureax_home / ".aureax" / sd).stat().st_mode)
        assert mode == 0o700, f"{sd} is {oct(mode)}"


def test_auth_status_reports_every_state(tmp_aureax_home, tmp_path):
    """The Auth column has four outcomes and asserting on the label covers one.

    It could report "disabled" for every keyed node in the fleet and a test
    that only looks for "Auth:" would still pass. Exact strings here, because
    "disabled" is a substring of both "disabled (no config_file)" and the
    "(disabled)" marker printed for a disabled target.
    """
    from orchestrator.config import keystore

    (tmp_path / "keyed.yaml").write_text(
        'model:\n  server: {port: 8080, api_key: "${api_key}"}\n'
    )
    (tmp_path / "plain.yaml").write_text("model:\n  server: {port: 8080}\n")
    net = tmp_path / "network.yaml"
    net.write_text(
        _FLEET_HEADER
        + "    - {node_id: keyed, host: 10.77.0.1, config_file: keyed.yaml}\n"
        + "    - {node_id: plain, host: 10.77.0.2, config_file: plain.yaml}\n"
        + "    - {node_id: bare,  host: 10.77.0.3}\n"
    )

    before = CliRunner().invoke(cli, ["-c", str(net), "nodes"])
    assert before.exit_code == 0, before.output
    # The four states, now behind the per-kind prefix: _auth_status' return
    # values are unchanged, so a regression in either half is still visible.
    assert "Auth:    llama: disabled (no config_file)" in before.output
    assert "Auth:    llama: disabled\n" in before.output
    assert "Auth:    llama: enabled (key pending — run `aureax generate`)" in before.output

    key = keystore.ensure_key("keyed")
    after = CliRunner().invoke(cli, ["-c", str(net), "nodes"])
    assert after.exit_code == 0, after.output
    assert f"Auth:    llama: enabled (…{key[-4:]})" in after.output
    assert "key pending" not in after.output
    # Never the full key: this lands in terminal scrollback and shell history.
    assert key not in after.output


def test_nodes_reports_the_mcp_key_separately_from_the_llama_key(tmp_aureax_home, tmp_path):
    """Two ports, two keys, two lines.

    A node can enforce a key on :30800 and none on :30900 — that was the state
    of the whole fleet before MCP auth existed — so one collapsed line would
    report "enabled" for a node whose MCP port is open to anyone who can reach
    it. And a node that serves no MCP at all (a shard worker, the planner) must
    not read as "disabled": there is no `mcp.server` block there to turn on.
    """
    from orchestrator.config import keystore

    (tmp_path / "head.yaml").write_text(
        'model:\n  server: {port: 8080, api_key: "${api_key}"}\n'
        'mcp:\n  server: {enabled: true, api_key: "${mcp_api_key}"}\n'
    )
    (tmp_path / "worker.yaml").write_text(
        "node: {role: shard_worker}\nmcp:\n  server: {enabled: false}\n"
    )
    net = tmp_path / "network.yaml"
    net.write_text(
        _FLEET_HEADER
        + "    - {node_id: head, host: 10.77.0.1, config_file: head.yaml}\n"
        + "    - {node_id: w1,   host: 10.77.0.2, config_file: worker.yaml}\n"
    )

    before = CliRunner().invoke(cli, ["-c", str(net), "nodes"])
    assert before.exit_code == 0, before.output
    assert "         mcp:   enabled (key pending — run `aureax generate`)" in before.output
    assert "         mcp:   unsupported (no MCP server)" in before.output

    llama_key = keystore.ensure_key("head", kind=keystore.KIND_LLAMA)
    mcp_key = keystore.ensure_key("head", kind=keystore.KIND_MCP)
    after = CliRunner().invoke(cli, ["-c", str(net), "nodes"])
    assert after.exit_code == 0, after.output
    assert f"Auth:    llama: enabled (…{llama_key[-4:]})" in after.output
    assert f"         mcp:   enabled (…{mcp_key[-4:]})" in after.output
    # Two distinct keys, and neither printed in full.
    assert llama_key not in after.output and mcp_key not in after.output


@posix_only
@not_root
def test_auth_status_tells_a_missing_key_from_an_unreadable_one(tmp_aureax_home, tmp_path):
    """The display helpers collapse both to None, and the two call for OPPOSITE
    actions: run `aureax generate`, or fix the file's ownership (a .key left
    root-owned by a sudo run is the usual cause). Telling an operator to
    regenerate a key that already exists is how a running node's clients get
    401s."""
    from orchestrator.config import keystore

    (tmp_path / "keyed.yaml").write_text(
        'model:\n  server: {port: 8080, api_key: "${api_key}"}\n'
    )
    net = tmp_path / "network.yaml"
    net.write_text(
        _FLEET_HEADER + "    - {node_id: keyed, host: 10.77.0.1, config_file: keyed.yaml}\n"
    )

    keystore.ensure_key("keyed")
    keystore.key_path("keyed").chmod(0o000)
    try:
        result = CliRunner().invoke(cli, ["-c", str(net), "nodes"])
    finally:
        keystore.key_path("keyed").chmod(0o600)

    assert result.exit_code == 0, result.output
    assert "unreadable" in result.output, result.output
    assert "key pending" not in result.output, "an existing key must not read as absent"


def test_an_unexported_password_env_is_named_not_reported_as_key_only(monkeypatch):
    """`password_env` resolving to None is indistinguishable at the call site
    from a genuinely key-only target, and reporting the second for the first
    sends the operator into an SSH auth failure with nothing naming the cause.

    `password_configured` is what tells them apart.
    """
    from orchestrator.cli import _no_password_reason
    from orchestrator.config.schema import SSHTarget

    unexported = SSHTarget(node_id="n", host="10.77.0.1", password_env="AUREAX_SSH_PASSWORD")
    monkeypatch.delenv("AUREAX_SSH_PASSWORD", raising=False)
    assert unexported.password_configured is True
    reason = _no_password_reason(unexported)
    assert reason is not None
    assert "password_env AUREAX_SSH_PASSWORD is not set" in reason

    monkeypatch.setenv("AUREAX_SSH_PASSWORD", "pw")
    assert _no_password_reason(unexported) is None

    keyonly = SSHTarget(node_id="k", host="10.77.0.2")
    assert keyonly.password_configured is False
    assert _no_password_reason(keyonly) == "no password — key-only target"


def _kubectl_fleet(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(_FLEET_HEADER + "    - {node_id: n1, host: 10.77.0.1}\n")
    return net


@pytest.mark.parametrize(
    "command",
    [
        ["deploy", "--orchestrator-ip", "10.77.0.99"],
        ["apply", "--orchestrator-ip", "10.77.0.99", "--node", "n1"],
        ["stop"],
        ["start"],
    ],
)
def test_a_kubectl_failure_is_one_line_not_a_traceback(monkeypatch, tmp_path, command):
    """Every kubectl/k3d call raises CommandError — a CalledProcessError
    subclass, so NOT a RuntimeError, so it escaped handlers that listed only
    RuntimeError/ValueError and reached the operator as a stack trace. Its
    str() carries the captured stderr, which is the part that says what
    kubectl actually objected to.
    """
    import orchestrator.deploy.deployer as deployer_mod
    import orchestrator.lifecycle as lifecycle_mod
    from orchestrator.cluster._subprocess import CommandError

    boom = CommandError(1, ["kubectl", "get", "nodes"], "", "the server could not find the requested resource")

    class _FakeDeployer:
        def __init__(self, *a, **kw):
            pass

        def generate_configs(self):
            raise boom

        def full_deploy(self):
            raise boom

        def apply_to_nodes(self, *a, **kw):
            raise boom

    monkeypatch.setattr(deployer_mod, "Deployer", _FakeDeployer)
    for name in ("do_stop", "do_stop_node", "do_start", "do_start_node"):
        monkeypatch.setattr(lifecycle_mod, name, lambda *a, **kw: (_ for _ in ()).throw(boom))

    result = CliRunner().invoke(cli, ["-c", str(_kubectl_fleet(tmp_path)), *command])

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert "could not find the requested resource" in result.output, result.output


def _heal_fleet(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(
        _FLEET_HEADER
        + "    - {node_id: n1, host: 10.77.0.1}\n"
        + "    - {node_id: n2, host: 10.77.0.2}\n"
        + "    - {node_id: off, host: 10.77.0.3, enabled: false}\n"
    )
    return net


def test_heal_nodes_refuses_to_run_against_a_loopback_url(monkeypatch, tmp_path):
    """Without the guard, one run points the whole fleet at the Mac's loopback.

    heal-nodes rewrites /etc/systemd/system/k3s-agent.service.env and restarts
    k3s-agent on every enabled target, so a LAN-IP detection that fell back to
    127.0.0.1 must stop the command, not be written out.
    """
    from orchestrator.cluster import k3d
    from orchestrator.deploy import k3s_agent

    contacted = []
    monkeypatch.setattr(k3d, "get_server_url", lambda *a, **kw: "https://127.0.0.1:6443")
    monkeypatch.setattr(
        k3s_agent, "reconcile_server_url",
        lambda host, url, **kw: contacted.append(host) or (True, "ok"),
    )

    result = CliRunner().invoke(cli, ["-c", str(_heal_fleet(tmp_path)), "heal-nodes"])
    assert result.exit_code == 1
    assert "could not detect the orchestrator LAN IP" in result.output
    assert contacted == [], "no node may be touched once detection has failed"


def test_heal_nodes_reconciles_every_enabled_target(monkeypatch, tmp_path):
    """The mirror case: with a real LAN IP, every enabled node is healed."""
    from orchestrator.cluster import k3d
    from orchestrator.deploy import k3s_agent

    calls = []
    monkeypatch.setattr(k3d, "get_server_url", lambda *a, **kw: "https://10.9.9.9:6443")
    monkeypatch.setattr(
        k3s_agent, "reconcile_server_url",
        lambda host, url, **kw: calls.append((host, url)) or (True, "already correct"),
    )

    result = CliRunner().invoke(cli, ["-c", str(_heal_fleet(tmp_path)), "heal-nodes"])
    assert result.exit_code == 0, result.output
    # Reconciled concurrently, so compare as a set; the disabled target is out.
    assert set(calls) == {
        ("10.77.0.1", "https://10.9.9.9:6443"),
        ("10.77.0.2", "https://10.9.9.9:6443"),
    }


def test_heal_nodes_honours_the_configured_connect_timeout(monkeypatch, tmp_path):
    """heal-nodes runs when nodes are slow to answer, so it must not connect on
    the 15s default after the operator raised ssh.connect_timeout_seconds.

    It goes in as ``connect_timeout``, never as reconcile_server_url's own
    ``timeout``, which bounds the install-and-restart and must not shrink.
    """
    from orchestrator.cluster import k3d
    from orchestrator.deploy import k3s_agent

    seen = []
    net = tmp_path / "network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n"
        "  connect_timeout_seconds: 45\n"
        "  targets:\n"
        "    - {node_id: n1, host: 10.77.0.1}\n"
    )
    monkeypatch.setattr(k3d, "get_server_url", lambda *a, **kw: "https://10.9.9.9:6443")
    monkeypatch.setattr(
        k3s_agent, "reconcile_server_url",
        lambda host, url, **kw: seen.append(kw) or (True, "already correct"),
    )

    result = CliRunner().invoke(cli, ["-c", str(net), "heal-nodes"])
    assert result.exit_code == 0, result.output
    assert seen[0]["connect_timeout"] == 45
    assert "timeout" not in seen[0], "the install-and-restart budget was overwritten"


def test_cli_logger_tee_forwards_the_source_field():
    """SystemLogger.emit is emit(level, text, source="lifecycle").

    Roughly twenty lifecycle call sites pass source="nodes"/"pods", and a tee
    that did not accept it raised TypeError straight through `aureax start`'s
    `except RuntimeError` about 30s into any slow cold start.
    """
    from orchestrator.cli import _cli_logger

    log = _cli_logger()
    log.emit("warn", "kwarg form", source="nodes")
    log.emit("info", "positional form", "pods")
    log.emit("info", "default form")

    rows = {line.text: line.source for line in log.snapshot(0)}
    assert rows["kwarg form"] == "nodes"
    assert rows["positional form"] == "pods"
    assert rows["default form"] == "lifecycle"


def _apply_fleet(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(_FLEET_HEADER + "    - {node_id: h1, host: 10.77.0.1}\n")
    return net


def test_apply_requires_an_explicit_scope(tmp_path):
    """A forgotten --node used to mean the whole fleet, silently.

    That rotates every key and restarts every head — both 35B groups reloading
    cold at once, with rotation recoverable only through the single .key.prev.
    """
    result = CliRunner().invoke(
        cli, ["-c", str(_apply_fleet(tmp_path)), "apply", "--orchestrator-ip", "10.77.0.99"]
    )
    assert result.exit_code == 2
    assert "--all" in result.output


def test_apply_rejects_node_together_with_all(tmp_path):
    result = CliRunner().invoke(cli, [
        "-c", str(_apply_fleet(tmp_path)), "apply",
        "--orchestrator-ip", "10.77.0.99", "--all", "--node", "h1",
    ])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def _recording_deployer(monkeypatch, outcomes: dict):
    """A Deployer stand-in recording every apply_to_nodes call; installs itself."""
    import orchestrator.deploy.deployer as deployer_mod

    calls: list[tuple] = []

    class _FakeDeployer:
        def __init__(self, *a, **kw):
            pass

        def apply_to_nodes(self, node_ids, **kw):
            calls.append((node_ids, kw))
            return outcomes

    monkeypatch.setattr(deployer_mod, "Deployer", _FakeDeployer)
    return _FakeDeployer, calls


@pytest.mark.parametrize(
    ("outcomes", "expected_code"),
    [
        (
            {"h1": "config pushed; pod restarted but did not become Ready in time — "
                   "a sharded head may have failed to re-attach to its RPC workers"},
            1,
        ),
        (
            {"h1": "config pushed; pod restarting (pre-delete pod list "
                   "unavailable — not waiting)"},
            1,
        ),
        (
            {"h1": "applied (pod Ready on new config)",
             "w1": "config pushed; restart skipped (shard workers run no llama-server)"},
            0,
        ),
    ],
)
def test_apply_exit_code_tracks_the_wait_outcome(monkeypatch, tmp_path, outcomes, expected_code):
    """Two of the three failure outcomes carry no FAILED marker.

    A substring test for "FAILED" alone exits 0 for a head that never
    re-attached to its RPC workers, and for a --wait run whose readiness poll
    was skipped because the pre-delete pod snapshot came back empty — a gate
    that never ran must not report success to a caller that asked to be gated.
    Any script keyed on the exit code would read the broken node as applied.
    "restart skipped" must stay a success.
    """
    _recording_deployer(monkeypatch, outcomes)
    result = CliRunner().invoke(cli, [
        "-c", str(_apply_fleet(tmp_path)), "apply",
        "--orchestrator-ip", "10.77.0.99", "--node", "h1", "--wait",
    ])
    assert result.exit_code == expected_code, result.output


@pytest.mark.parametrize(
    ("flags", "expected_ids", "expected_kwargs"),
    [
        (["--node", "h1"], ["h1"], {"restart": True, "rotate_keys": False, "wait": False}),
        (["--node", "h1", "--node", "w1"], ["h1", "w1"],
         {"restart": True, "rotate_keys": False, "wait": False}),
        (["--all"], None, {"restart": True, "rotate_keys": False, "wait": False}),
        (["--node", "h1", "--no-restart"], ["h1"],
         {"restart": False, "rotate_keys": False, "wait": False}),
        (["--node", "h1", "--rotate-key", "--wait"], ["h1"],
         {"restart": True, "rotate_keys": True, "wait": True}),
    ],
)
def test_apply_flags_reach_the_deployer(monkeypatch, tmp_path, flags, expected_ids,
                                        expected_kwargs) -> None:
    """Every flag on `apply` inverts or scopes a destructive action.

    `--no-restart` becomes `restart=False`, `--rotate-key` becomes
    `rotate_keys=True` (note the plural), and `--all` becomes a node_ids of
    None — the sentinel the deployer reads as "the whole fleet". A flag wired
    to the wrong keyword is invisible from either side: the CLI still exits 0
    and the deployer still does something, just not what was asked. Nothing
    covered the Click wrapper before; test_apply.py drives apply_to_nodes
    directly.
    """
    _, calls = _recording_deployer(monkeypatch, {"h1": "applied (pod Ready on new config)"})

    result = CliRunner().invoke(cli, [
        "-c", str(_apply_fleet(tmp_path)), "apply", "--orchestrator-ip", "10.77.0.99", *flags,
    ])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1, calls
    node_ids, kwargs = calls[0]
    assert node_ids == expected_ids
    for key, value in expected_kwargs.items():
        assert kwargs[key] is value, f"{key} mapped to {kwargs[key]!r}"


@pytest.mark.parametrize("value", ["127.0.0.1", "192.168.1.42", "fd00::1", "orchestrator.local"])
def test_orchestrator_ip_accepts_addresses_and_hostnames(value):
    from orchestrator.cli import _validate_orchestrator_ip

    assert _validate_orchestrator_ip(None, None, value) == value


@pytest.mark.parametrize("value", ["", "192.168.1..42", "10.77.0.1 ", "http://10.77.0.1", "10.77.0.1:7001"])
def test_orchestrator_ip_rejects_values_no_resolver_could_use(value):
    """This is substituted into every node's telemetry endpoint.

    An unusable value regenerates and pushes the whole fleet with an endpoint
    that never resolves, visible only as status printing "(no metrics received
    yet)".
    """
    from orchestrator.cli import _validate_orchestrator_ip

    with pytest.raises(click.BadParameter):
        _validate_orchestrator_ip(None, None, value)


def test_invalid_config_path_fails_cleanly(tmp_path):
    """A missing `-c` path is the loader's error, not Click's.

    `-c` deliberately does NOT carry `exists=True`: the fleet definition is
    operator-owned and gitignored, so "it isn't there" is the normal state of a
    fresh clone and the message has to name the file and the fix. Click's own
    existence check fires before any of our code and says only "Path ... does
    not exist", which is why it was removed — so the exit code is the 1 the
    group's ConfigError handler raises, not a UsageError's 2.

    Exit 1 alone would also be what CliRunner reports for an unhandled
    exception — the "crashes with a traceback" case this test exists to
    exclude — so the SystemExit assertion below is what separates the two.
    """
    bad = tmp_path / "missing.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, ["-c", str(bad), "nodes"])
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "could not read" in result.output.lower(), result.output
    assert "missing.yaml" in result.output, result.output


def test_init_and_help_work_before_any_fleet_definition_exists(tmp_path, monkeypatch):
    """`aureax init` is the first command on a fresh clone, before the operator
    has copied the example config. The group callback must not load the fleet
    eagerly: Click runs it before the subcommand's own `--help` is parsed, so
    an eager load broke both `init` and every `<cmd> --help` on a bare tree."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runner = CliRunner()
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "home" / ".aureax" / "keys").is_dir()
    result = runner.invoke(cli, ["nodes", "--help"])
    assert result.exit_code == 0, result.output
    assert "could not read" not in result.output.lower()


def test_a_missing_default_config_points_at_the_shipped_example(tmp_path, monkeypatch):
    """A fresh clone has no configs/aureax-network.yaml at all.

    It is gitignored — it carries the fleet's SSH credentials and real
    addresses — so the very first `aureax` command anyone runs after cloning
    hits this path. "No such file or directory" is a dead end there; the error
    has to name configs/aureax-network.example.yaml and the copy that fixes it.
    Nothing else asserts that hint, and it is emitted only for the DEFAULT
    path, so it cannot be reached through the `-c` case above.
    """
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["nodes"])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "aureax-network.example.yaml" in result.output, result.output


def test_malformed_yaml_config_fails_cleanly(tmp_path):
    """ruamel's YAMLError is not a ValueError, so ConfigError never covered it.

    The parse happens outside loader's try, so one mistyped line made every
    command — nodes, status, stop — die with a ruamel traceback instead of the
    one-liner the group's handler exists to print.
    """
    from ruamel.yaml.error import YAMLError

    bad = tmp_path / "network.yaml"
    bad.write_text("network: {name: t\nssh: [oops\n")

    result = CliRunner().invoke(cli, ["-c", str(bad), "nodes"])
    assert result.exit_code != 0
    assert not isinstance(result.exception, YAMLError), "raw parser error escaped"
    assert "Error" in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["deploy", "--orchestrator-ip", "10.77.0.99"],
        ["apply", "--orchestrator-ip", "10.77.0.99", "--all"],
    ],
)
def test_deploy_and_apply_report_a_bad_config_like_every_other_command(tmp_path, argv):
    """Both build a Deployer, which loads the fleet definition itself.

    Without going through ctx.obj first, that load raises ConfigError (or a
    ruamel YAMLError) from inside the constructor: exit 1 with an empty output
    and a traceback, instead of the one-line "could not read ..." that `nodes`
    prints for the same file.
    """
    bad = tmp_path / "missing.yaml"

    result = CliRunner().invoke(cli, ["-c", str(bad), *argv])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit), result.exception
    assert "could not read" in result.output.lower(), result.output
    assert "missing.yaml" in result.output, result.output


def test_an_unreadable_bootstrap_key_is_reported_not_swallowed(tmp_path):
    """`status` documents the key install as a side effect of running it.

    The schema default names a path plenty of orchestrators do not have, so an
    unreadable key is the ordinary first-run case — and returning "" alone
    prints a green fleet on which nothing was installed while `ssh -i` keeps
    failing for the bash scripts.
    """
    from orchestrator.cli import _bootstrap_public_key_text

    text, reason = _bootstrap_public_key_text(str(tmp_path / "absent.pub"))
    assert text == ""
    assert "absent.pub" in reason

    # An empty ssh.bootstrap_public_key is a deliberate opt-out: nothing to
    # warn about on every `aureax status`.
    assert _bootstrap_public_key_text("") == ("", "")

    present = tmp_path / "id.pub"
    present.write_text("ssh-ed25519 AAAA test\n")
    text, reason = _bootstrap_public_key_text(str(present))
    assert text == "ssh-ed25519 AAAA test"
    assert reason == ""


# --- `aureax tls` -----------------------------------------------------------


def test_tls_status_reports_an_empty_store_without_minting(tmp_aureax_home):
    """`tls status` must never create the fleet's root of trust.

    A status command that minted a CA as a side effect would hand out a root no
    node has ever been given: every TLS failure afterwards points at the nodes
    instead of at this Mac, and the real CA — if one existed on a backup — can
    never be restored over it.
    """
    result = CliRunner().invoke(cli, ["tls", "status"])

    assert result.exit_code == 0, result.output
    assert "absent" in result.output
    assert not (tmp_aureax_home / ".aureax" / "tls" / "ca.pem").exists()
    assert not (tmp_aureax_home / ".aureax" / "tls" / "ca.key").exists()


def test_tls_status_reports_the_ca_and_every_leaf(tmp_aureax_home):
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("10.9.9.9")

    result = CliRunner().invoke(cli, ["tls", "status"])

    assert result.exit_code == 0, result.output
    assert "Aureax Fleet CA" in result.output
    assert "orchestrator" in result.output
    # The SAN list is the whole reason a leaf goes stale after a network move,
    # so it is printed rather than summarised.
    assert "IP:10.9.9.9" in result.output
    assert "IP:127.0.0.1" in result.output


def test_tls_ca_path_prints_one_line_and_nothing_else(tmp_aureax_home):
    """Shaped for `curl --cacert "$(aureax tls ca-path)"`, so it must stay bare
    even before the store exists."""
    result = CliRunner().invoke(cli, ["tls", "ca-path"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(tmp_aureax_home / ".aureax" / "tls" / "ca.pem")


def test_tls_regenerate_reissues_only_the_named_leaf(tmp_aureax_home):
    """Reissuing one leaf is safe on its own — the CA is untouched, so only
    that node needs pushing. Reissuing every leaf would mean a fleet-wide apply
    for one node's certificate."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("10.9.9.9")
    tlsstore.ensure_leaf("edge-01", ["IP:10.77.0.31"])
    ca_before = tlsstore.ca_cert_path().read_bytes()
    orchestrator_before = tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes()
    edge_before = tlsstore.leaf_paths("edge-01")[0].read_bytes()

    result = CliRunner().invoke(cli, ["tls", "regenerate", "--node", "edge-01"])

    assert result.exit_code == 0, result.output
    assert tlsstore.leaf_paths("edge-01")[0].read_bytes() != edge_before
    assert tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes() == orchestrator_before
    assert tlsstore.ca_cert_path().read_bytes() == ca_before
    assert "aureax apply --all" in result.output


def test_tls_regenerate_ca_warns_about_the_apply_and_the_browser_trust(tmp_aureax_home, tmp_path):
    """Replacing the CA breaks every node until it has been pushed AND every
    browser until the new root is trusted again. Neither is recoverable by
    re-running anything, so the warning has to name both before it happens."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("192.0.2.9")
    ca_before = tlsstore.ca_cert_path().read_bytes()
    leaf_before = tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes()

    result = CliRunner().invoke(cli, [
        "-c", str(_tls_fleet(tmp_path)),
        "tls", "regenerate", "--ca", "--orchestrator-ip", "192.0.2.9",
    ])

    assert result.exit_code == 0, result.output
    assert "aureax apply --all" in result.output
    assert "browser" in result.output
    assert tlsstore.ca_cert_path().read_bytes() != ca_before
    # Every leaf is reissued in the same call: a leaf signed by the old CA
    # verifies against nothing once the root under it has changed.
    assert tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes() != leaf_before


def _tls_fleet(tmp_path):
    """A fleet on 192.0.2.x — documentation range, deliberately NOT private, so
    the CA's name constraints have to carry it explicitly to stay usable."""
    net = tmp_path / "tls-fleet.yaml"
    net.write_text(_FLEET_HEADER + "    - {node_id: n1, host: 192.0.2.5}\n")
    return net


def test_tls_regenerate_ca_requires_an_orchestrator_ip(tmp_aureax_home, tmp_path):
    """The new root is name-constrained to this fleet, and the dashboard's leaf
    is minted under it — both need the address the fleet's configs were
    generated against. Guessing it here would guess on the fleet's behalf."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("10.9.9.9")
    ca_before = tlsstore.ca_cert_path().read_bytes()

    result = CliRunner().invoke(cli, ["-c", str(_tls_fleet(tmp_path)), "tls", "regenerate", "--ca"])

    assert result.exit_code == 2, result.output
    assert "--orchestrator-ip" in result.output
    assert tlsstore.ca_cert_path().read_bytes() == ca_before, "nothing may be replaced on a refusal"


def test_tls_regenerate_ca_prints_the_whole_rotation_procedure(tmp_aureax_home, tmp_path):
    """`regenerate --ca` + `apply --all` — the pair the docs used to give — does
    NOT restore the fleet: apply never restarts a shard worker, and every
    node's telemetry and log reporter holds the SSLContext it built at startup,
    so those pushes fail verification at DEBUG level and the nodes simply
    vanish from the dashboard. The remaining steps have to be printed where the
    damage is done."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("192.0.2.9")
    ca_before = tlsstore.ca_cert_path().read_bytes()
    leaf_before = tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes()

    result = CliRunner().invoke(cli, [
        "-c", str(_tls_fleet(tmp_path)),
        "tls", "regenerate", "--ca", "--orchestrator-ip", "192.0.2.9",
    ])

    assert result.exit_code == 0, result.output
    assert "aureax apply --all --orchestrator-ip 192.0.2.9" in result.output
    assert "aureax stop && aureax start" in result.output
    assert "restart `aureax dashboard`" in result.output
    assert "re-trust" in result.output and "browser" in result.output
    assert tlsstore.ca_cert_path().read_bytes() != ca_before
    assert tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes() != leaf_before
    # And the dashboard can serve the fleet the moment this returns: the store
    # reissues that leaf with its loopback names only, so the address has to be
    # minted back in here rather than at the next `aureax generate`.
    assert "IP:192.0.2.9" in tlsstore.status()["leaves"][0]["sans"]


def test_tls_regenerate_ca_constrains_the_new_root_to_this_fleet(tmp_aureax_home, tmp_path):
    """The constraints are baked at creation and nothing rotates the CA again,
    so this call is the only chance to scope the replacement root — a CA minted
    without them is a universal signing credential on whichever Mac trusted the
    old one."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("192.0.2.9")

    result = CliRunner().invoke(cli, [
        "-c", str(_tls_fleet(tmp_path)),
        "tls", "regenerate", "--ca", "--orchestrator-ip", "192.0.2.9",
    ])

    assert result.exit_code == 0, result.output
    permitted_ips, permitted_dns = tlsstore.ca_name_constraints()
    # Both addresses are outside every private range, so each has to be
    # permitted as itself or the fleet cannot be certified at all.
    assert "192.0.2.5/32" in permitted_ips
    assert "192.0.2.9/32" in permitted_ips
    assert "localhost" in permitted_dns


def test_tls_regenerate_node_orchestrator_resets_the_accumulated_addresses(tmp_aureax_home,
                                                                           tmp_path):
    """The dashboard's leaf only ever grows — `aureax generate` and `aureax
    dashboard` union their addresses into it so neither can delete the other's
    SAN. This is the one lever that shrinks it back to the address in use."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("10.9.9.9")
    tlsstore.ensure_orchestrator_leaf("10.8.8.8")
    sans_before = tlsstore.status()["leaves"][0]["sans"]
    assert {"IP:10.9.9.9", "IP:10.8.8.8"} <= set(sans_before), sans_before

    result = CliRunner().invoke(cli, [
        "tls", "regenerate", "--node", "orchestrator", "--orchestrator-ip", "10.8.8.8",
    ])

    assert result.exit_code == 0, result.output
    sans = tlsstore.status()["leaves"][0]["sans"]
    assert "IP:10.8.8.8" in sans
    assert "IP:10.9.9.9" not in sans, "the point of the reset is dropping the stale address"
    # uvicorn read the pair at startup, so the operator has to be told.
    assert "restarted" in result.output


def test_tls_regenerate_node_orchestrator_requires_an_orchestrator_ip(tmp_aureax_home):
    """Reissuing that leaf from its own SANs would keep every address it has
    ever carried, which is the opposite of a reset — so the address is not
    optional here either."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("10.9.9.9")
    before = tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes()

    result = CliRunner().invoke(cli, ["tls", "regenerate", "--node", "orchestrator"])

    assert result.exit_code == 2, result.output
    assert "--orchestrator-ip" in result.output
    assert tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes() == before


def test_tls_regenerate_reports_a_bad_node_name_as_one_line(tmp_aureax_home):
    """--node is joined into a path inside the store, so the store refuses
    anything that is not one path segment. That refusal reaches the operator as
    a message naming what they typed, not as a ValueError traceback."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("10.9.9.9")

    result = CliRunner().invoke(cli, ["tls", "regenerate", "--node", "../../id_ed25519"])

    assert result.exit_code == 1, result.output
    assert "../../id_ed25519" in result.output
    assert "Traceback" not in result.output


def test_tls_status_names_what_the_ca_may_sign(tmp_aureax_home, tmp_path):
    """README has this root trusted in the macOS keychain, where what it may
    certify IS the security question — and the file itself shows nothing."""
    from orchestrator.config import tlsstore

    tlsstore.ensure_orchestrator_leaf("192.0.2.9")
    # A CA minted with no constraints (as every CA from before this change was)
    # must say so rather than printing nothing.
    unconstrained = CliRunner().invoke(cli, ["tls", "status"])
    assert unconstrained.exit_code == 0, unconstrained.output
    assert "limits: NONE" in unconstrained.output

    CliRunner().invoke(cli, [
        "-c", str(_tls_fleet(tmp_path)),
        "tls", "regenerate", "--ca", "--orchestrator-ip", "192.0.2.9",
    ])
    result = CliRunner().invoke(cli, ["tls", "status"])

    assert result.exit_code == 0, result.output
    assert "192.0.2.5/32" in result.output
    assert "DNS localhost" in result.output


# --- `aureax dashboard` over TLS --------------------------------------------


def _dashboard_fleet(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(_FLEET_HEADER + "    - {node_id: n1, host: 10.77.0.1}\n")
    return net


def _patch_lan_ip(monkeypatch, value: str) -> None:
    from orchestrator.cluster import k3d

    monkeypatch.setattr(k3d, "probe_host", lambda config: "10.77.0.1")
    monkeypatch.setattr(k3d, "mac_lan_ip", lambda probe=None: value)


def test_dashboard_serves_https_with_the_orchestrator_leaf(tmp_aureax_home, tmp_path, monkeypatch):
    """The dashboard is the fleet's telemetry sink and its control plane, and
    it is now TLS-only: `run()` takes the cert and key as required keyword
    arguments, and the URL printed for the operator has to be the one that
    actually answers."""
    import orchestrator.dashboard.app as app_mod

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "10.9.9.9")

    result = CliRunner().invoke(cli, ["-c", str(_dashboard_fleet(tmp_path)), "dashboard"])

    assert result.exit_code == 0, result.output
    assert "https://10.9.9.9:7001/" in result.output
    # The only plaintext URL anywhere is the loopback one (below); nothing
    # here may offer an http:// address the fleet could dial.
    assert "http://10.9.9.9" not in result.output
    assert len(calls) == 1, calls
    # server.dashboard.host defaults to 0.0.0.0, which means "the address the
    # fleet reaches this orchestrator at" — never a literal wildcard bind, or
    # the loopback listener would have no port left to take.
    assert calls[0]["host"] == "10.9.9.9"
    cert, key = calls[0]["ssl_certfile"], calls[0]["ssl_keyfile"]
    from pathlib import Path as _Path
    assert _Path(cert).exists() and _Path(key).exists()
    # Minted on the way in, so a Mac that changed networks reissues for its new
    # address instead of serving a certificate nothing on the LAN can verify.
    assert _Path(cert) == tmp_aureax_home / ".aureax" / "tls" / "orchestrator.pem"


def test_dashboard_prints_the_loopback_url_and_the_fleet_url(tmp_aureax_home, tmp_path,
                                                             monkeypatch):
    """Both doors, named for who uses them.

    The operator sitting at the Mac should not have to trust a private CA to
    open their own dashboard — that listener is plain HTTP on loopback, and the
    banner has to say so, or the printed https:// URL sends them to a browser
    warning they then work around. The https:// line is the one the fleet uses
    and the one another machine needs, so it names the CA to trust.
    """
    import orchestrator.dashboard.app as app_mod

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "10.9.9.9")

    result = CliRunner().invoke(cli, ["-c", str(_dashboard_fleet(tmp_path)), "dashboard"])

    assert result.exit_code == 0, result.output
    assert "Browse from this machine: http://localhost:7001/" in result.output
    assert "no certificate needed" in result.output
    assert "Fleet and other machines: https://10.9.9.9:7001/" in result.output
    assert str(tmp_aureax_home / ".aureax" / "tls" / "ca.pem") in result.output
    assert calls[0]["loopback_http"] is True


def test_dashboard_serves_https_alone_when_the_bind_host_is_loopback(tmp_aureax_home, tmp_path,
                                                                     monkeypatch):
    """An explicit loopback server.dashboard.host is the SSH-tunnel setup the
    schema describes: HTTPS already holds the port on loopback, so there is no
    plaintext listener to offer beside it — and promising one in the banner
    would send the operator to a port that answers TLS."""
    import orchestrator.dashboard.app as app_mod

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "10.9.9.9")
    net = tmp_path / "network.yaml"
    net.write_text(
        _FLEET_HEADER
        + "    - {node_id: n1, host: 10.77.0.1}\n"
        + "server:\n  dashboard:\n    host: 127.0.0.1\n"
    )

    result = CliRunner().invoke(cli, ["-c", str(net), "dashboard"])

    assert result.exit_code == 0, result.output
    assert "http://localhost" not in result.output
    assert "https://127.0.0.1:7001/" in result.output
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["loopback_http"] is False


def test_dashboard_reports_a_busy_port_as_a_message_not_a_traceback(tmp_aureax_home, tmp_path,
                                                                    monkeypatch):
    """`run()` names the pid holding the port (dashboard/app.py::_listen_socket).
    Raised through Click that sentence is the whole output; unhandled it is the
    last line of a stack that reads like a crash in the dashboard."""
    import errno as _errno

    import orchestrator.dashboard.app as app_mod

    def busy(**_kw):
        raise OSError(_errno.EADDRINUSE, "address already in use — held by PID 45235")

    monkeypatch.setattr(app_mod, "run", busy)
    _patch_lan_ip(monkeypatch, "10.9.9.9")

    result = CliRunner().invoke(cli, ["-c", str(_dashboard_fleet(tmp_path)), "dashboard"])

    assert result.exit_code == 1, result.output
    assert "45235" in result.output
    assert "Traceback" not in result.output


def test_dashboard_refuses_to_serve_without_a_detectable_lan_ip(tmp_aureax_home, tmp_path,
                                                                monkeypatch):
    """Falling back to the bind host would mint a certificate for 0.0.0.0 or
    loopback: a padlock for nobody, and every node's telemetry POST failing
    verification against a name the certificate does not carry."""
    import orchestrator.dashboard.app as app_mod

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "")

    result = CliRunner().invoke(cli, ["-c", str(_dashboard_fleet(tmp_path)), "dashboard"])

    assert result.exit_code == 1, result.output
    assert "LAN IP" in result.output
    # The way out has to be in the message: detection is macOS-only and the
    # operator otherwise has no lever at all on a machine it cannot answer for.
    assert "--orchestrator-ip" in result.output
    assert calls == [], "nothing may be served on a certificate the fleet cannot verify"


def test_dashboard_takes_an_explicit_orchestrator_ip_over_detection(tmp_aureax_home, tmp_path,
                                                                    monkeypatch):
    """Detection answers with whichever interface reaches the first node, which
    on a multi-homed Mac is not necessarily the address every node's
    ${orchestrator_url} was generated against — and a certificate for the other
    one fails every telemetry push."""
    import orchestrator.dashboard.app as app_mod
    from orchestrator.config import tlsstore

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "10.9.9.9")

    result = CliRunner().invoke(cli, [
        "-c", str(_dashboard_fleet(tmp_path)), "dashboard", "--orchestrator-ip", "10.8.8.8",
    ])

    assert result.exit_code == 0, result.output
    assert "https://10.8.8.8:7001/" in result.output
    assert len(calls) == 1, calls
    assert "IP:10.8.8.8" in tlsstore.status()["leaves"][0]["sans"]


def test_dashboard_serves_when_detection_fails_but_an_address_was_given(tmp_aureax_home, tmp_path,
                                                                        monkeypatch):
    """The hard fail is about not having an address, not about detection: a
    laptop off the nodes' network can still serve stored telemetry when the
    operator names the address the fleet knows it by."""
    import orchestrator.dashboard.app as app_mod

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "")

    result = CliRunner().invoke(cli, [
        "-c", str(_dashboard_fleet(tmp_path)), "dashboard", "--orchestrator-ip", "aureax.local",
    ])

    assert result.exit_code == 0, result.output
    assert "https://aureax.local:7001/" in result.output
    assert len(calls) == 1, calls


def test_dashboard_reports_an_address_the_fleet_ca_will_not_certify(tmp_aureax_home, tmp_path,
                                                                    monkeypatch):
    """The CA is name-constrained to the fleet, so an address outside it cannot
    be given a certificate at all. The store's message names the rotation that
    widens the CA; a traceback here would read as a crash in the TLS store."""
    import orchestrator.dashboard.app as app_mod
    from orchestrator.config import tlsstore

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "10.9.9.9")
    # A CA that covers only 192.0.2.0/24-shaped names, minted the way
    # `aureax generate` mints it for a fleet on public addresses.
    tlsstore.ensure_ca(permitted_ips=["192.0.2.5/32"], permitted_dns=["localhost"])

    result = CliRunner().invoke(cli, [
        "-c", str(_dashboard_fleet(tmp_path)), "dashboard", "--orchestrator-ip", "203.0.113.7",
    ])

    assert result.exit_code == 1, result.output
    assert "203.0.113.7" in result.output
    assert "regenerate --ca" in result.output
    assert "Traceback" not in result.output
    assert calls == []


def test_dashboard_rejects_a_malformed_orchestrator_ip(tmp_aureax_home, tmp_path, monkeypatch):
    """Same validator as generate/deploy/apply: a value those three refuse must
    not become a SAN here, or the dashboard's certificate carries a name the
    fleet's configs can never contain."""
    import orchestrator.dashboard.app as app_mod

    calls: list[dict] = []
    monkeypatch.setattr(app_mod, "run", lambda **kw: calls.append(kw))
    _patch_lan_ip(monkeypatch, "10.9.9.9")

    result = CliRunner().invoke(cli, [
        "-c", str(_dashboard_fleet(tmp_path)), "dashboard", "--orchestrator-ip", "192.168.1..42",
    ])

    assert result.exit_code == 2, result.output
    assert calls == []


# --- --accept-worker-reorder ------------------------------------------------


def test_generate_accepts_the_worker_reorder_flag(tmp_path, monkeypatch):
    """The refusal message names this exact spelling, so the flag has to exist
    on `generate` and reach the generator — a flag Click accepts and drops
    leaves the operator re-running the same refused command."""
    import orchestrator.cli as cli_mod

    seen: list[dict] = []
    monkeypatch.setattr(
        cli_mod, "generate_all_edge_configs",
        lambda *a, **kw: seen.append(kw) or [],
    )
    net = _apply_fleet(tmp_path)

    default = CliRunner().invoke(cli, ["-c", str(net), "generate", "--orchestrator-ip", "10.77.0.9"])
    assert default.exit_code == 0, default.output
    assert seen[0]["accept_worker_reorder"] is False

    accepted = CliRunner().invoke(cli, [
        "-c", str(net), "generate", "--orchestrator-ip", "10.77.0.9", "--accept-worker-reorder",
    ])
    assert accepted.exit_code == 0, accepted.output
    assert seen[1]["accept_worker_reorder"] is True


def _recording_deployer_ctor(monkeypatch):
    """A Deployer stand-in recording its constructor kwargs; installs itself."""
    import orchestrator.deploy.deployer as deployer_mod

    ctor: list[dict] = []

    class _FakeDeployer:
        def __init__(self, *a, **kw):
            ctor.append(kw)

        def generate_configs(self):
            return []

        def deploy_configs(self, paths):
            return None

        def apply_manifests(self):
            return None

        def apply_to_nodes(self, node_ids, **kw):
            return {"h1": "applied (pod Ready on new config)"}

    monkeypatch.setattr(deployer_mod, "Deployer", _FakeDeployer)
    return ctor


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["deploy", "--orchestrator-ip", "10.77.0.9"], False),
        (["deploy", "--orchestrator-ip", "10.77.0.9", "--accept-worker-reorder"], True),
        (["apply", "--orchestrator-ip", "10.77.0.9", "--node", "h1"], False),
        (["apply", "--orchestrator-ip", "10.77.0.9", "--node", "h1", "--accept-worker-reorder"],
         True),
    ],
)
def test_deploy_and_apply_thread_the_reorder_flag_to_the_deployer(monkeypatch, tmp_path, argv,
                                                                  expected):
    """Both regenerate every config, so both hit the reorder check — and a flag
    wired to neither is invisible: the command still exits 0 and still refuses
    the reorder, with the remedy it printed not working."""
    ctor = _recording_deployer_ctor(monkeypatch)

    result = CliRunner().invoke(cli, ["-c", str(_apply_fleet(tmp_path)), *argv])

    assert result.exit_code == 0, result.output
    assert ctor[0]["accept_worker_reorder"] is expected


# --- `aureax status` surfaces a double-labelled node ------------------------


def test_status_warns_about_a_node_carrying_both_daemonset_labels(monkeypatch, capsys):
    """Nothing else notices two pods on one node.

    A node matching both DaemonSet selectors runs llama-server AND rpc-server,
    each sized for the whole machine, and both look perfectly healthy while
    they compete for its RAM. The node listing `status` already fetched carries
    every label, so this costs no extra API call.
    """
    from orchestrator.cli import _status_cluster
    from orchestrator.cluster import kubectl
    from tests.conftest import make_network_config, make_targets

    monkeypatch.setattr(kubectl, "get_nodes", lambda: [
        {"name": "host-a", "status": "True", "roles": "worker",
         "labels": {"aureax.io/role": "edge", "aureax.io/shard": "worker"}},
        {"name": "host-b", "status": "True", "roles": "worker",
         "labels": {"aureax.io/role": "edge"}},
    ])
    monkeypatch.setattr(kubectl, "get_pods", lambda ns: [])

    _status_cluster(make_network_config(make_targets("n1")))

    out = capsys.readouterr().out
    assert "WARNING: host-a carries both aureax.io/role and aureax.io/shard" in out
    assert "aureax deploy" in out
    assert "host-b carries both" not in out


# --- `aureax group doctor` verdicts -----------------------------------------
#
# Every case below is a fleet that answers every probe and looks healthy to
# kubectl. The failures these pin are the ones whose only other symptom is a
# head sitting 0/1 Running until its 3015s startupProbe budget expires.

_HEAD_HOST, _W1_HOST, _W2_HOST = "10.77.0.30", "10.77.0.31", "10.77.0.32"


def _doctor_fleet(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(
        _FLEET_HEADER
        + f"    - {{node_id: h1, host: {_HEAD_HOST}}}\n"
        + f"    - {{node_id: w1, host: {_W1_HOST}}}\n"
        + f"    - {{node_id: w2, host: {_W2_HOST}}}\n"
        + "node_groups:\n"
        + "  - {name: g1, head: h1, workers: [w1, w2], rpc_port: 50052}\n"
    )
    return net


# The sha256 of llama-server and rpc-server inside the node's image — what the
# doctor and the pre-start gate actually compare across a group. NOT the image
# digest: every node builds `aureax-client:dev` natively, so those differ across
# a perfectly coherent group (orchestrator/cluster/buildcheck.py).
_LLAMA_SHA, _RPC_SHA = "a" * 64, "b" * 64


def _probe_out(build: str = "b10375", docker_digest: str = "1a2b3c4d5e6f",
               k3s_digest: str = "1a2b3c4d5e6f", k3s_image: str = "present",
               rpc_listen: int = 1, rpc_peers: str = "",
               llama_sha: str = _LLAMA_SHA, rpc_sha: str = _RPC_SHA,
               build_probe_error: str = "", arch: str = "aarch64",
               commit: str = "0f1e2d3c") -> str:
    """One member's stdout, in _DOCTOR_PROBE's exact shape.

    ``arch`` is the IMAGE's `uname -m` (BUILD_PROBE's ``image_arch`` line); the
    first line's ``arch=`` is the host's and stays aarch64 here.
    """
    return (
        "arch=aarch64 cores=4 mem=13.2Gi\n"
        "nic=eth0 speed=1000Mb\n"
        "isa=3 dp/i8mm/bf16 flags\n"
        f'llama_build_docker={{"version":"{build}","commit":"{commit}"}}\n'
        f"llama_server_sha={llama_sha}\n"
        f"rpc_server_sha={rpc_sha}\n"
        f"image_arch={arch}\n"
        f"build_probe_error={build_probe_error}\n"
        f"llama_build_k3s={k3s_image}\n"
        f"image_digest_docker={docker_digest}\n"
        f"image_digest_k3s={k3s_digest}\n"
        "gguf=/opt/aureax/models/qwen.gguf (20GB)\n"
        f"rpc_listen={rpc_listen}\n"
        f"rpc_peers={rpc_peers}\n"
        "--- last log lines ---\n"
        "[/opt/aureax/logs/llama-server.log]\n"
        "llama_context: n_ctx = 131072\n"
    )


def _healthy_fleet_output() -> dict[str, str]:
    return {
        _HEAD_HOST: _probe_out(rpc_listen=0, rpc_peers=f"{_W1_HOST},{_W2_HOST},"),
        _W1_HOST: _probe_out(rpc_peers=f"{_HEAD_HOST},"),
        _W2_HOST: _probe_out(rpc_peers=f"{_HEAD_HOST},"),
    }


def _patch_probe(monkeypatch, outputs: dict) -> None:
    """Answer the doctor's one SSH round trip per host from a canned table.

    Values are stdout strings, an (rc, stdout) pair, or an exception to raise —
    the three shapes run_command's callers actually see.
    """
    import orchestrator.deploy.ssh as ssh_mod

    def run_command(host, command, **kwargs):
        canned = outputs[host]
        if isinstance(canned, Exception):
            raise canned
        if isinstance(canned, tuple):
            return canned[0], canned[1], ""
        return 0, canned, ""

    monkeypatch.setattr(ssh_mod, "run_command", run_command)


def _doctor(monkeypatch, tmp_path, outputs):
    _patch_probe(monkeypatch, outputs)
    return CliRunner().invoke(cli, ["-c", str(_doctor_fleet(tmp_path)), "group", "doctor"])


def test_doctor_parses_probe_facts():
    """Split on the FIRST '=' only, and stop at the log separator.

    llama_build_docker's value is JSON full of colons and its fallback text is
    prose; both must survive intact because the verdict quotes them back. And
    the log tail below the separator is llama-server output full of `key =
    value` lines ("n_ctx = 131072") that would otherwise shadow a real fact.
    """
    from orchestrator.cli import _parse_doctor_facts

    facts = _parse_doctor_facts(_probe_out(rpc_peers="10.77.0.31,10.77.0.32,"))

    assert facts["llama_build_docker"] == '{"version":"b10375","commit":"0f1e2d3c"}'
    assert facts["rpc_peers"] == "10.77.0.31,10.77.0.32,"
    assert facts["rpc_listen"] == "1"
    assert facts["image_digest_k3s"] == "1a2b3c4d5e6f"
    # The first probe line packs three facts; the first key is the one that
    # matters and the rest of the line rides along with its value.
    assert facts["arch"].startswith("aarch64")
    assert "llama_context" not in facts and "n_ctx" not in facts

    unavailable = _parse_doctor_facts(
        "llama_build_docker=unavailable (image missing, or docker/sudo failed)\n"
    )
    assert unavailable["llama_build_docker"] == "unavailable (image missing, or docker/sudo failed)"


def test_the_doctor_fan_out_states_one_per_member_budget():
    """The comment justifying the concurrency quotes the per-member timeout, so
    a second number in the same function sends a later reader hunting for a
    timeout that does not exist."""
    import inspect

    from orchestrator import cli as cli_mod

    src = inspect.getsource(cli_mod.group_doctor.callback)
    timeouts = set(re.findall(r"timeout=(\d+)", src))
    quoted = set(re.findall(r"whole (\d+)s budget", src))
    assert quoted, "the fan-out no longer explains its per-member budget"
    assert quoted <= timeouts, f"comment says {quoted}, the call passes {timeouts}"


def test_doctor_exits_zero_on_a_coherent_group(monkeypatch, tmp_path):
    result = _doctor(monkeypatch, tmp_path, _healthy_fleet_output())

    assert result.exit_code == 0, result.output
    assert "1 group(s) coherent." in result.output
    assert "FAIL" not in result.output
    assert "2/2 workers listening on :50052" in result.output
    assert "head h1 is attached to all 2 worker(s)" in result.output


def test_doctor_reports_the_checksum_verdict(monkeypatch, tmp_path):
    """The verdict has to say the check RAN. "No output" and "checked, all
    good" look identical otherwise, and this is the check that stands between a
    group and a 50-minute silent hang at the RPC handshake."""
    result = _doctor(monkeypatch, tmp_path, _healthy_fleet_output())

    assert result.exit_code == 0, result.output
    assert "OK   llama-server/rpc-server checksums identical on every member" in result.output


def test_doctor_exits_one_when_a_members_binaries_differ(monkeypatch, tmp_path):
    """Stronger than the build id above it: a build id is a git tag, so two
    members can report the same one and still hold different binaries."""
    outputs = _healthy_fleet_output()
    outputs[_W2_HOST] = _probe_out(rpc_sha="d" * 64, rpc_peers=f"{_HEAD_HOST},")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL w2: llama-server/rpc-server do not match head h1" in result.output
    assert "d" * 12 in result.output and "b" * 12 in result.output, "name both checksums"
    assert f"./scripts/build-on-node.sh {_W2_HOST}" in result.output
    assert "1/1 group(s) have problems: g1" in result.output


def test_doctor_passes_a_cuda_member_built_from_the_heads_commit(monkeypatch, tmp_path):
    """An x86_64 CUDA image and an aarch64 one never share bytes, so a mixed
    group is compared by commit across architectures — and the verdict says so
    on the member's own line instead of passing it silently."""
    outputs = _healthy_fleet_output()
    outputs[_W2_HOST] = _probe_out(llama_sha="c" * 64, rpc_sha="d" * 64, arch="x86_64",
                                   rpc_peers=f"{_HEAD_HOST},")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 0, result.output
    assert "OK   w2: x86_64 member compared by commit: same commit 0f1e2d3c" in result.output
    assert "checksums identical on every aarch64 member (2/3 read)" in result.output
    assert "1 group(s) coherent." in result.output


def test_doctor_exits_one_when_a_cuda_member_is_on_another_commit(monkeypatch, tmp_path):
    """The version lock holds across architectures: a different commit is a
    hang at the handshake whatever the member's CPU."""
    outputs = _healthy_fleet_output()
    outputs[_W2_HOST] = _probe_out(llama_sha="c" * 64, rpc_sha="d" * 64, arch="x86_64",
                                   commit="9a8b7c6d", rpc_peers=f"{_HEAD_HOST},")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL w2: llama.cpp commit 9a8b7c6d does not match head h1's 0f1e2d3c" in result.output
    assert f"./scripts/build-on-node.sh {_W2_HOST}" in result.output
    assert "./scripts/update-all.sh" in result.output
    assert "1/1 group(s) have problems: g1" in result.output


def test_doctor_reads_the_image_architecture_in_the_same_container_start():
    """Still one SSH round trip and ONE container start per member, and the
    image's architecture lands under its own key — not over the host `arch`
    fact the doctor's first line already reports."""
    from orchestrator.cli import _DOCTOR_PROBE, _parse_doctor_facts

    assert _DOCTOR_PROBE.count("docker run") == 1
    assert '"image_arch=$(uname -m)"' in _DOCTOR_PROBE

    facts = _parse_doctor_facts(_probe_out(arch="x86_64"))
    assert facts["image_arch"] == "x86_64"
    assert facts["arch"].startswith("aarch64"), "the host fact must survive the spliced-in probe"


def test_doctor_notes_a_member_whose_checksums_could_not_be_read(monkeypatch, tmp_path):
    """An unreadable checksum is NOT a mismatch. The probe merges the remote's
    stderr into its output so a sudo refusal survives the trip, and reading that
    as a mismatch would send the operator rebuilding images to fix sudo."""
    outputs = _healthy_fleet_output()
    outputs[_W2_HOST] = _probe_out(
        llama_sha="", rpc_sha="", build_probe_error="sudo: a password is required",
        rpc_peers=f"{_HEAD_HOST},",
    )

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 0, result.output
    assert "SKIP w2: the build check did not cover this node" in result.output
    assert "sudo: a password is required" in result.output
    assert "FAIL w2: llama-server/rpc-server" not in result.output


def test_doctor_asks_each_member_for_its_binaries_in_one_round_trip(monkeypatch, tmp_path):
    """One SSH connection and one container start per member: the checksum
    command is spliced into the doctor's own probe rather than run separately,
    so the doctor and `aureax start`'s gate can never disagree about what they
    measured."""
    commands: list[str] = []

    import orchestrator.deploy.ssh as ssh_mod

    outputs = _healthy_fleet_output()

    def run_command(host, command, **_kwargs):
        commands.append(command)
        return 0, outputs[host], ""

    monkeypatch.setattr(ssh_mod, "run_command", run_command)
    result = CliRunner().invoke(cli, ["-c", str(_doctor_fleet(tmp_path)), "group", "doctor"])

    assert result.exit_code == 0, result.output
    assert len(commands) == 3, "one round trip per member"
    assert all(
        "sha256sum /usr/local/bin/llama-server /usr/local/bin/rpc-server" in c
        for c in commands
    )


def test_doctor_exits_one_on_a_build_mismatch(monkeypatch, tmp_path):
    """The RPC wire format is version-locked: the head connects to a peer built
    from another commit and then stops, with no error on either side."""
    outputs = _healthy_fleet_output()
    outputs[_W2_HOST] = _probe_out(build="b10201", rpc_peers=f"{_HEAD_HOST},")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL w2: llama.cpp build b10201, head h1 runs b10375" in result.output
    assert "HANGS at the handshake" in result.output
    assert "./scripts/update-all.sh" in result.output
    assert "1/1 group(s) have problems: g1" in result.output


def test_doctor_exits_one_when_docker_and_k3s_digests_differ(monkeypatch, tmp_path):
    """Pods run the containerd copy, so a rebuild that was never re-imported is
    invisible: `docker images` shows the new build and the node keeps running
    the old one."""
    outputs = _healthy_fleet_output()
    outputs[_W1_HOST] = _probe_out(k3s_digest="9f8e7d6c5b4a", rpc_peers=f"{_HEAD_HOST},")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "sha256:1a2b3c4d5e6f" in result.output
    assert "sha256:9f8e7d6c5b4a" in result.output
    assert "k3s ctr images import" in result.output


def test_doctor_exits_one_on_a_missing_k3s_image(monkeypatch, tmp_path):
    """imagePullPolicy is Never, so an absent containerd copy is a pod that can
    never start — not a pull that is merely slow."""
    outputs = _healthy_fleet_output()
    outputs[_W2_HOST] = _probe_out(k3s_image="MISSING", k3s_digest="",
                                   rpc_peers=f"{_HEAD_HOST},")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL w2: aureax-client:dev is absent from K3s containerd" in result.output
    assert "imagePullPolicy is Never" in result.output


def test_doctor_exits_one_when_a_worker_has_no_rpc_listener(monkeypatch, tmp_path):
    """The head waits out worker_wait_timeout_seconds and exits 1, which reads
    as "the worker never came up" with nothing saying which one."""
    outputs = _healthy_fleet_output()
    outputs[_W1_HOST] = _probe_out(rpc_listen=0, rpc_peers="")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL w1: nothing is listening on :50052" in result.output
    assert "300s" in result.output


def test_doctor_exits_one_when_the_heads_peers_are_a_subset_of_its_workers(monkeypatch, tmp_path):
    """A group serving on fewer devices than configured answers every probe and
    every request — it is just lopsided, and nothing else reports it. The
    missing host is named because that is the machine to go and look at."""
    outputs = _healthy_fleet_output()
    outputs[_HEAD_HOST] = _probe_out(rpc_listen=0, rpc_peers=f"{_W1_HOST},")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL h1: attached to 1 RPC peer(s)" in result.output
    assert f"Missing: {_W2_HOST}" in result.output
    assert "lopsided" in result.output


def test_doctor_skips_the_peer_check_when_the_head_is_not_attached(monkeypatch, tmp_path):
    """A stopped group and a head still streaming tensors both look like an
    empty peer list. A doctor that fails on a stopped fleet is one nobody runs
    — and running it against a stopped fleet is the normal case."""
    outputs = _healthy_fleet_output()
    outputs[_HEAD_HOST] = _probe_out(rpc_listen=0, rpc_peers="")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 0, result.output
    assert "SKIP head h1 is attached to no peers" in result.output
    assert "FAIL" not in result.output


def test_doctor_exits_zero_on_a_stopped_group(monkeypatch, tmp_path):
    """`aureax stop` leaves every machine up with nothing listening and the head
    holding no peer connection. That is the state an operator runs the doctor
    in — comparing builds before `aureax start` — and one FAIL per worker for it
    made the exit code say "this group is broken" about a group nobody started.
    """
    outputs = {
        _HEAD_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
        _W1_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
        _W2_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
    }

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 0, result.output
    assert "SKIP group is stopped" in result.output
    assert "FAIL" not in result.output
    # The static half is what the operator came for, so it still runs.
    assert "all 3 member(s) run llama.cpp build b10375" in result.output
    assert "image digests match on every member" in result.output


def test_a_stopped_group_still_fails_on_a_build_mismatch(monkeypatch, tmp_path):
    """Only the runtime checks are skipped. A stopped group split across two
    llama.cpp builds hangs at the RPC handshake the moment it is started, which
    is precisely what a pre-start doctor run is for."""
    outputs = {
        _HEAD_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
        _W1_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
        _W2_HOST: _probe_out(build="b10201", rpc_listen=0, rpc_peers=""),
    }

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "SKIP group is stopped" in result.output
    assert "FAIL w2: llama.cpp build b10201" in result.output


def test_a_half_started_group_is_not_treated_as_stopped(monkeypatch, tmp_path):
    """One worker listening and one not is the failure the listener check exists
    for: the head waits out its worker_wait_timeout_seconds and exits. Only a
    group with NO listeners anywhere is stopped."""
    outputs = {
        _HEAD_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
        _W1_HOST: _probe_out(rpc_listen=1, rpc_peers=""),
        _W2_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
    }

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "SKIP group is stopped" not in result.output
    assert "FAIL w2: nothing is listening on :50052" in result.output


def test_an_unprobed_head_is_unknown_not_stopped(monkeypatch, tmp_path):
    """"Stopped" is a claim about the head, and a head that did not answer
    cannot make it: reporting a group as stopped because its head is
    unreachable would hide the one member the whole verdict depends on."""
    outputs = {
        _HEAD_HOST: OSError("Connection refused"),
        _W1_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
        _W2_HOST: _probe_out(rpc_listen=0, rpc_peers=""),
    }

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "SKIP group is stopped" not in result.output
    assert "FAIL h1: could not be probed" in result.output


# A fleet whose targets are DNS names, which the schema permits and
# tlsstore._host_san explicitly anticipates. The head's probe reports the
# ADDRESSES its kernel holds connections to, never the names.
_NAMED_HOSTS = {"h1.lan": "10.77.0.30", "w1.lan": "10.77.0.31", "w2.lan": "10.77.0.32"}


def _named_doctor_fleet(tmp_path):
    net = tmp_path / "named.yaml"
    net.write_text(
        _FLEET_HEADER
        + "    - {node_id: h1, host: h1.lan}\n"
        + "    - {node_id: w1, host: w1.lan}\n"
        + "    - {node_id: w2, host: w2.lan}\n"
        + "node_groups:\n"
        + "  - {name: g1, head: h1, workers: [w1, w2], rpc_port: 50052}\n"
    )
    return net


def _patch_resolver(monkeypatch, table: dict[str, list[str]]) -> None:
    """Answer the doctor's name lookups from a table; anything absent fails."""
    import socket as socket_mod

    def getaddrinfo(host, _port, *_a, **_kw):
        if host not in table:
            raise socket_mod.gaierror(socket_mod.EAI_NONAME, f"no address for {host}")
        return [
            (socket_mod.AF_INET, socket_mod.SOCK_STREAM, socket_mod.IPPROTO_TCP, "", (addr, 0))
            for addr in table[host]
        ]

    monkeypatch.setattr("orchestrator.cli.socket.getaddrinfo", getaddrinfo)


def test_doctor_matches_peer_addresses_against_resolved_worker_names(monkeypatch, tmp_path):
    """A fleet named by DNS reports peers as IP literals, because that is what
    the head's kernel holds. Compared verbatim against `host:` those never
    match, and a fully attached group was reported lopsided — exit 1 on a
    healthy fleet."""
    _patch_resolver(monkeypatch, {h: [ip] for h, ip in _NAMED_HOSTS.items()})
    _patch_probe(monkeypatch, {
        "h1.lan": _probe_out(rpc_listen=0, rpc_peers="10.77.0.31,10.77.0.32,"),
        "w1.lan": _probe_out(rpc_peers="10.77.0.30,"),
        "w2.lan": _probe_out(rpc_peers="10.77.0.30,"),
    })

    result = CliRunner().invoke(
        cli, ["-c", str(_named_doctor_fleet(tmp_path)), "group", "doctor"])

    assert result.exit_code == 0, result.output
    assert "head h1 is attached to all 2 worker(s)" in result.output
    assert "FAIL" not in result.output


def test_doctor_says_so_when_a_worker_host_could_not_be_resolved(monkeypatch, tmp_path):
    """An unresolvable name falls back to a literal comparison — best effort —
    but the verdict has to name it: otherwise a resolver problem on this Mac
    reads as "the head is attached to the wrong machines"."""
    _patch_resolver(monkeypatch, {"h1.lan": ["10.77.0.30"], "w1.lan": ["10.77.0.31"]})
    _patch_probe(monkeypatch, {
        "h1.lan": _probe_out(rpc_listen=0, rpc_peers="10.77.0.31,"),
        "w1.lan": _probe_out(rpc_peers="10.77.0.30,"),
        "w2.lan": _probe_out(rpc_peers="10.77.0.30,"),
    })

    result = CliRunner().invoke(
        cli, ["-c", str(_named_doctor_fleet(tmp_path)), "group", "doctor"])

    assert result.exit_code == 1, result.output
    assert "Missing: w2.lan" in result.output
    assert "w2.lan could not be resolved" in result.output


def test_doctor_exits_one_when_a_member_is_unreachable(monkeypatch, tmp_path):
    """Consistency is UNKNOWN, not fine: the whole question the doctor answers
    is whether these machines agree, and one silent member makes the answer
    unavailable rather than yes."""
    outputs = _healthy_fleet_output()
    outputs[_W2_HOST] = OSError("Connection refused")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL w2: could not be probed" in result.output
    assert "UNKNOWN" in result.output


def test_doctor_exits_one_when_a_member_times_out(monkeypatch, tmp_path):
    """rc 124 is run_command's timeout convention and it returns no stdout at
    all, so the member has no facts — the same UNKNOWN as an unreachable one."""
    outputs = _healthy_fleet_output()
    outputs[_W1_HOST] = (124, "")

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "PROBE TIMED OUT" in result.output
    assert "FAIL w1: could not be probed" in result.output


def test_a_below_floor_build_warns_but_does_not_change_the_exit_code(monkeypatch, tmp_path):
    """It is a security advisory about a knowingly-run rollback, not a peer
    mismatch: every member agrees, so the group works. Gating the exit on it
    would make `group doctor` unusable as a pre-flight for anyone running the
    122B rollback image."""
    outputs = {
        _HEAD_HOST: _probe_out(build="b10201", rpc_listen=0,
                               rpc_peers=f"{_W1_HOST},{_W2_HOST},"),
        _W1_HOST: _probe_out(build="b10201", rpc_peers=f"{_HEAD_HOST},"),
        _W2_HOST: _probe_out(build="b10201", rpc_peers=f"{_HEAD_HOST},"),
    }

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 0, result.output
    assert "below the security floor" in result.output
    assert "FAIL" not in result.output


def test_doctor_reports_an_unparseable_build_rather_than_comparing_nothing(monkeypatch, tmp_path):
    """"unavailable (image missing, or docker/sudo failed)" carries no build
    number, and silently dropping that member from the comparison would report
    a coherent group whose consistency was never checked."""
    outputs = _healthy_fleet_output()
    outputs[_W1_HOST] = (
        "llama_build_docker=unavailable (image missing, or docker/sudo failed)\n"
        "llama_build_k3s=present\n"
        "image_digest_docker=?\n"
        "image_digest_k3s=1a2b3c4d5e6f\n"
        "rpc_listen=1\n"
        f"rpc_peers={_HEAD_HOST},\n"
    )

    result = _doctor(monkeypatch, tmp_path, outputs)

    assert result.exit_code == 1, result.output
    assert "FAIL w1: llama.cpp build unreported" in result.output
    assert "image missing, or docker/sudo failed" in result.output


def test_group_problems_is_pure_and_names_the_group_it_was_given():
    """The helper takes facts, not a fleet: every verdict above is reachable in
    a unit test with no SSH, which is what keeps them cheap enough to have one
    per failure mode."""
    from orchestrator.cli import _group_problems
    from tests.conftest import make_network_config, make_targets

    config = make_network_config(
        make_targets("h1", "w1"),
        groups=[{"name": "g1", "head": "h1", "workers": ["w1"], "rpc_port": 50052}],
    )
    group = config.node_groups[0]
    facts = {
        "h1": {"ssh_host": "10.77.0.1", "llama_build_docker": '{"version":"b10375"}',
               "llama_build_k3s": "present", "image_digest_docker": "aaaa", "image_digest_k3s":
               "aaaa", "rpc_listen": "0", "rpc_peers": "10.77.0.2,"},
        "w1": {"ssh_host": "10.77.0.2", "llama_build_docker": '{"version":"b10375"}',
               "llama_build_k3s": "present", "image_digest_docker": "aaaa", "image_digest_k3s":
               "aaaa", "rpc_listen": "1", "rpc_peers": "10.77.0.1,"},
    }

    problems, notes = _group_problems(group, facts, "h1")
    assert problems == []
    assert any("attached to all 1 worker(s)" in n for n in notes)

    facts["w1"]["rpc_listen"] = "0"
    problems, _ = _group_problems(group, facts, "h1")
    assert len(problems) == 1 and problems[0].startswith("FAIL w1: nothing is listening")


# --- `aureax wg` and the tunnel in `group list` / `group doctor` --------------


def _wg_fleet(tmp_path, *, enabled: bool = True):
    net = tmp_path / "wg-fleet.yaml"
    net.write_text(
        _FLEET_HEADER
        + "    - {node_id: h1, host: 10.77.0.30}\n"
        "    - {node_id: w1, host: 10.77.0.31}\n"
        "    - {node_id: w2, host: 10.77.0.32}\n"
        "    - {node_id: h2, host: 10.77.0.40}\n"
        "    - {node_id: w3, host: 10.77.0.41}\n"
        "node_groups:\n"
        f"  - {{name: g1, head: h1, workers: [w1, w2], wireguard: {{enabled: {str(enabled).lower()}}}}}\n"
        "  - {name: g2, head: h2, workers: [w3]}\n"
    )
    return net


def test_wg_status_reports_each_member_without_minting(tmp_aureax_home, tmp_path):
    result = CliRunner().invoke(cli, ["-c", str(_wg_fleet(tmp_path)), "wg", "status"])

    assert result.exit_code == 0, result.output
    assert "g1  [on] subnet 10.254.0.0/24  udp/51820  mtu 1420  keepalive 25s" in result.output
    for node_id, addr in (("h1", "10.254.0.1"), ("w1", "10.254.0.2"), ("w2", "10.254.0.3")):
        assert re.search(rf"{node_id}\s+{re.escape(addr)}\s+key file absent", result.output), node_id
    assert "no key — run `aureax generate" in result.output
    assert "g2  [off]" in result.output
    assert not (tmp_aureax_home / ".aureax" / "wireguard").exists(), "status minted a key"


def test_wg_status_shows_the_public_key_and_the_key_file(tmp_aureax_home, tmp_path):
    from orchestrator.config import wgstore

    pub = wgstore.ensure_key("w1")
    result = CliRunner().invoke(cli, ["-c", str(_wg_fleet(tmp_path)), "wg", "status"])

    assert result.exit_code == 0, result.output
    assert re.search(r"w1\s+10\.254\.0\.2\s+key file present", result.output)
    assert f"public key {pub}" in result.output
    assert wgstore.private_key("w1") not in result.output, "the private key must never be printed"


def test_wg_regenerate_rotates_and_names_both_follow_up_steps(tmp_aureax_home, tmp_path):
    from orchestrator.config import wgstore

    before = wgstore.ensure_key("w1")
    old_private = wgstore.private_key("w1")
    result = CliRunner().invoke(
        cli, ["-c", str(_wg_fleet(tmp_path)), "wg", "regenerate", "--node", "w1"]
    )

    assert result.exit_code == 0, result.output
    after = wgstore.public_key("w1")
    assert after != before
    assert f"Rotated w1: public key {after}" in result.output
    assert wgstore.previous_private_key("w1") == old_private
    assert "aureax apply --node w1 --orchestrator-ip <ip>" in result.output
    assert "aureax stop && aureax start" in result.output


@pytest.mark.parametrize("node_id", ["w3", "nobody", "../x"])
def test_wg_regenerate_refuses_a_node_outside_an_enabled_group(tmp_aureax_home, tmp_path, node_id):
    result = CliRunner().invoke(
        cli, ["-c", str(_wg_fleet(tmp_path)), "wg", "regenerate", "--node", "w1", "--node", node_id]
    )

    assert result.exit_code == 1
    assert "not a member of a node group with wireguard.enabled" in result.output
    assert not (tmp_aureax_home / ".aureax" / "wireguard").exists(), "a refusal rotated something"


def test_wg_regenerate_needs_a_node(tmp_aureax_home, tmp_path):
    result = CliRunner().invoke(cli, ["-c", str(_wg_fleet(tmp_path)), "wg", "regenerate"])
    assert result.exit_code == 2


def test_group_list_names_the_tunnel_the_head_dials(tmp_path):
    result = CliRunner().invoke(cli, ["-c", str(_wg_fleet(tmp_path)), "group", "list"])

    assert result.exit_code == 0, result.output
    assert ("rpc:     over wireguard 10.254.0.0/24 (head 10.254.0.1; dials "
            "w1 10.254.0.2:50052, w2 10.254.0.3:50052)") in result.output
    assert result.output.count("over wireguard") == 1, "g2 has no tunnel"

    plain = CliRunner().invoke(cli, ["-c", str(_wg_fleet(tmp_path, enabled=False)), "group", "list"])
    assert "wireguard" not in plain.output


def test_doctor_accepts_a_head_attached_over_the_tunnel():
    """The head's rpc_peers are tunnel addresses when the group is on
    WireGuard; group_doctor injects each member's as `wg_host` so the peer
    check does not call a fully attached group lopsided."""
    from orchestrator.cli import _group_problems
    from tests.conftest import make_network_config, make_targets

    config = make_network_config(
        make_targets("h1", "w1"),
        groups=[{"name": "g1", "head": "h1", "workers": ["w1"], "wireguard": {"enabled": True}}],
    )
    group = config.node_groups[0]
    common = {"llama_build_docker": '{"version":"b10375"}', "llama_build_k3s": "present",
              "image_digest_docker": "aaaa", "image_digest_k3s": "aaaa"}
    facts = {
        "h1": {**common, "ssh_host": "10.77.0.1", "wg_host": "10.254.0.1",
               "rpc_listen": "0", "rpc_peers": "10.254.0.2,"},
        "w1": {**common, "ssh_host": "10.77.0.2", "wg_host": "10.254.0.2",
               "rpc_listen": "1", "rpc_peers": "10.254.0.1,"},
    }

    problems, notes = _group_problems(group, facts, "h1")
    assert problems == []
    assert any("attached to all 1 worker(s)" in n for n in notes)

    del facts["w1"]["wg_host"]
    problems, _ = _group_problems(group, facts, "h1")
    assert problems and "10.254.0.2, which is not in this group" in problems[0]


def test_doctor_injects_each_members_tunnel_address(monkeypatch, tmp_path):
    seen: dict[str, dict] = {}
    from orchestrator import cli as cli_mod

    def fake_problems(group, facts_by_node, head_id):
        seen.update(facts_by_node)
        return [], []

    monkeypatch.setattr(cli_mod, "_group_problems", fake_problems)
    monkeypatch.setattr("orchestrator.deploy.ssh.run_command", lambda *_a, **_k: (0, "", ""))
    result = CliRunner().invoke(cli, ["-c", str(_wg_fleet(tmp_path)), "group", "doctor"])

    assert result.exit_code == 0, result.output
    assert seen["h1"]["wg_host"] == "10.254.0.1"
    assert seen["w2"]["wg_host"] == "10.254.0.3"
    assert "wg_host" not in seen["w3"], "g2 is not on WireGuard"
