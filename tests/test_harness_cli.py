"""`aureax-harness` reports failures a wrapper can act on, and reaches a node
the way the fleet requires: TLS against the fleet CA, a bearer key that never
appears in output, and a positional that may name a node instead of an address.

Four contracts are pinned here.

*Exit status*: scripts/test-all.sh grades harness invocations on exit status
alone, so a tool-level failure printed in red on stdout while the process exits
0 is counted as a passing test.

*Detail text*: the MCP SDK wraps every transport failure in an ExceptionGroup
whose own message names no cause, so the harness renders the leaves rather than
the group — and a TLS verification failure arrives in exactly that shape, so
telling the two apart is a walk down ``__cause__``, not a type check. The two
remedies are opposites (wait for the pod / fix the CA on this Mac), which is
why one hint must never be printed for the other's failure.

*Credentials*: the key has five possible sources with a fixed precedence, and
none of them may reach the terminal.

*Endpoint*: the positional resolves an enabled node id through the fleet
config, which is what lets the key come from the keystore with nothing typed.

Everything runs against ``tmp_aureax_home`` and a real CA minted by
``tlsstore``: the harness verifies with ``ssl.create_default_context(cafile=)``,
so a stand-in that is not a parseable bundle would not exercise the path.
"""

from __future__ import annotations

import asyncio
import os
import ssl
from pathlib import Path
from unittest import mock

import click
import httpx
import pytest
from click.testing import CliRunner

from orchestrator.config import keystore, tlsstore
from orchestrator.harness import cli as harness
from tests.conftest import make_targets, write_network_yaml

# Shaped like a real key (keystore.KEY_PREFIX) so the leak test greps for the
# token scripts/test-all.sh greps for.
FAKE_KEY = "aureax_sk_" + "z" * 43

# File modes are the mechanism under test in one case below, and root ignores
# them — same guard tests/test_tlsstore.py uses.
unprivileged_posix_only = pytest.mark.skipif(
    os.name != "posix" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file modes only, and root reads a 0000 file regardless",
)


@pytest.fixture
def runner() -> CliRunner:
    """A Click runner with stdout and stderr kept separate."""
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_store(tmp_aureax_home, monkeypatch) -> Path:
    """A redirected $HOME holding a real fleet CA, and no ambient credentials.

    The harness reads ``$AUREAX_MCP_API_KEY``, ``$AUREAX_CA`` and the
    orchestrator's own key store, all of which exist on the machine this suite
    is developed on — without clearing them a precedence test would pass
    because of the developer's environment rather than because of the code.

    The CA is minted for real rather than stubbed because ``_ca_context``
    builds an ``ssl.SSLContext`` from the file, so anything unparseable fails
    before the code under test runs.
    """
    monkeypatch.delenv(harness.ENV_API_KEY, raising=False)
    monkeypatch.delenv(harness.ENV_CA, raising=False)
    harness._ssl_contexts.clear()
    return tlsstore.ensure_ca()


@pytest.fixture
def fleet(tmp_path) -> Path:
    """A two-target fleet file: `edge-01` enabled at 10.77.0.1, `edge-02` disabled."""
    return write_network_yaml(
        tmp_path / "configs", make_targets("edge-01", "edge-02", disabled=("edge-02",))
    )


def _aclose(client: httpx.AsyncClient) -> None:
    """Close an AsyncClient built outside a running loop."""
    asyncio.run(client.aclose())


def _tls_group() -> ExceptionGroup:
    """The exact shape httpx 0.28 produces for an untrusted certificate.

    Built by raising rather than by assigning ``__cause__``, so the chain is
    the one the interpreter builds: an ``httpx.ConnectError`` raised ``from``
    the ssl error, wrapped in the MCP SDK's task-group ExceptionGroup.
    """
    try:
        try:
            raise ssl.SSLCertVerificationError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                "unable to get local issuer certificate"
            )
        except ssl.SSLError as ssl_error:
            raise httpx.ConnectError(str(ssl_error)) from ssl_error
    except httpx.ConnectError as connect_error:
        return ExceptionGroup("unhandled errors in a TaskGroup", [connect_error])


def _auth_group(status: int = 401) -> ExceptionGroup:
    """A `status` response from POST /mcp, in the shape the MCP SDK delivers.

    httpx raises ``HTTPStatusError`` out of ``raise_for_status()`` inside the
    SDK's task group, so it reaches the harness as an ExceptionGroup leaf —
    indistinguishable, before the classification under test, from a refused
    connection.
    """
    request = httpx.Request("POST", "https://10.77.0.1:30900/mcp")
    response = httpx.Response(status, request=request)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as status_error:
        return ExceptionGroup("unhandled errors in a TaskGroup", [status_error])
    raise AssertionError(f"{status} did not raise")


# --- exit status and failure detail (pre-existing contracts) ----------------

# Both shapes `_call_tool` can return for a failure: a call the server marked
# isError, and a tool that caught its own exception and returned it as an
# ordinary result (run_inference does this whenever llama-server is silent).
FAILURE_SHAPES = [
    {"error": True, "content": "llama_server unhealthy"},
    {"error": "Inference failed: connection refused"},
]

TOOL_COMMANDS = [
    ("call", ["node-a", "get_health"]),
    ("sensor", ["node-a"]),
    ("infer", ["node-a", "hello"]),
]


@pytest.mark.parametrize("result", FAILURE_SHAPES)
@pytest.mark.parametrize(("command", "argv"), TOOL_COMMANDS)
def test_tool_level_failure_sets_a_non_zero_exit_status(runner, command, argv, result) -> None:
    """A failed tool call exits non-zero, so a wrapper can see it."""
    with mock.patch.object(harness, "_call_tool", mock.AsyncMock(return_value=result)):
        outcome = runner.invoke(harness.cli, [command, *argv])
    assert outcome.exit_code == 1, outcome.output


@pytest.mark.parametrize(("command", "argv"), TOOL_COMMANDS)
def test_successful_tool_call_still_exits_zero(runner, command, argv) -> None:
    """The failure exit is keyed on the error shape, not on every result."""
    ok = {"content": "4", "tokens_predicted": 1, "generation_time_ms": 12}
    with mock.patch.object(harness, "_call_tool", mock.AsyncMock(return_value=ok)):
        outcome = runner.invoke(harness.cli, [command, *argv])
    assert outcome.exit_code == 0, outcome.output


def test_ping_prints_the_leaf_cause_under_its_verdict(runner) -> None:
    """A refused connection and an unready pod must not read identically."""
    group = ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("refused")])
    with mock.patch.object(harness, "_ping", mock.AsyncMock(return_value=group)):
        outcome = runner.invoke(harness.cli, ["ping", "node-a"])
    assert outcome.exit_code == 1
    assert "FAILED" in outcome.output
    assert "ConnectError: refused" in outcome.output


def test_the_endpoint_hint_is_only_attached_to_transport_failures() -> None:
    """A bug in a tool handler must not be diagnosed as an unreachable node."""
    transport = ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("refused")])
    assert harness._is_connection_failure(transport) is True
    assert harness._is_connection_failure(TypeError("bad argument")) is False
    assert harness._describe(transport) == "ConnectError: refused"


def test_the_spinner_writes_nothing_when_stderr_is_not_a_terminal(capsys) -> None:
    """Redirected stderr must not collect thousands of repainted frames."""
    with harness._Spinner("Generating"):
        pass
    assert capsys.readouterr().err == ""


# --- TLS ---------------------------------------------------------------------


def test_make_url_is_https() -> None:
    """There is no plaintext MCP listener left to fall back to."""
    assert harness._make_url("10.77.0.1", 30900) == "https://10.77.0.1:30900/mcp"


def test_make_url_brackets_an_ipv6_literal() -> None:
    """A bare literal's own colons read as the port separator, so httpx cannot
    parse the authority at all. The fleet CA's name constraints admit fc00::/7,
    so an IPv6 node address is a supported shape, not a typo."""
    assert harness._make_url("fd00::5", 30900) == "https://[fd00::5]:30900/mcp"


def test_a_certificate_failure_is_tls_not_unreachable() -> None:
    """The node answered; the CA simply did not sign what it served."""
    group = _tls_group()
    assert isinstance(harness._tls_failure(group), ssl.SSLCertVerificationError)
    # Mutually exclusive on purpose: the two hints send the operator to
    # opposite machines.
    assert harness._is_connection_failure(group) is False


def test_a_plain_connection_failure_is_not_tls() -> None:
    """A refused connection keeps the unreachable hint it has always had."""
    group = ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("refused")])
    assert harness._tls_failure(group) is None
    assert harness._is_connection_failure(group) is True


def test_ping_reports_a_tls_failure_as_tls(runner, fleet, _isolated_store) -> None:
    """The verdict names TLS, and the remedy names the CA and the node."""
    with mock.patch.object(harness, "_ping", mock.AsyncMock(return_value=_tls_group())):
        outcome = runner.invoke(harness.cli, ["ping", "edge-01", "--config", str(fleet)])
    assert outcome.exit_code == 1
    assert "TLS verification failed" in outcome.output
    assert str(_isolated_store) in outcome.output
    assert "aureax apply --node edge-01 --orchestrator-ip" in outcome.output
    assert "MCP server not reachable" not in outcome.output


def test_the_tls_hint_names_a_command_that_runs(runner, fleet, _isolated_store) -> None:
    """This is the remedy printed at the exact moment verification fails, so it
    has to be runnable: `aureax apply` has a REQUIRED option of its own, and a
    hint that omits it exits 2 on Click's argument parsing instead."""
    from orchestrator.cli import apply as apply_command

    with mock.patch.object(harness, "_ping", mock.AsyncMock(return_value=_tls_group())):
        outcome = runner.invoke(harness.cli, ["ping", "edge-01", "--config", str(fleet)])

    required = [
        p for p in apply_command.params if isinstance(p, click.Option) and p.required
    ]
    assert required, "apply has no required option — this test no longer guards anything"
    for param in required:
        assert param.opts[0] in outcome.output, f"the hint omits {param.opts[0]}"


# --- a rejected key --------------------------------------------------------


def test_a_401_is_a_rejected_key_not_an_unreachable_node() -> None:
    """The node answered. Reporting it as unreachable hides the only remedy.

    All three classifications have to disagree at once: a 401 is not TLS, not a
    transport failure (httpx.HTTPStatusError is neither a TransportError nor an
    OSError), and is the one thing _auth_failure recognises.
    """
    group = _auth_group()
    rejected = harness._auth_failure(group)
    assert isinstance(rejected, httpx.HTTPStatusError)
    assert rejected.response.status_code == 401
    assert harness._tls_failure(group) is None
    assert harness._is_connection_failure(group) is False


def test_other_http_statuses_are_not_read_as_a_rejected_key() -> None:
    """A 404 on /mcp is a wrong path or an unpublished route, not a key problem.

    Widening this to "any HTTPStatusError" would answer a mis-routed NodePort
    with advice about keys, which is the mirror of the bug being fixed.
    """
    assert harness._auth_failure(_auth_group(404)) is None
    assert harness._auth_failure(_auth_group(500)) is None


def test_a_plain_connection_failure_is_not_a_rejected_key() -> None:
    """The unreachable verdict still owns the case it has always owned."""
    group = ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("refused")])
    assert harness._auth_failure(group) is None


def test_ping_reports_a_401_as_a_rejected_key(runner, fleet) -> None:
    """The verdict names the 401, and the hint names every way to supply a key."""
    with mock.patch.object(harness, "_ping", mock.AsyncMock(return_value=_auth_group())):
        outcome = runner.invoke(harness.cli, ["ping", "10.77.0.1", "--config", str(fleet)])
    assert outcome.exit_code == 1
    assert "MCP key rejected (401 Unauthorized)" in outcome.output
    # The misleading headline this replaced, and the hint belonging to the
    # other verdict, must both be gone.
    assert "MCP server not reachable" not in outcome.output
    assert "TLS verification failed" not in outcome.output
    assert "~/.aureax/keys/<node_id>.mcp.key" in outcome.output
    assert f"${harness.ENV_API_KEY}" in outcome.output
    # The leaf cause is still printed under the verdict, as for every failure.
    assert "HTTPStatusError" in outcome.output


def test_a_401_reaching_main_gets_the_key_hint_and_no_endpoint_advice(capsys) -> None:
    """`infer` raises out of the command, and main() is the single renderer.

    scripts/test-all.sh's end-to-end check is an `infer`, so main() is the path
    an operator reads when that check fails — and the unreachable hint it used
    to get talks about pods still loading and shard workers, none of which
    applies to a node that answered.
    """
    with (
        mock.patch.object(harness, "cli", mock.Mock(side_effect=_auth_group())),
        pytest.raises(SystemExit) as exit_info,
    ):
        harness.main()
    assert exit_info.value.code == 1

    err = capsys.readouterr().err
    assert "401 Unauthorized" in err
    assert "~/.aureax/keys/<node_id>.mcp.key" in err
    assert f"${harness.ENV_API_KEY}" in err
    assert "the MCP endpoint may be unreachable" not in err


def test_ping_still_reports_a_refused_connection_as_unreachable(runner, fleet) -> None:
    """The new branch must not capture the failure it sits next to."""
    group = ExceptionGroup("unhandled errors in a TaskGroup", [httpx.ConnectError("refused")])
    with mock.patch.object(harness, "_ping", mock.AsyncMock(return_value=group)):
        outcome = runner.invoke(harness.cli, ["ping", "10.77.0.1", "--config", str(fleet)])
    assert outcome.exit_code == 1
    assert "MCP server not reachable" in outcome.output
    assert "401" not in outcome.output


def test_ca_precedence_is_flag_then_env_then_default(monkeypatch, tmp_path) -> None:
    """--ca beats $AUREAX_CA beats the store default."""
    monkeypatch.setenv(harness.ENV_CA, str(tmp_path / "env-ca.pem"))
    assert harness._ca_path(str(tmp_path / "flag-ca.pem")) == tmp_path / "flag-ca.pem"
    assert harness._ca_path(None) == tmp_path / "env-ca.pem"

    monkeypatch.delenv(harness.ENV_CA)
    assert harness._ca_path(None) == Path.home() / ".aureax" / "tls" / "ca.pem"


def test_the_ca_context_trusts_the_named_bundle(_isolated_store) -> None:
    """`verify=` is a real context built from the file, never a disabled check."""
    ctx = harness._ca_context(str(_isolated_store))
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    assert ctx.get_ca_certs(), "the bundle loaded but holds no certificates"


def test_the_default_ca_is_the_store(_isolated_store) -> None:
    """With the store in place, no --ca and no env var is the working case."""
    assert harness._ca_context(None).get_ca_certs()


def test_a_missing_ca_names_the_path_and_the_remedy(runner) -> None:
    """A CA this Mac never generated must not read as an unreachable node."""
    outcome = runner.invoke(harness.cli, ["ping", "node-a", "--ca", "/nonexistent/ca.pem"])
    assert outcome.exit_code == 1
    assert "/nonexistent/ca.pem" in outcome.output
    assert "aureax generate" in outcome.output
    assert "unreachable" not in outcome.output


# --- endpoint and key resolution ---------------------------------------------


def test_an_enabled_node_id_resolves_to_that_targets_address(fleet) -> None:
    """`ping edge-01` dials 10.77.0.1 and remembers which node that was."""
    url, _key, node_id = harness._resolve_endpoint("edge-01", 30900, fleet_config=str(fleet))
    assert url == "https://10.77.0.1:30900/mcp"
    assert node_id == "edge-01"


def test_anything_that_is_not_an_enabled_node_id_is_a_literal_host(fleet) -> None:
    """An address still works, and a DISABLED target's id is not a node id.

    Disabled targets are skipped by every other pass — config generation, both
    label passes, the pod waits — so resolving one here would hand the operator
    an address nothing has been deployed to.
    """
    url, _key, node_id = harness._resolve_endpoint("10.77.0.9", 30900, fleet_config=str(fleet))
    assert (url, node_id) == ("https://10.77.0.9:30900/mcp", None)

    url, _key, node_id = harness._resolve_endpoint("edge-02", 30900, fleet_config=str(fleet))
    assert (url, node_id) == ("https://edge-02:30900/mcp", None)


def test_the_resolved_node_id_is_printed(runner, fleet) -> None:
    """A typo'd id silently becoming a hostname is the failure this prevents."""
    with mock.patch.object(harness, "_ping", mock.AsyncMock(return_value=None)):
        outcome = runner.invoke(harness.cli, ["ping", "edge-01", "--config", str(fleet)])
    assert "https://10.77.0.1:30900/mcp (node edge-01)" in outcome.output


def test_a_missing_default_fleet_file_is_not_an_error(monkeypatch, tmp_path) -> None:
    """A machine that only ever talks to one address needs no fleet file."""
    monkeypatch.chdir(tmp_path)
    assert harness._resolve_endpoint("10.77.0.9", 30900) == (
        "https://10.77.0.9:30900/mcp", None, None,
    )


def test_a_named_fleet_file_that_cannot_be_read_is_an_error(runner, tmp_path) -> None:
    """A typo'd --config must not silently degrade into a literal host."""
    outcome = runner.invoke(
        harness.cli, ["ping", "edge-01", "--config", str(tmp_path / "nope.yaml")]
    )
    assert outcome.exit_code == 1
    assert "nope.yaml" in outcome.output


def test_key_precedence_table(fleet, tmp_path, monkeypatch) -> None:
    """--api-key > --key-file > $AUREAX_MCP_API_KEY > keystore > none."""
    key_file = tmp_path / "from-file.key"
    key_file.write_text("aureax_sk_from_file\n")
    monkeypatch.setenv(harness.ENV_API_KEY, "aureax_sk_from_env")
    stored = keystore.ensure_key("edge-01", kind=keystore.KIND_MCP)

    def resolve(host: str, **kwargs) -> str | None:
        return harness._resolve_endpoint(host, 30900, fleet_config=str(fleet), **kwargs)[1]

    # 1. the flag beats everything below it
    assert resolve("edge-01", api_key="aureax_sk_from_flag",
                   key_file=str(key_file)) == "aureax_sk_from_flag"
    # 2. the file beats the environment and the keystore, and is stripped
    assert resolve("edge-01", key_file=str(key_file)) == "aureax_sk_from_file"
    # 3. the environment beats the keystore
    assert resolve("edge-01") == "aureax_sk_from_env"
    # 4. the keystore — reachable only because the positional resolved a node id
    monkeypatch.delenv(harness.ENV_API_KEY)
    assert resolve("edge-01") == stored
    # 5. nothing at all for a literal host, which is correct for an auth-off node
    assert resolve("10.77.0.9") is None


def test_the_keystore_is_only_read_for_the_mcp_kind(fleet) -> None:
    """A node's llama key must never be sent to its MCP server.

    Both kinds carry the `aureax_sk_` prefix and live in the same directory, so
    a wrong ``kind=`` would send a key that looks right and 401s on every call.
    """
    llama_only = keystore.ensure_key("edge-01", kind=keystore.KIND_LLAMA)
    assert harness._resolve_key(None, None, "edge-01") is None

    mcp = keystore.ensure_key("edge-01", kind=keystore.KIND_MCP)
    assert harness._resolve_key(None, None, "edge-01") == mcp
    assert mcp != llama_only


@unprivileged_posix_only
def test_an_unreadable_key_file_is_not_reported_as_an_unreachable_node(runner, tmp_path) -> None:
    """A .key left root-owned by a sudo run is a usage error, not a dead node.

    Click's own ``Path(readable=True)`` catches it first and names the file;
    what matters is that the endpoint hint — which would send the operator to
    the node for a permissions problem on this Mac — is nowhere in the output.
    """
    key_file = tmp_path / "locked.key"
    key_file.write_text(FAKE_KEY)
    key_file.chmod(0o000)
    outcome = runner.invoke(harness.cli, ["ping", "node-a", "--key-file", str(key_file)])
    assert outcome.exit_code == 2
    assert str(key_file) in outcome.output
    assert "unreachable" not in outcome.output
    assert FAKE_KEY not in outcome.output


@unprivileged_posix_only
def test_reading_a_key_file_that_fails_late_is_still_not_a_transport_error(tmp_path) -> None:
    """OSError is in _CONNECTION_ERRORS, so the read has to catch its own.

    Click's readability check happens at parse time; the read happens later, so
    a file whose permissions change in between — or any direct caller — must
    not hand `main` an OSError to diagnose as an unreachable endpoint.
    """
    key_file = tmp_path / "locked.key"
    key_file.write_text(FAKE_KEY)
    key_file.chmod(0o000)
    with pytest.raises(click.ClickException) as raised:
        harness._read_key_file(str(key_file))
    assert str(key_file) in str(raised.value)
    assert FAKE_KEY not in str(raised.value)


# --- the Authorization header ------------------------------------------------


def test_the_bearer_header_is_set_when_a_key_is_resolved() -> None:
    """One place applies it: streamable_http_client takes no headers of its own."""
    client = harness._make_http_client(api_key=FAKE_KEY)
    try:
        assert client.headers["authorization"] == f"Bearer {FAKE_KEY}"
    finally:
        _aclose(client)


def test_no_bearer_header_without_a_key() -> None:
    """An auth-off node must not be sent an empty or literal-None header."""
    client = harness._make_http_client()
    try:
        assert "authorization" not in client.headers
    finally:
        _aclose(client)


# --- the key never reaches the terminal --------------------------------------


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("ping", ["edge-01"]),
        ("call", ["edge-01", "get_health"]),
        ("sensor", ["edge-01"]),
        ("infer", ["edge-01", "hello"]),
    ],
)
def test_the_key_is_never_echoed(runner, fleet, command, argv) -> None:
    """Not in the endpoint line, not in a failure, not in a hint.

    Harness output is pasted into issues and transcripts, and
    scripts/test-all.sh greps captured output for `aureax_sk_` precisely
    because a key that reaches a terminal is a leaked key.
    """
    failure = {"error": True, "content": "llama_server unhealthy"}
    with (
        mock.patch.object(harness, "_call_tool", mock.AsyncMock(return_value=failure)),
        mock.patch.object(harness, "_ping", mock.AsyncMock(return_value=_tls_group())),
    ):
        outcome = runner.invoke(harness.cli, [
            command, *argv, "--config", str(fleet), "--api-key", FAKE_KEY,
        ])
    assert FAKE_KEY not in outcome.output
    assert "aureax_sk_" not in outcome.output
