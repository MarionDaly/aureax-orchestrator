"""Guards for ``orchestrator.config.loader`` and the schema it validates.

Covers happy-path loading, bad YAML raising ``ConfigError`` rather than
``SystemExit``, the ``config_file`` path-segment and duplicate-node-id rules,
and the one orchestrator-host rule the CLI and the dashboard both apply.
"""

from __future__ import annotations

import os
import re

import pytest
from pydantic import ValidationError

from orchestrator.config.loader import (
    ConfigError,
    load_edge_config,
    load_network_config,
    validate_orchestrator_host,
    validate_yaml_text,
)
from orchestrator.config.schema import NetworkConfig, SSHConfig, SSHTarget
from tests.conftest import PROJECT_ROOT, REAL_NETWORK_CONFIG


def test_loads_example_network_config(network_config_path):
    """The tracked example — the file a fresh clone actually has."""
    cfg = load_network_config(network_config_path)
    assert cfg.network.name
    assert len(cfg.ssh.targets) >= 1


@pytest.mark.skipif(
    not REAL_NETWORK_CONFIG.exists(),
    reason="operator-owned fleet config not present",
)
def test_operator_network_config_validates_if_present():
    """The operator's own gitignored config still has to load, when it is there.

    configs/aureax-network.yaml carries real addresses and SSH credentials, so
    it is gitignored and absent from a fresh clone — which is why every other
    test in the suite reads the example instead. This is the one check that
    keeps the operator's file honest: it validates against the same schema, and
    every enabled target that names a per-node `config_file` names one that is
    actually in configs/. A dangling name is not a load error; it surfaces as a
    failed deploy against a live node.
    """
    cfg = load_network_config(REAL_NETWORK_CONFIG)
    assert isinstance(cfg, NetworkConfig)
    assert cfg.ssh.enabled_targets, "the operator config has no enabled targets"

    configs_dir = PROJECT_ROOT / "configs"
    for target in cfg.ssh.enabled_targets:
        if not target.config_file:
            continue  # optional in the schema; such a node just has no role
        assert (configs_dir / target.config_file).exists(), (
            f"{target.node_id!r} names config_file={target.config_file!r}, "
            f"which is not in {configs_dir}"
        )


def test_invalid_network_yaml_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("network:\n  name: 1\n")  # missing required keys + wrong types
    with pytest.raises(ConfigError) as exc_info:
        load_network_config(bad)
    assert "Invalid network config" in str(exc_info.value)


def test_config_error_is_value_error():
    """ConfigError must subclass ValueError so existing handlers catch it."""
    assert issubclass(ConfigError, ValueError)


def test_validate_yaml_text_accepts_example_network_config(network_config_path):
    text = network_config_path.read_text()
    ok, errors = validate_yaml_text(text, "network")
    assert ok, errors


def test_validate_yaml_text_rejects_malformed_yaml():
    ok, errors = validate_yaml_text("a:\n - 1\n -\nbroken", "network")
    assert not ok
    assert errors


def test_validate_yaml_text_unknown_kind_passes_silently():
    """Unknown kinds are treated as OK (loader returns ``(True, [])``)."""
    ok, errors = validate_yaml_text("anything: 1\n", "no-such-kind")
    assert ok
    assert errors == []


def test_concurrent_loads_do_not_share_a_parser(network_config_path):
    """ruamel's YAML instance is not reentrant, so each load gets its own.

    load() keeps the stream and its per-document state on the instance, and the
    dashboard reaches this module from both sides at once — the sync config
    routes run in Starlette's threadpool while an inline validate_yaml_text runs
    on the event loop. Sharing one parser corrupts both loads, and the resulting
    error is swallowed upstream into an empty Config tab that names no cause.
    """
    from concurrent.futures import ThreadPoolExecutor

    text = network_config_path.read_text()

    def work(n: int):
        return (
            load_network_config(network_config_path).network.name
            if n % 2
            else validate_yaml_text(text, "network")[0]
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(64)))

    assert all(results), results


def test_load_edge_config_with_template_placeholders():
    """The shared edge template uses ${} placeholders; load_edge_config
    must substitute dummies before validation so the template parses.

    The template is asserted to exist rather than skipped over. Returning early
    when it was missing meant this reported PASS having checked nothing, which
    is the one outcome a config-loading test must never produce.
    """
    template = PROJECT_ROOT / "configs" / "aureax-edge.yaml"
    assert template.exists(), f"shared edge template missing at {template}"
    cfg = load_edge_config(template)
    assert cfg.node.id  # placeholders resolved to dummy values


def test_planner_config_validates_with_its_placeholders_resolved():
    """aureax-planner.yaml carries ${tags} and ${mcp_endpoints}.

    The "planner" kind skipped placeholder substitution, so validation saw a
    string where a list belongs and always failed — the dashboard config editor
    could open the planner config but every save was rejected.
    """
    planner = PROJECT_ROOT / "configs" / "aureax-planner.yaml"
    assert planner.exists(), f"planner config missing at {planner}"

    ok, errors = validate_yaml_text(planner.read_text(), "planner")
    assert ok, f"planner config rejected: {errors}"


def test_planner_validation_still_rejects_a_real_error():
    """Substituting dummies must not turn validation into a rubber stamp."""
    bad = "node:\n  id: 'x'\nmodel:\n  server:\n    port: 'not-a-port'\n"
    ok, errors = validate_yaml_text(bad, "planner")
    assert not ok
    assert errors


def test_edge_kinds_are_dispatched_to_a_real_validator() -> None:
    """"edge-node" is what the dashboard passes for every per-node file.

    The kind->validator dict is a closed set that silently passes anything it
    does not know, so a removed or misspelled key turns every save of a per-node
    config into a rubber stamp — the same class of bug as the planner one above.
    """
    template = PROJECT_ROOT / "configs" / "aureax-edge.yaml"
    assert template.exists(), f"shared edge template missing at {template}"
    for kind in ("edge-node", "edge-template"):
        ok, errors = validate_yaml_text(template.read_text(), kind)
        assert ok, f"{kind}: {errors}"

    bad = "node:\n  id: 'x'\nmodel:\n  server:\n    port: 'not-a-port'\n"
    for kind in ("edge-node", "edge-template"):
        ok, errors = validate_yaml_text(bad, kind)
        assert not ok, f"{kind} rubber-stamped an invalid config"
        assert errors


# --- every failure mode is a ConfigError --------------------------------------


@pytest.mark.parametrize("load", [load_network_config, load_edge_config])
def test_malformed_yaml_raises_config_error(load, tmp_path) -> None:
    """A parse error used to escape as a raw ruamel exception.

    Both loaders caught only ValidationError while parsing OUTSIDE the try, so a
    half-written file reached cli.py as a traceback and the dashboard's config
    API as an unhandled 500 — on the unauthenticated bind.
    """
    bad = tmp_path / "bad.yaml"
    bad.write_text("a:\n - 1\n -\nbroken\n")
    with pytest.raises(ConfigError) as exc_info:
        load(bad)
    assert "Could not read" in str(exc_info.value)


@pytest.mark.parametrize("load", [load_network_config, load_edge_config])
def test_a_missing_file_raises_config_error(load, tmp_path) -> None:
    with pytest.raises(ConfigError):
        load(tmp_path / "does-not-exist.yaml")


def test_config_error_names_the_field_without_echoing_its_value(tmp_path) -> None:
    """pydantic renders `input_value=` for the failing field's parent object.

    For a network config that parent is an ssh.targets entry, so an unquoted
    factory password (`password: 12345678` parses as an int) was reproduced
    verbatim into the message — which the dashboard both logs to its persisted
    system log and returns as an HTTP 400 body.
    """
    net = tmp_path / "net.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n    - {node_id: a, host: 10.77.0.1, password: 12345678}\n"
    )
    with pytest.raises(ConfigError) as exc_info:
        load_network_config(net)

    message = str(exc_info.value)
    assert "12345678" not in message
    # Still diagnostic: the field and the reason survive.
    assert "ssh.targets.0.password" in message
    assert "valid string" in message


# --- ssh.targets identity and shape -------------------------------------------


# --- encoding ----------------------------------------------------------------

# Read in a child interpreter under -X warn_default_encoding, where every
# open()/read_text() that omits an encoding raises. Nothing about the parent's
# locale can make this test lie, which matters because CPython coerces a C
# locale to UTF-8 by itself: the failure this pins belongs to an environment
# that suppresses the coercion (PYTHONCOERCECLOCALE=0, an ASCII-only locale in
# a minimal image), and asserting on the ambient locale would pass either way.
_ENCODING_PROBE = """
import sys
from pathlib import Path

from client.config.parser import load_edge_config_raw
from orchestrator.config import group_state
from orchestrator.config.loader import load_network_config, load_raw_text

fleet, edge, home = (Path(a) for a in sys.argv[1:4])
assert load_network_config(fleet).ssh.targets[0].node_id == "a"
assert "\u2014" in load_raw_text(fleet)
assert load_edge_config_raw(edge)["node"]["id"] == "a"
group_state._read_baseline()
"""


def test_every_config_read_names_its_encoding(tmp_path, monkeypatch) -> None:
    """Configs are UTF-8, whatever the ambient locale claims.

    Every shipped config carries non-ASCII comment text, so a read that falls
    back to an ASCII locale encoding fails on the FIRST config any command
    touches — with a byte offset for a message. The three readers are the
    orchestrator's YAML loader, its raw-text reader (the dashboard's editor),
    and the node-side parser, whose container locale is POSIX.
    """
    import subprocess
    import sys

    fleet = tmp_path / "aureax-network.yaml"
    fleet.write_text(
        "# em dash \u2014 and an arrow \u2192\n"
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n    - {node_id: a, host: 10.77.0.1}\n",
        encoding="utf-8",
    )
    edge = tmp_path / "aureax-edge.yaml"
    edge.write_text("# arrow \u2192\nnode: {id: a}\n", encoding="utf-8")

    home = tmp_path / "home"
    (home / ".aureax" / "state").mkdir(parents=True)
    (home / ".aureax" / "state" / "node_groups.json").write_text("{}", encoding="utf-8")

    probe = tmp_path / "probe.py"
    probe.write_text(_ENCODING_PROBE, encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning",
         str(probe), str(fleet), str(edge), str(home)],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "HOME": str(home)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_duplicate_node_ids_are_rejected() -> None:
    """node_id keys group membership, generated filenames, hostname resolution
    and dashboard addressing — a duplicate makes whichever loses the dict build
    silently invisible, so it never deploys and nothing says why."""
    with pytest.raises(ValidationError, match=re.escape("duplicate ssh.targets node_id")):
        SSHConfig(targets=[
            SSHTarget(node_id="a", host="10.77.0.1"),
            SSHTarget(node_id="a", host="10.77.0.2"),
        ])


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"node_id": "../evil"}, "a traversal in the key-file and output name"),
        ({"node_id": "a/b"}, "a separator makes the join escape the store"),
        ({"node_id": ""}, "empty equals mcp_endpoints' exclude_node_id default"),
        ({"config_file": "/etc/passwd.yaml"}, "an absolute path replaces the join"),
        ({"config_file": "../outside.yaml"}, "a traversal out of configs/"),
    ],
)
def test_node_id_and_config_file_must_be_path_segments(kwargs, why) -> None:
    """Both are joined straight into paths, and `Path('/a') / '/etc/x'` is
    `/etc/x` — so these values decide where the key file and the key-bearing
    resolved config land. aureax-network.yaml is writable over the dashboard."""
    base = {"node_id": "a", "host": "10.77.0.1", "config_file": "a.yaml"}
    with pytest.raises(ValidationError, match="must be a single path segment"):
        SSHConfig(targets=[SSHTarget(**{**base, **kwargs})])


def test_node_ids_differing_only_in_case_are_a_duplicate() -> None:
    """The orchestrator's own filesystem decides what these ids mean, and the
    default macOS volume is case-insensitive: two targets here share one
    ~/.aureax/keys/edge-01.key and one TLS leaf, so the second node runs on the
    first one's credentials with nothing anywhere saying so."""
    with pytest.raises(ValidationError, match=re.escape("already declared as 'edge-01'")):
        SSHConfig(targets=[
            SSHTarget(node_id="edge-01", host="10.77.0.1"),
            SSHTarget(node_id="Edge-01", host="10.77.0.2"),
        ])


@pytest.mark.parametrize("node_id", ["ca", "orchestrator", "CA", "Orchestrator"])
def test_reserved_node_ids_are_rejected(node_id) -> None:
    """~/.aureax/tls/<node_id>.pem is a plain join too. "ca" is the fleet's
    issuer; "orchestrator" is the dashboard's own pair — and `aureax apply`
    pushes leaf_paths(node_id) to the node, so a node with that id would be
    handed the orchestrator's private key and could impersonate the dashboard
    to every node's telemetry shipper."""
    with pytest.raises(ValidationError, match="reserved by the TLS store"):
        SSHConfig(targets=[SSHTarget(node_id=node_id, host="10.77.0.1")])


def test_the_schema_and_the_tls_store_reserve_the_same_names() -> None:
    """Two fences, one list. schema cannot import tlsstore (both Dockerfiles
    copy schema.py into the edge image, which has no orchestrator package), so
    the names are spelled twice and pinned equal here."""
    from orchestrator.config import schema, tlsstore

    assert set(schema._RESERVED_NODE_IDS) == {tlsstore._CA_STEM, tlsstore.ORCHESTRATOR_NAME}


def test_a_host_with_a_comma_is_rejected() -> None:
    """Worker hosts are comma-joined into --rpc, where position names the RPC
    device — an extra comma renames every later device and a hand-written
    tensor_split silently starts addressing other machines."""
    with pytest.raises(ValidationError, match="must not contain commas"):
        SSHTarget(node_id="a", host="10.77.0.1,10.77.0.2")
    with pytest.raises(ValidationError, match="must not contain commas"):
        SSHTarget(node_id="a", host="10.77.0.1 ")


@pytest.mark.parametrize("host", [
    "10.77.0.5@evil.example",     # userinfo: the authority is evil.example
    "10.77.0.5/../evil.example",  # path: the authority ends at the slash
    "10.77.0.5\\evil.example",    # a backslash, which browsers read as a slash
    "10.77.0.5?evil.example",     # query
    "10.77.0.5#evil.example",     # fragment
])
def test_a_host_that_breaks_out_of_the_url_authority_is_rejected(host) -> None:
    """Every fleet URL is built by interpolating this value into
    https://<host>:<port>, with that node's bearer key attached — the dashboard's
    chat and access services, and each peer entry in a planner's
    ${mcp_endpoints}. A character that ends the authority early sends the key
    somewhere else."""
    with pytest.raises(ValidationError, match="ends the authority early"):
        SSHTarget(node_id="a", host=host)


@pytest.mark.parametrize("host", ["node_01.local", "10.77.0.5", "fd00::1", "NODE-01"])
def test_an_odd_but_legal_host_is_left_alone(host) -> None:
    """aureax-network.yaml is an operator's file. Underscores are illegal in a
    hostname and perfectly common in one, and an IPv6 literal is unbracketed
    here (templating._url_host brackets it for the URL) — refusing either would
    be this check inventing a fleet policy rather than closing the key leak."""
    assert SSHTarget(node_id="a", host=host).host == host


def test_an_unexported_password_env_is_not_a_key_only_target(monkeypatch) -> None:
    """resolve_password() has one falsy return, so "no password" and
    "password_env names a variable nobody exported" look identical — and the
    caller reports the second as the first, skipping the key bootstrap."""
    monkeypatch.delenv("AUREAX_TEST_PW", raising=False)
    t = SSHTarget(node_id="a", host="10.77.0.1", password_env="AUREAX_TEST_PW")
    assert t.resolve_password() is None
    assert t.password_configured is True

    monkeypatch.setenv("AUREAX_TEST_PW", "s3cret")
    assert t.resolve_password() == "s3cret"

    assert SSHTarget(node_id="a", host="10.77.0.1").password_configured is False


# --- orchestrator host: one rule, two entry points ----------------------------


@pytest.mark.parametrize(
    "value", ["127.0.0.1", "192.168.1.42", "10.77.0.99", "fd00::1", "orchestrator.local", "mac"]
)
def test_orchestrator_host_accepts_addresses_and_hostnames(value) -> None:
    """The schema types the telemetry endpoint as a bare str and an mDNS name is
    a plausible operator input, so only clearly malformed values are refused."""
    assert validate_orchestrator_host(value) == value


@pytest.mark.parametrize(
    "value", ["", "192.168.1..42", "10.77.0.1 ", "http://10.77.0.1", "10.77.0.1:7001", "-lead", "trail-"]
)
def test_orchestrator_host_rejects_values_no_resolver_could_use(value) -> None:
    """This is substituted into every node's telemetry and logs endpoint, so an
    unusable value regenerates and pushes the WHOLE fleet with an endpoint that
    never resolves — visible only as status printing "(no metrics received
    yet)".
    """
    with pytest.raises(ValueError, match="neither an IP address nor a hostname"):
        validate_orchestrator_host(value)


def test_the_cli_callback_applies_the_same_rule_as_the_dashboard() -> None:
    """The CLI callback and POST /api/config/regenerate must agree.

    A stricter copy on either side — a bare ``ipaddress.ip_address``, say —
    means a fleet whose orchestrator is named by an mDNS hostname regenerates
    from one entry point and is rejected by the other. This is the CLI half;
    ``tests/test_dashboard_app.py`` asserts the same values over the route.
    """
    import click

    from orchestrator.cli import _validate_orchestrator_ip

    for value in ("10.77.0.99", "fd00::1", "orchestrator.local"):
        assert _validate_orchestrator_ip(None, None, value) == value

    with pytest.raises(click.BadParameter, match="neither an IP address nor a hostname"):
        _validate_orchestrator_ip(None, None, "http://10.77.0.1")
