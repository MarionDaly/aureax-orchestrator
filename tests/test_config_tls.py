"""What generation stamps onto a config so the node can speak TLS at all.

TLS is mandatory fleet-wide: there is no plaintext listener anywhere, so a node
that reached the fleet without certificate paths does not degrade — it cannot
build a llama-server command line, and it crash-loops. That makes the `tls:`
block the generator's responsibility rather than the template's: a placeholder
can be typo'd (the leftover check catches that) but nothing catches a template
that simply LACKS one, and the frozen 122B rollback pair must keep validating
without gaining new keys.

The other half is where the credentials come from. `generate_all_edge_configs`
is the one place that mints them — the CA, the orchestrator leaf and one leaf
per enabled target. `generate_edge_config` mints none, because the dashboard's
Access tab calls it once per node on a 20 s browser poll: a page left open must
not create the fleet's root of trust.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from orchestrator.config import tlsstore
from orchestrator.config.loader import validate_edge_data
from orchestrator.config.templating import (
    PLACEHOLDER_DUMMIES,
    build_variables,
    generate_all_edge_configs,
    generate_edge_config,
    mcp_endpoints,
)
from tests.conftest import make_network_config, make_targets, write_network_yaml

yaml = YAML()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_CONFIGS = PROJECT_ROOT / "configs"

# The three node-side paths Deployer.deploy_configs installs. Spelled out here
# rather than imported so this test fails when the constants MOVE — the client,
# the deployer and the DaemonSet's read-only /etc/aureax mount all agree on
# these strings, and a silent change to one of them is a fleet that starts with
# certificates nothing reads.
EXPECTED_TLS = {
    "ca_file": "/etc/aureax/tls/ca.pem",
    "cert_file": "/etc/aureax/tls/node.pem",
    "key_file": "/etc/aureax/tls/node.key",
}

HEAD_YAML = """
node:
  id: "${node_id}"
  role: "edge_executor"
  worker_wait_timeout_seconds: ${worker_wait_timeout_seconds}
  control_plane: {endpoint: "x"}
model:
  name: M
  filename: m.gguf
  server: {gpu_layers: 99, rpc_servers: "${rpc_servers}", tensor_split: []}
telemetry:
  remote: {endpoint: "${orchestrator_url}/api/v1/telemetry"}
mcp: {server: {enabled: true}}
"""
WORKER_YAML = """
node: {id: "${node_id}", role: "shard_worker", control_plane: {endpoint: "x"}}
model: {name: M, filename: m.gguf, rpc_server: {port: 50052}}
telemetry:
  remote: {endpoint: "${orchestrator_url}/api/v1/telemetry"}
mcp: {server: {enabled: false}}
"""


def _fleet(tmp_path: Path, dashboard_port: int | None = None) -> tuple[Path, Path]:
    """h + w1 + w2 as one group, with the templates on disk. Returns (net, out)."""
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "h.yaml").write_text(HEAD_YAML)
    for w in ("w1", "w2"):
        (configs / f"{w}.yaml").write_text(WORKER_YAML)

    net = configs / "aureax-network.yaml"
    lines = [
        "network: {name: t, version: '0.1.0'}",
        "model_registry: {models: []}",
        "ssh:",
        "  targets:",
        "    - {node_id: h, host: 10.77.0.1, config_file: h.yaml}",
        "    - {node_id: w1, host: 10.77.0.2, config_file: w1.yaml}",
        "    - {node_id: w2, host: 10.77.0.3, config_file: w2.yaml}",
        "node_groups:",
        "  - {name: g, head: h, workers: [w1, w2]}",
    ]
    if dashboard_port is not None:
        lines += ["server:", "  dashboard:", f"    port: {dashboard_port}"]
    net.write_text("\n".join(lines) + "\n")
    return net, tmp_path / "out"


def _generated(out: Path, node_id: str) -> dict:
    with open(out / f"aureax-edge-{node_id}.yaml") as f:
        return yaml.load(f)


# --- the injected block -------------------------------------------------------


def test_every_generated_config_carries_the_tls_block(tmp_aureax_home, tmp_path) -> None:
    """Including the shard workers.

    A worker runs rpc-server, which speaks raw TCP and takes no certificate —
    but it still ships telemetry and logs to the dashboard over TLS, and
    _require_tls_files refuses to start any node without all three files. A
    block emitted only for llama-serving nodes would leave every worker exiting
    78 at boot.
    """
    net, out = _fleet(tmp_path)

    generate_all_edge_configs(net, out, "10.77.0.99")

    for node_id in ("h", "w1", "w2"):
        assert _generated(out, node_id)["tls"] == EXPECTED_TLS, node_id


def test_the_block_is_stamped_not_merged(tmp_aureax_home, tmp_path) -> None:
    """Byte-exact, three keys, nothing else.

    The node reads these paths verbatim; an extra or renamed key is a path the
    deployer never populates, which is one crash-looping pod discovered at
    deploy time rather than here.
    """
    net, out = _fleet(tmp_path)

    generate_all_edge_configs(net, out, "10.77.0.99")

    block = _generated(out, "h")["tls"]
    assert dict(block) == EXPECTED_TLS
    assert sorted(block) == ["ca_file", "cert_file", "key_file"]


def test_a_template_carrying_its_own_tls_block_is_overridden_and_warned(
    tmp_aureax_home, tmp_path, caplog
) -> None:
    """Assignment, not setdefault.

    A template pointing at paths the deployer never installs would ship a node
    that cannot open its own key, and that template is now lying about what it
    deploys — so the override is loud rather than silent.
    """
    net, out = _fleet(tmp_path)
    (tmp_path / "configs" / "h.yaml").write_text(
        HEAD_YAML + 'tls: {ca_file: "/tmp/ca.pem", cert_file: "/tmp/c.pem", key_file: "/tmp/k"}\n'
    )

    with caplog.at_level(logging.WARNING, logger="aureax.config.templating"):
        generate_all_edge_configs(net, out, "10.77.0.99")

    assert _generated(out, "h")["tls"] == EXPECTED_TLS
    assert any("h.yaml" in r.getMessage() and "tls" in r.getMessage() for r in caplog.records)


def test_the_stamped_config_still_passes_edge_validation(tmp_aureax_home, tmp_path) -> None:
    """The RESOLVED config is what the node runs and nothing between here and
    the node validates it, so the block the generator adds has to satisfy the
    same schema as everything a template wrote."""
    net, out = _fleet(tmp_path)

    generate_all_edge_configs(net, out, "10.77.0.99")

    cfg = validate_edge_data(_generated(out, "h"))
    assert cfg.tls.ca_file == EXPECTED_TLS["ca_file"]
    assert cfg.tls.cert_file == EXPECTED_TLS["cert_file"]
    assert cfg.tls.key_file == EXPECTED_TLS["key_file"]


def test_the_block_is_inside_the_config_hash(tmp_aureax_home, tmp_path) -> None:
    """The hash drives the Access tab's staged/applied badge, and both sides of
    that comparison go through generate_edge_config — so the block has to be
    present BEFORE the hash is taken or the badge would read "applied" for a
    node whose certificates have never been pushed.
    """
    import hashlib
    import json

    from orchestrator.config.loader import load_network_config

    net, out = _fleet(tmp_path)
    generate_all_edge_configs(net, out, "10.77.0.99")
    with_block = _generated(out, "h")["deployment"]["config_hash"]

    network_config = load_network_config(net)
    target = next(t for t in network_config.ssh.targets if t.node_id == "h")
    resolved = generate_edge_config(net.parent, target, network_config, "10.77.0.99")
    del resolved["deployment"]
    del resolved["tls"]

    without = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    assert with_block != without


# --- ${orchestrator_url} ------------------------------------------------------


def test_orchestrator_url_carries_the_scheme_and_the_configured_port(tmp_path) -> None:
    """server.dashboard.port is configurable but was hardcoded as 7001 in every
    template — so a fleet that moved the dashboard shipped telemetry to a closed
    port, with the sender logging a failed POST and carrying on."""
    cfg = make_network_config(make_targets("solo"), models=[])
    assert (
        build_variables(cfg.ssh.targets[0], cfg, "10.77.0.99")["orchestrator_url"]
        == "https://10.77.0.99:7001"
    )

    moved = make_network_config(make_targets("solo"), models=[])
    moved.server.dashboard.port = 8443
    assert (
        build_variables(moved.ssh.targets[0], moved, "10.77.0.99")["orchestrator_url"]
        == "https://10.77.0.99:8443"
    )


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("10.77.0.5", "https://10.77.0.5:7001"),
        ("orchestrator.local", "https://orchestrator.local:7001"),
        ("fd00::1", "https://[fd00::1]:7001"),
    ],
)
def test_an_ipv6_orchestrator_is_bracketed_inside_the_url(given, expected) -> None:
    """Unbracketed, the address's own colons read as the port separator, so no
    HTTP client parses the endpoint and the whole fleet reports "(no metrics
    received yet)" with nothing naming the cause."""
    cfg = make_network_config(make_targets("solo"), models=[])
    assert build_variables(cfg.ssh.targets[0], cfg, given)["orchestrator_url"] == expected


def test_the_endpoint_a_node_actually_ships_to_is_https(tmp_aureax_home, tmp_path) -> None:
    """End to end: what ${orchestrator_url} puts in the resolved config."""
    net, out = _fleet(tmp_path, dashboard_port=8443)

    generate_all_edge_configs(net, out, "10.77.0.99")

    endpoint = _generated(out, "w1")["telemetry"]["remote"]["endpoint"]
    assert endpoint == "https://10.77.0.99:8443/api/v1/telemetry"


# --- the dummies stay in lockstep with the builder ----------------------------


def test_the_dummy_urls_are_https_like_the_real_ones(tmp_aureax_home, tmp_path) -> None:
    """PLACEHOLDER_DUMMIES is what the dashboard's editor validates a template
    against. A plaintext dummy would let the editor accept a template the real
    builder then resolves to https — the same drift the key-set lockstep test in
    tests/test_node_groups.py exists to prevent, one level down in the values.
    """
    assert str(PLACEHOLDER_DUMMIES["orchestrator_url"]).startswith("https://")
    for entry in PLACEHOLDER_DUMMIES["mcp_endpoints"]:
        assert entry["url"].startswith("https://")

    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    cfg = make_network_config(make_targets("h"), models=[])
    assert mcp_endpoints(cfg, tmp_path)[0]["url"].startswith("https://")


def test_the_mcp_endpoints_dummy_has_the_same_shape_as_a_real_entry(
    tmp_aureax_home, tmp_path
) -> None:
    """The dummy has to carry api_key too: a peer with MCP auth on contributes
    that field, and a dummy without it means the editor validates a planner
    template against a shape the generator never produces."""
    from orchestrator.config import keystore

    keyed = HEAD_YAML.replace(
        "mcp: {server: {enabled: true}}",
        'mcp: {server: {enabled: true, api_key: "${mcp_api_key}"}}',
    )
    (tmp_path / "h.yaml").write_text(keyed)
    keystore.ensure_key("h", kind=keystore.KIND_MCP)
    cfg = make_network_config(make_targets("h"), models=[])

    real = mcp_endpoints(cfg, tmp_path)[0]
    dummy = PLACEHOLDER_DUMMIES["mcp_endpoints"][0]
    assert set(real) == set(dummy)


# --- who is allowed to mint ---------------------------------------------------


def test_generate_all_mints_the_ca_the_orchestrator_leaf_and_every_node_leaf(
    tmp_aureax_home, tmp_path
) -> None:
    """The deployer refuses to push a node whose leaf is missing, so a generate
    that produced configs but no certificates would be a fleet that cannot
    start — and the operator would be told about it one node at a time."""
    net, out = _fleet(tmp_path)

    generate_all_edge_configs(net, out, "10.77.0.99")

    assert tlsstore.ca_cert_path().exists()
    assert tlsstore.ca_key_path().exists()
    for name in ("orchestrator", "h", "w1", "w2"):
        cert, key = tlsstore.leaf_paths(name, allow_reserved=True)
        assert cert.exists() and key.exists(), name


def test_generate_edge_config_alone_mints_nothing_under_the_tls_store(
    tmp_aureax_home, tmp_path
) -> None:
    """The Access tab calls this once per node every 20 s through
    desired_config_hash. A browser left open on that tab must not create the
    fleet's root of trust — and a CA minted from a poll would be one no node has
    ever been given, so every request would still fail, just with a
    verification error pointing at the node instead of at the empty store.
    """
    from orchestrator.config.loader import load_network_config

    net, _ = _fleet(tmp_path)
    network_config = load_network_config(net)
    target = next(t for t in network_config.ssh.targets if t.node_id == "h")

    resolved = generate_edge_config(net.parent, target, network_config, "10.77.0.99")

    assert resolved["tls"] == EXPECTED_TLS, "the block is stamped without minting"
    assert not tlsstore.store_dir().exists() or list(tlsstore.store_dir().iterdir()) == []


def test_a_second_generate_leaves_the_certificates_untouched(
    tmp_aureax_home, tmp_path
) -> None:
    """Every deploy and every apply regenerates the whole fleet. Reissuing there
    would hand each node a new identity while the running pods still present the
    old one — and nothing would report the mismatch until a client failed to
    verify."""
    net, out = _fleet(tmp_path)
    generate_all_edge_configs(net, out, "10.77.0.99")
    cert, key = tlsstore.leaf_paths("h")
    before = (cert.read_bytes(), key.read_bytes(), tlsstore.ca_cert_path().read_bytes())

    generate_all_edge_configs(net, out, "10.77.0.99")

    assert (cert.read_bytes(), key.read_bytes(), tlsstore.ca_cert_path().read_bytes()) == before


# --- the CA's name constraints -------------------------------------------------
# The operator trusts ca.pem as a root in the macOS keychain, so what this CA
# may certify is what a stolen ca.key can forge.


def test_generate_mints_a_ca_constrained_to_this_fleet(tmp_aureax_home, tmp_path) -> None:
    net, out = _fleet(tmp_path)

    generate_all_edge_configs(net, out, "10.77.0.99")

    permitted_ips, permitted_dns = tlsstore.ca_name_constraints()
    assert "10.0.0.0/8" in permitted_ips
    assert permitted_dns == ["localhost"]
    assert tlsstore.san_constraint_violations(
        ["DNS:login.yourbank.com"], tlsstore.load_ca_cert()
    ) != []


def test_generate_refuses_a_fleet_the_existing_ca_cannot_cover(
    tmp_aureax_home, tmp_path
) -> None:
    """The CA is minted once and never rotated automatically, so a fleet that
    later gains an address outside it is a real state. Refusing names the
    address and the one command that replaces a root; minting would hand out
    leaves that verify for none of their names, half a fleet at a time."""
    net, out = _fleet(tmp_path)
    generate_all_edge_configs(net, out, "10.77.0.99")
    net.write_text(net.read_text().replace("host: 10.77.0.2", "host: 8.8.8.8"))

    with pytest.raises(ValueError, match=r"8\.8\.8\.8"):
        generate_all_edge_configs(net, out, "10.77.0.99")


def test_the_refusal_names_the_command_that_replaces_the_ca(tmp_aureax_home, tmp_path) -> None:
    net, out = _fleet(tmp_path)
    generate_all_edge_configs(net, out, "10.77.0.99")
    net.write_text(net.read_text().replace("host: 10.77.0.2", "host: 8.8.8.8"))

    with pytest.raises(ValueError, match=re.escape("tls regenerate --ca --orchestrator-ip 10.77.0.99")):
        generate_all_edge_configs(net, out, "10.77.0.99")


def test_nothing_is_minted_before_the_constraint_check(tmp_aureax_home, tmp_path) -> None:
    """The check runs over every planned SAN set before the first mint: a
    verifier rejects a certificate outright when one SAN is outside the
    permitted subtrees, so a partial fleet of such leaves is worse than a
    generate that refuses."""
    net, out = _fleet(tmp_path)
    generate_all_edge_configs(net, out, "10.77.0.99")
    net.write_text(
        net.read_text()
        .replace("host: 10.77.0.2", "host: 8.8.8.8")
        .replace("host: 10.77.0.3", "host: 10.77.0.33")
    )
    before = {n: tlsstore.leaf_paths(n)[0].read_bytes() for n in ("h", "w1", "w2")}

    with pytest.raises(ValueError):
        generate_all_edge_configs(net, out, "10.77.0.99")

    # w2 moved legitimately and would have been reissued; it was not, because
    # nothing was minted at all.
    assert {n: tlsstore.leaf_paths(n)[0].read_bytes() for n in ("h", "w1", "w2")} == before


def test_a_public_fleet_address_is_covered_by_the_generated_ca(
    tmp_aureax_home, tmp_path
) -> None:
    """The shipped example fleet lives on 192.0.2.x, which is not private. A CA
    that permitted only the RFC 1918 ranges would refuse to generate it."""
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "h.yaml").write_text(HEAD_YAML)
    (configs / "w1.yaml").write_text(WORKER_YAML)
    net = configs / "aureax-network.yaml"
    net.write_text("\n".join([
        "network: {name: t, version: '0.1.0'}",
        "model_registry: {models: []}",
        "ssh:",
        "  targets:",
        "    - {node_id: h, host: 192.0.2.10, config_file: h.yaml}",
        "    - {node_id: w1, host: 192.0.2.11, config_file: w1.yaml}",
        "node_groups:",
        "  - {name: g, head: h, workers: [w1]}",
    ]) + "\n")

    generate_all_edge_configs(net, tmp_path / "out", "192.0.2.1")

    permitted_ips, _ = tlsstore.ca_name_constraints()
    assert {"192.0.2.10/32", "192.0.2.11/32", "192.0.2.1/32"} <= set(permitted_ips)


# --- the orchestrator leaf's two producers --------------------------------------


def test_generate_does_not_strip_a_name_the_dashboard_added(tmp_aureax_home, tmp_path) -> None:
    """`aureax dashboard` mints for the LAN address it detects, `aureax
    generate` for the typed --orchestrator-ip. Whichever ran last used to
    delete the other's SAN — and every node's telemetry endpoint carries the
    generate-time value, so a dashboard start could end fleet ingest with the
    dashboard's own URL still working."""
    net, out = _fleet(tmp_path)
    generate_all_edge_configs(net, out, "10.77.0.99")
    tlsstore.ensure_orchestrator_leaf("192.168.1.20")  # what `aureax dashboard` does

    generate_all_edge_configs(net, out, "10.77.0.99")

    cert_path, _ = tlsstore.leaf_paths("orchestrator", allow_reserved=True)
    sans = set(tlsstore._sans_of(cert_path))
    assert {"IP:10.77.0.99", "IP:192.168.1.20"} <= sans


# --- the frozen rollback pair --------------------------------------------------


def test_the_122b_rollback_pair_still_generates(tmp_aureax_home, tmp_path) -> None:
    """configs/aureax-shard-head.yaml and aureax-shard-worker.yaml are the
    rollback target and are kept unedited apart from the one transport line.

    They carry no `tls:` block of their own — which is exactly why the generator
    stamps one rather than asking templates for it. A rollback that produced
    configs missing the paths would be a rollback that cannot start, discovered
    at the worst possible moment.

    Seven workers because the 122B ran as ONE group of 1 head + 7 workers, and
    the head's tensor_split is a literal list of seven that validate_groups
    length-checks per group.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(REPO_CONFIGS / "aureax-shard-head.yaml", configs / "head.yaml")
    shutil.copy(REPO_CONFIGS / "aureax-shard-worker.yaml", configs / "worker.yaml")

    workers = [f"w{i}" for i in range(1, 8)]
    targets = [{"node_id": "head", "host": "10.77.0.1", "config_file": "head.yaml"}]
    targets += [
        {"node_id": w, "host": f"10.77.0.{10 + i}", "config_file": "worker.yaml"}
        for i, w in enumerate(workers)
    ]
    net = write_network_yaml(
        configs, targets, groups=[{"name": "g122", "head": "head", "workers": workers}]
    )

    paths = generate_all_edge_configs(net, tmp_path / "out", "10.77.0.99")

    assert len(paths) == 8
    for node_id in ("head", *workers):
        generated = _generated(tmp_path / "out", node_id)
        assert generated["tls"] == EXPECTED_TLS, node_id
        assert generated["telemetry"]["remote"]["endpoint"] == (
            "https://10.77.0.99:7001/api/v1/telemetry"
        ), node_id


def test_the_rollback_pair_grew_no_api_key_lines(tmp_aureax_home, tmp_path) -> None:
    """Both stay auth-off by design: every auth field defaults to the pre-auth
    behaviour precisely so the rollback keeps validating and serving. A
    ${mcp_api_key} line on the worker would also be refused by validate_groups.
    """
    head = (REPO_CONFIGS / "aureax-shard-head.yaml").read_text()
    worker = (REPO_CONFIGS / "aureax-shard-worker.yaml").read_text()

    for text, name in ((head, "aureax-shard-head.yaml"), (worker, "aureax-shard-worker.yaml")):
        assert "${api_key}" not in text, name
        assert "${mcp_api_key}" not in text, name
        assert "tls:" not in text, f"{name} must not carry its own TLS paths"
