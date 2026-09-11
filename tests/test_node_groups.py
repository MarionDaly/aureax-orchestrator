"""node_groups schema, validators, and placeholder resolution.

A group is a set of names cross-referenced across two YAML blocks plus N
per-node files, so most mistakes here surface far from their cause — as a head
that waits forever for a worker that was never labelled, or a lopsided tensor
split that looks perfectly healthy. These tests pin the checks that turn those
into errors at generate time.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from orchestrator.config.cache import load_edge_config_cached
from orchestrator.config.loader import load_network_config
from orchestrator.config.schema import DEFAULT_WORKER_WAIT_SECONDS, NetworkConfig
from orchestrator.config.templating import (
    PLACEHOLDER_DUMMIES,
    build_variables,
    generate_all_edge_configs,
    mcp_endpoints,
    resolve_host_literals,
    validate_groups,
    validate_planners,
)
from tests.conftest import make_network_config, make_targets

MODELS = [
    {"name": "M", "filename": "m.gguf", "size_gb": 1.0,
     "quantization": "Q4_K_M", "target_roles": ["edge_executor"]},
]

def _net(targets: list[dict], groups: list[dict]) -> NetworkConfig:
    """conftest's builder plus the non-empty model_registry this module needs —
    the only thing it varies. Targets come straight from ``make_targets``."""
    return make_network_config(targets, groups, models=MODELS)


GROUP = {"name": "g", "head": "h", "workers": ["w1", "w2"]}


def test_valid_group_parses() -> None:
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    group = cfg.node_groups[0]
    assert group.members == ["h", "w1", "w2"]
    assert cfg.group_for_node("w2") is group
    assert cfg.group_for_node("nobody") is None


def test_rpc_endpoints_follow_worker_order() -> None:
    """--rpc order defines RPC0/RPC1, which tensor_split indexes positionally."""
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    hosts = cfg.host_by_node_id()
    assert cfg.node_groups[0].rpc_endpoints(hosts) == ["10.77.0.2:50052", "10.77.0.3:50052"]


def test_include_head_as_worker_is_rejected() -> None:
    """The option cannot work, and its failure looks like a network problem.

    Setting it puts the head's own host:50052 in the head's --rpc list, but
    client/entrypoint.py starts an rpc-server only for role "shard_worker" and
    the head is "edge_executor". The head would wait out its peer timeout on a
    port nobody listens on and crash-loop, so generation refuses instead.
    """
    with pytest.raises(ValidationError, match="include_head_as_worker is not implemented"):
        _net(make_targets("h", "w1", "w2"), [{**GROUP, "include_head_as_worker": True}])


def test_no_groups_is_valid_and_inert() -> None:
    """Every pre-existing config must keep working untouched."""
    cfg = make_network_config(make_targets("a"), models=MODELS)
    assert cfg.node_groups == []
    assert cfg.group_for_node("a") is None


@pytest.mark.parametrize(
    ("groups", "targets", "message"),
    [
        ([GROUP, {"name": "g", "head": "x", "workers": ["y"]}],
         make_targets("h", "w1", "w2", "x", "y"), "duplicate node_group name"),
        ([{"name": "g", "head": "h", "workers": []}],
         make_targets("h"), "has no workers"),
        ([{"name": "g", "head": "h", "workers": ["h", "w1"]}],
         make_targets("h", "w1"), "is also listed in workers"),
        ([{"name": "g", "head": "h", "workers": ["w1", "w1"]}],
         make_targets("h", "w1"), "listed more than once"),
        ([GROUP], make_targets("h", "w1"), "unknown node_id"),
        ([GROUP], make_targets("h", "w1", "w2", disabled=("w2",)), "references disabled node"),
        ([GROUP, {"name": "g2", "head": "w2", "workers": ["w3"]}],
         make_targets("h", "w1", "w2", "w3"), "at most one group"),
    ],
)
def test_invalid_groups_are_rejected(groups, targets, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _net(targets, groups)


def test_placeholder_dummies_cover_every_built_variable() -> None:
    """The dummies and the real builder must expose the same names.

    A name known to one and not the other means the dashboard rejects configs
    that deploy fine, or accepts configs that blow up at deploy time.
    """
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    built = build_variables(cfg.ssh.targets[0], cfg, "127.0.0.1")
    assert set(built) == set(PLACEHOLDER_DUMMIES)


def test_only_the_head_gets_rpc_servers() -> None:
    cfg = _net(make_targets("h", "w1", "w2"), [{**GROUP, "worker_wait_timeout_seconds": 900}])
    by_id = {t.node_id: t for t in cfg.ssh.targets}

    head = build_variables(by_id["h"], cfg, "127.0.0.1")
    assert head["rpc_servers"] == ["10.77.0.2:50052", "10.77.0.3:50052"]
    assert head["group_id"] == "g"

    # Workers listen; they never dial peers.
    worker = build_variables(by_id["w1"], cfg, "127.0.0.1")
    assert worker["rpc_servers"] == []
    assert worker["group_id"] == "g"

    # The timeout is configured on the GROUP and read by the client from the
    # node's own `node:` block — without this hop the setting is a silent no-op
    # and the head sits on the default while every config still looks right.
    assert head["worker_wait_timeout_seconds"] == 900
    assert worker["worker_wait_timeout_seconds"] == 900


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("10.77.0.5", "10.77.0.5"),
        ("orchestrator.local", "orchestrator.local"),
        ("fd00::1", "[fd00::1]"),
        ("::1", "[::1]"),
    ],
)
def test_an_ipv6_orchestrator_is_bracketed_for_the_url(given, expected) -> None:
    """Every template puts ${orchestrator_ip} straight into a URL authority.

    Unbracketed, an IPv6 literal's own colons read as the port separator, so no
    HTTP client parses the telemetry endpoint and the whole fleet reports "(no
    metrics received yet)" — the exact failure the host validator exists to
    prevent, arriving through a value it accepts.
    """
    cfg = _net(make_targets("solo"), [])
    assert build_variables(cfg.ssh.targets[0], cfg, given)["orchestrator_ip"] == expected


def test_standalone_node_gets_no_group_variables() -> None:
    cfg = _net(make_targets("solo"), [])
    built = build_variables(cfg.ssh.targets[0], cfg, "127.0.0.1")
    assert built["rpc_servers"] == []
    assert built["group_id"] == ""
    assert built["worker_wait_timeout_seconds"] == DEFAULT_WORKER_WAIT_SECONDS


# --- cross-file validation (roles + mcp flags live in the per-node YAMLs) ----

# Carries `worker_wait_timeout_seconds` as the placeholder because a real head
# template must — validate_groups checks for it, and every test below that
# asserts an empty problem list depends on it being there.
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
mcp: {server: {enabled: true}}
"""
WORKER_YAML = """
node: {id: "${node_id}", role: "shard_worker", control_plane: {endpoint: "x"}}
model: {name: M, filename: m.gguf, rpc_server: {port: 50052}}
mcp: {server: {enabled: false}}
"""


PLANNER_YAML = """
node: {id: "${node_id}", role: "cognitive_planner", control_plane: {endpoint: "x"}}
model: {name: M, filename: m.gguf}
mcp:
  server: {enabled: false}
  client: {enabled: true, remote_servers: "${mcp_endpoints}"}
"""


def _write_group_configs(tmp_path, head=HEAD_YAML, worker=WORKER_YAML, workers=("w1", "w2")):
    (tmp_path / "h.yaml").write_text(head)
    for w in workers:
        (tmp_path / f"{w}.yaml").write_text(worker)


def _bump_mtime(path: Path, seconds: int = 1) -> None:
    """Push a file's mtime forward so the mtime-keyed config cache sees a change.

    load_edge_config_cached memoizes on st_mtime_ns, so two writes to one path
    inside a single test can share a timestamp on any filesystem coarser than
    nanoseconds and the second parse is served from the first.
    """
    stamp = path.stat().st_mtime_ns + seconds * 10**9
    os.utime(path, ns=(stamp, stamp))


def _write_network(
    configs: Path,
    workers: tuple[str, ...] = ("w1", "w2"),
    extra_nodes: tuple[str, ...] = (),
    filename: str = "aureax-network.yaml",
) -> Path:
    """aureax-network.yaml for the h/w1/w2 group, beside its per-node configs.

    ``workers`` is both the group's list AND the ssh.targets order, because the
    reorder check compares exactly that list against the recorded baseline.
    Hosts are pinned to the worker NAME (w1 -> 10.77.0.31) rather than to the
    position, so swapping two entries moves the ids without also moving the
    addresses — otherwise a reorder would look like a host change too and the
    tests could not tell the two verdicts apart.
    """
    nodes = ("h", *workers, *extra_nodes)
    net = configs / filename
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n"
        + "".join(
            f"    - {{node_id: {n}, host: {_host_for(n)}, config_file: {n}.yaml}}\n"
            for n in nodes
        )
        + f"node_groups:\n  - {{name: g, head: h, workers: [{', '.join(workers)}]}}\n"
    )
    return net


def _host_for(node_id: str) -> str:
    """A stable address per NAME — ``w1`` is always 10.77.0.31, wherever it sits."""
    if node_id == "h":
        return "10.77.0.1"
    digits = "".join(c for c in node_id if c.isdigit()) or "0"
    return f"10.77.0.{10 + int(digits)}"


def test_validate_groups_accepts_correct_roles(tmp_path) -> None:
    _write_group_configs(tmp_path)
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert validate_groups(cfg, tmp_path) == []


def test_validate_groups_rejects_worker_with_edge_role(tmp_path) -> None:
    """A mislabelled worker gets a llama-server pod with no model file."""
    _write_group_configs(tmp_path, worker=WORKER_YAML.replace("shard_worker", "edge_executor"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("expected 'shard_worker'" in p for p in problems)


def test_validate_groups_rejects_worker_serving_mcp(tmp_path) -> None:
    _write_group_configs(tmp_path, worker=WORKER_YAML.replace("enabled: false", "enabled: true"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("mcp.server.enabled: false" in p for p in problems)


def test_validate_groups_rejects_head_with_worker_role(tmp_path) -> None:
    """A head that isn't edge_executor is never labelled, so never gets a pod."""
    _write_group_configs(tmp_path, head=HEAD_YAML.replace("edge_executor", "shard_worker"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("expected 'edge_executor'" in p for p in problems)


def test_validate_groups_rejects_mismatched_tensor_split(tmp_path) -> None:
    """A wrong split fails silently as a lopsided one, not as an error."""
    _write_group_configs(tmp_path, head=HEAD_YAML.replace("tensor_split: []", "tensor_split: [1, 1]"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert validate_groups(cfg, tmp_path) == []  # 2 workers, 2 entries — correct

    # Rewriting the same path a second time within one test: the config cache
    # is keyed on st_mtime_ns, so on a filesystem that stamps at 1s granularity
    # the second parse would be served from the first write and this assertion
    # would fail against a correct validator. Age the file past that.
    _write_group_configs(tmp_path, head=HEAD_YAML.replace("tensor_split: []", "tensor_split: [1, 1, 1]"))
    _bump_mtime(tmp_path / "h.yaml")
    problems = validate_groups(cfg, tmp_path)
    assert any("tensor_split has 3 entries but 2 device(s)" in p for p in problems)


def test_validate_groups_rejects_head_without_gpu_layers(tmp_path) -> None:
    """-ngl 0 keeps every layer on the head's CPU while the workers idle.

    The group loads, /health answers and every probe is green, so nothing about
    this surfaces as an error — hence the generate-time check.
    """
    _write_group_configs(tmp_path, head=HEAD_YAML.replace("gpu_layers: 99", "gpu_layers: 0"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert any("gpu_layers" in p for p in validate_groups(cfg, tmp_path))


def test_validate_groups_rejects_head_missing_rpc_servers(tmp_path) -> None:
    """Without the key the head starts with no --rpc flag at all."""
    _write_group_configs(
        tmp_path, head=HEAD_YAML.replace('rpc_servers: "${rpc_servers}", ', "")
    )
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert any("rpc_servers" in p for p in validate_groups(cfg, tmp_path))


@pytest.mark.parametrize(
    ("literal", "why"),
    [
        ("[]", "empty list drops --rpc entirely: the head loads the model alone"),
        ('["10.9.9.9:50052"]', "a hardcoded list addresses machines the group does not name"),
    ],
)
def test_validate_groups_rejects_a_head_that_hardcodes_its_rpc_servers(
    tmp_path, literal, why
) -> None:
    """The group's `workers:` list reaches the head only through the placeholder.

    Presence of the key is not enough: client/config/parser.py emits --rpc only
    for a non-empty list, so neither of these is an error anywhere downstream —
    the workers simply idle while looking perfectly healthy.
    """
    _write_group_configs(
        tmp_path,
        head=HEAD_YAML.replace('rpc_servers: "${rpc_servers}"', f"rpc_servers: {literal}"),
    )
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("hardcodes model.server.rpc_servers" in p for p in problems), why


def test_validate_groups_reports_a_non_mapping_config_instead_of_raising(tmp_path) -> None:
    """A per-node config whose top level is a list parses fine and reaches here.

    validate_groups is called from generation with no try, so an AttributeError
    escaping it aborts the whole fleet's generate — and reaches the dashboard's
    regenerate route as a 500 with a stack trace instead of a message.
    """
    _write_group_configs(tmp_path)
    (tmp_path / "h.yaml").write_text("- a\n- b\n")
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("is not a mapping" in p for p in problems), problems


def test_validate_groups_reports_non_numeric_numbers_instead_of_raising(tmp_path) -> None:
    """Same rule for the two int() coercions: report, never raise."""
    _write_group_configs(
        tmp_path,
        head=HEAD_YAML.replace("gpu_layers: 99", 'gpu_layers: "all"'),
        worker=WORKER_YAML.replace("port: 50052", 'port: "default"'),
    )
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("gpu_layers: 'all'" in p and "expected a number" in p for p in problems), problems
    assert any(
        "model.rpc_server.port: 'default'" in p and "expected a number" in p for p in problems
    ), problems


def test_validate_groups_rejects_a_head_that_hardcodes_its_worker_wait(tmp_path) -> None:
    """The NodeGroup owns this timeout; a literal in the template silences it.

    Nothing errors and nothing looks different — the head simply waits its own
    number while tests/test_probe_budget.py and daemonset-edge.yaml's
    startupProbe budget are both sized against the group's.
    """
    _write_group_configs(
        tmp_path,
        head=HEAD_YAML.replace(
            "worker_wait_timeout_seconds: ${worker_wait_timeout_seconds}",
            "worker_wait_timeout_seconds: 900",
        ),
    )
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("hardcodes node.worker_wait_timeout_seconds: 900" in p for p in problems), problems


def test_validate_groups_rejects_a_head_that_drops_its_worker_wait(tmp_path) -> None:
    """Omitting it is the same no-op, just via the client-side default."""
    _write_group_configs(
        tmp_path,
        head=HEAD_YAML.replace(
            "  worker_wait_timeout_seconds: ${worker_wait_timeout_seconds}\n", ""
        ),
    )
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("has no node.worker_wait_timeout_seconds" in p for p in problems), problems


def test_validate_groups_rejects_worker_port_mismatch(tmp_path) -> None:
    """The head dials group.rpc_port; a worker binding another port just never answers."""
    _write_group_configs(tmp_path, worker=WORKER_YAML.replace("port: 50052", "port: 50053"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = validate_groups(cfg, tmp_path)
    assert any("binds model.rpc_server.port 50053" in p for p in problems)


def test_unresolved_placeholder_fails_generation(tmp_aureax_home, tmp_path) -> None:
    """A typo'd ${name} must not ship to the node as a literal string.

    Nothing downstream rejects one — a mistyped ${rpc_servers} becomes a list of
    one-character RPC peers and costs the head its whole peer timeout before it
    exits. The typo here is in node.id, a key validate_groups does not inspect,
    so what catches it is generation's own leftover check.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs, head=HEAD_YAML.replace("${node_id}", "${node_idz}"))
    net = _write_network(configs)

    with pytest.raises(ValueError, match="unresolved placeholder"):
        generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")


def test_generate_fails_loudly_on_invalid_group(tmp_aureax_home, tmp_path) -> None:
    """Generation must refuse rather than emit a subtly wrong fleet."""
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs, worker=WORKER_YAML.replace("shard_worker", "edge_executor"))
    net = _write_network(configs)
    load_network_config(net)  # schema-level checks pass; the role check must not

    with pytest.raises(ValueError, match="node_groups validation failed"):
        generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")


def test_structurally_invalid_resolved_config_fails_generation(tmp_aureax_home, tmp_path) -> None:
    """The RESOLVED config is what the node runs, and nothing downstream checks it.

    It is dumped to disk, SSH-pushed to /etc/aureax/aureax-edge.yaml and read by
    the client with no Pydantic at all — so a wrong type used to generate
    cleanly and surface minutes later as a crash-looping pod.
    """
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(
        configs, head=HEAD_YAML.replace("gpu_layers: 99", "gpu_layers: 99, port: 'not-a-port'")
    )
    net = _write_network(configs)

    with pytest.raises(ValueError, match="not a valid edge config"):
        generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")


def test_validate_groups_parses_each_config_file_once(tmp_path, monkeypatch) -> None:
    """Workers share one template, and the generation pass re-reads the same files.

    Both used to parse independently — a hand-rolled cache inside
    validate_groups plus the module cache mcp_endpoints fills — so every group
    member's YAML was read twice per `aureax generate`.
    """
    from client.config import parser as parser_mod

    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "worker.yaml").write_text(WORKER_YAML)
    cfg = make_network_config(
        [
            {"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"},
            {"node_id": "w1", "host": "10.77.0.2", "config_file": "worker.yaml"},
            {"node_id": "w2", "host": "10.77.0.3", "config_file": "worker.yaml"},
        ],
        [GROUP],
        models=MODELS,
    )

    real = parser_mod.load_edge_config_raw
    parsed: list[str] = []

    def counting(path):
        parsed.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(parser_mod, "load_edge_config_raw", counting)

    assert validate_groups(cfg, tmp_path) == []
    assert parsed == ["h.yaml", "worker.yaml"], "one parse per file, not per node"

    # Generation reads the same files immediately afterwards, through the same
    # cache — it must not parse them again.
    mcp_endpoints(cfg, tmp_path)
    assert parsed == ["h.yaml", "worker.yaml"], "the generation pass re-parsed"


def test_validate_groups_rejects_a_head_that_serves_no_mcp(tmp_path) -> None:
    """MCP reaches a group only through its head; every worker sets it false.

    A head with `mcp.server.enabled: false` therefore leaves a group that serves
    inference happily and has no MCP endpoint at all.
    """
    _write_group_configs(
        tmp_path, head=HEAD_YAML.replace("mcp: {server: {enabled: true}}",
                                         "mcp: {server: {enabled: false}}")
    )
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert any("only MCP endpoint" in p for p in validate_groups(cfg, tmp_path))


def test_validate_groups_rejects_a_worker_template_carrying_the_api_key(tmp_path) -> None:
    """rpc-server takes no key, so a worker template that asks for one mints a
    key file per worker and ships a bearer token nothing on the node reads —
    while the Access tab keeps reporting workers as unsupported.

    Reported once per FILE: the six real workers share one template.
    """
    keyed_worker = WORKER_YAML.replace(
        "rpc_server: {port: 50052}", 'rpc_server: {port: 50052}, server: {api_key: "${api_key}"}'
    )
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "worker.yaml").write_text(keyed_worker)
    cfg = make_network_config(
        [
            {"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"},
            {"node_id": "w1", "host": "10.77.0.2", "config_file": "worker.yaml"},
            {"node_id": "w2", "host": "10.77.0.3", "config_file": "worker.yaml"},
        ],
        [GROUP],
        models=MODELS,
    )

    problems = [p for p in validate_groups(cfg, tmp_path) if "api_key" in p]
    assert len(problems) == 1, problems
    assert "worker.yaml" in problems[0]


def test_validate_groups_reports_a_bad_tensor_split_instead_of_raising(tmp_path) -> None:
    """A scalar `tensor_split: 3` must not skip the length check — skipping is
    exactly the silent lopsided split this validator exists to prevent."""
    _write_group_configs(tmp_path, head=HEAD_YAML.replace("tensor_split: []", "tensor_split: 3"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert any("expected a list" in p for p in validate_groups(cfg, tmp_path))


def test_a_null_worker_port_reads_as_the_default_and_never_raises(tmp_path) -> None:
    """`port:` with no value parses to None, and `.get("port", 50052)` returns
    that None rather than the default — int(None) raised TypeError out of a
    function called with no try, killing the whole generate."""
    _write_group_configs(tmp_path, worker=WORKER_YAML.replace("port: 50052", "port:"))
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert [p for p in validate_groups(cfg, tmp_path) if "rpc_server.port" in p] == []


def test_an_unreadable_shared_template_is_reported_once(tmp_path) -> None:
    """Both workers share one file, so a naive loop reports it per node."""
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "worker.yaml").write_text("model: {server: [unclosed\n")
    cfg = make_network_config(
        [
            {"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"},
            {"node_id": "w1", "host": "10.77.0.2", "config_file": "worker.yaml"},
            {"node_id": "w2", "host": "10.77.0.3", "config_file": "worker.yaml"},
        ],
        [GROUP],
        models=MODELS,
    )
    problems = [p for p in validate_groups(cfg, tmp_path) if "worker.yaml" in p]
    assert len(problems) == 1, problems
    assert "could not parse" in problems[0]


def test_a_missing_config_file_is_reported_per_file(tmp_path) -> None:
    (tmp_path / "h.yaml").write_text(HEAD_YAML)  # w1.yaml/w2.yaml never written
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    problems = [p for p in validate_groups(cfg, tmp_path) if "not found" in p]
    assert len(problems) == 2, problems  # two DIFFERENT files, one report each


def test_a_placeholder_in_a_mapping_key_is_refused(tmp_aureax_home, tmp_path) -> None:
    """Resolution rebuilds dicts as {k: resolve(v)}, so a ${...} in key position
    is substituted by nothing and scanned by nothing — it used to ship to the
    node as the literal string."""
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(
        configs, head=HEAD_YAML.replace("  server: {", '  "${node_id}": 1\n  server: {')
    )
    net = _write_network(configs)

    with pytest.raises(ValueError, match="mapping KEYS"):
        generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")


def test_deployment_metadata_is_stamped_without_a_deployment_block(
    tmp_aureax_home, tmp_path
) -> None:
    """A per-node config written by hand may carry no `deployment:` block —
    adding a node is documented as two YAML edits and no code change (see
    ARCHITECTURE.md, "2. Configuration: per-node files, not one shared
    template"). Unstamped, the Access tab compares None against the applied
    hash and reads the node as never-applied forever.
    """
    from ruamel.yaml import YAML

    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs)
    assert "deployment" not in HEAD_YAML
    net = _write_network(configs)

    out = tmp_path / "out"
    generate_all_edge_configs(net, out, "127.0.0.1")

    with open(out / "aureax-edge-h.yaml") as f:
        generated = YAML().load(f)
    assert generated["deployment"]["config_hash"]
    assert generated["deployment"]["config_version"] == "0.1.0"


# --- the worker RPC allowlist -------------------------------------------------
# model.rpc_server.allowed_clients is generator output, not operator input: it
# is read on the node by client/rpc_firewall.py in a NET_ADMIN init container
# that resolves nothing, so the head's address has to arrive as a literal.


def _write_fleet_with_hosts(configs: Path, hosts: dict[str, str]) -> Path:
    """The h/w1 group with hand-chosen hosts, so a hostname-shaped head is testable."""
    _write_group_configs(configs, workers=("w1",))
    net = configs / "aureax-network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n"
        + "".join(
            f"    - {{node_id: {n}, host: {h}, config_file: {n}.yaml}}\n"
            for n, h in hosts.items()
        )
        + "node_groups:\n  - {name: g, head: h, workers: [w1]}\n"
    )
    return net


def _worker_allowlist(out: Path, worker: str = "w1") -> list[str]:
    from ruamel.yaml import YAML

    with open(out / f"aureax-edge-{worker}.yaml") as f:
        return YAML().load(f)["model"]["rpc_server"]["allowed_clients"]


def test_resolve_host_literals_normalises_an_address_without_resolving(monkeypatch) -> None:
    """An IP literal must never reach the resolver — a fleet addressed by IP
    generates on a machine with no DNS at all."""
    from orchestrator.config import templating as templating_mod

    def explode(*a, **kw):
        raise AssertionError("an IP literal was sent to the resolver")

    monkeypatch.setattr(templating_mod.socket, "getaddrinfo", explode)

    assert resolve_host_literals("10.77.0.1", "h") == ["10.77.0.1"]
    # Canonicalised, because the firewall compares these to what the kernel
    # prints back.
    assert resolve_host_literals("FD00:0:0::0009", "h") == ["fd00::9"]


def test_an_ip_literal_head_reaches_the_worker_allowlist_unchanged(
    tmp_aureax_home, tmp_path
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    net = _write_fleet_with_hosts(configs, {"h": "10.77.0.1", "w1": "10.77.0.31"})

    out = tmp_path / "out"
    generate_all_edge_configs(net, out, "127.0.0.1")

    assert _worker_allowlist(out) == ["10.77.0.1"]


def test_a_hostname_head_is_resolved_to_literals_at_generate_time(
    tmp_aureax_home, tmp_path, monkeypatch
) -> None:
    """The init container has no mDNS and its DNS answers for cluster names, so
    a `.local` head resolves on the orchestrator or not at all — and a firewall
    that never installs means the worker never starts."""
    import socket as socket_mod

    from orchestrator.config import templating as templating_mod

    def fake_getaddrinfo(host, *a, **kw):
        assert host == "head.example"
        return [
            (socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("10.77.0.9", 0)),
            # Same address twice (one entry per protocol) and a second family:
            # the allowlist must de-duplicate and carry both.
            (socket_mod.AF_INET, socket_mod.SOCK_STREAM, 17, "", ("10.77.0.9", 0)),
            (socket_mod.AF_INET6, socket_mod.SOCK_STREAM, 6, "", ("fd00::9", 0, 0, 0)),
        ]

    monkeypatch.setattr(templating_mod.socket, "getaddrinfo", fake_getaddrinfo)

    configs = tmp_path / "configs"
    configs.mkdir()
    net = _write_fleet_with_hosts(configs, {"h": "head.example", "w1": "10.77.0.31"})

    out = tmp_path / "out"
    generate_all_edge_configs(net, out, "127.0.0.1")

    assert _worker_allowlist(out) == ["10.77.0.9", "fd00::9"]


def test_a_head_that_does_not_resolve_fails_generation(
    tmp_aureax_home, tmp_path, monkeypatch
) -> None:
    """Naming the node here beats a worker init container crash-looping with a
    getaddrinfo traceback on a machine nobody is watching."""
    import socket as socket_mod

    from orchestrator.config import templating as templating_mod

    def refuse(host, *a, **kw):
        raise socket_mod.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(templating_mod.socket, "getaddrinfo", refuse)

    configs = tmp_path / "configs"
    configs.mkdir()
    net = _write_fleet_with_hosts(configs, {"h": "head.example", "w1": "10.77.0.31"})

    with pytest.raises(ValueError, match=r"h: host 'head.example' does not resolve"):
        generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")


def test_a_head_that_resolves_to_nothing_fails_generation(
    tmp_aureax_home, tmp_path, monkeypatch
) -> None:
    """An empty answer would otherwise ship an empty allowlist, which
    rpc_firewall reads as "no head may connect" — the group hangs at the RPC
    handshake with every part looking healthy."""
    from orchestrator.config import templating as templating_mod

    monkeypatch.setattr(templating_mod.socket, "getaddrinfo", lambda *a, **kw: [])

    configs = tmp_path / "configs"
    configs.mkdir()
    net = _write_fleet_with_hosts(configs, {"h": "head.example", "w1": "10.77.0.31"})

    with pytest.raises(ValueError, match="resolves to no address"):
        generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")


def test_the_worker_allowlist_is_inside_the_config_hash(tmp_aureax_home, tmp_path) -> None:
    """A head that moves has to restage its workers: their firewalls name the
    old address and would drop every packet the head sends."""
    from ruamel.yaml import YAML

    def generate(head_host: str, out_name: str) -> dict:
        configs = tmp_path / out_name / "configs"
        configs.mkdir(parents=True)
        net = _write_fleet_with_hosts(configs, {"h": head_host, "w1": "10.77.0.31"})
        out = tmp_path / out_name / "out"
        generate_all_edge_configs(net, out, "127.0.0.1")
        with open(out / "aureax-edge-w1.yaml") as f:
            return YAML().load(f)

    before = generate("10.77.0.1", "before")
    after = generate("10.77.0.2", "after")

    assert before["model"]["rpc_server"]["allowed_clients"] == ["10.77.0.1"]
    assert after["model"]["rpc_server"]["allowed_clients"] == ["10.77.0.2"]
    assert before["deployment"]["config_hash"] != after["deployment"]["config_hash"]


def test_the_deployment_block_itself_stays_out_of_the_hash(
    tmp_aureax_home, tmp_path
) -> None:
    """The hash is computed before the stamp is added, so two generates of an
    unchanged fleet agree even though `deployed_at` differs."""
    from ruamel.yaml import YAML

    configs = tmp_path / "configs"
    configs.mkdir()
    net = _write_fleet_with_hosts(configs, {"h": "10.77.0.1", "w1": "10.77.0.31"})
    out = tmp_path / "out"

    def head_hash() -> str:
        generate_all_edge_configs(net, out, "127.0.0.1")
        with open(out / "aureax-edge-h.yaml") as f:
            return YAML().load(f)["deployment"]["config_hash"]

    assert head_hash() == head_hash()


# --- the mtime-keyed caches ---------------------------------------------------


def test_cached_config_is_reparsed_after_an_edit(tmp_path, monkeypatch) -> None:
    """The dashboard is long-running, so an operator edit has to win.

    Keying on (path, mtime) rather than path alone is the whole reason — and a
    regression to path-only keying passes every other test in the suite while
    serving a stale config forever.
    """
    from client.config import parser as parser_mod

    path = tmp_path / "n.yaml"
    path.write_text(HEAD_YAML)

    real = parser_mod.load_edge_config_raw
    parses: list[str] = []

    def counting(p):
        parses.append(str(p))
        return real(p)

    monkeypatch.setattr(parser_mod, "load_edge_config_raw", counting)

    assert load_edge_config_cached(path)["node"]["role"] == "edge_executor"
    assert load_edge_config_cached(path)["node"]["role"] == "edge_executor"
    assert len(parses) == 1, "an unchanged file was parsed twice"

    path.write_text(WORKER_YAML)
    _bump_mtime(path)

    assert load_edge_config_cached(path)["node"]["role"] == "shard_worker"
    assert len(parses) == 2


def test_generation_parses_each_template_once_per_file(
    tmp_aureax_home, tmp_path, monkeypatch
) -> None:
    """Templates are shared on purpose — both heads share one file and all six
    workers share another — but generation parsed per TARGET."""
    from orchestrator.config import loader as loader_mod

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "h.yaml").write_text(HEAD_YAML)
    (configs / "worker.yaml").write_text(WORKER_YAML)
    net = configs / "aureax-network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n"
        "    - {node_id: h, host: 10.77.0.1, config_file: h.yaml}\n"
        "    - {node_id: w1, host: 10.77.0.2, config_file: worker.yaml}\n"
        "    - {node_id: w2, host: 10.77.0.3, config_file: worker.yaml}\n"
        "node_groups:\n  - {name: g, head: h, workers: [w1, w2]}\n"
    )

    real = loader_mod.load_yaml
    parsed: list[str] = []

    def counting(path):
        parsed.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(loader_mod, "load_yaml", counting)
    generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")

    assert parsed.count("worker.yaml") == 1, "parsed once per target, not per file"
    assert parsed.count("h.yaml") == 1


# --- validate_planners: the MCP-client mirror of the head's rpc_servers rule ---
# mcp.client.remote_servers reaches a node ONLY through ${mcp_endpoints}. A
# hardcoded list does not error and does not look different — the planner dials
# whatever was typed, forever, while every reader of the fleet file believes the
# list is derived. It is also where each peer's MCP bearer key arrives, so a
# hardcoded list means a planner that 401s on every call after one rotation.


def _planner_fleet(tmp_path: Path, planner_yaml: str, planners=("p",)) -> NetworkConfig:
    """Head + N planners sharing one template, with the files on disk."""
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "planner.yaml").write_text(planner_yaml)
    targets = [{"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"}]
    targets += [
        {"node_id": p, "host": f"10.77.0.{20 + i}", "config_file": "planner.yaml"}
        for i, p in enumerate(planners)
    ]
    return make_network_config(targets, models=MODELS)


def test_validate_planners_rejects_a_hardcoded_remote_servers_list(tmp_path) -> None:
    cfg = _planner_fleet(
        tmp_path,
        PLANNER_YAML.replace(
            'remote_servers: "${mcp_endpoints}"',
            "remote_servers: [{name: edge-a, url: 'https://10.9.9.9:30900/mcp'}]",
        ),
    )
    problems = validate_planners(cfg, tmp_path)
    assert len(problems) == 1, problems
    assert "hardcodes mcp.client.remote_servers" in problems[0]
    assert "${mcp_endpoints}" in problems[0], "the message must carry the fix"


def test_validate_planners_rejects_an_mcp_client_with_no_remote_servers(tmp_path) -> None:
    """A client that dials nothing is not an error anywhere at runtime — the
    planner simply has no remotes and reports none."""
    cfg = _planner_fleet(
        tmp_path,
        PLANNER_YAML.replace(', remote_servers: "${mcp_endpoints}"', ""),
    )
    problems = validate_planners(cfg, tmp_path)
    assert len(problems) == 1, problems
    assert "but no mcp.client.remote_servers" in problems[0]
    assert "dials nothing" in problems[0]
    assert "${mcp_endpoints}" in problems[0], "the message must carry the fix"


def test_validate_planners_ignores_a_node_that_is_not_an_mcp_client(tmp_path) -> None:
    """Absent means NO on this predicate — the opposite of mcp.server.enabled.

    Every ordinary edge node and every shard worker omits the client block, so
    a default of True here would demand the placeholder from the whole fleet.
    """
    _write_group_configs(tmp_path)
    cfg = _net(make_targets("h", "w1", "w2"), [GROUP])
    assert validate_planners(cfg, tmp_path) == []


def test_validate_planners_reports_a_shared_planner_template_once(tmp_path) -> None:
    """One problem per FILE: two planners can share one template, and reporting
    per node would print the same fix twice."""
    cfg = _planner_fleet(
        tmp_path,
        PLANNER_YAML.replace('"${mcp_endpoints}"', "[{name: a, url: 'https://h:30900/mcp'}]"),
        planners=("p1", "p2"),
    )
    assert len(validate_planners(cfg, tmp_path)) == 1


def test_a_null_mcp_client_block_does_not_raise(tmp_path) -> None:
    """`client:` with nothing under it parses to None, not {}.

    ``.get("client", {})`` returns that None, so reading `.enabled` off it
    raised AttributeError out of a function generation calls with no try —
    one malformed-but-parseable file taking the whole fleet's generate down.
    """
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "p.yaml").write_text(
        "node: {id: p, role: cognitive_planner, control_plane: {endpoint: x}}\n"
        "model: {name: M, filename: m.gguf}\n"
        "mcp:\n  client:\n"
    )
    cfg = make_network_config(
        [
            {"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"},
            {"node_id": "p", "host": "10.77.0.20", "config_file": "p.yaml"},
        ],
        models=MODELS,
    )
    assert validate_planners(cfg, tmp_path) == []


def test_validate_planners_skips_a_disabled_target(tmp_path) -> None:
    """A disabled node is documented, not deployed — every other pass skips it,
    so refusing to generate the whole fleet over one is the wrong severity."""
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "p.yaml").write_text(PLANNER_YAML.replace('"${mcp_endpoints}"', "[]"))
    cfg = make_network_config(
        [
            {"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"},
            {"node_id": "p", "host": "10.77.0.20", "config_file": "p.yaml", "enabled": False},
        ],
        models=MODELS,
    )
    assert validate_planners(cfg, tmp_path) == []


def test_generate_runs_the_planner_validator_too(tmp_aureax_home, tmp_path) -> None:
    """The planner rule has to reach generation, not just its own unit test —
    validate_groups returns early for a fleet with no node_groups, so folding
    this check into it would have silently skipped every group-less fleet."""
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "h.yaml").write_text(HEAD_YAML)
    (configs / "p.yaml").write_text(
        PLANNER_YAML.replace('"${mcp_endpoints}"', "[{name: a, url: 'https://h:30900/mcp'}]")
    )
    net = configs / "aureax-network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n"
        "    - {node_id: h, host: 10.77.0.1, config_file: h.yaml}\n"
        "    - {node_id: p, host: 10.77.0.20, config_file: p.yaml}\n"
    )

    with pytest.raises(ValueError, match=re.escape("hardcodes mcp.client.remote_servers")):
        generate_all_edge_configs(net, tmp_path / "out", "127.0.0.1")


def test_validate_groups_rejects_a_worker_template_carrying_the_mcp_api_key(tmp_path) -> None:
    """Mirror of the ${api_key} rule: a worker sets mcp.server.enabled: false,
    so it runs no MCP server for a bearer check to protect. The line would mint
    a key file per worker and ship a token nothing on the node ever reads.

    Reported once per FILE, and separately from the llama rule — a template
    carrying both lines needs both edits undone.
    """
    keyed_worker = WORKER_YAML.replace(
        "mcp: {server: {enabled: false}}",
        'mcp: {server: {enabled: false, api_key: "${mcp_api_key}"}}',
    )
    (tmp_path / "h.yaml").write_text(HEAD_YAML)
    (tmp_path / "worker.yaml").write_text(keyed_worker)
    cfg = make_network_config(
        [
            {"node_id": "h", "host": "10.77.0.1", "config_file": "h.yaml"},
            {"node_id": "w1", "host": "10.77.0.2", "config_file": "worker.yaml"},
            {"node_id": "w2", "host": "10.77.0.3", "config_file": "worker.yaml"},
        ],
        [GROUP],
        models=MODELS,
    )

    problems = [p for p in validate_groups(cfg, tmp_path) if "mcp_api_key" in p]
    assert len(problems) == 1, problems
    assert "worker.yaml" in problems[0]
    assert "runs no MCP server" in problems[0]


# --- worker order against the persisted baseline ------------------------------
# Worker order IS --rpc order, which names the devices RPC0..RPCn that
# tensor_split and devices index POSITIONALLY. A reordered list is a perfectly
# valid list, so the only thing that can catch it is what the order was last
# time — ~/.aureax/state/node_groups.json. Every test here needs tmp_aureax_home
# for that reason.


def _generate(net: Path, out: Path, **kw) -> list[Path]:
    return generate_all_edge_configs(net, out, "127.0.0.1", **kw)


def _regenerate_with_workers(configs: Path, workers: tuple[str, ...]) -> Path:
    """Rewrite the fleet file with a new worker list, past the mtime cache."""
    net = _write_network(configs, workers=workers)
    _bump_mtime(net)
    return net


@pytest.fixture
def group_fleet(tmp_aureax_home, tmp_path):
    """configs/ + out/ for the h + w1..w3 group, generated once at [w1, w2]."""
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs, workers=("w1", "w2", "w3"))
    out = tmp_path / "out"
    _generate(_write_network(configs, workers=("w1", "w2")), out)
    return configs, out


def test_a_first_generate_records_a_baseline_and_reports_nothing(
    tmp_aureax_home, tmp_path, caplog
) -> None:
    """A fresh clone has nothing to compare against; refusing there would mean
    no one could generate at all."""
    from orchestrator.config import group_state

    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs)

    with caplog.at_level(logging.WARNING, logger="aureax.config.templating"):
        _generate(_write_network(configs), tmp_path / "out")

    assert caplog.records == []
    assert group_state.baseline_path().exists()


def test_a_reordered_workers_list_fails_generation(group_fleet) -> None:
    """The one case that corrupts a tensor_split silently: same machines, new
    positions, so every entry now addresses a different node."""
    configs, out = group_fleet

    with pytest.raises(ValueError, match="REORDERED") as excinfo:
        _generate(_regenerate_with_workers(configs, ("w2", "w1")), out)

    message = str(excinfo.value)
    assert "--accept-worker-reorder" in message, "the message must carry the override"
    assert "w1, w2" in message and "w2, w1" in message, "was/now must both be shown"


def test_a_reordered_workers_list_is_accepted_with_the_flag_and_rebaselined(
    group_fleet,
) -> None:
    """Accepting records the NEW order, so the next generate is silent — a
    reorder that stayed reported forever would train the operator to pass the
    flag every time."""
    configs, out = group_fleet

    net = _regenerate_with_workers(configs, ("w2", "w1"))
    _generate(net, out, accept_worker_reorder=True)

    _bump_mtime(net, 2)
    _generate(net, out)  # must not raise: the new order IS the baseline now


@pytest.mark.parametrize(
    ("workers", "why"),
    [
        (("w1", "w2", "w3"), "appending keeps every existing index"),
        (("w1",), "removing the tail keeps every retained index"),
    ],
)
def test_changing_the_worker_SET_is_a_warning_not_an_error(
    group_fleet, caplog, workers, why
) -> None:
    """Resizing a group costs a cold load — the head re-streams every tensor —
    but it is the documented way to resize, so it must not refuse."""
    configs, out = group_fleet

    with caplog.at_level(logging.WARNING, logger="aureax.config.templating"):
        _generate(_regenerate_with_workers(configs, workers), out)

    assert any("COLD" in r.getMessage() or "cold" in r.getMessage() for r in caplog.records), why


def test_the_baseline_is_keyed_by_network_config_path(tmp_aureax_home, tmp_path) -> None:
    """`aureax -c configs/aureax-network.example.yaml generate` is what every
    fresh clone and scripts/test-all.sh runs. Sharing one baseline key would let
    the example's worker order accuse the real fleet's next generate of a
    reorder — and vice versa."""
    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs)
    out = tmp_path / "out"

    _generate(_write_network(configs, workers=("w1", "w2")), out)
    # Same directory, different fleet file, deliberately opposite order.
    other = _write_network(configs, workers=("w2", "w1"), filename="other-network.yaml")

    _generate(other, out)  # no baseline of its own yet — must not raise


def test_the_baseline_is_not_written_when_generation_fails(
    tmp_aureax_home, tmp_path, monkeypatch
) -> None:
    """A baseline for an order that never reached a node makes the retry look
    like a legitimate second run, hiding the reorder for good."""
    from orchestrator.config import group_state
    from orchestrator.config import templating as templating_mod

    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs)

    def die(data, stream, **kw):
        raise RuntimeError("no space left on device")

    monkeypatch.setattr(templating_mod.yaml, "dump", die)
    with pytest.raises(RuntimeError):
        _generate(_write_network(configs), tmp_path / "out")

    assert not group_state.baseline_path().exists()


def test_an_unreadable_baseline_does_not_block_generation(
    tmp_aureax_home, tmp_path, caplog
) -> None:
    """Nothing in the reorder check may abort a deploy. A corrupt JSON file
    holds an order this process cannot see, so it degrades to "no baseline" —
    the same rule access_service._load_journal follows."""
    from orchestrator.config import group_state

    configs = tmp_path / "configs"
    configs.mkdir()
    _write_group_configs(configs)
    group_state.state_dir().mkdir(parents=True, exist_ok=True)
    group_state.baseline_path().write_text("{ not json")

    with caplog.at_level(logging.ERROR, logger="aureax.config.templating"):
        paths = _generate(_write_network(configs, workers=("w2", "w1")), tmp_path / "out")

    assert len(paths) == 3


def test_resolve_host_literals_drops_scoped_and_link_local_answers(monkeypatch) -> None:
    """mDNS answers a `.local` head with `fe80::…%en0` first; ip6tables rejects a
    scope-qualified source, so passing it through would crash the worker's
    init container. Only routable answers reach the allowlist."""
    import socket as socket_mod

    from orchestrator.config import templating as templating_mod

    def fake_getaddrinfo(host, *a, **kw):
        return [
            (socket_mod.AF_INET6, socket_mod.SOCK_STREAM, 6, "", ("fe80::1c:2:3:4%en0", 0, 0, 5)),
            (socket_mod.AF_INET6, socket_mod.SOCK_STREAM, 6, "", ("::1", 0, 0, 0)),
            (socket_mod.AF_INET, socket_mod.SOCK_STREAM, 6, "", ("10.77.0.9", 0)),
        ]

    monkeypatch.setattr(templating_mod.socket, "getaddrinfo", fake_getaddrinfo)
    assert templating_mod.resolve_host_literals("head.local", "w1") == ["10.77.0.9"]


def test_a_head_with_only_link_local_answers_fails_generation(monkeypatch) -> None:
    import socket as socket_mod

    from orchestrator.config import templating as templating_mod

    def fake_getaddrinfo(host, *a, **kw):
        return [(socket_mod.AF_INET6, socket_mod.SOCK_STREAM, 6, "", ("fe80::1%en0", 0, 0, 5))]

    monkeypatch.setattr(templating_mod.socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(ValueError, match="resolves to no address"):
        templating_mod.resolve_host_literals("head.local", "w1")
