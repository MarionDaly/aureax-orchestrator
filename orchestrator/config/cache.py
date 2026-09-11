"""Mtime-keyed caches for the config files the orchestrator re-reads.

Both the generation pass and the dashboard resolve the same handful of YAML
files many times per operation — ``mcp_endpoints()`` reads every target's
config and is itself called once per target, so an N-node fleet paid N^2 YAML
parses (measured: 375 ms for 9 nodes), and building one chat engine re-read and
re-validated the network config three times.

Keyed on ``(path, mtime)`` rather than path alone: a long-running process (the
dashboard) must still pick up a config the operator just edited.

The cached values are shared between callers and must be treated as read-only —
mutating one changes what every later reader sees.

The per-node and template wrappers stat the path before consulting their
loader, so a MISSING file raises FileNotFoundError from them rather than the
loader's error. The network-config wrapper instead hands a missing file to
``load_network_config`` so the operator sees its ConfigError, which names the
shipped example when the default path is the one missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from orchestrator.config.schema import NetworkConfig

logger = logging.getLogger("aureax.config.cache")

# str(path) -> (mtime_ns, parsed)
_RAW_CONFIG_CACHE: dict[str, tuple[int, dict]] = {}
_TEMPLATE_CACHE: dict[str, tuple[int, dict]] = {}
_NETWORK_CONFIG_CACHE: dict[str, tuple[int, NetworkConfig]] = {}


def load_edge_config_cached(path: Path | str) -> dict:
    """Parse a per-node config, memoized on (path, mtime)."""
    from client.config.parser import load_edge_config_raw

    path = Path(path)
    key = str(path)
    mtime = path.stat().st_mtime_ns
    hit = _RAW_CONFIG_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    parsed = load_edge_config_raw(key)
    _RAW_CONFIG_CACHE[key] = (mtime, parsed)
    return parsed


def load_template_cached(path: Path | str) -> dict:
    """Parse a per-node template via ``loader.load_yaml``, memoized on (path, mtime).

    Deliberately a second cache rather than ``load_edge_config_cached``: that
    one parses through ``client.config.parser``, whose ``YAML()`` has no
    ``preserve_quotes``. Generation DUMPS what it parses, so reusing it would
    silently restyle every generated file (``id: "${node_id}"`` -> ``id:
    ${node_id}``). Templates are shared by several targets on purpose (see
    ARCHITECTURE.md, "2. Configuration: per-node files, not one shared
    template"), so this is once per file instead of once per node.
    """
    from orchestrator.config.loader import load_yaml

    path = Path(path)
    key = str(path)
    mtime = path.stat().st_mtime_ns
    hit = _TEMPLATE_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    parsed = load_yaml(path)
    _TEMPLATE_CACHE[key] = (mtime, parsed)
    return parsed


def load_network_config_cached(path: Path | str) -> NetworkConfig:
    """Load and validate aureax-network.yaml, memoized on (path, mtime)."""
    from orchestrator.config.loader import load_network_config

    path = Path(path)
    key = str(path)
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        # Let the loader raise: its ConfigError carries the copy-the-example
        # hint for a fresh clone, which a bare stat error would hide.
        return load_network_config(path)
    hit = _NETWORK_CONFIG_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    parsed = load_network_config(path)
    _NETWORK_CONFIG_CACHE[key] = (mtime, parsed)
    return parsed


def load_node_config_resolved(configs_dir: Path | str, target) -> dict:
    """A target's per-node config, RESOLVED where possible. ``{}`` on failure.

    Prefers the generated file in ``~/.aureax/configs`` over the source
    template in ``configs_dir``. The template is only a fallback, so that a
    fleet which has never run `aureax generate` still resolves the values that
    are literals in both forms — roles, ports, flags.

    Anything a template spells as ``${...}`` is a plain STRING there, and
    callers must treat every such value as absent. Two of them bite hard: the
    planner's ``mcp.client.remote_servers: ${mcp_endpoints}`` is a list only
    after generation, so iterating it walks the characters of the placeholder
    and calls ``.get()`` on ``"$"``; and ``model.server.api_key`` sent as a
    bearer token is the literal "${api_key}", which 401s every request.

    One lookup, shared by both dashboard services that need it (chat_service's
    target list and access_service's per-node view), so they cannot drift on
    which file wins.
    """
    from orchestrator.config.loader import generated_config_path

    candidates = [generated_config_path(target.node_id)]
    if target.config_file:
        candidates.append(Path(configs_dir) / target.config_file)
    for path in candidates:
        if not path.exists():
            continue
        try:
            return load_edge_config_cached(path)
        except Exception:
            logger.exception("failed to load per-node config %s", path)
    return {}
