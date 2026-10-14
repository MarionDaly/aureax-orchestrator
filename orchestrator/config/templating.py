"""Generate per-node edge configs from the network config and per-node config files."""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import logging
import os
import re
import socket
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML

from orchestrator.config import placement
from orchestrator.config.atomic import write_private_file
from orchestrator.config.cache import (
    load_edge_config_cached,
    load_network_config_cached,
    load_template_cached,
)
from orchestrator.config.loader import (
    describe_validation_error,
    generated_config_path,
    validate_edge_data,
)
from orchestrator.config.schema import (
    DEFAULT_WORKER_WAIT_SECONDS,
    HEAD_ROLE,
    MCP_NODE_PORT,
    SHARD_WORKER_ROLE,
    WG_INTERFACE,
    NetworkConfig,
    SSHTarget,
    mcp_client_enabled,
    mcp_server_enabled,
    node_role,
)

logger = logging.getLogger("aureax.config.templating")

yaml = YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False


_PLACEHOLDER_RE = re.compile(r"\$\{([^}]*)\}")

# Spelled out rather than f-string-built from the key: these are the literals a
# head's or planner's template must carry, and validate_groups/validate_planners
# compare raw template text against them.
_WORKER_WAIT_PLACEHOLDER = "${worker_wait_timeout_seconds}"
_RPC_SERVERS_PLACEHOLDER = "${rpc_servers}"
# The two launch values llama.cpp's placement rule turns into a layer layout
# (orchestrator/config/placement.py). A head template may carry literals — the
# frozen rollback pair does — or these placeholders, which resolve per group
# from node_groups.head_layers / .placement. Any head-resident layers REQUIRE
# the placeholders: a literal split cannot follow a changed layer range.
_TENSOR_SPLIT_PLACEHOLDER = "${tensor_split}"
_GPU_LAYERS_PLACEHOLDER = "${gpu_layers}"
_MCP_ENDPOINTS_PLACEHOLDER = "${mcp_endpoints}"
_MCP_API_KEY_PLACEHOLDER = "${mcp_api_key}"


def _unresolved_placeholders(obj: object) -> set[str]:
    """Every ``${name}`` appearing in a string VALUE, at any depth.

    Resolution-agnostic: run on a resolved config these are genuine leftovers,
    run on a template they are the names the template references — both uses
    are load-bearing below. Mapping keys are NOT covered; that is
    _key_placeholders' job.
    """
    if isinstance(obj, str):
        return set(_PLACEHOLDER_RE.findall(obj))
    if isinstance(obj, dict):
        return set().union(*(_unresolved_placeholders(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(_unresolved_placeholders(v) for v in obj)) if obj else set()
    return set()


def _key_placeholders(obj: object) -> set[str]:
    """Every ``${name}`` written as a MAPPING KEY, at any depth.

    resolve_placeholders rebuilds dicts as ``{k: resolve(v)}``, so keys are
    never substituted — deliberately, since exact-match injection of e.g.
    ``${tags}`` would yield an unhashable list as a key. Scanned separately so
    the leftover guard can say a key placeholder is unsupported rather than
    report a perfectly known name as "unresolved".
    """
    if isinstance(obj, dict):
        names: set[str] = set()
        for k, v in obj.items():
            if isinstance(k, str):
                names |= set(_PLACEHOLDER_RE.findall(k))
            names |= _key_placeholders(v)
        return names
    if isinstance(obj, list):
        return set().union(*(_key_placeholders(v) for v in obj)) if obj else set()
    return set()


def resolve_placeholders(obj: object, variables: dict[str, object]) -> object:
    """Recursively resolve ${...} placeholders in a data structure.

    If the entire value is a single placeholder (e.g. "${tags}"), the raw
    variable value is injected directly (preserving type). If a placeholder
    appears inside a larger string, it is stringified.
    """
    if isinstance(obj, str):
        # Neither branch below can match without a literal "${", and most
        # leaves have none — this skips one str.replace per known variable on
        # each of them.
        if "${" not in obj:
            return obj
        stripped = obj.strip()
        # Exact match: inject raw value (preserves lists, dicts, etc.)
        if stripped.startswith("${") and stripped.endswith("}") and stripped.count("${") == 1:
            key = stripped[2:-1]
            if key in variables:
                return variables[key]
        # Partial match: string substitution
        result = obj
        for key, value in variables.items():
            result = result.replace(f"${{{key}}}", str(value))
        return result
    if isinstance(obj, dict):
        return {k: resolve_placeholders(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_placeholders(item, variables) for item in obj]
    return obj


# Type-correct stand-ins used when validating a template in isolation (the
# dashboard editor, and load_edge_config on an unresolved file). These must
# stay in lockstep with build_variables() below — a placeholder known to one
# and not the other means the dashboard rejects configs that deploy fine, or
# accepts configs that fail at deploy time.
PLACEHOLDER_DUMMIES: dict[str, object] = {
    "node_id": "dummy",
    "display_name": "Dummy Node",
    "tags": [],
    "orchestrator_ip": "127.0.0.1",
    # https, and carrying the dashboard's default port, because every template
    # substitutes this straight into a URL and TLS is mandatory fleet-wide. A
    # plaintext dummy here would let the dashboard editor accept a template that
    # the real build_variables then resolves to https — the same class of
    # drift the lockstep rule above exists to prevent.
    "orchestrator_url": "https://127.0.0.1:7001",
    "group_id": "",
    "rpc_servers": [],
    # Shaped like a resolved placement: an even split over one device and
    # llama.cpp's offload-everything value. The real build_variables computes
    # both per group (orchestrator/config/placement.py).
    "tensor_split": [1],
    "gpu_layers": 99,
    # int, not "": NodeConfig types this field, so a string stand-in would make
    # the dashboard reject a head template that deploys perfectly well.
    "worker_wait_timeout_seconds": DEFAULT_WORKER_WAIT_SECONDS,
    # Shaped like a real remote_servers entry so MCPClient validates against it,
    # api_key included: mcp_endpoints() emits that field for every peer whose
    # template enables MCP auth, and RemoteServer validates it.
    "mcp_endpoints": [
        {"name": "dummy", "url": f"https://127.0.0.1:{MCP_NODE_PORT}/mcp",
         "transport": "streamable-http",
         "api_key": "aureax_sk_validation_dummy_peer_key_00000000"},
    ],
    # Shaped to pass LlamaServerConfig's api-key pattern (16+ chars of
    # [A-Za-z0-9_.-]) so a template carrying `api_key: ${api_key}` validates in
    # the dashboard editor without touching the key store.
    "api_key": "aureax_sk_validation_dummy_not_a_real_key_00000",
    # Deliberately DIFFERENT text from the llama dummy above: the two keys land
    # in different blocks of the same file (model.server vs mcp.server), and a
    # test asserting which one resolved where needs to be able to tell them
    # apart.
    "mcp_api_key": "aureax_sk_validation_dummy_mcp_key_000000000",
}


def _template_references(template_path: Path, name: str) -> bool:
    """Whether a node's TEMPLATE carries the ``${name}`` placeholder.

    One reader for every "is this feature on for this node?" question, because
    each such question is asked from three places that must agree — the CLI's
    Auth column, the dashboard's Access tab and generation itself. An
    unreadable or absent template reads as OFF: the display callers legitimately
    run before configs exist and must not raise.
    """
    try:
        data = load_template_cached(template_path)
    except FileNotFoundError:
        # `aureax nodes` legitimately runs before configs are generated, and it
        # asks per target — a warning here would be per-node noise.
        return False
    except Exception as e:
        logger.warning(
            "could not read %s for the auth check (%s) — reporting auth as OFF",
            template_path, e,
        )
        return False
    return name in _unresolved_placeholders(data)


def template_references_api_key(template_path: Path) -> bool:
    """Whether a node's TEMPLATE enables llama-server auth.

    The `api_key: ${api_key}` line under model.server is the single source of
    truth for llama auth being on.
    """
    return _template_references(template_path, "api_key")


def template_references_mcp_api_key(template_path: Path) -> bool:
    """Whether a node's TEMPLATE enables the MCP server's bearer check.

    The `api_key: ${mcp_api_key}` line under mcp.server, exactly mirroring the
    llama line above — there is no separate enabled flag on either side.
    """
    return _template_references(template_path, "mcp_api_key")


def template_references_mcp_endpoints(template_path: Path) -> bool:
    """Whether a node's TEMPLATE consumes the derived peer list.

    Read by the deployer to work out who else must be pushed when a peer's MCP
    key changes: a planner's resolved config carries every peer's key, so a
    rotated key leaves an unpushed consumer dialling with the old one and
    401ing on every call.
    """
    return _template_references(template_path, "mcp_endpoints")


def mcp_endpoints(
    network_config: NetworkConfig,
    configs_dir: Path,
    exclude_node_id: str = "",
) -> list[dict[str, str]]:
    """Every reachable MCP server in the fleet, as remote_servers entries.

    This is what an MCP *client* — the cognitive planner — needs to dial. A node
    serves MCP only if its own config says so, which excludes shard workers
    (they run rpc-server) and other planners (they are clients).

    Derived rather than hand-listed on purpose: a hand-written remote_servers
    list goes stale silently. A host reassigned as a shard worker stops serving
    MCP and a decommissioned one stops answering at all, and the planner then
    retries dead endpoints forever with nothing reporting it.

    Each entry carries the peer's MCP bearer key when that peer's TEMPLATE
    references ``${mcp_api_key}`` and the key already exists. Two rules make
    this safe, and both are load-bearing:

    * ``read_key``, never ``ensure_key``. This function runs inside
      build_variables, which access_service.desired_config_hash calls from a
      route the browser polls every 20 s — minting fleet credentials from a
      poll is the rule that path already observes for llama keys.
    * The gate is the peer's TEMPLATE, not "a key file exists". A node whose
      MCP auth was turned off still has its old ``.mcp.key`` on the Mac, and
      shipping that into a planner's config would hand out a credential for a
      door that is now open anyway.
    """
    out: list[dict[str, str]] = []
    for t in network_config.ssh.enabled_targets:
        if t.node_id == exclude_node_id or not t.config_file:
            continue
        path = configs_dir / t.config_file
        if not path.exists():
            # Debug, not warning: this runs once per target, so a fleet-wide
            # miss would log N-1 identical lines per generate.
            logger.debug("mcp_endpoints: no config at %s, skipping %s", path, t.node_id)
            continue
        try:
            serves_mcp = mcp_server_enabled(load_edge_config_cached(path))
        except Exception as e:
            # Broad on purpose: a parseable-but-malformed config (`mcp: "yes"`)
            # raises AttributeError, and letting that escape takes the whole
            # fleet's generation down with a traceback. Logged rather than
            # swallowed — dropping an endpoint silently is the exact failure
            # this function exists to end.
            logger.warning("mcp_endpoints: skipping %s: %s", path, e)
            continue
        if not serves_mcp:
            continue
        entry = {
            "name": t.node_id,
            # https, always: the MCP listener serves the node's own leaf from
            # /etc/aureax/tls and there is no plaintext port to fall back to.
            "url": f"https://{_url_host(t.host)}:{MCP_NODE_PORT}/mcp",
            "transport": "streamable-http",
        }
        if template_references_mcp_api_key(path):
            from orchestrator.config import keystore

            peer_key = keystore.read_key(t.node_id, kind=keystore.KIND_MCP)
            if peer_key:
                entry["api_key"] = peer_key
        out.append(entry)
    return out


def _url_host(host: str) -> str:
    """``host`` in the form a URL authority accepts.

    ``${orchestrator_url}`` and every peer URL in ``${mcp_endpoints}`` put a
    host straight into a URL authority, and an IPv6 literal
    must be bracketed there or no HTTP client can parse it — the address's own
    colons read as the port separator, so the whole fleet ships telemetry to an
    endpoint that never resolves and `aureax status` prints "(no metrics
    received yet)" everywhere.

    Bracketing happens here rather than in ``loader.validate_orchestrator_host``
    because that function returns the operator's value verbatim: the CLI and the
    dashboard both echo it back, and only this URL needs the brackets.
    """
    try:
        parsed = ipaddress.ip_address(host)
    except ValueError:
        return host
    return f"[{host}]" if parsed.version == 6 else host


def resolve_host_literals(host: str, node_id: str) -> list[str]:
    """Every IP literal ``host`` names, resolved on the orchestrator.

    An IP literal passes through as itself. A hostname is resolved here, at
    generate time, because the only consumer is a worker's firewall allowlist
    and that runs where no resolver can help: the init container image carries
    no mDNS and its cluster DNS answers for cluster names, so a ``.local`` head
    fails there and the worker never starts. Failing on the Mac instead names
    the node and the command to re-run.

    Both families, sorted and de-duplicated: getaddrinfo returns one entry per
    socket type and can return A and AAAA records for one name, and the
    firewall builds a v4 and a v6 chain from the same list.
    """
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError(
            f"{node_id}: host {host!r} does not resolve on this machine ({e}). Its address is "
            "baked into the worker firewall allowlist at generate time, because the worker "
            "resolves nothing itself. Give the node an IP address in ssh.targets, or make the "
            "name resolvable here, then re-run `aureax generate`."
        ) from e
    usable: set[str] = set()
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        # mDNS on the orchestrator answers a `.local` name with its link-local
        # IPv6 first, scope-qualified (`fe80::…%en0`). ip_address() accepts the
        # scope id and round-trips it, and ip6tables then rejects the literal —
        # an allowlist entry that kills the worker's init container. Link-local
        # and loopback answers are never the address the head dials from, so
        # they are dropped rather than passed through.
        if getattr(addr, "scope_id", None) or addr.is_loopback or (addr.version == 6 and addr.is_link_local):
            continue
        usable.add(str(addr))
    addresses = sorted(usable)
    if not addresses:
        raise ValueError(
            f"{node_id}: host {host!r} resolves to no address the allowlist can use (link-local "
            "and loopback answers are ignored). Give the node an IP address in ssh.targets, then "
            "re-run `aureax generate`."
        )
    return addresses


def build_variables(
    target: SSHTarget,
    network_config: NetworkConfig,
    orchestrator_ip: str,
    configs_dir: Path | None = None,
    api_key: str = "",
    mcp_api_key: str = "",
) -> dict[str, object]:
    """Placeholder values for one target.

    Single source of truth for what `${...}` names exist; PLACEHOLDER_DUMMIES
    mirrors the same key set for validation-only paths.

    ``configs_dir`` is needed only for ``${mcp_endpoints}``, which has to read
    each node's own config to know whether it serves MCP. Omitting it yields an
    empty list; generate_edge_config, the only production caller, always passes
    it, and the default exists for the tests that assert on the KEY SET alone.

    ``api_key`` and ``mcp_api_key`` are passed in rather than read from the key
    store here: this function runs for every target including shard workers, so
    touching the store from inside it would create key files for nodes whose
    template never asks for one. generate_edge_config resolves the real keys,
    and only for templates that reference ``${api_key}`` / ``${mcp_api_key}``.
    """
    group = network_config.group_for_node(target.node_id)

    rpc_servers: list[str] = []
    placement_values: dict[str, object] = {"tensor_split": [], "gpu_layers": placement.GPU_LAYERS_ALL}
    if group is not None and target.node_id == group.head:
        # Only the head dials out; workers listen and are dialled. Over the
        # group's WireGuard tunnel when it has one: each worker's rpc-server
        # binds its tunnel address only, so its LAN address would be refused.
        wg_addresses = network_config.wireguard_addresses(group)
        rpc_servers = group.rpc_endpoints(
            wg_addresses if wg_addresses is not None else network_config.host_by_node_id()
        )
        placed = group_placement(group, network_config, configs_dir)
        placement_values = {"tensor_split": placed.tensor_split, "gpu_layers": placed.gpu_layers}

    return {
        "node_id": target.node_id,
        "display_name": target.display_name or target.node_id,
        "tags": target.tags,
        # Bracketed when it is an IPv6 literal — see _url_host; every template
        # puts this straight into a URL authority.
        "orchestrator_ip": _url_host(orchestrator_ip),
        # Scheme and PORT included, so a template never spells either one.
        # server.dashboard.port is configurable but was hardcoded as 7001 in
        # every template, so a fleet that moved the dashboard shipped telemetry
        # to a closed port with nothing reporting it. https because there is no
        # plaintext listener to fall back to.
        "orchestrator_url": (
            f"https://{_url_host(orchestrator_ip)}:{network_config.server.dashboard.port}"
        ),
        "group_id": group.name if group else "",
        "rpc_servers": rpc_servers,
        # The head's layer layout, as llama.cpp's own launch values. Every
        # non-head gets an empty split and the offload-everything count, which
        # is inert: client/config/parser.py emits --tensor-split only beside
        # --rpc, and a standalone node's gpu_layers literal is what it wrote.
        **placement_values,
        # The head reads this from its own `node:` block; the group is where it
        # is configured. Without this hop the setting is a silent no-op and the
        # head always uses the client's hardcoded default.
        "worker_wait_timeout_seconds": (
            group.worker_wait_timeout_seconds if group else DEFAULT_WORKER_WAIT_SECONDS
        ),
        # For MCP *clients* (the planner). Excludes the node itself so a target
        # never dials its own endpoint.
        "mcp_endpoints": (
            mcp_endpoints(network_config, configs_dir, exclude_node_id=target.node_id)
            if configs_dir is not None else []
        ),
        # The node's llama-server bearer key. "" for every template that never
        # references ${api_key} — resolve_placeholders only substitutes names
        # that appear, so the empty value is inert there.
        "api_key": api_key,
        # The node's own MCP bearer key, same rule. Distinct from ${api_key}:
        # they protect different ports and rotate independently, and one value
        # serving both would make a leaked llama key an MCP key too.
        "mcp_api_key": mcp_api_key,
    }


def group_device_count(group, network_config: NetworkConfig, configs_dir: Path | None) -> int:
    """How many RPC devices the group's workers expose in total.

    One per worker unless its template's ``model.rpc_server.devices`` names
    several: each exposed device is one RPC device on the head, numbered in
    ``--rpc`` order, and it is this count — not the worker count — that a
    ``tensor_split`` must match. A template that cannot be read counts as one
    device; validate_groups reports the unreadable file itself.
    """
    by_id = {t.node_id: t for t in network_config.ssh.targets}
    total = 0
    for worker in group.workers:
        count = 1
        wtarget = by_id.get(worker)
        if configs_dir is not None and wtarget is not None and wtarget.config_file:
            try:
                wcfg = load_edge_config_cached(configs_dir / wtarget.config_file)
                devices = ((wcfg.get("model") or {}).get("rpc_server") or {}).get("devices") or []
                if isinstance(devices, list) and devices:
                    count = len(devices)
            except Exception:
                count = 1
        total += count
    return total


def group_placement(group, network_config: NetworkConfig, configs_dir: Path | None):
    """The layer layout for one group: ``gpu_layers`` and an integer ``tensor_split``.

    Reads what the policy needs and no more. ``head_layers 0`` with the
    ``even`` policy needs nothing and yields ``[1, ...]`` with 99 — today's
    command line exactly. Head-resident layers need the layer count, taken
    from the GGUF header when the file is in ``<project>/models/`` and from
    the registry entry's ``layers`` hint otherwise; ``balanced`` needs the
    file. Context size, slots and KV cache type come from the head template
    because KV lives on the device that owns the layer and counts toward its
    footprint. Raises ``ValueError`` naming the missing input.
    """
    n_devices = group_device_count(group, network_config, configs_dir)
    if group.head_layers == 0 and group.placement == "even":
        return placement.plan(n_devices=n_devices)

    by_id = {t.node_id: t for t in network_config.ssh.targets}
    head = by_id.get(group.head)
    head_cfg: dict = {}
    if configs_dir is not None and head is not None and head.config_file:
        try:
            head_cfg = load_edge_config_cached(configs_dir / head.config_file) or {}
        except Exception:
            head_cfg = {}
    model = head_cfg.get("model") or {}
    server = model.get("server") or {}
    filename = str(model.get("filename") or "").strip()

    profile = None
    n_layer = None
    source = "none"
    gguf_path = (configs_dir.parent / "models" / filename) if (configs_dir is not None and filename) else None
    if gguf_path is not None and gguf_path.is_file():
        profile = placement.profile_from_gguf(
            gguf_path,
            cache_type_k=str(server.get("cache_type_k") or "f16"),
            cache_type_v=str(server.get("cache_type_v") or "f16"),
        )
    else:
        entry = next((m for m in network_config.model_registry.models if m.filename == filename), None)
        if entry is not None and entry.layers:
            n_layer, source = int(entry.layers), "registry"

    def _int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    try:
        return placement.plan(
            n_devices=n_devices, head_layers=group.head_layers, policy=group.placement,
            profile=profile, n_layer=n_layer, source=source,
            ctx_tokens=_int(server.get("context_size"), 4096),
            n_seq=_int(server.get("parallel_slots"), 1),
        )
    except ValueError as e:
        raise ValueError(
            f"group {group.name!r}: cannot place head_layers={group.head_layers} "
            f"({group.placement}) over {n_devices} RPC device(s): {e}"
        ) from e


def wireguard_node_block(node_id: str, network_config: NetworkConfig) -> dict | None:
    """The ``node.wireguard`` block for ``node_id``, or None outside an enabled group.

    ``address`` carries the prefix length, exactly as wg-quick's ``Address=``
    does, so the node can bind it and the deployer can render it from one value.
    """
    group = network_config.group_for_node(node_id)
    if group is None:
        return None
    addresses = network_config.wireguard_addresses(group)
    if addresses is None:
        return None
    net = network_config.wireguard_network(group)
    return {"interface": WG_INTERFACE, "address": f"{addresses[node_id]}/{net.prefixlen}"}


def _stamp_wireguard(node_block: dict, target: SSHTarget, network_config: NetworkConfig) -> None:
    """Assign or remove ``node.wireguard`` in a resolved config, in place."""
    stamped = wireguard_node_block(target.node_id, network_config)
    if stamped is None:
        if "wireguard" in node_block:
            logger.warning(
                "%s carries a node.wireguard block but %s is not in a group with "
                "wireguard.enabled — removing it, or rpc-server would bind an address this "
                "node does not have. Remove the block from the template.",
                target.config_file, target.node_id,
            )
            del node_block["wireguard"]
        return
    if "wireguard" in node_block and node_block["wireguard"] != stamped:
        logger.warning(
            "%s carries its own node.wireguard block (%r) — overriding it with %r, which is "
            "what the group's tunnel actually assigns. Remove the block from the template.",
            target.config_file, node_block["wireguard"], stamped,
        )
    node_block["wireguard"] = stamped


def _wg_endpoint(host: str, node_id: str, port: int) -> str:
    """``host:port`` for a worker's Endpoint= line, as an IP literal.

    Resolved on the orchestrator, like the firewall allowlist: wg-quick on the
    node would otherwise need a resolver that answers for the head's name at
    every boot, and a name it cannot resolve leaves the tunnel up with no peer
    address. IPv4 first when the name has both, because the tunnel's own
    addresses are IPv4 and the underlay rarely differs; an IPv6 literal is
    bracketed, as wg-quick requires.
    """
    literals = resolve_host_literals(host, node_id)
    chosen = next((a for a in literals if ipaddress.ip_address(a).version == 4), literals[0])
    if ipaddress.ip_address(chosen).version == 6:
        chosen = f"[{chosen}]"
    return f"{chosen}:{port}"


def render_wg_config(node_id: str, network_config: NetworkConfig) -> str:
    """The node's ``/etc/wireguard/wg0.conf``, private key inline.

    A star: on the HEAD one ``[Peer]`` per worker with that worker's /32 and no
    Endpoint (workers initiate, and the head learns each address from the
    worker's authenticated packets); on a WORKER one ``[Peer]`` for the head,
    with the head's physical address as Endpoint and a keepalive so the head
    can always reach it. Workers never peer with each other — all RPC traffic
    is head<->worker, the head relays.

    Reads keys, never mints them: ``aureax generate`` mints every member's pair
    and this raises ValueError naming that command when one is missing.
    Deterministic — no timestamp — because the deployer restarts wg-quick only
    when the installed file differs from this text.
    """
    from orchestrator.config import wgstore

    group = network_config.group_for_node(node_id)
    addresses = network_config.wireguard_addresses(group) if group is not None else None
    if group is None or addresses is None:
        raise ValueError(f"{node_id} is not a member of a node group with wireguard.enabled")
    net = network_config.wireguard_network(group)
    wg = group.wireguard

    def key_of(member: str, private: bool = False) -> str:
        value = wgstore.private_key(member) if private else wgstore.public_key(member)
        if value is None:
            raise ValueError(
                f"{member} has no WireGuard key in {wgstore.store_dir()} — run "
                "`aureax generate --orchestrator-ip <ip>`, which mints one for every member of "
                f"group {group.name!r}"
            )
        return value

    role = "head" if node_id == group.head else "worker"
    lines = [
        f"# {WG_INTERFACE}.conf for {node_id} ({role} of group {group.name}).",
        "# Written by `aureax apply`; edits are overwritten. Carries rpc-server's traffic;",
        '# see ARCHITECTURE.md, "WireGuard transport for rpc-server".',
        "[Interface]",
        f"PrivateKey = {key_of(node_id, private=True)}",
        f"Address = {addresses[node_id]}/{net.prefixlen}",
        f"ListenPort = {wg.listen_port}",
        f"MTU = {wg.mtu}",
    ]
    if node_id == group.head:
        for worker in group.workers:
            lines += [
                "",
                "[Peer]",
                f"# {worker} (worker)",
                f"PublicKey = {key_of(worker)}",
                f"AllowedIPs = {addresses[worker]}/32",
            ]
    else:
        head_host = network_config.host_by_node_id()[group.head]
        lines += [
            "",
            "[Peer]",
            f"# {group.head} (head)",
            f"PublicKey = {key_of(group.head)}",
            f"AllowedIPs = {addresses[group.head]}/32",
            f"Endpoint = {_wg_endpoint(head_host, group.head, wg.listen_port)}",
            f"PersistentKeepalive = {wg.persistent_keepalive}",
        ]
    return "\n".join(lines) + "\n"


def generate_edge_config(
    configs_dir: Path,
    target: SSHTarget,
    network_config: NetworkConfig,
    orchestrator_ip: str,
) -> dict:
    """Generate a resolved edge config for a specific node."""
    if not target.config_file:
        raise ValueError(
            f"Target '{target.node_id}' has no config_file field. "
            "Each target in aureax-network.yaml must specify its own config_file."
        )

    template_path = configs_dir / target.config_file
    if not template_path.exists():
        raise FileNotFoundError(
            f"Config file '{target.config_file}' for target '{target.node_id}' "
            f"not found at {template_path}"
        )

    template_data = load_template_cached(template_path)

    # Run against the PRE-resolution template, _unresolved_placeholders yields
    # the names the template REFERENCES. A key file is created iff the template
    # asks for one — so shard workers (whose template must never carry
    # ${api_key}; they run rpc-server, which takes no key) never grow one, and
    # enabling auth on a node is exactly "add the line, regenerate".
    #
    # The two kinds are minted INDEPENDENTLY off one scan: a head carries both
    # lines, the planner carries neither (it serves no MCP and its llama port
    # is open), and a node with only one of them must not grow the other's file
    # — `aureax nodes` reads key existence per kind and would report auth that
    # nothing enforces.
    refs = _unresolved_placeholders(template_data)
    api_key = ""
    mcp_api_key = ""
    if "api_key" in refs or "mcp_api_key" in refs:
        from orchestrator.config import keystore

        # ensure_key returns a non-empty key or raises RuntimeError; it has no
        # path that yields "", which would ship auth silently disabled.
        if "api_key" in refs:
            api_key = keystore.ensure_key(target.node_id)
        if "mcp_api_key" in refs:
            mcp_api_key = keystore.ensure_key(target.node_id, kind=keystore.KIND_MCP)

    variables = build_variables(
        target, network_config, orchestrator_ip, configs_dir=configs_dir,
        api_key=api_key, mcp_api_key=mcp_api_key,
    )

    resolved = resolve_placeholders(template_data, variables)

    # Keys are substituted by nothing (see _key_placeholders), so one written
    # in key position ships to the node as the literal string. Named on its own
    # rather than folded into the leftover list below, where a KNOWN name would
    # otherwise be reported as unresolved against a list containing it.
    key_names = _key_placeholders(resolved)
    if key_names:
        raise ValueError(
            f"{target.config_file}: placeholder(s) {sorted(key_names)} are written as "
            "mapping KEYS — resolution substitutes values only, so they would ship to "
            "the node verbatim. Put the placeholder on the value side."
        )

    # A name resolve_placeholders doesn't know is left untouched, so a typo ships
    # to the node as the literal string "${rpc_serverz}". Nothing downstream
    # rejects it: the head turns it into a list of one-character RPC peers, waits
    # out the full timeout, and exits — a five-minute detour to learn about a
    # typo. Fail here instead, naming the key and what is actually available.
    leftovers = _unresolved_placeholders(resolved)
    if leftovers:
        raise ValueError(
            f"{target.config_file}: unresolved placeholder(s) {sorted(leftovers)} — "
            f"known names are {sorted(variables)}"
        )

    # TLS is mandatory fleet-wide, so the generator STAMPS the paths rather than
    # asking every template to carry them. A placeholder can be forgotten — the
    # leftover check above catches a typo but nothing catches a template that
    # simply lacks the block — and a node with no tls paths cannot build a
    # llama-server command line at all, which surfaces as one crash-looping pod
    # discovered at deploy time. Injection makes that state unreachable, and it
    # leaves the frozen 122B rollback pair (ARCHITECTURE.md, "A frozen rollback
    # target...") needing no new keys.
    #
    # Assignment, not setdefault: a template that carried its own paths would
    # point the node at files the deployer never installs there. Warned rather
    # than silently accepted, because that template is now lying about what it
    # deploys.
    #
    # Skipped for a non-mapping top level so validate_edge_data below still
    # reports "not a valid edge config" instead of a TypeError from this
    # assignment. Constants, not literals, so the node-side paths have one
    # source of truth with what Deployer.deploy_configs installs.
    if isinstance(resolved, dict):
        from orchestrator.config import tlsstore

        stamped = {
            "ca_file": tlsstore.NODE_CA_FILE,
            "cert_file": tlsstore.NODE_CERT_FILE,
            "key_file": tlsstore.NODE_KEY_FILE,
        }
        if "tls" in resolved and resolved["tls"] != stamped:
            logger.warning(
                "%s carries its own tls: block (%r) — overriding it with %r, "
                "which is where Deployer.deploy_configs actually installs the "
                "certificates. Remove the block from the template.",
                target.config_file, resolved["tls"], stamped,
            )
        resolved["tls"] = stamped

    # The WireGuard block, stamped for the same reasons as tls: — never written
    # by hand, never forgettable. Present exactly on members of a group with
    # `wireguard.enabled`, removed from everything else, so a template carrying
    # a stale block cannot point rpc-server at an address the node does not
    # have. A fleet that never enables the tunnel carries no such block anywhere
    # and this changes nothing in its configs.
    if isinstance(resolved, dict) and isinstance(resolved.get("node"), dict):
        _stamp_wireguard(resolved["node"], target, network_config)

    # The RESOLVED config is what the node actually runs, and nothing between
    # here and the node validates it: it is dumped to disk, SSH-pushed to
    # /etc/aureax/aureax-edge.yaml and read by the client with no Pydantic at
    # all. A wrong type or a missing required key therefore generated cleanly
    # and surfaced minutes later as a crash-looping pod.
    try:
        validate_edge_data(resolved)
    except ValidationError as e:
        # Rendered without pydantic's input values: this config carries
        # model.server.api_key, and the message is logged and returned as an
        # HTTP body by the dashboard's regenerate route.
        raise ValueError(
            f"{target.config_file}: resolved config is not a valid edge config:\n"
            + describe_validation_error(e)
        ) from e

    # A worker's RPC allowlist, consumed on the node by client/rpc_firewall.py
    # as model.rpc_server.allowed_clients. Derived from the group head rather
    # than configured, so a head that moves cannot leave a worker firewalled
    # against it. Written AFTER validate_edge_data above, because it is
    # generator output rather than operator input, and deliberately BEFORE the
    # hash below, so a head address change restages the worker.
    #
    # IP literals, resolved here: the allowlist is applied in the worker's
    # NET_ADMIN init container, which has no mDNS resolver and no cluster DNS
    # for a LAN name, so a hostname-shaped head resolves on the orchestrator or
    # not at all. rpc_firewall.install() refuses a non-literal for the same
    # reason.
    if (resolved.get("node") or {}).get("role") == SHARD_WORKER_ROLE:
        group = network_config.group_for_node(target.node_id)
        if group is None:
            raise ValueError(f"Worker {target.node_id} has no group head for its RPC firewall")
        wg_addresses = network_config.wireguard_addresses(group)
        if wg_addresses is not None:
            # The head's TUNNEL address and nothing else. Its LAN address must
            # not stay here: the point of the tunnel is that the only way to
            # rpc-server is a packet WireGuard authenticated against the head's
            # key, and an allowlisted LAN address would be a second way.
            allowed = [wg_addresses[group.head]]
        else:
            head = network_config.host_by_node_id()[group.head]
            allowed = resolve_host_literals(head, group.head)
        resolved.setdefault("model", {}).setdefault("rpc_server", {})["allowed_clients"] = allowed
    group = network_config.group_for_node(target.node_id)
    if group is not None and group.head == target.node_id:
        # The head's own topology check compares its split against this, not
        # against the peer count: a worker exposing two devices is two slots.
        resolved.setdefault("model", {}).setdefault("server", {})["rpc_device_count"] = (
            group_device_count(group, network_config, configs_dir)
        )

    # Stamp deployment metadata. The hash is over
    # json.dumps(..., sort_keys=True), so every mapping is sorted recursively
    # and key order cannot affect it — only values and LIST order do. It is a
    # "did this file change" marker, not a content address: default=str coerces
    # any non-JSON scalar to its repr, which is lossy.
    # access_service.desired_config_hash/generated_config_hash compare it to
    # drive the Access tab's staged/applied badge, so a node whose template
    # simply omits a `deployment:` block must still get one: unstamped reads as
    # "unknown" there forever. Computed BEFORE the block is added, so adding it
    # never changes the hash of a config that already carried one.
    config_str = json.dumps(resolved, sort_keys=True, default=str)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    resolved.setdefault("deployment", {})
    resolved["deployment"]["deployed_at"] = datetime.now(UTC).isoformat()
    resolved["deployment"]["deployed_by"] = f"aureax-orchestrator@{network_config.network.name}"
    resolved["deployment"]["config_version"] = network_config.network.version
    resolved["deployment"]["config_hash"] = config_hash

    return resolved


def validate_groups(network_config: NetworkConfig, configs_dir: Path) -> list[str]:
    """Check group invariants that live in the per-node YAMLs.

    These can't be expressed in the Pydantic schema because they need to read
    other files. Each one is load-bearing rather than cosmetic:

    - The head must be `edge_executor`. Three separate places filter on that
      literal — Deployer.label_nodes, lifecycle._split_hostnames_by_role and
      setup-all.sh Step 4 — and any other value means the head is never
      labelled and never gets a pod.
    - The head must not set `mcp.server.enabled: false` (absent counts as
      true). Every worker sets it false, so the head is the group's only MCP
      endpoint: the group would serve inference and be invisible to MCP.
    - Workers must be `shard_worker`, which is exactly what makes those same
      passes skip them.
    - A worker's template must not reference `${api_key}`. rpc-server takes no
      key, so generation would mint a key file per worker and ship a bearer
      token into /etc/aureax/aureax-edge.yaml that nothing on the node reads or
      enforces — while the Access tab keeps reporting workers as unsupported.
    - Nor `${mcp_api_key}`, for the mirror reason: a worker sets
      `mcp.server.enabled: false`, so it runs no MCP server for a bearer check
      to protect and nothing on the node ever reads the key.
    - Workers must have `mcp.server.enabled: false`, because parse-config.py
      exports it as NODE_HAS_MCP_SERVER and the shell scripts use it to skip
      MCP health checks against nodes that serve no MCP.
    - The head must offload to the RPC devices (`gpu_layers` > 0) and must still
      carry the `rpc_servers` placeholder, or it runs the whole model locally
      while every worker sits idle looking perfectly healthy.
    - The head must keep the `worker_wait_timeout_seconds` placeholder for the
      same reason: the NodeGroup is where that value is configured, and a
      literal in the template makes the group's setting a silent no-op — while
      tests/test_probe_budget.py, daemonset-edge.yaml's startupProbe budget and
      `aureax status` all keep quoting the group number the node never uses.
    - Every worker must bind the port the head dials.
    - In a group with `wireguard.enabled`, every worker's rpc_server.host must
      be `auto`, the only value that binds the stamped tunnel address.
    - tensor_split, if given, must have one entry per participating device.

    Returns a list of human-readable problems; empty means valid.
    """
    problems: list[str] = []
    if not network_config.node_groups:
        return problems

    by_id = {t.node_id: t for t in network_config.ssh.targets}
    # One problem per FILE, not per node: the shard-worker template is shared by
    # every worker in the fleet, so a 3-worker group would report the same
    # unreadable file three times. Doubles as a negative cache — a file already
    # known to be unreadable is not retried.
    unreadable: set[Path] = set()
    # Same one-problem-per-FILE reasoning for the two worker api_key checks
    # below: six workers share one template in the real fleet. Two sets, not
    # one, so a template carrying BOTH lines reports both — they are two
    # separate edits to undo.
    keyed_worker_templates: set[Path] = set()
    keyed_worker_mcp_templates: set[Path] = set()

    def node_cfg(node_id: str) -> dict | None:
        target = by_id.get(node_id)
        if target is None or not target.config_file:
            return None
        path = configs_dir / target.config_file
        if path in unreadable:
            return None
        try:
            cfg = load_edge_config_cached(path)
        except FileNotFoundError:
            problems.append(f"{node_id}: config_file not found at {path}")
        except Exception as e:
            problems.append(f"{node_id}: could not parse {path.name}: {e}")
        else:
            if isinstance(cfg, dict):
                return cfg
            # A YAML file whose top level is a list or a scalar parses cleanly
            # and the loader hands it straight through, so every `.get()` below
            # would raise AttributeError out of a function generation calls with
            # no try — aborting the whole fleet instead of naming the one bad
            # file. Report it like any other problem.
            problems.append(
                f"{node_id}: {path.name} is not a mapping — a per-node config's top "
                "level must be keys such as `node:` and `model:`"
            )
        unreadable.add(path)
        return None

    for group in network_config.node_groups:
        head_cfg = node_cfg(group.head)
        if head_cfg is not None:
            role = node_role(head_cfg)
            if role != HEAD_ROLE:
                problems.append(
                    f"group {group.name!r}: head {group.head!r} has role {role!r}, "
                    f"expected {HEAD_ROLE!r} — any other value and the node is never "
                    "labelled, so it never gets a pod"
                )
            if not mcp_server_enabled(head_cfg):
                problems.append(
                    f"group {group.name!r}: head {group.head!r} has mcp.server.enabled: false — "
                    "the head is the group's only MCP endpoint"
                )

            head_server = (head_cfg.get("model") or {}).get("server") or {}

            # RPC devices are GPU-class devices. With gpu_layers 0 every layer
            # stays on the head's own CPU: the group loads, /health answers, all
            # probes are green, and the workers sit idle holding nothing. There
            # is no error anywhere — the only symptom is that a model which
            # should not fit somehow does, slowly.
            # int() in a try, not bare: this function is called from generation
            # with no try of its own, so a non-numeric value here would abort the
            # whole fleet's generate with a ValueError instead of reporting one
            # file. Same rule as the worker port below.
            raw_ngl = head_server.get("gpu_layers", 0) or 0
            ngl_is_placeholder = isinstance(raw_ngl, str) and _GPU_LAYERS_PLACEHOLDER in raw_ngl
            if ngl_is_placeholder:
                pass  # resolved per group by build_variables from node_groups.head_layers
            else:
                try:
                    gpu_layers = int(raw_ngl)
                except (TypeError, ValueError):
                    problems.append(
                        f"group {group.name!r}: head {group.head!r} has "
                        f"gpu_layers: {raw_ngl!r} — expected a number or `${{gpu_layers}}`"
                    )
                else:
                    if gpu_layers <= 0:
                        problems.append(
                            f"group {group.name!r}: head {group.head!r} has "
                            f"gpu_layers: {raw_ngl!r} — RPC devices are "
                            "GPU-class devices, so every layer would stay on the head's CPU "
                            "while the workers idle. Set gpu_layers: 99."
                        )

            # The head is the only member that dials out, and it dials what this
            # key resolves to. A template that has lost it generates a head with
            # no --rpc flag at all: an unsharded node quietly running (or failing
            # to run) the whole model alone.
            if "rpc_servers" not in head_server:
                problems.append(
                    f"group {group.name!r}: head {group.head!r} has no "
                    "model.server.rpc_servers — the head would start with no --rpc flag "
                    "and never reach its workers. Restore `rpc_servers: ${rpc_servers}`."
                )
            else:
                # Presence is not enough: the group's `workers:` list is the
                # single source of truth for the peers, and it reaches the head
                # only through this placeholder. An empty literal drops --rpc
                # entirely (client/config/parser.py emits it only for a non-empty
                # list), so the head loads the whole model alone while every
                # worker idles; a hardcoded list is worse, silently addressing
                # machines the group does not name.
                raw_rpc = head_server.get("rpc_servers")
                if not (isinstance(raw_rpc, str) and _RPC_SERVERS_PLACEHOLDER in raw_rpc):
                    problems.append(
                        f"group {group.name!r}: head {group.head!r} hardcodes "
                        f"model.server.rpc_servers: {raw_rpc!r} — the group's workers "
                        f"{list(group.workers)!r} would never be dialled. Restore "
                        "`rpc_servers: ${rpc_servers}`."
                    )

            # The group is the single source of truth for this timeout; it
            # reaches the node ONLY through build_variables resolving the
            # placeholder. A literal here does not error and does not differ
            # visibly — the head just waits its own number while every reader
            # of the group's (the probe-budget test, daemonset-edge.yaml's
            # failureThreshold) sizes itself against the other one.
            head_node = head_cfg.get("node") or {}
            if "worker_wait_timeout_seconds" not in head_node:
                problems.append(
                    f"group {group.name!r}: head {group.head!r} has no "
                    "node.worker_wait_timeout_seconds, so it waits the "
                    f"{DEFAULT_WORKER_WAIT_SECONDS}s client default instead of the group's "
                    f"{group.worker_wait_timeout_seconds}s. Add "
                    "`worker_wait_timeout_seconds: ${worker_wait_timeout_seconds}`."
                )
            else:
                raw_wait = head_node.get("worker_wait_timeout_seconds")
                if not (isinstance(raw_wait, str) and _WORKER_WAIT_PLACEHOLDER in raw_wait):
                    problems.append(
                        f"group {group.name!r}: head {group.head!r} hardcodes "
                        f"node.worker_wait_timeout_seconds: {raw_wait!r} — that makes the "
                        f"group's {group.worker_wait_timeout_seconds}s a no-op. Restore "
                        "`worker_wait_timeout_seconds: ${worker_wait_timeout_seconds}`."
                    )

            split = head_server.get("tensor_split") or []
            split_is_placeholder = isinstance(split, str) and _TENSOR_SPLIT_PLACEHOLDER in split
            expected = group_device_count(group, network_config, configs_dir)
            if split_is_placeholder:
                pass  # resolved per group by build_variables; its length is the device count
            elif split and not isinstance(split, list):
                # Reported rather than skipped: skipping would drop the length
                # check below, which is the whole point of this validator.
                problems.append(
                    f"group {group.name!r}: head {group.head!r} has "
                    f"model.server.tensor_split {split!r} — expected a list with one "
                    "entry per RPC device, or `${tensor_split}`"
                )
            elif split and len(split) != expected:
                # One entry per RPC device: one per worker unless a worker's
                # rpc_server.devices exposes several. The head takes no device
                # slot even when it keeps layers — head-resident layers are what
                # gpu_layers leaves on the CPU, not a device in the split.
                problems.append(
                    f"group {group.name!r}: tensor_split has {len(split)} entries but "
                    f"{expected} device(s) participate"
                )
            # Head-resident layers (or a balanced split) move the layout with
            # the group's settings; a literal in the template cannot follow and
            # would silently place the layers the old way.
            if (group.head_layers > 0 or group.placement != "even") and not (
                split_is_placeholder and ngl_is_placeholder
            ):
                problems.append(
                    f"group {group.name!r}: head_layers {group.head_layers} / placement "
                    f"{group.placement!r} need the head template to carry "
                    "`tensor_split: ${tensor_split}` and `gpu_layers: ${gpu_layers}` so the "
                    "layout follows the group; a literal would place the layers the old way"
                )

        for worker in group.workers:
            wcfg = node_cfg(worker)
            if wcfg is None:
                continue
            role = node_role(wcfg)
            if role != SHARD_WORKER_ROLE:
                problems.append(
                    f"group {group.name!r}: worker {worker!r} has role {role!r}, "
                    f"expected {SHARD_WORKER_ROLE!r} — otherwise it is labelled as a "
                    "normal edge node and gets a llama-server pod with no model"
                )
            if mcp_server_enabled(wcfg):
                problems.append(
                    f"group {group.name!r}: worker {worker!r} must set "
                    "mcp.server.enabled: false (it runs rpc-server, not an MCP server)"
                )

            wtarget = by_id.get(worker)
            wpath = (
                configs_dir / wtarget.config_file
                if wtarget is not None and wtarget.config_file
                else None
            )
            if (
                wpath is not None
                and wpath not in keyed_worker_templates
                and template_references_api_key(wpath)
            ):
                keyed_worker_templates.add(wpath)
                problems.append(
                    f"group {group.name!r}: worker {worker!r} template {wpath.name} "
                    "references ${api_key} — rpc-server takes no key, so this mints a "
                    "key file per worker and ships a bearer token nothing on the node "
                    "reads or enforces"
                )
            if (
                wpath is not None
                and wpath not in keyed_worker_mcp_templates
                and template_references_mcp_api_key(wpath)
            ):
                keyed_worker_mcp_templates.add(wpath)
                problems.append(
                    f"group {group.name!r}: worker {worker!r} template {wpath.name} "
                    f"references {_MCP_API_KEY_PLACEHOLDER} — the worker sets "
                    "mcp.server.enabled: false, so it runs no MCP server and nothing on "
                    "the node would ever read the key"
                )

            # The head dials group.rpc_port; the worker binds this one. A
            # mismatch is not a connection error anyone sees — the head simply
            # waits out worker_wait_timeout_seconds and exits 1, which reads as
            # "the worker never came up".
            # `.get("port") or 50052`, not `.get("port", 50052)`: a present but
            # null `port:` returns None, and the int() is guarded for the same
            # reason as the head's gpu_layers — generation calls this function
            # with no try, so an escaping coercion error aborts the whole fleet
            # instead of reporting a problem.
            rpc_block = (wcfg.get("model") or {}).get("rpc_server") or {}
            raw_port = rpc_block.get("port") or 50052
            try:
                worker_port = int(raw_port)
            except (TypeError, ValueError):
                problems.append(
                    f"group {group.name!r}: worker {worker!r} has "
                    f"model.rpc_server.port: {raw_port!r} — expected a number"
                )
            else:
                if worker_port != group.rpc_port:
                    problems.append(
                        f"group {group.name!r}: worker {worker!r} binds "
                        f"model.rpc_server.port {worker_port} but the head dials "
                        f"rpc_port {group.rpc_port} — the head would wait out its peer "
                        "timeout on a port nothing is listening on"
                    )

            # With the tunnel on, `host: auto` is what binds rpc-server to the
            # stamped tunnel address (client/config/parser.py::resolve_rpc_host).
            # A literal host overrides that and either binds the LAN address the
            # head no longer dials, or one the node does not have.
            if group.wireguard.enabled:
                raw_host = rpc_block.get("host")
                if raw_host is not None and raw_host != "auto":
                    problems.append(
                        f"group {group.name!r}: worker {worker!r} sets "
                        f"model.rpc_server.host: {raw_host!r}, but the group has "
                        "wireguard.enabled — rpc-server must bind its tunnel address, which "
                        "only `host: auto` resolves to. Remove the line or set it to auto."
                    )

    return problems


def validate_planners(network_config: NetworkConfig, configs_dir: Path) -> list[str]:
    """Every MCP CLIENT must DERIVE its remote_servers, not list them.

    The exact mirror of the head's `rpc_servers` rule, and a sibling of
    validate_groups rather than part of it: a fleet with no groups at all can
    still run a planner, and the hazard is identical there.

    ``mcp.client.remote_servers`` reaches a node only through
    ``${mcp_endpoints}`` resolving at generation. A hardcoded list does not
    error and does not differ visibly: the planner dials whatever was typed,
    forever, while every reader of the fleet config believes the list is
    derived. A decommissioned or re-roled node then stays in it silently —
    which is the exact failure :func:`mcp_endpoints` exists to end. It is also
    where the peers' MCP bearer keys arrive, so a hardcoded list means a
    planner that 401s on every call after the first key rotation.

    One problem per FILE, like validate_groups: several planners can share a
    template.

    Returns a list of human-readable problems; empty means valid.
    """
    problems: list[str] = []
    seen: set[Path] = set()

    for target in network_config.ssh.enabled_targets:
        if not target.config_file:
            continue
        path = configs_dir / target.config_file
        if path in seen:
            continue
        try:
            cfg = load_edge_config_cached(path)
        except FileNotFoundError:
            # Reported by validate_groups for group members, and by
            # generate_edge_config for everyone else. Naming it twice would put
            # two lines about one missing file in front of the operator.
            continue
        except Exception:
            # Same reasoning: an unparseable file is already reported by the
            # validator that owns it, and this one cannot tell whether the node
            # is even a planner.
            continue
        if not isinstance(cfg, dict) or not mcp_client_enabled(cfg):
            continue

        seen.add(path)
        client = (cfg.get("mcp") or {}).get("client") or {}
        if "remote_servers" not in client:
            problems.append(
                f"{target.node_id}: {path.name} has mcp.client.enabled: true but no "
                "mcp.client.remote_servers — this node is an MCP client that dials "
                "nothing, and nothing anywhere reports it. Add "
                "`remote_servers: ${mcp_endpoints}`."
            )
            continue
        raw = client.get("remote_servers")
        if not (isinstance(raw, str) and _MCP_ENDPOINTS_PLACEHOLDER in raw):
            problems.append(
                f"{target.node_id}: {path.name} hardcodes mcp.client.remote_servers: "
                f"{raw!r} — the list would never track the fleet, so a decommissioned "
                "or re-roled node stays in it and the planner retries a dead endpoint "
                "forever. Restore `remote_servers: ${mcp_endpoints}`."
            )

    return problems


def generate_all_edge_configs(
    network_config_path: Path,
    output_dir: Path,
    orchestrator_ip: str,
    *,
    accept_worker_reorder: bool = False,
) -> list[Path]:
    """Generate edge configs for all enabled SSH targets and write to output_dir.

    Filenames encode the node (``aureax-edge-<node_id>.yaml``), which is how
    ``Deployer.deploy_configs`` pairs each file with its target — order is not
    part of the contract.

    This is also the ONE place that mints fleet credentials — every node's MCP
    key, the fleet CA and every TLS leaf. ``generate_edge_config`` deliberately
    mints none of the TLS material: the Access tab calls it once per node on a
    20 s poll to compute the desired config hash, and a browser refresh must not
    create the fleet's root of trust.

    ``accept_worker_reorder`` records a reordered ``node_groups`` workers list
    as the new ``--rpc`` order instead of refusing. Reordering renames every RPC
    device, so it is refused by default; see :mod:`orchestrator.config.group_state`.
    """
    # Cached: callers have normally just parsed and validated this same file
    # (Deployer.__init__, the dashboard's regenerate route), and the (path,
    # mtime) key still picks up an operator edit between runs.
    network_config = load_network_config_cached(network_config_path)
    configs_dir = network_config_path.parent
    # Resolved configs may carry model.server.api_key, so the directory and
    # every file are tightened unconditionally — doing it only for keyed nodes
    # would make a file's mode advertise which nodes have auth. mode= covers
    # creation, chmod repairs a directory that already exists (mode= is ignored
    # then, and files written before auth landed are 0644).
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)

    errors = validate_groups(network_config, configs_dir) + validate_planners(
        network_config, configs_dir
    )
    if errors:
        raise ValueError(
            "node_groups validation failed:\n  " + "\n  ".join(errors)
        )

    # Order check second: it compares against a baseline on THIS machine, so it
    # is only meaningful once the config itself is internally consistent.
    # order_problems never writes and never raises.
    from orchestrator.config import group_state

    problems, order_warnings = group_state.order_problems(network_config_path, network_config)
    for warning in order_warnings:
        logger.warning("%s", warning)
    if problems and not accept_worker_reorder:
        raise ValueError(
            "node_groups worker order changed since the last generate:\n" + "\n".join(problems)
        )

    # Pre-mint every MCP key BEFORE any target is resolved. ${mcp_endpoints}
    # copies each peer's key into the planner's config, and the target loop
    # below walks the fleet file in order — so a planner listed before a head
    # would ship a remote_servers entry with no key for that head on the very
    # first generate, and 401 on every call until something else happened to
    # regenerate it. Llama keys need no such pass: nothing but the node itself
    # ever reads one.
    from orchestrator.config import keystore

    for target in network_config.ssh.enabled_targets:
        if target.config_file and template_references_mcp_api_key(
            configs_dir / target.config_file
        ):
            keystore.ensure_key(target.node_id, kind=keystore.KIND_MCP)

    # WireGuard pairs, in the same "before any target is resolved" slot and for
    # the same kind of reason: a node's wg0.conf carries its PEERS' public keys
    # (every worker's on the head, the head's on each worker), so a member
    # rendered before its peers were minted would ship a tunnel with a missing
    # peer. Only members of a group with wireguard.enabled get a pair — a fleet
    # that never enables the tunnel never grows ~/.aureax/wireguard at all.
    # ensure_key never replaces an existing key; `aureax wg regenerate` does.
    if any(g.wireguard.enabled for g in network_config.node_groups):
        from orchestrator.config import wgstore

        for group in network_config.node_groups:
            if group.wireguard.enabled:
                for node_id in group.members:
                    wgstore.ensure_key(node_id)

    # TLS material, for the same "before any config is written" reason: the
    # deployer refuses to push a node whose leaf is missing, and a generate that
    # produced configs but no certificates would be a fleet that cannot start.
    # ensure_* are idempotent — a healthy pair is left byte-identical, so this
    # costs nothing on the repeat runs every deploy and apply do.
    from orchestrator.config import tlsstore

    permitted_ips, permitted_dns = tlsstore.fleet_name_constraints(network_config, orchestrator_ip)
    tlsstore.ensure_ca(permitted_ips=permitted_ips, permitted_dns=permitted_dns)
    # The constraints are baked at CA creation and nothing rotates the CA
    # silently, so a fleet that has since gained an address outside them is a
    # real possibility. Check every SAN this run would mint BEFORE minting any:
    # a verifier rejects a certificate outright when ONE of its SANs is outside
    # the permitted subtrees, so minting anyway would hand out leaves that work
    # for none of their names — and it would do it to half the fleet before the
    # first failure was visible.
    ca_cert = tlsstore.load_ca_cert()
    planned = [(tlsstore.ORCHESTRATOR_NAME, tlsstore.orchestrator_leaf_sans(orchestrator_ip))]
    planned += [(t.node_id, tlsstore.node_sans(t)) for t in network_config.ssh.enabled_targets]
    forbidden = [
        f"{name}: {violation}"
        for name, sans in planned
        for violation in tlsstore.san_constraint_violations(sans, ca_cert)
    ]
    if forbidden:
        raise ValueError(
            "the fleet CA's name constraints do not cover this fleet:\n  "
            + "\n  ".join(forbidden)
            + f"\nThe CA is minted once and never rotated automatically. Run `aureax tls "
            f"regenerate --ca --orchestrator-ip {orchestrator_ip}` to mint one that covers "
            "today's fleet, then `aureax apply --all` and re-trust the CA in every browser."
        )

    tlsstore.ensure_orchestrator_leaf(orchestrator_ip)
    for target in network_config.ssh.enabled_targets:
        tlsstore.ensure_node_leaf(target)

    output_paths = []
    for target in network_config.ssh.enabled_targets:
        resolved = generate_edge_config(
            configs_dir, target, network_config, orchestrator_ip
        )

        output_path = generated_config_path(target.node_id, output_dir)
        # Through write_private_file rather than a plain open(): this file is
        # read concurrently — the Access tab polls it while the dashboard's own
        # regenerate route rewrites it — and a truncated YAML prefix is usually
        # still VALID YAML, so an in-place truncate hands a reader a silently
        # partial config instead of an error. The helper's rename makes the
        # partial state unreachable, its fsync makes the published bytes
        # durable, and it creates at 0600 from the first byte, which matters
        # because the whole api_key is written before any later chmod could
        # land. Dumped into memory first for the same reason: the helper takes
        # bytes, and a dump that failed partway must not reach the real name,
        # which cache.load_node_config_resolved prefers over the template for
        # every later read.
        buf = io.StringIO()
        yaml.dump(resolved, buf)
        write_private_file(output_path, buf.getvalue())

        output_paths.append(output_path)

    # Only now. A baseline recorded for an order that never reached a node makes
    # the retry after a half-failed generate look like a legitimate second run,
    # which hides the reorder permanently.
    group_state.write_baseline(network_config_path, network_config)

    return output_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m orchestrator.config.templating "
            "<network.yaml> <orchestrator_ip>"
        )
        sys.exit(1)

    network_path = Path(sys.argv[1])
    orch_ip = sys.argv[2]

    output_dir = Path("build/configs")
    paths = generate_all_edge_configs(network_path, output_dir, orch_ip)

    print(f"Generated {len(paths)} edge configs in {output_dir}/:")
    for p in paths:
        print(f"  - {p.name}")
