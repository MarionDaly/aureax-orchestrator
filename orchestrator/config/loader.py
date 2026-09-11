"""Load and validate Aureax YAML configuration files."""

from __future__ import annotations

import ipaddress
import re
from io import StringIO
from pathlib import Path

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from orchestrator.config.schema import EdgeConfig, NetworkConfig, to_plain_dict

# Where the network config lives relative to the project root — the default
# every entry point falls back to when the operator names no path (the CLI's
# --config, the dashboard app factory, lifecycle's helpers that take an
# optional path, and config_api's no-lifecycle-manager fallback, which resolves
# it against its own configs/ instead of the CWD). One spelling, so the four
# could not drift apart.
DEFAULT_NETWORK_CONFIG = Path("configs/aureax-network.yaml")

# The tracked, scrubbed copy an operator starts from. DEFAULT_NETWORK_CONFIG
# itself is gitignored because it carries SSH credentials and real addresses,
# so a fresh clone has only this file until the operator copies it over.
EXAMPLE_NETWORK_CONFIG = Path("configs/aureax-network.example.yaml")


def generated_config_dir() -> Path:
    """Where `aureax generate` writes the resolved per-node configs.

    Resolved per call, not at import: tests redirect ``$HOME``, and a
    module-level constant would freeze whatever it was at import time.

    The directory is 0700 and the files inside it are 0600 — a resolved config
    carries the node's ``model.server.api_key`` — so everything that reads or
    writes one comes through here rather than re-spelling the path.
    """
    return Path.home() / ".aureax" / "configs"


def generated_config_path(node_id: str, directory: Path | None = None) -> Path:
    """The generated config file for one node.

    The filename encodes the node (``aureax-edge-<node_id>.yaml``), which is
    what ``Deployer.deploy_configs`` pairs paths to targets by — position
    matching silently hands a node another node's config. ``directory``
    defaults to :func:`generated_config_dir`; generation passes its own output
    directory so the name is built in exactly one place either way.
    """
    return (generated_config_dir() if directory is None else directory) / f"aureax-edge-{node_id}.yaml"


class ConfigError(ValueError):
    """Raised when a YAML config fails to load or validate.

    Subclasses ``ValueError`` so existing handlers that catch broader exception
    families (e.g. dashboard config_api) continue to work.
    """


def _new_yaml() -> YAML:
    """A fresh ruamel parser, one per load.

    ruamel's ``YAML`` is not reentrant: ``load()`` stores the stream and its
    per-document state on the instance, so two concurrent loads through one
    instance corrupt each other. A shared module-level parser is therefore a
    race, and the dashboard reaches this module from both sides at once — the
    sync config routes run in Starlette's threadpool while an inline
    ``validate_yaml_text`` runs on the event loop. The symptom is silent: the
    resulting parse error is caught upstream and reported as "no configs",
    naming nothing. Construction is cheap beside a parse, and the mtime-keyed
    caches in :mod:`orchestrator.config.cache` already collapse repeat reads.

    ``preserve_quotes`` matters because generation dumps what it parses: without
    it ``id: "${node_id}"`` is restyled to ``id: ${node_id}`` on every write.
    """
    parser = YAML()
    parser.preserve_quotes = True
    return parser


# One label of a DNS name: no leading/trailing hyphen, 1-63 chars.
_HOSTNAME_RE = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$")


def validate_orchestrator_host(value: str) -> str:
    """Return ``value`` if it could name the orchestrator; raise ValueError if not.

    Templating substitutes this into every node's telemetry and logs endpoint
    (``${orchestrator_url}/api/v1/telemetry``, which resolves to
    ``https://<value>:7001/api/v1/telemetry``), so a typo like
    ``192.168.1..42`` regenerates and pushes the whole fleet with an endpoint
    that never resolves — visible only as status printing "(no metrics
    received yet)".

    Hostnames stay legal: the schema types the endpoint as a bare str and an
    mDNS name is a plausible operator input. Only clearly malformed values are
    refused.

    The value comes back verbatim, because the CLI and the dashboard both echo
    it to the operator. Shaping it for a URL authority — bracketing an IPv6
    literal — belongs to ``templating._url_host``, which is where the URL is
    built.

    Lives here rather than in cli.py because both entry points that accept this
    value must agree: `aureax generate/deploy/apply` via the CLI callback, and
    the dashboard's POST /api/config/regenerate. Split, they diverge on exactly
    the input the fleet uses — a fleet named by a hostname regenerating from the
    CLI while the Apply button answers 400.
    """
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass
    if not _HOSTNAME_RE.match(value):
        raise ValueError(f"{value!r} is neither an IP address nor a hostname")
    return value


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    # Explicit utf-8, never the locale default: every shipped config carries
    # non-ASCII comment text, and an ASCII locale (LC_ALL=C in a cron job, a
    # systemd unit or a minimal image) would turn the first read of the fleet
    # file into a UnicodeDecodeError naming a byte offset.
    with open(path, encoding="utf-8") as f:
        data = _new_yaml().load(f)
    if data is None:
        return {}
    # ruamel returns CommentedMap; convert to plain dict for Pydantic
    return to_plain_dict(data)


def load_raw_text(path: Path | str) -> str:
    """Return a YAML file's raw text unchanged (preserves comments & formatting)."""
    return Path(path).read_text(encoding="utf-8")


def validate_edge_data(data: object) -> EdgeConfig:
    """Validate already-parsed edge-config data against ``EdgeConfig``.

    The shared template, per-node configs and the planner config all use
    ``${...}`` placeholders that are resolved at deployment time, so
    type-correct stand-ins are substituted first and an unresolved file
    validates the same as a resolved one. Substitution is a no-op on a resolved
    config, which is why the generator can validate its own output with this.

    The planner belongs here too: aureax-planner.yaml carries ``${tags}`` and
    ``${mcp_endpoints}``, and validating it raw failed on types that are
    correct once resolved — the dashboard editor could load the file but never
    save it.

    Raises ``pydantic.ValidationError``; each caller shapes that into its own
    error type.
    """
    from orchestrator.config.templating import PLACEHOLDER_DUMMIES, resolve_placeholders

    return EdgeConfig.model_validate(resolve_placeholders(data, PLACEHOLDER_DUMMIES))


def validate_yaml_text(text: str, kind: str) -> tuple[bool, list[dict]]:
    """Validate raw YAML text against the schema for ``kind``.

    kind: "network" | "edge-template" | "edge-node" | "planner"

    Returns ``(ok, errors)`` where errors is a list of ``{"loc": str, "msg": str}``.
    Edge templates are validated after substituting dummy placeholder values so
    ``${node_id}`` etc. don't trigger spurious type errors.

    Anything else reports ok with no errors — that means "no schema for this
    file", not "valid": the dispatch table below is the closed set, so a kind
    the dashboard starts passing must be added there too or every save of that
    file becomes a silent rubber stamp.
    """
    try:
        parsed = _new_yaml().load(StringIO(text))
    except YAMLError as e:
        return False, [{"loc": "<yaml>", "msg": str(e)}]

    data = to_plain_dict(parsed) if parsed else {}

    validate = {
        "network": NetworkConfig.model_validate,
        "edge-template": validate_edge_data,
        "edge-node": validate_edge_data,
        "planner": validate_edge_data,
    }.get(kind)
    if validate is None:
        # No schema for this kind — see the docstring; adding a kind to the
        # dashboard means adding it to the dict above.
        return True, []

    try:
        validate(data)
        return True, []
    except ValidationError as e:
        return False, [
            {"loc": ".".join(str(x) for x in err["loc"]), "msg": err["msg"]}
            for err in e.errors()
        ]


def describe_validation_error(e: ValidationError) -> str:
    """Render a pydantic error as ``loc: msg [type]`` lines, WITHOUT input values.

    Pydantic renders ``input_value=`` for the failing field's parent object, and
    these messages travel: into ConfigError, into the dashboard's persisted
    system log and into its HTTP 400 bodies. A network config carries
    ``ssh.targets[].password`` (a short dict, so nothing is truncated) and a
    resolved edge config carries ``model.server.api_key``. loc + msg + type is
    the same shape ``validate_yaml_text`` already returns, and keeps the message
    diagnostic — it names the field and what is wrong with it, just not its
    value.
    """
    return "\n".join(
        f"  {'.'.join(str(x) for x in err['loc']) or '<root>'}: {err['msg']} [{err['type']}]"
        for err in e.errors(include_url=False, include_input=False)
    )


def load_network_config(path: Path | str) -> NetworkConfig:
    """Load and validate an aureax-network.yaml file.

    Every failure mode is a ``ConfigError``: callers (cli.py, the dashboard's
    config_api) handle that one type, and a raw ruamel parse error escaping here
    surfaced as a traceback or an unhandled 500.
    """
    path = Path(path)
    try:
        return NetworkConfig.model_validate(load_yaml(path))
    except ValidationError as e:
        raise ConfigError(
            f"Invalid network config at {path}:\n{describe_validation_error(e)}"
        ) from e
    except (YAMLError, OSError) as e:
        hint = ""
        if isinstance(e, FileNotFoundError) and path.resolve() == DEFAULT_NETWORK_CONFIG.resolve():
            # The one path a fresh clone is guaranteed to lack: the fleet
            # definition is operator-owned and never shipped.
            hint = (
                f"\nNo fleet definition yet. Create one from the shipped example and edit it:\n"
                f"    cp {EXAMPLE_NETWORK_CONFIG} {DEFAULT_NETWORK_CONFIG}\n"
                "Both paths are relative to the project root, so run this from the "
                "checkout — or pass --config with the path to your own copy."
            )
        raise ConfigError(f"Could not read {path}: {e}{hint}") from e


def load_edge_config(path: Path | str) -> EdgeConfig:
    """Load and validate an aureax-edge.yaml file.

    Templates and per-node configs use ${...} placeholders that are
    resolved at deployment time; ``validate_edge_data`` substitutes dummy
    values so structural validation succeeds regardless of which form the
    file is in.
    """
    path = Path(path)
    try:
        return validate_edge_data(load_yaml(path))
    except ValidationError as e:
        raise ConfigError(
            f"Invalid edge config at {path}:\n{describe_validation_error(e)}"
        ) from e
    except (YAMLError, OSError) as e:
        raise ConfigError(f"Could not read {path}: {e}") from e


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m orchestrator.config.loader <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    data = load_yaml(config_path)

    # Auto-detect config type by top-level keys
    if "ssh" in data and "kubernetes" in data:
        config = load_network_config(config_path)
        print(f"Network config loaded: {config.network.name} v{config.network.version}")
        print(f"  SSH targets: {len(config.ssh.targets)}")
        for target in config.ssh.targets:
            print(f"    - {target.node_id} @ {target.host}")
        print(f"  Models: {len(config.model_registry.models)}")
        for m in config.model_registry.models:
            print(f"    - {m.name} ({m.quantization}, {m.size_gb}GB)")
    elif "node" in data and "model" in data:
        config = load_edge_config(config_path)
        print(f"Edge config loaded: {config.node.id} ({config.node.role})")
        print(f"  Model: {config.model.name} ({config.model.quantization})")
        print(f"  MCP tools: {len(config.mcp.server.tools)}")
        for tool in config.mcp.server.tools:
            print(f"    - {tool.name}")
    else:
        print(f"Unknown config type in {config_path}")
        sys.exit(1)
