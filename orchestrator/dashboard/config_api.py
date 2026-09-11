"""Dashboard API routes for viewing, editing, and regenerating config files."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from orchestrator.config.atomic import write_private_file
from orchestrator.config.loader import (
    DEFAULT_NETWORK_CONFIG,
    generated_config_dir,
    load_raw_text,
    validate_orchestrator_host,
    validate_yaml_text,
)
from orchestrator.dashboard._http_utils import emit, network_config_path, read_json
from orchestrator.dashboard.config_secrets import REDACTION_MARKER, redact, restore
from orchestrator.resources import resource_root

router = APIRouter()

# The project root as ``resource_root`` finds it: the process CWD when that is
# itself a checkout or a workspace, else the tree this package lives in, else
# the configs bundled into an installed orchestrator. Resolved once at import,
# and it is both the directory this tab LISTS and the one a PUT WRITES into —
# so a dashboard started from a second checkout edits that tree's templates,
# and an installed deployment edits orchestrator/_resources/configs.
CONFIGS_DIR = resource_root() / "configs"

# Fixed-name entries, each listed when the file is present.
#
# The example is here because aureax-network.yaml is untracked: on a fresh
# clone it is the only file describing a fleet, and the one a new operator has
# to copy and edit. Without an entry it is neither listable nor openable, and
# _edge_node_names returns an empty set with no network config to read — so the
# Config tab would show two templates and nothing else. It carries no literal
# `password:` value, so the "network" kind's redaction is a no-op on it.
STATIC_ENTRIES: dict[str, str] = {
    "aureax-network.yaml":         "network",
    "aureax-network.example.yaml": "network",
    "aureax-edge.yaml":            "edge-template",
    "aureax-planner.yaml":         "planner",
}


@router.get("/api/config")
def list_configs(request: Request) -> list[dict]:
    """List all known config files — source (editable) and generated (view-only)."""
    entries: list[dict] = []

    # Fixed source files
    for name, kind in STATIC_ENTRIES.items():
        path = CONFIGS_DIR / name
        if path.exists():
            # The shipped example opens but never saves in place (see _resolve).
            entries.append(_entry(name, kind, path, editable=not name.endswith(".example.yaml")))

    # Per-node source files, derived from SSH targets in network config.
    # Skip any name that already appears in STATIC_ENTRIES (e.g. the planner
    # YAML, which is both an SSH target's config_file AND a static entry —
    # we keep the static entry's "planner" kind so the UI groups it correctly).
    for name in _edge_node_names(request):
        if name in STATIC_ENTRIES:
            continue
        path = CONFIGS_DIR / name
        if path.exists():
            entries.append(_entry(name, "edge-node", path, editable=True))

    # Generated per-node configs under ~/.aureax/configs/
    generated = generated_config_dir()
    if generated.exists():
        entries.extend(
            _entry(path.name, "generated", path, editable=False)
            for path in sorted(generated.glob("aureax-edge-*.yaml"))
        )

    return entries


@router.get("/api/config/{name}")
def read_config(name: str, request: Request) -> dict:
    kind, path, editable = _resolve(name, request)
    if not path.exists():
        raise HTTPException(404, f"not found: {name}")
    content = load_raw_text(path)
    content, redacted = redact(content)
    return {
        "name": name,
        "kind": kind,
        "editable": editable,
        "content": content,
        "redacted": redacted,
    }


# One save at a time. ``restore`` re-binds each redaction marker against the
# text on disk, so two concurrent saves of one file could bind against a copy
# the other has already replaced. The work runs on a worker thread, which is
# why the serialisation has to be explicit.
_save_lock = threading.Lock()


def _save_serialised(name: str, request: Request, content: str) -> tuple[list[str], int, int]:
    """The blocking half of a save: an empty error list means it was written.

    Everything here parses YAML or touches the disk: ``_resolve`` loads and
    validates the whole network config, ``restore`` parses the submitted text
    and the on-disk original, ``validate_yaml_text`` parses and validates once
    more, and then the file is written. Same reason ``regenerate`` goes off the
    loop — while this runs, nothing else on the loop does, including the SSE
    consoles and every node's telemetry POST.
    """
    with _save_lock:
        kind, path, editable = _resolve(name, request)
        if not editable:
            raise HTTPException(403, "config is view-only")

        # Before validation, so what is validated is what is written.
        if REDACTION_MARKER in content:
            content = restore(content, load_raw_text(path))

        ok, errors = validate_yaml_text(content, kind)
        if not ok:
            return errors, 0, 0

        before = path.stat().st_size if path.exists() else 0
        after = len(content.encode())
        # The fleet file can carry an SSH password, so it stays 0600; a
        # template is git-tracked source and keeps the 0644 the checkout gave
        # it (the Access tab's toggle writes templates the same way).
        write_private_file(path, content, mode=0o600 if kind == "network" else 0o644, dir_mode=None)
        return [], before, after


@router.put("/api/config/{name}")
async def save_config(name: str, request: Request) -> JSONResponse:
    body = await read_json(request)
    content = body.get("content", "")
    if not isinstance(content, str):
        raise HTTPException(400, "'content' must be a string")

    errors, before, after = await asyncio.to_thread(_save_serialised, name, request, content)
    if errors:
        emit(request, "warn", f"Config rejected: {name} ({len(errors)} error(s))",
             source="config")
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)

    # Back on the loop, as in regenerate: the log write is in-process state.
    emit(request, "info", f"Config saved: {name} ({before} → {after} bytes)", source="config")
    return JSONResponse({"ok": True})


# Serialize multi-file generations; individual files are already written atomically.
_generate_lock = threading.Lock()


def _generate_serialised(
    network_cfg: Path, ip: str, accept_worker_reorder: bool = False
) -> list[Path]:
    from orchestrator.config.templating import generate_all_edge_configs

    with _generate_lock:
        output_dir = generated_config_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        return generate_all_edge_configs(
            network_cfg, output_dir, ip, accept_worker_reorder=accept_worker_reorder
        )


@router.post("/api/config/regenerate")
async def regenerate(request: Request) -> dict:
    """Regenerate per-node configs into ~/.aureax/configs/ using the templates."""
    body = await read_json(request)
    raw_ip = body.get("orchestrator_ip")
    if not isinstance(raw_ip, str):
        raise HTTPException(400, "orchestrator_ip must be a string")
    ip = raw_ip.strip()
    if not ip:
        raise HTTPException(400, "orchestrator_ip required")
    try:
        # This value is written verbatim into every node's telemetry and log
        # endpoint, so it is where the whole fleet ships to after the next
        # Apply. Nothing downstream validates it.
        #
        # The SAME rule the CLI's --orchestrator-ip applies, from the same
        # function, so an address legal in `aureax generate` is legal here —
        # a hostname included, since a fleet may name its orchestrator by mDNS.
        validate_orchestrator_host(ip)
    except ValueError as e:
        raise HTTPException(400, f"orchestrator_ip is unusable: {e}") from e

    # The override for a REORDERED worker list. Generation refuses one by
    # default: `--rpc` order defines RPC0..n, which `tensor_split` and
    # `devices` index positionally, so swapping two workers silently
    # re-addresses every entry. Off unless the caller says otherwise, and there
    # is no checkbox for it — the 400 below carries the refusal's own message,
    # which names the CLI flag and shows the before/after order.
    accept_worker_reorder = body.get("accept_worker_reorder", False)
    if not isinstance(accept_worker_reorder, bool):
        raise HTTPException(400, "'accept_worker_reorder' must be a boolean")

    network_cfg = _network_config_path(request)
    # generate_all_edge_configs raises on the things this endpoint exists to
    # catch: an unresolved ${placeholder}, a missing per-node config file, a
    # group referencing an unknown node, a reordered worker list. Its own
    # message names the file and the placeholder (or the flag), so it is
    # returned as a 400 — unhandled it would be a 500 with a stack trace, and
    # the dashboard would show only "Regenerate failed: 500".
    #
    # RuntimeError is in the tuple for the TLS store and the keystore: a half
    # present CA pair and a leaf that could not be minted both raise one whose
    # message names the remedy (`aureax tls regenerate --ca`). Same tuple the
    # CLI's `generate` catches, for the same reason.
    try:
        # Off the loop: one parse + placeholder resolution + validation + 0600
        # write per enabled target, ~375 ms for 9 nodes, during which nothing
        # else runs — including the SSE streams and the node ingest routes.
        paths = await asyncio.to_thread(
            _generate_serialised, network_cfg, ip, accept_worker_reorder
        )
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        emit(request, "error", f"Config regeneration failed: {type(e).__name__}: {e}",
             source="config")
        raise HTTPException(400, f"{type(e).__name__}: {e}") from e
    emit(request, "info", f"Regenerated {len(paths)} edge config(s) with orchestrator_ip={ip}",
         source="config")
    return {"generated": [str(p) for p in paths]}


def _entry(name: str, kind: str, path: Path, *, editable: bool) -> dict:
    stat = path.stat()
    return {
        "name": name,
        "path": str(path),
        "kind": kind,
        "size": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        "editable": editable,
    }


def _resolve(name: str, request: Request) -> tuple[str, Path, bool]:
    """Resolve a requested config name to (kind, absolute path, editable).

    Prevents directory traversal: only names matching the whitelist are
    accepted, and the final path must live under configs/ or the generated-
    config directory.
    """
    if name in STATIC_ENTRIES:
        path = (CONFIGS_DIR / name).resolve()
        if path.parent == CONFIGS_DIR:
            # The shipped example is listed so a fresh clone has something to
            # open and copy, never something to edit in place: it is
            # git-tracked, and a PUT on it from this unauthenticated route
            # would rewrite the repository.
            return STATIC_ENTRIES[name], path, not name.endswith(".example.yaml")
    if name in _edge_node_names(request):
        path = (CONFIGS_DIR / name).resolve()
        if path.parent == CONFIGS_DIR:
            return "edge-node", path, True
    if name.startswith("aureax-edge-") and name.endswith(".yaml"):
        generated = generated_config_dir()
        gen_root = generated.resolve() if generated.exists() else None
        if gen_root is not None:
            path = (gen_root / name).resolve()
            if path.parent == gen_root and path.exists():
                return "generated", path, False
    raise HTTPException(404, f"not found: {name}")


def _edge_node_names(request: Request) -> set[str]:
    """Set of per-node source YAML filenames from the network config's SSH targets.

    Returns an empty set on any failure (file missing, parse error, IO error
    from fd exhaustion, etc.) so callers like ``_resolve`` and ``list_configs``
    can degrade gracefully — a transient config-load failure shouldn't turn a
    valid 404 into a 500.
    """
    from orchestrator.config.loader import ConfigError, load_network_config

    try:
        cfg = load_network_config(str(_network_config_path(request)))
    except ConfigError:
        # One type is enough: the loader normalises every failure it can have
        # — parse (ruamel YAMLError), IO and validation — into ConfigError.
        return set()
    return {t.config_file for t in cfg.ssh.targets if t.config_file}


def _network_config_path(request: Request) -> Path:
    """This router's view of the network config.

    Falls back to ``configs/aureax-network.yaml`` beside this package rather
    than 503ing like the other routers: the Config tab lists and validates
    files whether or not the lifecycle manager came up, and _edge_node_names
    degrades to an empty set on any read failure below.
    """
    return network_config_path(request, fallback=CONFIGS_DIR / DEFAULT_NETWORK_CONFIG.name)
