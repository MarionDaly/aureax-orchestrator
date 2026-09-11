"""Dashboard API routes for the Access tab — per-node endpoints + API-key auth.

Security invariants (pinned by tests/test_access_api.py):
- No list/poll payload ever contains a plaintext key. Exactly FOUR routes
  return one — the current and previous key of each of the node's two keyed
  services: ``GET /api/access/{node_id}/key``,
  ``GET /api/access/{node_id}/key/previous`` (llama-server), and
  ``GET /api/access/{node_id}/mcp-key``,
  ``GET /api/access/{node_id}/mcp-key/previous`` (the MCP server). A .prev key
  is a live credential until the rotate is applied. All four are explicit user
  actions, and all four write an audit line naming the node and the caller —
  that line is the only accountability there is, since the dashboard itself is
  unauthenticated.
- Auth is toggled by editing the node's TEMPLATE (the ``api_key: ${api_key}``
  line under ``model.server``, or ``api_key: ${mcp_api_key}`` under
  ``mcp.server``) through a ruamel round-trip + the same validation the config
  editor uses — the git-tracked template stays the single source of truth,
  whether it was edited here or by hand.
- Apply delegates to ``Deployer.apply_to_nodes`` (push + targeted pod
  restart) in a worker thread, interlocked 409-both-ways with lifecycle
  start/stop.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.representer import RoundTripRepresenter
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from orchestrator.config import keystore
from orchestrator.config.atomic import write_private_file
from orchestrator.config.cache import load_network_config_cached
from orchestrator.config.loader import validate_yaml_text
from orchestrator.config.schema import SHARD_WORKER_ROLE
from orchestrator.config.templating import (
    template_references_api_key,
    template_references_mcp_api_key,
    template_references_mcp_endpoints,
)
from orchestrator.dashboard import access_service
from orchestrator.dashboard._http_utils import (
    emit,
    network_config_path,
    read_json,
    target_for,
)
from orchestrator.dashboard.config_api import STATIC_ENTRIES

logger = logging.getLogger("aureax.dashboard.access")

router = APIRouter()

# The toggle rewrites a git-tracked template in full, so the emitter is
# configured to reproduce this repo's YAML layout byte for byte. A default
# YAML() reflows ~190 lines of an untouched head template on every click:
#
# - indent(): ruamel's default puts block sequences at the parent's own column.
#   Every list in configs/ is indented one further with the dash offset two in,
#   so without this each toggle dedents mcp.server.tools, the telemetry collect
#   lists and process.managed_services by two columns.
# - width: the default is 80, at which the emitter folds long plain scalars
#   onto a second line — the fleet file's long comments and descriptions.
# - _KeepNull: round-tripped None is emitted as an empty value, so an explicit
#   `deployed_at: null` comes back as `deployed_at:`. A SUBCLASS, because
#   add_representer mutates the class it is called on and RoundTripRepresenter
#   is shared with generation's YAML — registering there would change every
#   config the deployer writes.
class _KeepNullRepresenter(RoundTripRepresenter):
    pass


_KeepNullRepresenter.add_representer(
    type(None),
    lambda self, data: self.represent_scalar("tag:yaml.org,2002:null", "null"),
)

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.default_flow_style = False
_yaml.indent(mapping=2, sequence=4, offset=2)
_yaml.width = 4096
_yaml.Representer = _KeepNullRepresenter

# Detected-IP cache: generation needs the orchestrator IP and detection shells
# out. Both read routes detect, and the tab polls /api/access/live every 20 s
# (static/sections/access.js), so the TTL turns ~3 detections a minute into one.
#
# -inf, not 0.0: time.monotonic()'s reference is boot, so a 0.0 stamp reads as
# fresh for the first 60 s of uptime and the sentinel None is served as if it
# had been detected — every badge "unknown" and every Apply a 400.
_IP_CACHE_EMPTY: tuple[float, str | None] = (float("-inf"), None)
_ip_cache: tuple[float, str | None] = _IP_CACHE_EMPTY
_IP_TTL_S = 60.0


def _orchestrator_ip(cfg=None) -> str | None:
    global _ip_cache
    ts, ip = _ip_cache
    if time.monotonic() - ts < _IP_TTL_S:
        return ip
    from orchestrator.lifecycle import detect_mac_lan_ip

    try:
        ip = detect_mac_lan_ip(cfg)
    except Exception as e:
        logger.warning("orchestrator IP detection failed: %s", e)
        ip = None
    # Only a config-backed reading is cached. Without the config,
    # detect_mac_lan_ip has no node route to probe and answers with the DEFAULT
    # route, which on a multi-homed host is not the address the fleet reaches —
    # and the apply route reads this same cache with no bust, so caching that
    # answer would bake it into every pushed config's telemetry endpoint for
    # the rest of the TTL.
    if cfg is not None:
        _ip_cache = (time.monotonic(), ip)
    return ip


def _detected_ip(config_path: Path) -> str | None:
    """The orchestrator IP as the FLEET reaches it. Blocking (shells out).

    Always detected against a real node route: without the config, mac_lan_ip
    picks the DEFAULT route, and on a Mac whose nodes are not on that interface
    the two answers differ. The IP is baked into every generated config
    (telemetry endpoint) and therefore into desired_config_hash, so a read route
    detecting differently from the apply leaves every badge reading "Saved but
    not applied" for a fleet that is in sync.
    """
    cfg = None
    try:
        cfg = load_network_config_cached(config_path)
    except Exception as e:
        logger.warning("IP detection is falling back to the default route: %s", e)
    return _orchestrator_ip(cfg)


# What each key kind means to the routes that stage it: which section of the
# template carries the line, which placeholder it carries, how the audit line
# names it, and which predicate reports it as on. One table rather than a
# branch per route, because the toggle, the rotate and their two audit lines
# have to agree on all four — a rotate gated on the llama line while the toggle
# writes the MCP one is a 409 the operator cannot act on.
_KINDS: dict[str, dict] = {
    keystore.KIND_LLAMA: {
        "section": ("model", "server"),
        "placeholder": "${api_key}",
        "label": "API key",
        "references": template_references_api_key,
        "worker_refusal": "workers run rpc-server, which has no API key.",
    },
    keystore.KIND_MCP: {
        "section": ("mcp", "server"),
        "placeholder": "${mcp_api_key}",
        "label": "MCP API key",
        "references": template_references_mcp_api_key,
        "worker_refusal": "workers run no MCP server, so there is no :30900 to key.",
    },
}


def _kind_from(body: dict) -> str:
    """The key kind a staging request names, defaulting to llama.

    Defaulted rather than required so a browser or script written against the
    single-key routes keeps working unchanged — the llama key is the one those
    callers meant.
    """
    kind = body.get("kind", keystore.KIND_LLAMA)
    if kind not in _KINDS:
        raise HTTPException(
            400,
            f"'kind' must be {keystore.KIND_LLAMA!r} or {keystore.KIND_MCP!r}, not {kind!r}",
        )
    return kind


def _client_host(request: Request) -> str:
    """Caller address for the audit lines. Every state change on this router
    names one: it is the only thing that says WHO re-keyed the fleet."""
    return request.client.host if request.client else "?"


# ---------------------------------------------------------------------------
# Read routes
# ---------------------------------------------------------------------------

@router.get("/api/access/nodes")
def access_nodes(request: Request) -> list[dict]:
    """Per-node endpoints + staged auth state. Config-derived, no fleet I/O,
    and never a plaintext key."""
    path = network_config_path(request)
    try:
        return access_service.node_access_list(path, _detected_ip(path))
    except Exception as e:
        # Polled by the tab — degrade like lifecycle_api does rather than 500
        # with a stack trace while the operator is mid-edit of the config.
        logger.warning("access_nodes failed: %s", e)
        raise HTTPException(503, f"config unreadable: {type(e).__name__}") from e


@router.get("/api/access/live")
async def access_live(request: Request) -> dict:
    """Live auth state per llama-serving node, with the final badge from the
    classify() truth table. Probes run concurrently; ~a second wall-clock."""
    path = network_config_path(request)
    # Off the loop, both of them: detection shells out to `route`/`ipconfig`
    # with no timeout of its own, and the listing regenerates every node's
    # config to hash it. This route is polled every 20 s per open tab, and the
    # loop it would block also carries the chat SSE streams and node ingest.
    try:
        ip = await asyncio.to_thread(_detected_ip, path)
        entries = await asyncio.to_thread(access_service.node_access_list, path, ip)
    except Exception as e:
        logger.warning("access_live failed: %s", e)
        raise HTTPException(503, f"config unreadable: {type(e).__name__}") from e
    # Both ports in one pass: a staged-but-unapplied MCP key is invisible to
    # the llama probe, and running the two fan-outs in sequence would double
    # the route's wall clock for a tab that polls it every 20 s.
    live, mcp_live = await asyncio.gather(
        access_service.probe_all(entries), access_service.probe_mcp_all(entries)
    )
    nodes: dict[str, dict] = {}
    for e in entries:
        auth = e.get("auth") or {}
        node_live = live.get(e["node_id"])
        badge, message = access_service.classify(
            bool(auth.get("desired_enabled")), auth.get("state", "unknown"), node_live
        )
        mcp_auth = auth.get("mcp") or {}
        mcp_node_live = mcp_live.get(e["node_id"])
        mcp_badge, mcp_message = access_service.classify(
            bool(mcp_auth.get("desired_enabled")),
            mcp_auth.get("state", "unknown"),
            mcp_node_live,
        )
        nodes[e["node_id"]] = {
            **(node_live or {"node_id": e["node_id"], "auth_live": None}),
            "badge": badge,
            "message": message or auth.get("message", ""),
            "mcp": {
                **(mcp_node_live or {"node_id": e["node_id"], "auth_live": None}),
                "badge": mcp_badge,
                "message": mcp_message or mcp_auth.get("message", ""),
            },
        }
    return {"checked_at": datetime.now(UTC).isoformat(), "nodes": nodes}


def _reveal(request: Request, node_id: str, kind: str, *, previous: bool) -> dict:
    """The body shared by the four routes that hand out a plaintext key.

    One implementation so the audit cannot drift between them: every reveal
    writes BOTH records, deliberately — ``emit``'s durable write is suppressed
    on failure (SystemLogger.emit), and none of these routes may hand out a
    credential leaving no trace of it anywhere.
    """
    target_for(request, node_id)
    key = (
        keystore.previous_key(node_id, kind=kind)
        if previous
        else keystore.read_key(node_id, kind=kind)
    )
    label = _KINDS[kind]["label"]
    if not key:
        raise HTTPException(
            404, f"no {'previous ' if previous else ''}{label} exists for {node_id}"
        )
    client = _client_host(request)
    text = f"{'PREVIOUS ' if previous else ''}{label} revealed for {node_id} (client {client})"
    logger.warning("%s", text)
    emit(request, "warn", text, source="access")
    return {"node_id": node_id, "api_key": key}


@router.get("/api/access/{node_id}/key")
def reveal_key(node_id: str, request: Request) -> dict:
    """One of the four routes that return a plaintext key. Explicit user
    action; audit-logged with the client address."""
    return _reveal(request, node_id, keystore.KIND_LLAMA, previous=False)


@router.get("/api/access/{node_id}/key/previous")
def reveal_previous_key(node_id: str, request: Request) -> dict:
    """Previous key after a rotate — what a still-running harness holds while
    the apply is pending. Audit-logged like the current key."""
    return _reveal(request, node_id, keystore.KIND_LLAMA, previous=True)


@router.get("/api/access/{node_id}/mcp-key")
def reveal_mcp_key(node_id: str, request: Request) -> dict:
    """The node's MCP bearer token. Separate file, separate rotation, same
    audit — holding it reaches inference through ``run_inference``, so it is
    exactly as sensitive as the llama key."""
    return _reveal(request, node_id, keystore.KIND_MCP, previous=False)


@router.get("/api/access/{node_id}/mcp-key/previous")
def reveal_previous_mcp_key(node_id: str, request: Request) -> dict:
    """Previous MCP key after a rotate — still live on the node until the
    apply lands."""
    return _reveal(request, node_id, keystore.KIND_MCP, previous=True)


# ---------------------------------------------------------------------------
# Staging routes (edit template / rotate key) — nothing touches the fleet
# ---------------------------------------------------------------------------

@router.put("/api/access/{node_id}/auth")
async def set_auth(node_id: str, request: Request) -> dict:
    """Enable/disable auth by editing the node's TEMPLATE — inserting or
    removing one ``api_key:`` line.

    ``kind`` picks which service: ``"llama"`` (the default) writes
    ``${api_key}`` under ``model.server``, ``"mcp"`` writes ``${mcp_api_key}``
    under ``mcp.server``. A field rather than a second route, so both kinds go
    through one ruamel round-trip, one ``validate_yaml_text`` call, one
    ``template_sharers`` computation and one audit-line shape.

    The round-trip rewrites the whole file, and ``_yaml`` is configured to
    re-emit it byte for byte so a disable touches exactly the one line. The one
    thing it cannot restore is placement: re-enabling puts the line back among
    the section's scalar settings, but below the comment block it used to sit
    above (see ``_insert_api_key``). No comment or value is lost either way.
    """
    cfg, target = target_for(request, node_id)
    body = await read_json(request)
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(400, "'enabled' must be a boolean")
    key_kind = _kind_from(body)
    spec = _KINDS[key_kind]
    parent_key, section_key = spec["section"]
    if not target.config_file:
        raise HTTPException(409, f"{node_id} has no config_file to edit")

    template_path = _template_path(request, target.config_file)
    if not template_path.exists():
        raise HTTPException(404, f"template not found: {target.config_file}")

    try:
        data = _yaml.load(template_path.read_text())
        if ((data.get("node") or {}).get("role")) == SHARD_WORKER_ROLE:
            # Neither service exists on a worker, and a worker shares its
            # template with every other worker in the fleet — writing a
            # placeholder here would stage auth for none of them while dirtying
            # all of them.
            raise HTTPException(
                400, f"{target.config_file} is a shard-worker template — "
                     f"{spec['worker_refusal']}",
            )
        server = (data.get(parent_key) or {}).get(section_key)
        if server is None:
            raise HTTPException(
                409, f"{target.config_file} has no {parent_key}.{section_key} section — "
                     f"nothing to attach an {spec['label']} to.",
            )
        if enabled:
            _insert_api_key(server, spec["placeholder"])
        else:
            _remove_api_key(server)
        buf = io.StringIO()
        _yaml.dump(data, buf)
        content = buf.getvalue()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"template edit failed: {type(e).__name__}: {e}") from e

    kind = _template_kind(target.config_file)
    ok, errors = validate_yaml_text(content, kind)
    if not ok:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)

    # The tree's atomic writer rather than a hand-rolled rename: it uses a
    # unique temporary name and unlinks it on failure, so a write that dies
    # mid-way cannot leave a stray `<template>.tmp` in the git-tracked
    # configs/ directory for a later `git add -A` to commit. mode=0o644
    # because templates are tracked and readable; dir_mode=None because
    # configs/ is a source directory whose permissions are not ours to set.
    write_private_file(template_path, content, mode=0o644, dir_mode=None)

    # A shared template (both 35B heads use one file) means this toggle just
    # changed every sharer too — say so instead of letting them find out.
    sharers = access_service.template_sharers(cfg, target)
    emit(request, "info",
         f"{spec['label']} auth {'enabled' if enabled else 'disabled'} in "
         f"{target.config_file} for {node_id}"
         + (f" (also affects: {', '.join(sharers)})" if sharers else "")
         + f" (client {_client_host(request)})", source="access")
    return {"ok": True, "node_id": node_id, "enabled": enabled, "kind": key_kind,
            "also_affects": sharers,
            "note": "Staged only — Apply pushes it to the node."}


def _remove_api_key(server: CommentedMap) -> None:
    """Drop the ``api_key`` line, keeping the comment block that followed it.

    ruamel files a comment against the key it FOLLOWS, so the five-line
    ``# RESERVED — documentation only`` block that sits between ``api_key`` and
    ``tools:`` is stored on ``api_key`` and is deleted with it — a toggle that
    silently eats the file's documentation, and a disable/enable cycle never
    brings it back. Carrying the token over to the next key (or, for a trailing
    ``api_key``, onto the previous one) keeps the removal to exactly one line.
    """
    keys = list(server.keys())
    if "api_key" not in keys:
        return
    i = keys.index("api_key")
    slot = server.ca.items.pop("api_key", None)
    server.pop("api_key")
    token = slot[2] if slot else None
    if token is None or "\n" not in token.value:
        return
    # The token opens with the newline that ended the api_key line itself;
    # everything after it belongs to the following key, not to the removed one.
    trailing = token.value.split("\n", 1)[1]
    if not trailing.strip():
        return
    token.value = trailing
    if i + 1 < len(keys):
        nxt = server.ca.items.setdefault(keys[i + 1], [None, None, None, None])
        nxt[1] = [token, *(nxt[1] or [])]
    else:
        prev = server.ca.items.get(keys[i - 1]) if i else None
        if prev and prev[2] is not None:
            prev[2].value += trailing


def _insert_api_key(server: CommentedMap, placeholder: str) -> None:
    """Add (or overwrite) the ``api_key`` line, quoted and above any block.

    Double-quoted because every shipped template writes ``"${api_key}"`` that
    way, so a re-enable reproduces the tracked line rather than a bare one.
    Placed before the section's first nested block — appending would drop the
    MCP key below the ~190-line ``tools`` list, far from the comment block that
    documents it.

    Caveat: a disable/enable cycle is NOT byte-identical. The trailing comment
    block is re-attached to the preceding key on the next read, so the restored
    line lands after that block rather than in front of it. Nothing is lost —
    only the one line moves.
    """
    value = DoubleQuotedScalarString(placeholder)
    if "api_key" in server:
        server["api_key"] = value
        return
    pos = len(server)
    for n, v in enumerate(server.values()):
        # dict/list covers ruamel's CommentedMap/CommentedSeq, which subclass
        # them — a nested block of any shape, however it was parsed.
        if isinstance(v, dict | list):
            pos = n
            break
    server.insert(pos, "api_key", value)


def _template_path(request: Request, config_file: str) -> Path:
    """Absolute path of a target's template, pinned to configs/.

    ``config_file`` is already constrained to a single path segment by
    ``SSHConfig._check_names_are_path_segments``, and every caller here reaches
    this through ``target_for``, which loads the fleet file through that
    schema. This resolve-and-compare is defence in depth for a caller that did
    not: it is the same containment rule as config_api._resolve, against the
    same directory, so a ``../../../.ssh/config`` value cannot make this router
    read and rewrite a file outside configs/ even if it ever got this far.
    """
    configs_dir = network_config_path(request).parent.resolve()
    path = (configs_dir / config_file).resolve()
    if path.parent != configs_dir:
        raise HTTPException(400, f"config_file must live in configs/: {config_file!r}")
    return path


def _template_kind(config_file: str) -> str:
    """Which schema validates this template — the same table the Config tab
    validates by, so a new fixed-name template cannot be checked against one
    schema here and another there."""
    return STATIC_ENTRIES.get(config_file, "edge-node")


@router.post("/api/access/{node_id}/auth/rotate")
async def rotate(node_id: str, request: Request) -> dict:
    """Stage a new key of one kind (old one retained as .prev). Takes effect on
    Apply. ``kind`` in the body picks the service; it defaults to llama.

    One kind per call, unlike ``aureax apply --rotate-key``, which re-keys
    both: the CLI is re-keying a node it is about to restart anyway, while the
    card's Rotate button sits inside one service's sub-block and must do what
    that block says.

    Refused when the node's template carries no placeholder for that kind: the
    key would enable nothing, while the card would grow a Reveal/Copy/Rotate
    block implying auth the node does not enforce.

    Deliberately NOT the CLI's rule. ``aureax apply --rotate-key``
    (Deployer.apply_to_nodes) gates on the node already holding a key file and
    never mints one; this route gates on the template and mints a key when the
    node has none, because the card offers Rotate exactly when auth is staged
    on. So an auth-on node with no key yet is served here and skipped there,
    and an auth-off node still holding a stale .key file is rotated there and
    409'd here.
    """
    _cfg, target = target_for(request, node_id)
    key_kind = _kind_from(await read_json(request))
    spec = _KINDS[key_kind]
    if not target.config_file or not spec["references"](
        _template_path(request, target.config_file)
    ):
        raise HTTPException(
            409,
            f"{spec['label']} auth is off for {node_id} "
            f"({target.config_file or 'no config_file'}) — enable it first, then rotate.",
        )
    if keystore.has_key(node_id, kind=key_kind):
        keystore.rotate_key(node_id, kind=key_kind)
        action = "rotated"
    else:
        keystore.ensure_key(node_id, kind=key_kind)
        action = "generated"
    emit(request, "info",
         f"{spec['label']} {action} for {node_id} (staged — Apply to activate) "
         f"(client {_client_host(request)})", source="access")
    return {
        "ok": True,
        "node_id": node_id,
        "kind": key_kind,
        "action": action,
        "key_masked": access_service.mask_key(keystore.read_key(node_id, kind=key_kind)),
        "key_last4": keystore.last4(node_id, kind=key_kind),
        "has_previous_key": keystore.previous_key(node_id, kind=key_kind) is not None,
    }


# ---------------------------------------------------------------------------
# Apply — the only routes that touch the fleet
# ---------------------------------------------------------------------------

class ApplyRunner:
    """One apply at a time, run off the event loop, with pollable state.

    ``running`` is the single source of truth for "an apply is in flight", and
    it is read from another module: ``lifecycle_api._apply_in_flight`` checks
    this attribute to serve the 409 both-ways interlock. ``snapshot()`` derives
    the flag the browser polls from it, so the two cannot disagree.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self.state: dict = {"run_id": None, "nodes": {}}

    def snapshot(self) -> dict:
        return {**self.state, "running": self.running}


def _runner(request: Request) -> ApplyRunner:
    runner = getattr(request.app.state, "access_apply", None)
    if runner is None:
        runner = ApplyRunner()
        request.app.state.access_apply = runner
    return runner


@router.post("/api/access/apply")
async def apply_bulk(request: Request) -> JSONResponse:
    """Apply to the listed nodes, or to the whole fleet with ``{"all": true}``.

    Fleet-wide is never inferred. An apply restarts every head it touches, each
    a multi-minute model reload, so an absent or empty ``node_ids`` is a client
    bug rather than a request to re-key the fleet — the CLI requires ``--all``
    for the same reason.
    """
    body = await read_json(request)
    node_ids = body.get("node_ids")
    if node_ids is not None and (
        not isinstance(node_ids, list) or not all(isinstance(n, str) for n in node_ids)
    ):
        raise HTTPException(400, "'node_ids' must be a list of strings")
    if node_ids is not None and not node_ids:
        raise HTTPException(
            400, "'node_ids' must not be empty — pass {\"all\": true} to apply to every node",
        )
    if node_ids is None and body.get("all") is not True:
        raise HTTPException(
            400, "pass 'node_ids' (a non-empty list), or {\"all\": true} for the whole fleet",
        )
    # Same 404 apply_one gives, and for the same reason: Deployer.apply_to_nodes
    # does reject an unknown id, but only after this route has answered 202, so
    # the caller reads success and the abort surfaces on the state poll — where
    # a mixed known/unknown list reports the REAL nodes as possibly half-written
    # when nothing was pushed at all.
    for nid in node_ids or []:
        target_for(request, nid)
    return await _start_apply(request, node_ids)


@router.post("/api/access/{node_id}/apply")
async def apply_one(node_id: str, request: Request) -> JSONResponse:
    target_for(request, node_id)
    return await _start_apply(request, [node_id])


@router.get("/api/access/apply/state")
def apply_state(request: Request) -> dict:
    return _runner(request).snapshot()


def _apply_push_set(config_path: Path, node_ids: list[str] | None) -> list[str]:
    """Every node an apply may write to: the requested ones plus their whole
    groups, since ``Deployer.apply_to_nodes`` pushes group-wide — plus every
    node whose config is DERIVED from another's key.

    The derived half is the planner fan-out: a template carrying
    ``${mcp_endpoints}`` gets every peer's MCP key baked into its resolved
    config, so ``apply_to_nodes`` pushes to it whenever an MCP key changed. Its
    membership is listed unconditionally here because this function runs
    BEFORE the run and cannot know whether a fingerprint will drift, and the
    failure mode it exists to prevent is under-reporting: this list is what an
    aborted run names as "may have been touched".

    Best effort — a config that will not load falls back to what was asked for.
    """
    try:
        cfg = load_network_config_cached(config_path)
    except Exception:
        return list(node_ids or [])
    enabled = [t.node_id for t in cfg.ssh.enabled_targets]
    requested = list(node_ids) if node_ids is not None else enabled
    out: list[str] = []
    for nid in requested:
        group = cfg.group_for_node(nid)
        for member in ([group.head, *group.workers] if group else [nid]):
            if member in enabled and member not in out:
                out.append(member)
    configs_dir = config_path.parent
    for t in cfg.ssh.enabled_targets:
        if t.node_id in out or not t.config_file:
            continue
        if template_references_mcp_endpoints(configs_dir / t.config_file):
            out.append(t.node_id)
    return out


async def _start_apply(request: Request, node_ids: list[str] | None) -> JSONResponse:
    runner = _runner(request)
    lifecycle = getattr(request.app.state, "lifecycle", None)

    # 409 both ways: a config push racing an aureax start would land
    # mid-label-pass; lifecycle_api carries the mirror-image guard.
    if lifecycle is not None and lifecycle.busy:
        return JSONResponse(
            {"error": "busy", "detail": "a lifecycle operation is running"},
            status_code=409,
        )
    if runner.running:
        return JSONResponse(
            {"error": "busy", "detail": "an apply is already running",
             **runner.snapshot()},
            status_code=409,
        )

    config_path = network_config_path(request)
    # Claimed in the same uninterrupted stretch as the two guards above, and
    # before the awaits below: every await yields the loop, so a claim placed
    # after them lets two POSTs both read running=False and dispatch two
    # applies at once — two SSH pushes and two `kubectl delete pods` over the
    # same nodes, a runner.state the second run overwrites, and the first run's
    # `finally` releasing the lifecycle interlock while the second still
    # pushes. It sits after network_config_path because that helper raises
    # 503: claiming ahead of it would latch `running` True with no `finally`
    # left to clear it.
    run_id = uuid.uuid4().hex[:12]
    runner.running = True
    previous_state = runner.state
    runner.state = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "node_ids": node_ids, "nodes": {}, "error": None,
    }
    try:
        # Off the loop: detection shells out with no timeout of its own, and
        # the value it returns is baked into every pushed config's telemetry
        # endpoint, so it is detected against a real node route (_detected_ip).
        ip = await asyncio.to_thread(_detected_ip, config_path)
        if not ip:
            raise HTTPException(
                400, "could not determine the orchestrator IP — configs would be "
                     "generated pointing telemetry at the wrong address",
            )
        touched = await asyncio.to_thread(_apply_push_set, config_path, node_ids)
    except BaseException:
        # Nothing has been scheduled yet, so no `finally` runs to release the
        # claim — and a latched True 409s every later apply and every
        # lifecycle start/stop for the life of the process. The state goes
        # back too, or /api/access/apply/state reports a run that never
        # happened as cleanly finished.
        runner.running = False
        runner.state = previous_state
        raise
    emit(request, "info",
         f"Apply started (run {run_id}) for "
         f"{', '.join(node_ids) if node_ids else 'ALL nodes'} "
         f"(client {_client_host(request)})", source="access")

    log = getattr(request.app.state, "system_log", None)

    def do_apply() -> None:
        def emit(msg: str) -> None:
            if log is not None:
                with contextlib.suppress(Exception):
                    log.emit("info", str(msg), source="access")

        try:
            # Imported here, INSIDE the try: deployer pulls in paramiko and
            # kubectl, so it stays off both the dashboard's import path and the
            # event loop — and an error raised while importing it must still
            # reach the finally that clears `running`, or every later apply and
            # every lifecycle start/stop 409s for the life of the process.
            from orchestrator.deploy.deployer import Deployer

            deployer = Deployer(config_path, ip)
            outcomes = deployer.apply_to_nodes(node_ids, restart=True, log=emit)
            runner.state["nodes"] = outcomes
            # Journal only what was PUSHED, with the hash actually written —
            # this is what turns the badge from "pending" to verified-later.
            entries = access_service.journal_entries(
                outcomes, run_id, datetime.now(UTC).isoformat()
            )
            if entries:
                access_service.record_applied(entries)
            emit(f"Apply finished (run {run_id}): "
                 + "; ".join(f"{n}: {o}" for n, o in outcomes.items()))
        except Exception as e:
            logger.exception("apply run %s failed", run_id)
            runner.state["error"] = f"{type(e).__name__}: {e}"
            # deploy_configs raises on the FIRST node that fails, so nodes
            # earlier in the push already hold the new config with an
            # unrestarted pod. Name them: an empty result tells the operator
            # nothing about which half of the fleet was touched.
            if not runner.state.get("nodes"):
                runner.state["nodes"] = {
                    nid: f"unknown — apply aborted: {type(e).__name__}" for nid in touched
                }
            emit(f"Apply FAILED (run {run_id}): {type(e).__name__}: {e}")
        finally:
            runner.running = False

    try:
        asyncio.get_running_loop().run_in_executor(None, do_apply)
    except Exception as e:
        # Nothing will run the finally above if the submit itself failed
        # (executor shut down, thread creation refused), and `running` latched
        # True blocks every later apply and lifecycle operation.
        runner.running = False
        runner.state["error"] = f"{type(e).__name__}: {e}"
        logger.exception("apply run %s could not be scheduled", run_id)
        raise HTTPException(500, f"could not start the apply: {type(e).__name__}") from e
    return JSONResponse({"run_id": run_id, "node_ids": node_ids}, status_code=202)
