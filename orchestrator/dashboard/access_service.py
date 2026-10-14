"""Pure logic behind the dashboard's Access tab (no FastAPI in here).

Three separate truths, kept in three separate columns so the UI cannot lie:

- DESIRED — what the git-tracked template says should be true (auth on/off via
  the ``api_key: ${api_key}`` line, webui, alias). Computed per target through
  the same generation code the deploy path uses, never read off the shared
  template directly — two heads share one template file and must not share a
  key.
- APPLIED — what was last pushed to the node, recorded in an apply journal at
  ``~/.aureax/access/apply-state.json``. The generated YAML alone cannot carry
  this: "regenerated but never pushed" and "pushed" are different states.
- LIVE — what the running llama-server actually enforces, measured by an
  unauthenticated probe. Unverifiable is reported as unknown, never as green.

Each node has TWO keyed services — llama-server on :30800 and the MCP server on
:30900 — so every column above is computed twice, once per
``keystore`` kind, and the MCP half is reported in a nested ``auth["mcp"]``
block with the identical field names. One config hash covers both: the two keys
live in the same resolved YAML, so a staged change to either moves it.

The testable core is :func:`classify` — the badge truth table.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
from pathlib import Path

import httpx

from orchestrator.config import keystore, tlsstore
from orchestrator.config.atomic import write_private_file
from orchestrator.config.cache import (
    load_edge_config_cached,
    load_network_config_cached,
    load_node_config_resolved,
)
from orchestrator.config.loader import generated_config_path
from orchestrator.config.schema import (
    LLAMA_NODE_PORT,
    MCP_NODE_PORT,
    SHARD_WORKER_ROLE,
    NetworkConfig,
    SSHTarget,
    mcp_server_enabled,
    node_role,
)
from orchestrator.config.templating import (
    # Borrowed rather than re-implemented: every URL this module hands out or
    # dials has to bracket an IPv6 literal exactly the way ${mcp_endpoints} and
    # ${orchestrator_url} already do, and two copies of that rule would drift.
    _url_host,
    generate_edge_config,
    template_references_api_key,
    template_references_mcp_api_key,
)

# lifecycle.tcp_reachable is the single reachability gate for the whole
# orchestrator — lifecycle's node-status column and this probe must agree on
# what "reachable" means, IPv6 hosts included, or the same node reads up in one
# view and down in the other.
from orchestrator.lifecycle import tcp_reachable

logger = logging.getLogger("aureax.dashboard.access")

# The live-auth probe path. Requirements: it must be BEHIND --api-key (so 401
# means secured), and cheap (no inference). Verified against the pinned build
# b11429's source (tools/server/server-http.cpp): the auth-exempt set there is
# exactly /health, /v1/health, / and the web-UI assets (plus any OPTIONS
# request) — /props is NOT in it, while GET /props is always served (the
# --props flag gates POST only). /props rather than /v1/models because the
# latter was exempt through b10375 (removed in b10518, #26347), so a node
# still on an old image would read "open" on it even with a key enforced.
_AUTH_PROBE_PATH = "/props"

# Probe budgets. The TCP gate keeps a node that is down (or a host that no
# longer exists) from costing the full HTTP timeout, and the HTTP timeout
# bounds one poll of the whole fleet, which is probed concurrently every 20 s
# by the Access tab.
_PROBE_TIMEOUT_S = 2.0
_TCP_GATE_TIMEOUT_S = 0.3

# The MCP probe's body is a literal "{}" — deliberately NOT a real `initialize`.
# StreamableHTTPSessionManager creates and REGISTERS a server-side transport for
# any POST that arrives without an Mcp-Session-Id, BEFORE the body is parsed,
# and only drops it when that session's task exits. A real initialize from a
# 20 s poll would therefore leak one session per node per poll for the life of
# the pod. "{}" is well-formed JSON that fails JSON-RPC validation, so it still
# passes through the bearer middleware (401 when a key is enforced) and
# otherwise comes back 400 "Validation error" — and the response carries the
# mcp-session-id header, which is what lets the probe DELETE what it created.
_MCP_PROBE_BODY = b"{}"
_MCP_PROBE_HEADERS = {
    # Both types, or the SDK's transport rejects the request before auth: it
    # requires the client to accept the SSE stream it may answer with.
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
_MCP_SESSION_HEADER = "mcp-session-id"

# What a probe reports when this Mac holds no fleet CA. Not a fallback to
# plaintext or to verify=False: a probe that skipped verification would report
# "secured" for a node presenting any certificate at all, which is the exact
# false green this whole module exists to prevent.
_NO_CA_DETAIL = "no fleet CA on this Mac — run `aureax generate --orchestrator-ip <ip>`"

# What probe_one reports when the handshake itself failed. Without this the
# failure lands in the generic `except` AFTER tcp_reachable already passed, so
# the card reads "reachable, auth unknown: ConnectError" and names nothing the
# operator can act on — while the actual cause is a node still holding the
# certificate of a previous CA, which one apply fixes.
_TLS_DETAIL = "TLS: certificate not trusted — run aureax apply"


# ---------------------------------------------------------------------------
# Apply journal
# ---------------------------------------------------------------------------

def journal_path() -> Path:
    return Path.home() / ".aureax" / "access" / "apply-state.json"


_JOURNAL_CORRUPT_SUFFIX = ".corrupt"


def _load_journal() -> tuple[dict[str, dict], bool]:
    """The journal, plus whether it was READABLE.

    A missing file is readable-and-empty — nothing has been applied from here.
    A file that will not parse is neither: it holds records this process cannot
    see, and :func:`record_applied` must not merge onto ``{}`` and replace it.

    The failure is logged rather than swallowed: without a line, an unreadable
    journal shows as every card dropping to "Never applied from here" for a
    fleet that is in sync, with nothing anywhere saying why.
    """
    path = journal_path()
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}, True
    except (json.JSONDecodeError, OSError):
        logger.exception("apply journal %s could not be read; reporting it empty", path)
        return {}, False
    if not isinstance(data, dict):
        logger.warning("apply journal %s is not a JSON object; reporting it empty", path)
        return {}, False
    return data, True


def read_journal() -> dict[str, dict]:
    """Per-node apply records, or ``{}`` when there are none to be read.

    Degrades instead of raising — a polled route reads this.
    """
    return _load_journal()[0]


def journal_entries(outcomes: dict[str, str], run_id: str, now: str) -> dict[str, dict]:
    """Journal rows for the nodes an apply actually reached.

    The substring test is the whole contract between ``Deployer.apply_to_nodes``
    outcome strings ("config pushed", "config pushed; restart FAILED (...)",
    "applied (pod Ready on new config)") and the badge leaving "pending": a node
    whose push failed is not recorded, so it keeps reading as not-applied.

    ``restarted`` carries the other half of that outcome. A push whose pod
    restart failed still journals the new hash — the file really is on the node
    — but the running pod predates it, so :func:`classify_staged` must not call
    that in_sync.

    Both key fingerprints are recorded because one apply installs both: a
    record naming only the llama key cannot say afterwards which MCP key the
    running pod is enforcing.
    """
    return {
        nid: {
            "applied_hash": generated_config_hash(nid),
            "applied_at": now,
            "key_fingerprint": keystore.fingerprint(nid, kind=keystore.KIND_LLAMA),
            "mcp_key_fingerprint": keystore.fingerprint(nid, kind=keystore.KIND_MCP),
            "status": outcome,
            "restarted": "FAILED" not in outcome,
            "run_id": run_id,
        }
        for nid, outcome in outcomes.items()
        if "pushed" in outcome or "applied" in outcome
    }


def record_applied(entries: dict[str, dict]) -> None:
    """Merge per-node apply records into the journal (atomic replace).

    A merge onto a journal that could not be READ would replace every other
    node's record with nothing, so an unreadable file is moved aside first and
    kept for recovery. If it cannot even be moved, this run goes unrecorded
    rather than destroying records nothing here has seen.
    """
    path = journal_path()
    merged, readable = _load_journal()
    if not readable:
        aside = path.with_name(path.name + _JOURNAL_CORRUPT_SUFFIX)
        try:
            os.replace(path, aside)
        except OSError:
            logger.exception(
                "refusing to overwrite the unreadable apply journal %s — this run is "
                "not recorded; move the file aside to let the journal rebuild", path,
            )
            return
        logger.warning("unreadable apply journal preserved as %s", aside)
    merged.update(entries)
    # The tree's atomic writer, not a hand-rolled rename: it fsyncs before the
    # replace (so a power cut cannot publish a zero-length journal at the real
    # name), uses a unique temporary that is unlinked on failure, and creates
    # ~/.aureax/access at 0700 — chmodding it even when the directory already
    # existed, which mkdir's mode argument does not.
    write_private_file(path, json.dumps(merged, indent=2))


# ---------------------------------------------------------------------------
# Config-derived state (the DESIRED column)
# ---------------------------------------------------------------------------

def mask_key(key: str | None) -> str | None:
    if not key:
        return None
    return f"{keystore.KEY_PREFIX}…{key[-4:]}"


def desired_config_hash(
    configs_dir: Path,
    target: SSHTarget,
    network_config: NetworkConfig,
    orchestrator_ip: str | None,
) -> str | None:
    """config_hash the node WOULD get if generated right now; None if that is
    not computable without side effects.

    Depends on ``orchestrator_ip`` on purpose: configs embed it, so a Mac that
    changed networks makes every node read "pending apply" — which is true.

    Never creates a key file (of either kind), and never mints TLS material: a
    browser polling the Access tab must not mint fleet credentials. A template
    that wants auth but has no key yet reports None, and the caller shows
    "pending generate" instead.
    """
    # Falsy, not "is None": detection returns "" when it fails (k3d.mac_lan_ip),
    # and generating against an empty ${orchestrator_ip} yields a hash that
    # matches nothing ever journalled — every node reading "Saved but not
    # applied" forever. Uncomputable is the honest answer for both.
    if not orchestrator_ip or not target.config_file:
        return None
    template = configs_dir / target.config_file
    if template_references_api_key(template) and not keystore.has_key(
        target.node_id, kind=keystore.KIND_LLAMA
    ):
        return None
    # The node's OWN MCP key, on the same rule as its llama key: generation
    # would mint it, and this path may not.
    #
    # Deliberately NOT extended to the keys of a planner's PEERS (which reach
    # its config through ${mcp_endpoints}). A planner whose peer was keyed a
    # moment ago genuinely is pending an apply, and "pending" is the honest
    # badge for that — returning None instead would report the one node whose
    # config is out of date as merely uncomputable.
    if template_references_mcp_api_key(template) and not keystore.has_key(
        target.node_id, kind=keystore.KIND_MCP
    ):
        return None
    try:
        resolved = generate_edge_config(
            configs_dir, target, network_config, orchestrator_ip
        )
    except Exception as e:
        logger.warning("desired-config generation failed for %s: %s", target.node_id, e)
        return None
    return ((resolved.get("deployment") or {}).get("config_hash")) or None


def generated_config_hash(node_id: str) -> str | None:
    """config_hash stamped in the last WRITTEN ~/.aureax/configs file.

    Logs rather than swallows: this value is what the apply journal records, and
    the only caller runs immediately after a successful regenerate + push, so
    None here is never expected — it leaves the node reading "never applied"
    forever for an apply that worked.
    """
    path = generated_config_path(node_id)
    try:
        ncfg = load_edge_config_cached(path)
    except Exception:
        logger.exception("could not read the generated config for %s (%s)", node_id, path)
        return None
    config_hash = ((ncfg.get("deployment") or {}).get("config_hash")) or None
    if config_hash is None:
        logger.warning("generated config %s carries no deployment.config_hash", path)
    return config_hash


def node_access_list(
    network_config_path: Path, orchestrator_ip: str | None
) -> list[dict]:
    """One NodeAccess entry per enabled target. NEVER contains a plaintext key
    — pinned by
    tests/test_access_service.py::test_node_access_list_never_returns_plaintext_keys.
    """
    # Cached on (path, mtime): this is a polled route, and re-reading plus
    # re-validating the fleet config per poll cost ~11 ms of loop time.
    cfg = load_network_config_cached(network_config_path)
    configs_dir = network_config_path.parent
    journal = read_journal()

    out: list[dict] = []
    for t in cfg.ssh.enabled_targets:
        ncfg = load_node_config_resolved(configs_dir, t)
        role = node_role(ncfg)
        group = cfg.group_for_node(t.node_id)
        serves_llama = role != SHARD_WORKER_ROLE
        server_cfg = (ncfg.get("model") or {}).get("server") or {}
        webui_enabled = bool(server_cfg.get("webui", True))
        serves_mcp = mcp_server_enabled(ncfg)

        alias = server_cfg.get("alias")
        model_id = alias or (ncfg.get("model") or {}).get("name") \
            or (ncfg.get("model") or {}).get("filename") or None

        entry: dict = {
            "node_id": t.node_id,
            "display_name": t.display_name or t.node_id,
            "host": t.host,
            "role": role,
            "group": group.name if group else None,
            "group_members": list(group.members) if group else [],
            "serves_llama": serves_llama,
            "served_by": None,
            # Workers keep every URL field null so a copy button structurally
            # cannot hand out an address that serves nothing.
            #
            # https, with no plaintext spelling anywhere: every listener in the
            # fleet is TLS-only, so an http:// address here would hand the
            # operator a URL that cannot connect at all — and a copy button is
            # exactly the place a wrong scheme survives longest.
            #
            # _url_host, not t.host: an IPv6 literal needs brackets in a URL
            # authority or its own colons read as the port separator, and a
            # copy button that hands out an unparseable URL is worse than none.
            "api_base_url": f"https://{_url_host(t.host)}:{LLAMA_NODE_PORT}/v1"
            if serves_llama
            else None,
            "chat_ui_url": (
                f"https://{_url_host(t.host)}:{LLAMA_NODE_PORT}/"
                if serves_llama and webui_enabled
                else None
            ),
            "webui_enabled": webui_enabled if serves_llama else False,
            "mcp_url": f"https://{_url_host(t.host)}:{MCP_NODE_PORT}/mcp" if serves_mcp else None,
            "model_id": model_id if serves_llama else None,
            "shared_template_with": template_sharers(cfg, t),
        }

        entry["auth"] = _auth_block(
            configs_dir, cfg, t, serves_llama, serves_mcp, journal, orchestrator_ip
        )
        out.append(entry)

    # Second pass: workers point at their head's ENTRY, not a re-derived URL —
    # this keeps served_by.chat_ui_url gated on the head's own webui setting
    # (a worker card must not advertise a chat UI its head has disabled).
    entries_by_id = {e["node_id"]: e for e in out}
    for e in out:
        if e["serves_llama"]:
            continue
        group = cfg.group_for_node(e["node_id"])
        head_entry = entries_by_id.get(group.head) if group else None
        if head_entry is not None:
            e["served_by"] = {
                "node_id": head_entry["node_id"],
                "display_name": head_entry["display_name"],
                "api_base_url": head_entry["api_base_url"],
                "chat_ui_url": head_entry["chat_ui_url"],
                # The worker's own auth block is "unsupported", so its env
                # snippet would otherwise tell the operator the head's endpoint
                # needs no key — and every call against a keyed head 401s.
                "auth_desired": bool((head_entry.get("auth") or {}).get("desired_enabled")),
            }

    for e in out:
        e["env_snippet"] = _env_snippet(e)
        e["mcp_snippet"] = _mcp_snippet(e)
    return out


def template_sharers(cfg: NetworkConfig, target: SSHTarget) -> list[str]:
    """Other enabled node_ids resolved from the SAME template file — a
    template-level auth toggle affects all of them, and the UI must say so.

    Public because the toggle route's ``also_affects`` and
    ``/api/access/nodes``' ``shared_template_with`` must mean the same thing,
    so the definition of "shared" lives here and nowhere else."""
    if not target.config_file:
        return []
    return [
        t.node_id
        for t in cfg.ssh.enabled_targets
        if t.config_file == target.config_file and t.node_id != target.node_id
    ]


_LLAMA_UNSUPPORTED = "Shard workers run rpc-server, which has no API key."
_MCP_UNSUPPORTED = "This node runs no MCP server, so it has no :30900 to key."


def _auth_block(
    configs_dir: Path,
    cfg: NetworkConfig,
    target: SSHTarget,
    serves_llama: bool,
    serves_mcp: bool,
    journal: dict[str, dict],
    orchestrator_ip: str | None,
) -> dict:
    """Staged auth for both of the node's keyed services.

    The llama half stays at the top level — three consumers read
    ``auth.desired_enabled`` / ``auth.state`` — and the MCP half is the same
    field set one level down under ``"mcp"``. Both come out of the same
    :func:`classify_staged`, against the same desired hash: the two keys travel
    in one resolved config, so one hash decides whether either is stale.

    ``supported`` differs per kind and per role, which is why it is computed
    twice: a planner serves llama and no MCP, a shard worker serves neither.
    """
    template = (configs_dir / target.config_file) if target.config_file else None
    jentry = journal.get(target.node_id)
    # One generation per node, not one per kind: generate_edge_config is the
    # expensive half of a route the tab polls every 20 s.
    d_hash = (
        desired_config_hash(configs_dir, target, cfg, orchestrator_ip)
        if template is not None
        else None
    )

    llama = _kind_block(
        target.node_id,
        keystore.KIND_LLAMA,
        supported=serves_llama and template is not None,
        unsupported_message=_LLAMA_UNSUPPORTED,
        desired=template is not None and template_references_api_key(template),
        d_hash=d_hash,
        jentry=jentry,
    )
    mcp = _kind_block(
        target.node_id,
        keystore.KIND_MCP,
        supported=serves_mcp and template is not None,
        unsupported_message=_MCP_UNSUPPORTED,
        desired=template is not None and template_references_mcp_api_key(template),
        d_hash=d_hash,
        jentry=jentry,
    )
    return {**llama, "mcp": mcp}


def _kind_block(
    node_id: str,
    kind: str,
    *,
    supported: bool,
    unsupported_message: str,
    desired: bool,
    d_hash: str | None,
    jentry: dict | None,
) -> dict:
    if not supported:
        return {"supported": False, "state": "unsupported", "message": unsupported_message}

    key_state = keystore.key_state(node_id, kind=kind)
    has_key = key_state == keystore.KEY_PRESENT
    state, message = classify_staged(desired, has_key, d_hash, jentry)
    if key_state == keystore.KEY_UNREADABLE:
        # Distinct from "no key": the two call for opposite actions. Without
        # this the card reads "no key exists yet — Apply will create one", and
        # Apply cannot: the file is there, it just cannot be read.
        state = "unknown"
        message = (
            f"A key file for {node_id} exists but cannot be read — fix its ownership "
            f"({keystore.key_path(node_id, kind=kind)}, 0600, owned by you). A .key left "
            "root-owned by a sudo run is the usual cause."
        )
    return {
        "supported": True,
        "desired_enabled": desired,
        "has_key": has_key,
        "key_state": key_state,
        "key_masked": mask_key(keystore.read_key(node_id, kind=kind)),
        "key_last4": keystore.last4(node_id, kind=kind),
        "key_fingerprint": keystore.fingerprint(node_id, kind=kind),
        "has_previous_key": keystore.previous_key(node_id, kind=kind) is not None,
        "desired_hash": d_hash,
        "applied_hash": (jentry or {}).get("applied_hash"),
        "applied_at": (jentry or {}).get("applied_at"),
        "state": state,
        "message": message,
    }


def _env_snippet(entry: dict) -> list[str]:
    base = entry["api_base_url"] or (entry["served_by"] or {}).get("api_base_url")
    if not base:
        return []
    auth = entry.get("auth") or {}
    served_by = entry.get("served_by") or {}
    if auth.get("desired_enabled"):
        key_line = "OPENAI_API_KEY=<copy from the key field above>"
    elif served_by.get("auth_desired"):
        # A worker's snippet points at its HEAD, so the head's auth decides
        # whether a key is needed — the worker's own block is "unsupported".
        key_line = f"OPENAI_API_KEY=<copy from {served_by.get('node_id')}'s key field>"
    else:
        # Many harnesses refuse to start with an EMPTY key var; llama-server
        # ignores the value entirely when no --api-key is configured.
        key_line = "OPENAI_API_KEY=sk-no-key-required  # endpoint is open"
    lines = [f"OPENAI_BASE_URL={base}", key_line]
    if entry.get("model_id"):
        lines.append(f"OPENAI_MODEL={entry['model_id']}")
    return lines


def _mcp_snippet(entry: dict) -> list[str]:
    """The MCP half of the copy-paste block, or [] for a node serving no MCP.

    A sibling of :func:`_env_snippet` rather than two more lines inside it: the
    MCP key is not an ``OPENAI_API_KEY``, and nothing that speaks the
    OpenAI-compatible endpoint reads it. ``AUREAX_MCP_API_KEY`` is the name
    ``aureax-harness`` and scripts/test-mcp-tools.py look up, so this is the
    variable an operator actually needs exported.
    """
    url = entry.get("mcp_url")
    if not url:
        return []
    mcp = (entry.get("auth") or {}).get("mcp") or {}
    if mcp.get("desired_enabled"):
        key_line = "AUREAX_MCP_API_KEY=<copy from the MCP key field above>"
    else:
        # A comment, not an empty assignment: an exported empty value would be
        # sent as `Authorization: Bearer `, which reads as a wrong key rather
        # than as no key at all if this port is ever secured later.
        key_line = "# no AUREAX_MCP_API_KEY — this MCP port is open to anyone who can reach it"
    return [f"AUREAX_MCP_URL={url}", key_line]


# ---------------------------------------------------------------------------
# Badge truth tables
# ---------------------------------------------------------------------------

def classify_staged(
    auth_desired: bool,
    has_key: bool,
    desired_hash: str | None,
    journal_entry: dict | None,
) -> tuple[str, str]:
    """Config-only half of the badge (no network I/O): is what we WANT the
    same as what we last PUSHED?"""
    if auth_desired and not has_key:
        return ("pending", "Auth is enabled in the template but no key exists yet — "
                           "Apply (or `aureax generate`) will create one.")
    if desired_hash is None:
        return ("unknown", "Could not compute the desired config "
                           "(orchestrator IP undetectable, or generation failed).")
    applied_hash = (journal_entry or {}).get("applied_hash")
    if applied_hash is None:
        return ("never_applied", "Never applied from here — the node may still run "
                                 "settings from a full deploy.")
    if applied_hash != desired_hash:
        return ("pending", "Saved but not applied — the node still enforces the "
                           "previous settings.")
    if (journal_entry or {}).get("restarted") is False:
        # The config file reached the node, so the hashes agree — but the pod
        # that reads it never restarted, and llama-server only reads its
        # settings at startup. Green here would claim a live alias, context
        # size or key that the running process does not have.
        return ("pending", "Config pushed but the pod never restarted — it still runs "
                           "the previous settings. Apply again.")
    return ("in_sync", "")


def classify(
    auth_desired: bool,
    staged_state: str,
    live: dict | None,
) -> tuple[str, str]:
    """Final badge, combining the staged verdict with the live probe.

    Never green on faith: 'applied' requires the staged states to agree AND
    the live probe to confirm. Unverifiable is 'unknown', not a color.
    """
    if staged_state == "unsupported":
        return ("unsupported", "")
    if staged_state in ("pending", "never_applied", "unknown"):
        # Staged problems outrank live readings — an out-of-date node probing
        # "secured" is still out of date.
        return (staged_state, "")
    auth_live = (live or {}).get("auth_live")
    if auth_live in (None, "unknown", "unreachable"):
        return ("unknown", "Could not verify the live state.")
    if auth_desired:
        if auth_live == "open":
            return ("drift", "Config sets a key but the node answers unauthenticated — "
                             "the running pod predates the change. Apply, or restart the pod.")
        if auth_live == "secured" and (live or {}).get("key_matches") is False:
            return ("drift", "The node is enforcing a key we do not hold. "
                             "Rotate + Apply to take back control.")
        return ("applied", "")
    if auth_live == "secured":
        return ("drift", "Config has auth off but the node still requires a key — "
                         "the running pod predates the change.")
    return ("applied", "")


# ---------------------------------------------------------------------------
# Live probe
# ---------------------------------------------------------------------------

def _blank_result(node_id: str) -> dict:
    return {
        "node_id": node_id,
        "reachable": False,
        "auth_live": "unreachable",
        "key_matches": None,
        "http_status": None,
        "detail": "",
    }


def _no_ca_result(node_id: str, *, reachable: bool = False) -> dict:
    """No verdict, and why. ``reachable`` is passed through rather than assumed:
    the per-node probes learn it from the TCP gate before they ever need a
    context, and reporting a node that answered on the port as unreachable
    would send the operator after the wrong problem."""
    return {**_blank_result(node_id), "reachable": reachable,
            "auth_live": "unknown", "detail": _NO_CA_DETAIL}


def tls_context() -> ssl.SSLContext | None:
    """The fleet-CA context every probe verifies against, or None if there is
    no CA on this Mac yet.

    None means "do not probe", never "probe without verification": a probe that
    skipped verification would report "secured" for any node presenting any
    certificate, which is precisely the false green this module exists to
    prevent. The callers turn None into an ``unknown`` verdict naming the
    missing store.
    """
    try:
        return tlsstore.ca_ssl_context()
    except Exception as e:
        logger.warning("live probes are disabled: %s", e)
        return None


def _is_tls_failure(exc: BaseException) -> bool:
    """Whether an ``ssl.SSLError`` is anywhere in this exception's cause chain.

    httpx does not re-raise the SSL error: it surfaces as ``ConnectError``
    wrapping an httpcore ``ConnectError`` wrapping the real
    ``SSLCertVerificationError``, two links down. Walking ``__context__`` as
    well as ``__cause__`` because the intermediate layers re-raise inside an
    ``except`` block without ``from``.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return True
        current = current.__cause__ or current.__context__
    return False


async def probe_one(
    host: str, node_id: str, client: httpx.AsyncClient | None = None
) -> dict:
    """Auth state of one llama endpoint: secured / open / unknown / unreachable,
    plus key_matches when we hold a key for a secured node.

    ``client`` is shared by probe_all: constructing an httpx.AsyncClient builds
    an SSL context (~5 ms), and that cost lands on the event loop once per node
    per 20 s poll.
    """
    result = _blank_result(node_id)
    if not await asyncio.to_thread(tcp_reachable, host, LLAMA_NODE_PORT, _TCP_GATE_TIMEOUT_S):
        return result
    result["reachable"] = True

    # Bracketed for IPv6: httpx raises InvalidURL on a bare literal, and the
    # generic handler below would file that as auth_live "unknown" forever —
    # i.e. an IPv6 fleet could never see live auth drift at all.
    url = f"https://{_url_host(host)}:{LLAMA_NODE_PORT}{_AUTH_PROBE_PATH}"
    async with contextlib.AsyncExitStack() as stack:
        if client is None:
            ctx = tls_context()
            if ctx is None:
                return _no_ca_result(node_id, reachable=True)
            client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S, verify=ctx)
            )
        try:
            resp = await client.get(url)
            result["http_status"] = resp.status_code
            if resp.status_code == 401:
                result["auth_live"] = "secured"
            elif 200 <= resp.status_code < 300:
                result["auth_live"] = "open"
            elif resp.status_code == 503:
                # A 503 during a sharded head's multi-minute model load comes
                # from the readiness middleware, which runs BEFORE auth — it
                # proves nothing about auth and must not read as "open".
                result["auth_live"] = "unknown"
                result["detail"] = "loading model (503 comes before auth)"
            else:
                result["auth_live"] = "unknown"
                result["detail"] = f"unexpected status {resp.status_code}"
        # Not just httpx.HTTPError: httpx.InvalidURL and UnicodeError (a host
        # that is not a valid authority) derive straight from Exception, and
        # probe_all fans this out with gather.
        except Exception as e:
            result["auth_live"] = "unknown"
            # A rejected certificate is a distinct, fixable state, and it lands
            # here AFTER tcp_reachable already passed — so without this branch
            # the card says "reachable, ConnectError" for a node whose only
            # problem is that it still holds a certificate from a previous CA.
            result["detail"] = _TLS_DETAIL if _is_tls_failure(e) else type(e).__name__
            return result

        if result["auth_live"] == "secured":
            key = keystore.read_key(node_id, kind=keystore.KIND_LLAMA)
            if key:
                # Its own guard: a failure of the key check degrades
                # key_matches, never the "secured" verdict already proven by
                # the 401 above.
                try:
                    authed = await client.get(url, headers={"Authorization": f"Bearer {key}"})
                    result["key_matches"] = 200 <= authed.status_code < 300
                except Exception as e:
                    result["detail"] = f"key check failed: {type(e).__name__}"
    return result


async def probe_all(entries: list[dict]) -> dict[str, dict]:
    """Probe every llama-serving node concurrently; keyed by node_id."""
    return await _probe_fanout(
        [e for e in entries if e.get("serves_llama") and e.get("host")], probe_one
    )


async def probe_mcp_one(
    host: str, node_id: str, client: httpx.AsyncClient | None = None
) -> dict:
    """Auth state of one node's MCP endpoint, in probe_one's result shape.

    ``secured`` is a 401 from the bearer middleware; ``open`` is anything the
    MCP transport itself answered (a 2xx, or the 400 the deliberately-invalid
    body earns), because reaching the transport at all means nothing rejected
    the missing Authorization header. Every other status is ``unknown`` — a
    proxy, a wrong port, or a pod that is not the MCP server.

    See ``_MCP_PROBE_BODY`` for why the body is "{}" and what the DELETE is
    cleaning up.
    """
    result = _blank_result(node_id)
    if not await asyncio.to_thread(tcp_reachable, host, MCP_NODE_PORT, _TCP_GATE_TIMEOUT_S):
        return result
    result["reachable"] = True

    # Bracketed for IPv6, as in probe_one.
    url = f"https://{_url_host(host)}:{MCP_NODE_PORT}/mcp"
    key = keystore.read_key(node_id, kind=keystore.KIND_MCP)
    async with contextlib.AsyncExitStack() as stack:
        if client is None:
            ctx = tls_context()
            if ctx is None:
                return _no_ca_result(node_id, reachable=True)
            client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S, verify=ctx)
            )
        try:
            resp = await client.post(url, headers=_MCP_PROBE_HEADERS, content=_MCP_PROBE_BODY)
            result["http_status"] = resp.status_code
            if resp.status_code == 401:
                result["auth_live"] = "secured"
            elif 200 <= resp.status_code < 300 or resp.status_code == 400:
                result["auth_live"] = "open"
                await _drop_mcp_session(client, url, resp, None)
            else:
                result["auth_live"] = "unknown"
                result["detail"] = f"unexpected status {resp.status_code}"
        except Exception as e:
            result["auth_live"] = "unknown"
            result["detail"] = _TLS_DETAIL if _is_tls_failure(e) else type(e).__name__
            return result

        if result["auth_live"] == "secured" and key:
            # Its own guard, like the llama probe's: a failure of the key check
            # degrades key_matches, never the "secured" verdict the 401 proved.
            try:
                authed = await client.post(
                    url,
                    headers={**_MCP_PROBE_HEADERS, "Authorization": f"Bearer {key}"},
                    content=_MCP_PROBE_BODY,
                )
                result["key_matches"] = authed.status_code != 401
                await _drop_mcp_session(client, url, authed, key)
            except Exception as e:
                result["detail"] = f"key check failed: {type(e).__name__}"
    return result


async def _drop_mcp_session(
    client: httpx.AsyncClient, url: str, resp: httpx.Response, key: str | None
) -> None:
    """Best-effort DELETE of the session the probe's POST just registered.

    Skipped silently when the response carries no session id (nothing was
    registered) and when the DELETE fails (the session expires with the pod).
    Without it a 20 s poll accumulates one server-side transport per node per
    poll, for the life of the pod — see ``_MCP_PROBE_BODY``.
    """
    session_id = resp.headers.get(_MCP_SESSION_HEADER)
    if not session_id:
        return
    headers = {_MCP_SESSION_HEADER: session_id}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with contextlib.suppress(Exception):
        await client.delete(url, headers=headers)


async def probe_mcp_all(entries: list[dict]) -> dict[str, dict]:
    """Probe every MCP-serving node concurrently; keyed by node_id.

    Gated on ``mcp_url`` rather than on a role: that field is already null for
    every node whose own config turns the MCP server off (shard workers and the
    planner), so the two cannot disagree.
    """
    return await _probe_fanout(
        [e for e in entries if e.get("mcp_url") and e.get("host")], probe_mcp_one
    )


async def _probe_fanout(targets: list[dict], probe) -> dict[str, dict]:
    """Run ``probe`` over ``targets`` concurrently on one shared client."""
    if not targets:
        return {}
    ctx = tls_context()
    if ctx is None:
        # No CA, no probe: reporting "unknown" for every node beats connecting
        # without verification, and beats a 500 on a route the tab polls.
        return {e["node_id"]: _no_ca_result(e["node_id"]) for e in targets}
    async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S, verify=ctx) as client:
        results = await asyncio.gather(
            *(probe(e["host"], e["node_id"], client) for e in targets),
            # One malformed host must not take the whole live view down with
            # it: this route has no error handler of its own.
            return_exceptions=True,
        )
    out: dict[str, dict] = {}
    for entry, result in zip(targets, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("live probe of %s failed: %r", entry["node_id"], result)
            out[entry["node_id"]] = {
                **_blank_result(entry["node_id"]),
                "auth_live": "unknown",
                "detail": type(result).__name__,
            }
            continue
        out[result["node_id"]] = result
    return out
