"""Pydantic models for Aureax YAML config validation."""

from __future__ import annotations

import os
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ============================================================================
# node.role values
# ============================================================================
# These literals are load-bearing in every place that must agree — at least the
# edge label pass filters on HEAD_ROLE (deployer.label_nodes), the lifecycle
# start/stop split filters on SHARD_WORKER_ROLE and PLANNER_ROLE and treats
# everything else as an edge node (lifecycle._split_hostnames_by_role), the
# client picks which binary to spawn from SHARD_WORKER_ROLE
# (client/entrypoint.py), the group cross-file validator rejects a head or
# worker carrying the wrong one (templating.validate_groups), the dashboard
# hides workers as chat targets and skips them as llama servers, and the shell
# scripts compare against the same strings via parse-config.py's NODE_ROLES
# export. No count here: a hard number goes stale on the next consumer.
#
# They live here, rather than being redeclared per module with "keep in sync"
# comments, because this is the one file both Dockerfiles copy into the edge
# image — so the client can import it too.
HEAD_ROLE = "edge_executor"
SHARD_WORKER_ROLE = "shard_worker"
PLANNER_ROLE = "cognitive_planner"

# ============================================================================
# NodePorts the edge Services publish
# ============================================================================
# The other side of this contract is k8s/service-mcp.yaml and
# k8s/service-llama.yaml, which keep their own literals — a change here needs
# the same change there. They live beside the role literals for the same
# reason: every module that dials these ports reads them from here rather than
# declaring its own copy.
MCP_NODE_PORT = 30900
LLAMA_NODE_PORT = 30800

# The HuggingFace ref a registry entry is fetched from when it pins none.
# Shared with scripts/parse-config.py, which reports it to download-model.sh:
# the two must agree or the log would name a ref the URLs do not use.
DEFAULT_MODEL_REVISION = "main"

# The ports the managed child binds INSIDE the pod, which the NodePorts above
# forward to. Declared here, and used as the Pydantic field defaults below, so
# there is one number rather than one per reader: the supervisor dials
# llama-server on its own port and probes the rpc-server listener before
# handing the head its peers, and client/config/parser.py and
# client/entrypoint.py import these rather than re-typing them.
DEFAULT_LLAMA_PORT = 8080
DEFAULT_RPC_PORT = 50052

# How long a sharded group's head waits for every worker's RPC port before
# starting llama-server. Declared once because three places must agree: the
# NodeGroup where it is configured, the NodeConfig that build_variables()
# injects it into, and the non-group fallback in build_variables() itself.
# daemonset-edge.yaml's startupProbe budget has to cover it (see
# ARCHITECTURE.md, "Probe budgets").
DEFAULT_WORKER_WAIT_SECONDS = 300

# The supervisor's fallback when a config omits model.server.health_timeout_seconds
# must equal the schema default, or a field-omitting config would wait a different
# time on the node than the orchestrator's probe-budget test assumes. Declared
# here (this file ships in the edge image) so client/entrypoint.py imports it.
DEFAULT_HEALTH_TIMEOUT_SECONDS = 120

# ============================================================================
# llama.cpp build floor
# ============================================================================
# Minimum llama.cpp build this fleet may run. b8492 fixed CVE-2026-34159 (an
# unauthenticated RCE in rpc-server's tensor-graph deserialization, CVSS 9.8,
# with a public 0-click exploit), and further server-side memory-corruption
# fixes landed through b9060. Both Dockerfiles pin LLAMA_CPP_VERSION=b10375
# (docker/Dockerfile.client and docker/Dockerfile.client.cuda), which is
# above every fixed-in build known as of 2026-08 — the floor makes a rollback
# to an older build a visible security decision instead of an accident.
# `aureax group doctor` warns when a member reports a build below it.
MIN_LLAMA_BUILD = 10375

_BUILD_TAG_RE = re.compile(r"\bb(\d{3,})\b")

# Shape of an acceptable model.server.api_key. The character class exists
# because llama.cpp's own parsing punishes anything outside it silently — see
# LlamaServerConfig._check_server_string_shapes for the per-character reasons.
_API_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]{16,512}$")


def _check_api_key(value: str | None, field: str) -> None:
    """Raise unless ``value`` is None or a key every consumer carries intact.

    Shared by all three api_key fields — llama-server's (model.server.api_key),
    the node's own MCP server (mcp.server.api_key) and each peer's copy in
    mcp.client.remote_servers[].api_key — so one alphabet governs the lot. That
    matters because a single minted key travels several of those paths at once:
    the generator writes a node's MCP key into that node's own config AND into
    every planner's peer list, so a value one side accepted and another
    rejected would ship a fleet that validates node by node and fails as a
    whole.

    llama-server's parsing makes several characters quiet foot-guns rather than
    errors: a comma SPLITS the value into multiple valid keys, '#' starts a
    comment in --api-key-file format, and whitespace/CR are kept verbatim (no
    trimming anywhere), producing a key nobody actually holds. The MCP side
    wants the same alphabet for its own reason: the bearer middleware compares
    raw bytes, and a non-ASCII token never reaches it to be compared — httpx
    encodes header values as ASCII and raises UnicodeEncodeError in the client
    instead, so the caller sees a crash rather than the 401 that would tell
    them the key is wrong.

    An empty string is rejected too — "" is falsy downstream, so `api_key: ""`
    would ship auth silently disabled while looking configured. Auth-off is
    spelled by OMITTING the field.

    A still-unresolved template reaches here only through the dummies in
    templating.PLACEHOLDER_DUMMIES, which are shaped to match this pattern.
    """
    if value is not None and not _API_KEY_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be 16-512 chars of [A-Za-z0-9_.-] "
            "(no commas, whitespace or '#' — llama.cpp treats those as key "
            "separators/comments); omit the field entirely to disable auth"
        )


# ============================================================================
# Shared YAML helper
# ============================================================================


def to_plain_dict(obj: object) -> object:
    """Recursively convert ruamel YAML objects to plain dicts/lists.

    Lives in the schema module because it is the one file both sides import:
    the orchestrator's loader and — since both Dockerfiles copy this file into
    the edge image — the client's config parser, which each carried their own
    copy of this walk.
    """
    if hasattr(obj, "items"):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(item) for item in obj]
    return obj


def node_role(cfg: dict) -> str:
    """A raw per-node config's ``node.role``; "" when absent or explicitly null.

    ``or {}`` at every level, not ``.get(k, {})``: a key present with a null
    value returns None, not the default, and `node:` with nothing under it is
    valid YAML — reading ``.role`` off that raised AttributeError inside
    `aureax generate`, taking the whole fleet's generation down.

    Lives beside the role literals, and in the one file both Dockerfiles copy
    into the edge image, so the deployer, the lifecycle split, templating, the
    dashboard and the supervisor all read the key the same way. Compare the
    result against HEAD_ROLE / SHARD_WORKER_ROLE / PLANNER_ROLE above.
    """
    return (cfg.get("node") or {}).get("role", "") or ""


def mcp_server_enabled(cfg: dict) -> bool:
    """Whether a raw per-node config runs an MCP server. Absent means yes.

    Same null-safety rule as :func:`node_role`, and the same reason for living
    here: shard workers turn this off, and templating, the Access tab and
    scripts/parse-config.py (which exports it as NODE_HAS_MCP_SERVER) must all
    answer it identically.
    """
    return bool(((cfg.get("mcp") or {}).get("server") or {}).get("enabled", True))


def mcp_client_enabled(cfg: dict) -> bool:
    """Whether a raw per-node config DIALS other MCP servers. Absent means no.

    The opposite default to :func:`mcp_server_enabled`, matching MCPClient's own
    schema default below — a node that does not say it is a client is not one,
    while every node is assumed to serve until it opts out.

    Same ``or {}`` null-safety at every level, and here for the same reason: a
    key present with a null value returns None rather than the default, and
    `mcp:` with nothing under it is valid YAML. Lives beside its sibling so the
    planner validator (templating.validate_planners) and anything else asking
    "is this a planner?" read the key the same way.
    """
    return bool(((cfg.get("mcp") or {}).get("client") or {}).get("enabled", False))


def parse_llama_build(text: str) -> int | None:
    """Extract the numeric llama.cpp build from a tag like ``b10375``.

    Returns None — never 0 — when the text carries no build number ("image not
    present", "master", ""). A false "below the floor" would send someone
    rebuilding images to fix what is actually an SSH or probe problem.
    """
    if not text:
        return None
    m = _BUILD_TAG_RE.search(text)
    return int(m.group(1)) if m else None

# ============================================================================
# aureax-network.yaml schema
# ============================================================================


class NetworkInfo(BaseModel):
    """Network identity — name, version, and description."""

    name: str
    version: str
    description: str = ""


class DefaultsModel(BaseModel):
    """Default model storage path for edge nodes."""

    local_path: str = "/opt/aureax/models/"


class DefaultsTelemetryRemote(BaseModel):
    """Default remote telemetry push endpoint."""

    endpoint: str
    protocol: str = "jsonl_http"
    compress: bool = False


class DefaultsTelemetry(BaseModel):
    """Default telemetry settings applied to all nodes."""

    remote: DefaultsTelemetryRemote


class DefaultsStorage(BaseModel):
    """Default storage base path for edge nodes."""

    base_path: str = "/opt/aureax"


class Defaults(BaseModel):
    """Network-wide defaults.

    NOT inherited automatically — nothing merges this block into a per-node
    config. Each per-node YAML sets these values itself (that is the whole point
    of the per-node-config pattern). Kept as the documented intent for the
    production reference; treat it as a comment, not a mechanism.
    """

    model: DefaultsModel = Field(default_factory=DefaultsModel)
    telemetry: DefaultsTelemetry | None = None
    storage: DefaultsStorage = Field(default_factory=DefaultsStorage)


# One filesystem path segment: no separator, no leading dot, bounded length.
# See SSHConfig._check_names_are_path_segments for why both names need it.
_PATH_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}")

# Node ids the TLS store owns: ~/.aureax/tls/ca.pem is the fleet's issuer and
# orchestrator.pem/.key is the dashboard's own pair. Spelled here rather than
# imported from tlsstore because both Dockerfiles copy this file into the edge
# image, which has no orchestrator package — tlsstore._check_leaf_name refuses
# the same two names at the path builder, and a test pins the two lists equal.
_RESERVED_NODE_IDS = ("ca", "orchestrator")

# Characters that terminate a URL authority. A host carrying one of these is
# not the host the resulting URL dials, and every fleet URL built from a target
# host carries that node's bearer key.
_URL_AUTHORITY_BREAKERS = ("@", "/", "\\", "?", "#")


class SSHTarget(BaseModel):
    """Connection details for a single edge node."""

    node_id: str
    host: str
    port: int = 22
    user: str = "aureax"
    display_name: str = ""
    tags: list[str] = Field(default_factory=list)
    key_file: str | None = None
    config_file: str | None = None

    # Set false to keep a node documented in this file without deploying to,
    # labeling, waiting for, or counting it. Decommissioned and not-yet-racked
    # hardware would otherwise stall every SSH preflight and pod-count loop.
    enabled: bool = True

    # Password auth, for nodes that ship with a factory password and no key.
    # `password` is a literal: a plaintext secret at rest in the fleet
    # definition, which the dashboard both serves over GET /api/config and lets
    # anyone reaching its unauthenticated bind rewrite. Prefer `password_env`,
    # which names an environment variable to read instead, so the secret never
    # lands in the file at all.
    # Either way, the first successful connect installs the orchestrator's
    # public key on the node (see ssh.ensure_key_installed), so subsequent
    # connects — and every bash script, which only knows `ssh -i` — use the
    # key and the password becomes a fallback.
    password: str | None = None
    password_env: str | None = None

    def resolve_password(self) -> str | None:
        """The effective password, or None if there is none to use.

        `password_env` WINS rather than falling back: when it names a variable
        nobody exported this returns None, which is indistinguishable from "no
        password configured" — so a caller reporting on the difference must ask
        `password_configured` too.
        """
        if self.password_env:
            return os.environ.get(self.password_env) or None
        return self.password or None

    @property
    def password_configured(self) -> bool:
        """Whether this target declares a password at all.

        Together with `resolve_password()` returning None this says "the
        password_env variable is not exported", which otherwise reads as
        "key-only target" — and silently skips the first-connect key bootstrap
        (see ARCHITECTURE.md, "SSH authentication").
        """
        return bool(self.password or self.password_env)

    @model_validator(mode="after")
    def _check_host_shape(self) -> SSHTarget:
        """``host`` reaches two places that parse it structurally.

        A group member's host is comma-joined into llama-server's ``--rpc``
        list. RPC0, RPC1, ... are positional device names, so a comma or a space
        here adds or renames devices and a hand-written `tensor_split` silently
        starts addressing other machines. validate_groups counts `group.workers`
        rather than the resolved list, so nothing downstream notices. Same
        list-parsing hazard as `alias`
        (LlamaServerConfig._check_server_string_shapes).

        Every node's URL is also built by interpolating this value into a URL
        authority — the dashboard's chat and access services, and each peer
        entry in a planner's ``${mcp_endpoints}`` — with that node's bearer key
        attached. ``@``, ``/``, ``\\``, ``?`` and ``#`` all end the authority
        early, so ``10.0.0.5@example.net`` is userinfo plus a *different* host
        and the key travels there instead. None of the five can appear in a
        hostname or an IP literal, so refusing them costs no legal value; the
        rest of the syntax is left alone deliberately, since aureax-network.yaml
        is an operator's file and an odd-but-working name is theirs to write.
        """
        if any(c in self.host for c in ",\t\r\n "):
            raise ValueError(
                "ssh.targets host must not contain commas or whitespace — worker "
                "hosts are comma-joined into llama-server's --rpc list, where the "
                "position of each entry names an RPC device"
            )
        if any(c in self.host for c in _URL_AUTHORITY_BREAKERS):
            raise ValueError(
                f"ssh.targets host {self.host!r} must not contain any of "
                f"{' '.join(_URL_AUTHORITY_BREAKERS)} — the host is interpolated into "
                "https://<host>:<port> URLs that carry this node's bearer key, and each "
                "of those characters ends the authority early, sending the key to a "
                "different host"
            )
        return self


class SSHConfig(BaseModel):
    """SSH connection settings and list of edge node targets."""

    key_store: str = "~/.aureax/ssh-keys/"
    connect_timeout_seconds: int = 15
    targets: list[SSHTarget]

    # Public key pushed to a node on first password-authenticated connect.
    # Set to "" to disable key bootstrapping entirely.
    bootstrap_public_key: str = "~/.ssh/id_ed25519.pub"

    @property
    def enabled_targets(self) -> list[SSHTarget]:
        """Targets that should actually be contacted."""
        return [t for t in self.targets if t.enabled]

    @model_validator(mode="after")
    def _check_unique_node_ids(self) -> SSHConfig:
        """node_id is the key everything downstream pairs on.

        Groups reference members by node_id, config generation writes one
        resolved file per node_id, lifecycle maps node_id to hostname, and the
        dashboard addresses nodes by it. A duplicate silently makes one of the
        two invisible — whichever loses the dict build — so it is worth
        catching in the config rather than as a node that mysteriously never
        deploys.

        Compared case-insensitively, because the stores these ids name are on
        the orchestrator's own filesystem and the default macOS volume is
        case-insensitive: ``edge-01`` and ``Edge-01`` are two targets here and
        one ``~/.aureax/keys/edge-01.key`` and one TLS leaf there, so the second
        node silently runs on the first one's credentials. The error reports the
        spelling the operator actually wrote.
        """
        seen: dict[str, str] = {}
        for t in self.targets:
            first = seen.get(t.node_id.casefold())
            if first is not None:
                same = "" if first == t.node_id else f" (already declared as {first!r})"
                raise ValueError(f"duplicate ssh.targets node_id: {t.node_id!r}{same}")
            seen[t.node_id.casefold()] = t.node_id
        return self

    @model_validator(mode="after")
    def _check_names_are_path_segments(self) -> SSHConfig:
        """node_id and config_file are joined into paths, so they must be safe.

        ``store_dir() / f"{node_id}.key"``, ``output_dir /
        f"aureax-edge-{node_id}.yaml"`` and ``configs_dir / config_file`` are
        all plain joins, and ``Path("/a") / "/etc/x"`` is ``/etc/x`` — a value
        carrying a separator or a leading dot writes the key file, or the
        key-bearing resolved config, outside the 0700 directories those sites
        create. aureax-network.yaml is editable over the dashboard's
        unauthenticated bind, so these values are not purely operator-authored.
        """
        for t in self.targets:
            for field, value in (("node_id", t.node_id), ("config_file", t.config_file)):
                if field == "config_file" and not value:
                    continue  # optional: a target may carry no config_file
                if not _PATH_SEGMENT_RE.fullmatch(value or ""):
                    raise ValueError(
                        f"ssh.targets {field} {value!r} must be a single path segment: "
                        "a letter or digit, then letters, digits, '.', '_' or '-' "
                        "(max 63 chars). It is joined straight into "
                        "~/.aureax/keys/<node_id>.key and configs/<config_file>."
                    )
            if t.node_id.casefold() in _RESERVED_NODE_IDS:
                # Same join, one directory over: ~/.aureax/tls/<node_id>.pem.
                # A node called "orchestrator" would share the dashboard's
                # certificate AND private key, and `aureax apply` would push
                # that key to it — an identity for the orchestrator's own
                # address, handed to a node. "ca" would overwrite the issuer.
                #
                # casefold, because the orchestrator's own filesystem decides
                # what the path means and the default macOS volume is
                # case-insensitive: "CA" passes an exact-match check and then
                # ~/.aureax/tls/CA.pem IS ca.pem, so generation replaces the
                # fleet's issuer with a node leaf.
                raise ValueError(
                    f"ssh.targets node_id {t.node_id!r} is reserved by the TLS store: "
                    f"~/.aureax/tls/{t.node_id}.pem is the fleet CA's or the dashboard's own "
                    "certificate, not a node's. Rename the node."
                )
        return self


class K3dConfig(BaseModel):
    """k3d cluster settings (the cluster runs on the orchestrator)."""

    api_port: int = 6443
    servers: int = 1  # passed to `k3d cluster create --servers`


class KubernetesConfig(BaseModel):
    """Kubernetes cluster configuration and node labeling rules."""

    cluster_name: str = "aureax"
    k3d: K3dConfig = Field(default_factory=K3dConfig)
    # CONTRACT: these two, and the shard pair below, are NOT the only copies of
    # their literals. The same strings are hardcoded in k8s/daemonset-edge.yaml
    # and k8s/daemonset-shard.yaml (nodeSelector), scripts/setup-all.sh (the
    # `kubectl label node` calls), scripts/test-all.sh (the label assertions)
    # and orchestrator/dashboard/static/sections/overview.js (the edge/shard
    # split in the node list). Nothing reconciles them, so changing a value
    # here alone leaves the Python side labelling one key while the manifests
    # select another — pods that are simply never scheduled, with no error.
    # Change all six places together.
    node_label: str = "aureax.io/role"
    edge_label_value: str = "edge"
    namespace: str = "aureax"

    # Value the DaemonSet nodeSelector is patched to in order to suspend
    # workloads. No node ever carries it, so the DaemonSet matches nothing and
    # K8s drains the pods while preserving the DaemonSet itself.
    stopped_label_value: str = "stopped"

    # Shard workers are selected by a SECOND, orthogonal label key. Keeping it
    # separate from `node_label` is what lets the existing stop/start machinery
    # keep working unmodified on heads and standalone nodes while workers get
    # an independent switch.
    shard_label: str = "aureax.io/shard"
    shard_label_value: str = "worker"


class NodeGroup(BaseModel):
    """A set of nodes that cooperatively serve ONE model via llama.cpp RPC.

    The head runs llama-server (with --rpc pointing at the workers) plus the
    MCP server, so the group is addressable exactly like a single node. The
    workers run only rpc-server and hold their share of the weights.

    `workers` ORDER IS LOAD-BEARING: it becomes the order of the --rpc list,
    which is what defines the device names RPC0, RPC1, ... that `--device` and
    `--tensor-split` index positionally. That is also why membership lives here
    rather than as a `group:` field on each SSH target, which would leave the
    ordering to however the targets happen to be listed.
    """

    name: str
    display_name: str = ""
    head: str
    workers: list[str] = Field(default_factory=list)
    rpc_port: int = DEFAULT_RPC_PORT

    # Whether the head also runs an rpc-server and takes a share of the
    # weights. Off by default: the head already streams the entire GGUF from
    # disk and runs sampling, KV, HTTP and MCP, and pointing it at itself adds
    # a loopback serialize/deserialize round-trip for tensors already in its
    # own address space.
    include_head_as_worker: bool = False

    # How long the head waits for every worker's RPC port before starting
    # llama-server. Without workers up, llama-server exits immediately.
    worker_wait_timeout_seconds: int = DEFAULT_WORKER_WAIT_SECONDS

    @property
    def members(self) -> list[str]:
        """Head first, then workers — the order `tensor_split` is written in."""
        return [self.head, *self.workers]

    def rpc_endpoints(self, host_by_node_id: dict[str, str]) -> list[str]:
        """`host:port` for each worker, in --rpc order.

        Workers only. The head never appears here: `include_head_as_worker` is
        rejected outright by the validator below, so a branch for it would be
        unreachable — and reading as though the head could be a peer is exactly
        the confusion the validator exists to prevent.

        A worker missing from the map raises rather than being skipped:
        silently shortening this list renames every later RPC device
        (RPC0, RPC1, ... are positional), so a hand-written tensor_split
        would start addressing other machines.
        """
        missing = [n for n in self.workers if n not in host_by_node_id]
        if missing:
            raise ValueError(
                f"group {self.name!r}: worker(s) {', '.join(missing)} have no "
                "ssh.targets entry — every worker must resolve to a host"
            )
        return [f"{host_by_node_id[n]}:{self.rpc_port}" for n in self.workers]


# A GGUF too large to publish as one file is split as `<stem>-00001-of-00003.gguf`.
# The name carries the part count, so nothing extra has to be configured — and it
# is why `filename` always points at part 1: llama.cpp is given part 1 and
# discovers the rest itself.
GGUF_SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$")

# A sha256 digest exactly as `shasum -a 256` / `sha256sum` print it. Lowercase
# only, so the value in the registry can be compared to the tool's output as a
# plain string on either side of the SSH boundary — a shell comparison has no
# case-insensitive form that is portable between bash and busybox ash.
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def gguf_part_names(filename: str) -> list[str]:
    """Every file making up a GGUF, given the name of its first part.

    Returns ``[filename]`` unchanged for a single-file model, so callers need no
    special case.
    """
    m = GGUF_SPLIT_RE.match(filename)
    if not m:
        return [filename]
    stem, total = m.group("stem"), int(m.group("total"))
    return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


class ModelRegistryEntry(BaseModel):
    """A single model available for deployment to edge nodes."""

    name: str
    # Local basename the file is stored under, and what llama.cpp is pointed at.
    # For a split GGUF this is PART 1.
    filename: str
    # DECIMAL GB (10^9 bytes), matching how model cards and HuggingFace quote
    # sizes — not GiB. download-model.sh's completeness check depends on this.
    size_gb: float
    quantization: str
    # RESERVED — no code reads this yet. Intended to validate a model against
    # node.role at deploy time; today model selection is purely per-node config.
    target_roles: list[str] = Field(default_factory=list)
    huggingface_repo: str | None = None
    # Path WITHIN the repo, which may include a directory — unsloth publishes
    # split quants under a per-quant subdirectory. May therefore differ from
    # `filename`, which is the flat local name.
    huggingface_file: str | None = None

    # Expected digest of a SINGLE-file GGUF. Optional: an operator who has no
    # digest to hand still gets the size checks, which is the behaviour every
    # existing registry entry keeps.
    sha256: str | None = None
    # One digest per part of a split GGUF, in the same positional order as
    # ``part_filenames`` and ``part_urls()``. A split model needs one per part
    # because each part is fetched separately and any one of them can be
    # truncated or substituted on its own.
    sha256_parts: list[str] = Field(default_factory=list)
    # Pin the download to an immutable HuggingFace ref — a commit sha, or a tag
    # the repo does not move. Unset means ``main``, which is mutable: the bytes
    # behind a URL can change between two provisions of the same fleet, which is
    # exactly what a declared digest then reports as a mismatch.
    revision: str | None = None

    @model_validator(mode="after")
    def _check_digests(self) -> ModelRegistryEntry:
        """Digests must be usable as written, or they verify nothing.

        A digest that is the wrong length, upper-cased, or attached to the wrong
        number of parts fails at download time on every node at once, after the
        multi-gigabyte fetch — so the shape is checked where the registry is
        parsed instead.
        """
        if self.sha256 and self.sha256_parts:
            raise ValueError(
                f"model {self.name!r}: set sha256 OR sha256_parts, not both — sha256 names the "
                "digest of a single-file GGUF, sha256_parts one digest per part of a split one"
            )
        for field, values in (("sha256", [self.sha256] if self.sha256 else []),
                              ("sha256_parts", self.sha256_parts)):
            for value in values:
                if not _SHA256_RE.fullmatch(value or ""):
                    raise ValueError(
                        f"model {self.name!r}: {field} entry {value!r} is not a sha256 digest — "
                        "64 lowercase hex characters, as `shasum -a 256` prints them"
                    )
        split = GGUF_SPLIT_RE.match(self.filename or "")
        if split and int(split.group("index")) != 1:
            return self  # _filename_is_first_part names the real fault
        parts = len(gguf_part_names(self.filename))
        if self.sha256 and parts > 1:
            raise ValueError(
                f"model {self.name!r}: sha256 names the digest of a single-file GGUF, but "
                f"{self.filename!r} declares {parts} parts. Use sha256_parts, one digest per "
                "part — each part is fetched separately and can be truncated on its own."
            )
        if self.sha256_parts and len(self.sha256_parts) != parts:
            raise ValueError(
                f"model {self.name!r}: sha256_parts has {len(self.sha256_parts)} digest(s) "
                f"but {self.filename!r} declares {parts} part(s). The lists are paired by "
                "position, so a mismatch verifies the wrong file or none at all."
            )
        return self

    @model_validator(mode="after")
    def _filename_is_first_part(self) -> ModelRegistryEntry:
        """A split model must be registered by its FIRST part.

        Pointing llama.cpp at part 2 loads a file with no metadata header and
        fails deep in the loader, long after deploy.
        """
        for field, value in (
            ("filename", self.filename),
            ("huggingface_file", self.huggingface_file),
        ):
            m = GGUF_SPLIT_RE.match(value or "")
            if not m:
                continue
            if int(m.group("index")) != 1:
                raise ValueError(
                    f"model {self.name!r}: {field} must name part 1 of a split GGUF, "
                    f"got {value!r} — llama.cpp is handed part 1 and finds the rest"
                )
            # `-of-00000` makes gguf_part_names() return NOTHING, so the entry
            # validates with an empty part list and download-model.sh reports
            # "no download URL available — model_registry needs an entry", which
            # sends the operator to the one place that is already correct.
            if int(m.group("total")) < 1:
                raise ValueError(
                    f"model {self.name!r}: {field} declares 0 parts "
                    f"({value!r}) — a split GGUF must declare at least one"
                )

        # `part_filenames` derives from `filename` and `part_urls` from
        # `huggingface_file`; download-model.sh zips the two lists POSITIONALLY.
        # If they disagree on part count the shorter one silently truncates the
        # download, leaving a model that is missing tensors — which surfaces
        # much later as a loader failure on the node, not as a config error.
        if self.huggingface_file:
            local_parts = len(gguf_part_names(self.filename))
            remote_parts = len(gguf_part_names(self.huggingface_file))
            if local_parts != remote_parts:
                raise ValueError(
                    f"model {self.name!r}: filename declares {local_parts} part(s) but "
                    f"huggingface_file declares {remote_parts} "
                    f"({self.filename!r} vs {self.huggingface_file!r}). "
                    "download-model.sh pairs these lists by position, so a mismatch "
                    "downloads an incomplete model."
                )
        return self

    @property
    def part_filenames(self) -> list[str]:
        """Every local file this model needs. One entry unless it is split."""
        return gguf_part_names(self.filename)

    def part_urls(self) -> list[str]:
        """HuggingFace download URL per part, in the same order as part_filenames.

        Empty when the entry carries no repo/file — a locally-staged model.

        ``revision`` selects the ref. Unset it is ``main``, which upstream is
        free to move, so two provisions of one fleet can fetch different bytes
        under the same URL; a commit sha there makes the fetch reproducible and
        gives a declared digest something stable to match.
        """
        if not (self.huggingface_repo and self.huggingface_file):
            return []
        ref = self.revision or DEFAULT_MODEL_REVISION
        base = f"https://huggingface.co/{self.huggingface_repo}/resolve/{ref}"
        return [f"{base}/{p}" for p in gguf_part_names(self.huggingface_file)]


class ModelRegistry(BaseModel):
    """Model catalog for the fleet. Where a model is STORED is per-node
    config (`model.local_path`), not a registry concern."""

    models: list[ModelRegistryEntry]


class TelemetryStorage(BaseModel):
    """Orchestrator-side telemetry storage settings."""

    # RESERVED — nothing reads this; the backend is hardcoded sqlite.
    engine: str = "sqlite"
    path: str = "~/.aureax/telemetry/metrics.db"
    retention_days: int = 30


class ServerTelemetry(BaseModel):
    """Telemetry receiver endpoint and storage config."""

    storage: TelemetryStorage = Field(default_factory=TelemetryStorage)


class ServerDashboard(BaseModel):
    """Dashboard web server settings."""

    # RESERVED — nothing consults this; the dashboard runs iff you start it.
    enabled: bool = True
    # Read by `aureax dashboard` (as the default bind port) and by `aureax
    # stop`, which kills whatever listens here — the two must stay coupled.
    port: int = 7001

    # Bind address for the HTTPS listener — the one the fleet uses. The
    # "0.0.0.0" default is NOT bound as a wildcard: `aureax dashboard` reads it
    # as "the address the fleet reaches this orchestrator at" and binds that
    # single address (the detected LAN IP, or --orchestrator-ip). It has to be
    # a specific address because the dashboard ALSO serves plain HTTP on
    # loopback at this same port, and a wildcard socket owns loopback too.
    #
    # That HTTPS bind is the load-bearing one: every edge node ships telemetry
    # to <orchestrator_url>/api/v1/telemetry and logs to /api/v1/logs, and
    # <orchestrator_url> names exactly this address — so a Mac with several LAN
    # interfaces serves the fleet on the one the certificate names, and
    # "127.0.0.1" is still NOT a safe drop-in: it silently kills fleet
    # telemetry and log shipping. Use loopback only with an SSH tunnel, and
    # expect the Overview tab to go empty (there is no second listener in that
    # case — HTTPS already holds the port on loopback).
    #
    # Setting a specific address here overrides only the BIND; the certificate
    # is still minted for --orchestrator-ip / the detected LAN IP, so give both
    # the same address or `aureax dashboard` warns about the mismatch it would
    # serve.
    #
    # The operator's own browser needs none of this: the second listener is
    # plain HTTP on 127.0.0.1 and ::1 at the same port, so http://localhost:7001
    # opens from this machine with no certificate and no trust store. Loopback
    # is not routable, so nothing there crosses the LAN and the fleet's
    # mandatory TLS is unchanged.
    #
    # The fleet's listener is HTTPS: `aureax dashboard` serves the orchestrator leaf
    # from ~/.aureax/tls, signed by the fleet CA every node already trusts, so
    # telemetry and logs are encrypted in transit and each node can tell it is
    # shipping to THIS orchestrator rather than to whatever answered on the
    # LAN.
    #
    # TLS authenticates the SERVER, not the caller. The dashboard remains
    # intentionally unauthenticated — no token, no login, no cookie — under the
    # NAT-plus-trusted-users assumption. Every route is still open to anything
    # that can reach the port, including GET /api/access/{node_id}/key and PUT
    # /api/config/..., and a browser that trusts the fleet CA can be driven
    # against the state-changing ones by a hostile page on the same LAN exactly
    # as before; encryption does nothing about that. It is the accepted risk
    # under that assumption. The LAN-address bind is load-bearing for node
    # ingest, so narrowing this knob is not the fix.
    host: str = "0.0.0.0"


class ServerConfig(BaseModel):
    """Orchestrator server settings — telemetry receiver and dashboard."""

    telemetry: ServerTelemetry = Field(default_factory=ServerTelemetry)
    dashboard: ServerDashboard = Field(default_factory=ServerDashboard)


class NetworkConfig(BaseModel):
    """Top-level schema for aureax-network.yaml."""

    network: NetworkInfo
    defaults: Defaults = Field(default_factory=Defaults)
    ssh: SSHConfig
    kubernetes: KubernetesConfig = Field(default_factory=KubernetesConfig)
    model_registry: ModelRegistry
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Empty by default, so every pre-existing config stays valid and a
    # deployment with no groups behaves exactly as it did before.
    node_groups: list[NodeGroup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_node_groups(self) -> NetworkConfig:
        """Validate group membership against ssh.targets.

        A group is a set of cross-referencing names spread over two YAML
        blocks, so a typo here otherwise surfaces much later as a head that
        waits forever for a worker that was never labelled.
        """
        if not self.node_groups:
            return self

        by_id = {t.node_id: t for t in self.ssh.targets}
        seen_names: set[str] = set()
        claimed: dict[str, str] = {}  # node_id -> group that already owns it

        for group in self.node_groups:
            if group.name in seen_names:
                raise ValueError(f"duplicate node_group name: {group.name!r}")
            seen_names.add(group.name)

            if not group.workers:
                raise ValueError(f"node_group {group.name!r} has no workers")

            if group.head in group.workers:
                raise ValueError(
                    f"node_group {group.name!r}: head {group.head!r} is also listed in workers"
                )

            if group.include_head_as_worker:
                # Not implemented, and it fails in a way that looks like a
                # network problem. Setting it puts the head's own host:50052
                # into the head's --rpc list, but nothing ever starts an
                # rpc-server there: client/entrypoint.py spawns one only when
                # node.role == "shard_worker", and the head is "edge_executor"
                # (which is also what gets it labelled at all). The head then
                # waits out worker_wait_timeout_seconds on a port nobody is
                # listening on and exits 1, over and over.
                raise ValueError(
                    f"node_group {group.name!r}: include_head_as_worker is not implemented — "
                    "nothing starts an rpc-server on the head, so it would wait out its peer "
                    "timeout on its own port and crash-loop. Leave it false."
                )

            dupes = {w for w in group.workers if group.workers.count(w) > 1}
            if dupes:
                raise ValueError(
                    f"node_group {group.name!r}: worker(s) listed more than once: {sorted(dupes)}"
                )

            for node_id in group.members:
                target = by_id.get(node_id)
                if target is None:
                    raise ValueError(
                        f"node_group {group.name!r} references unknown node_id {node_id!r}; "
                        f"known: {sorted(by_id)}"
                    )
                if not target.enabled:
                    raise ValueError(
                        f"node_group {group.name!r} references disabled node {node_id!r} — "
                        "a group cannot serve a model with a member excluded from deployment"
                    )
                if node_id in claimed:
                    raise ValueError(
                        f"node {node_id!r} is in both {claimed[node_id]!r} and {group.name!r}; "
                        "a node can belong to at most one group"
                    )
                claimed[node_id] = group.name

        return self

    def group_for_node(self, node_id: str) -> NodeGroup | None:
        """The group `node_id` belongs to, or None if it is standalone."""
        for group in self.node_groups:
            if node_id in group.members:
                return group
        return None

    def host_by_node_id(self) -> dict[str, str]:
        """node_id -> host, for resolving group members to RPC addresses."""
        return {t.node_id: t.host for t in self.ssh.targets}


# ============================================================================
# aureax-edge.yaml schema
# ============================================================================


class NodeConfig(BaseModel):
    """Edge node identity, role, and control plane connection."""

    id: str
    # Literal, not str: every label pass and lifecycle split FILTERS on these
    # exact strings, so a typo'd role that validated would fall out of all of
    # them silently — the node simply never gets a pod.
    role: Literal[HEAD_ROLE, SHARD_WORKER_ROLE, PLANNER_ROLE] = HEAD_ROLE
    display_name: str = ""
    tags: list[str] = Field(default_factory=list)

    # How long a sharded group's head waits for every worker's RPC port before
    # starting llama-server (which exits immediately if a peer is unreachable).
    # Configured on the NodeGroup and injected here by build_variables() — the
    # client reads it from this block, so without the injection the group-level
    # setting silently does nothing.
    worker_wait_timeout_seconds: int = DEFAULT_WORKER_WAIT_SECONDS

    # CPU cores the managed inference child is pinned to, by index. Empty (the
    # default) means no pinning.
    #
    # This exists for big.LITTLE SoCs, where the little cluster reaches roughly
    # half the memcpy bandwidth of the big one. ggml splits each operation's
    # rows evenly across its threads and barriers at every graph node, so one
    # thread parked on a little core gates the whole op — and because ggml's
    # barriers busy-spin, the fast threads burn their cores waiting for it.
    # Neither llama-server's `threads` nor rpc-server's `-t` constrains WHICH
    # cores the scheduler picks, so without this the placement is luck — list
    # the big cluster's core indices here.
    #
    # Applied in client/entrypoint.py::_spawn_child, which covers llama-server
    # and rpc-server in one place. Deliberately NOT an rpc-server flag: the
    # pinned build accepts only -h/-t/-d/-H/-p/-c and exits on anything else.
    cpu_affinity: list[int] = Field(default_factory=list)

    # Kill a leftover llama-server/rpc-server from a previous pod incarnation
    # before spawning our own. rpc-server serves ONE client, so an orphaned head
    # keeps the whole group's RPC slots and the replacement pod waits forever on
    # peers it can never reach — with no error, because from its side it is
    # simply still loading. Needs hostPID on the DaemonSet to see the orphan.
    reap_stale_children: bool = True


class LlamaServerConfig(BaseModel):
    """llama.cpp server launch parameters."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_LLAMA_PORT
    context_size: int = 4096
    batch_size: int = 512
    parallel_slots: int = 1
    gpu_layers: int = 0
    threads: int = 4
    # Batch and generation want the same core count on a node with one CPU pool,
    # so client/config/parser.py::get_llama_server_args MIRRORS `threads` here
    # when this key is absent or null in the per-node YAML — the default below
    # applies only to a config that reaches llama-server through this model.
    # Set it explicitly to split the two. A value of -1 (like `threads`) omits
    # the flag and lets llama-server auto-detect.
    threads_batch: int = 4
    use_mlock: bool = False
    use_mmap: bool = True

    # Tri-state: true / false / "auto". client/config/parser.py emits
    # --flash-attn on EVERY launch, `false` included, because an omitted flag
    # does not mean off: llama-server's own default is
    # LLAMA_FLASH_ATTN_TYPE_AUTO, and over RPC that probe always answers yes —
    # ggml_backend_rpc_device_supports_op is a bare `return true`. Omitting the
    # flag therefore resolves to flash attention ENABLED, so a head configured
    # `flash_attention: false` only gets what it asked for if the value is
    # passed explicitly.
    #
    # The default is "auto" because that is llama-server's own default: a node
    # that never sets the field is not forced off by this schema.
    flash_attention: bool | Literal["auto"] = "auto"
    # Omitted here means the FLAG is omitted (client/config/parser.py emits
    # -ctk/-ctv only for a present value) and llama.cpp's own default is f16 —
    # schema defaults are never serialised into a deployed config, so a default
    # of "q4_0" claimed a 4x smaller KV cache than an omitting config gets.
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    jinja: bool = False

    # Prometheus /metrics endpoint. ON by default: it is what
    # client/telemetry/collector.py derives tokens_per_second and
    # prompt_tokens_per_second from, and without it the fleet has no throughput
    # measurement at all. /health is not a substitute: the pinned build reports
    # no counters there, and the slot fields older builds exposed
    # (slots_idle/slots_processing) are an occupancy reading, not throughput.
    metrics: bool = True

    # --- API-key authentication --------------------------------------------
    # Bearer-token auth for llama-server's HTTP surface. In a TEMPLATE this is
    # `api_key: ${api_key}` — generation resolves it to the node's own key from
    # ~/.aureax/keys/<node_id>.key (created on demand). Absent/None = no auth,
    # which is what keeps the frozen 122B rollback pair valid and open.
    #
    # The key reaches llama-server via the LLAMA_API_KEY environment variable
    # set by the supervisor — NEVER as --api-key argv, which would be visible
    # in `ps` on a hostPID node and echoed into shipped logs. With a key set,
    # /health, /v1/health, /models, /v1/models and the web-UI assets stay
    # public on the pinned build (b10375; upstream removed the two /models
    # exemptions right after it), while /metrics, /props, /slots, /completion
    # and all /v1/* inference routes require `Authorization: Bearer <key>`.
    api_key: str | None = None

    # --- HTTP surface hardening --------------------------------------------
    # --cors-origins. None = flag omitted = llama-server's own default ('*',
    # echoing any Origin with credentials). Nothing in this repo is a
    # cross-origin browser consumer — the dashboard talks to llama-server
    # server-side and the embedded web UI is same-origin — so the shipped
    # default stays permissive and the API key is the actual control. Set this
    # if a node is ever reachable beyond the trusted LAN.
    cors_origins: str | None = None

    # /slots diagnostic endpoint. OFF by default (emits --no-slots): it leaks
    # the full prompt/response text of every in-flight request to anyone who
    # can reach the port, and nothing in this repo reads it. This is the one
    # new field whose default changes fleet behaviour.
    slots: bool = False

    # llama.cpp's embedded web UI at "/". Its static assets are EXEMPT from
    # api_key auth by design (the UI loads, then prompts for the key), so on a
    # node that should serve API traffic only, false (emits --no-webui) is the
    # control that actually removes the surface.
    webui: bool = True

    # --alias: the model id reported by /v1/models and /props. Unset, llama
    # reports the GGUF file path, which harnesses (aider, Continue, Open WebUI)
    # then require users to type verbatim. No commas: b10375 parses --alias as
    # a comma-separated LIST of aliases.
    alias: str | None = None

    # --backend-sampling keeps sampling on the compute device and returns a
    # token id instead of the full logits vector. The raw-logits path returns
    # one float per vocabulary entry per decoded token, over a blocking RPC
    # round trip — 993 KB per token at a 248,320-entry vocabulary, which is the
    # size the example fleet's models use. Scale that figure by the vocabulary
    # of whatever model this node actually serves.
    #
    # Off by default (upstream marks it experimental), and note it disables
    # ITSELF for any request carrying tools or a json_schema — which, with
    # jinja on, is most structured requests. Treat it as a small win on plain
    # chat, not a guaranteed one.
    backend_sampling: bool = False

    # How long to wait for /health after launch. Load time scales with model
    # size and, for a sharded group, with shipping every tensor to the RPC
    # peers over the network — minutes, not seconds. 120s is fine for a small
    # local model; a sharded head needs far more.
    health_timeout_seconds: int = DEFAULT_HEALTH_TIMEOUT_SECONDS

    # --- Distributed inference (llama.cpp RPC backend) ----------------------
    # Set only on a shard group's head. Empty everywhere else, which keeps the
    # generated command line byte-identical to the single-node case.
    #
    # rpc_servers: "host:port" of each rpc-server peer. ORDER IS LOAD-BEARING —
    #   it defines the device names RPC0, RPC1, ... that `devices` and
    #   `tensor_split` index positionally.
    # devices: explicit device list, e.g. ["RPC0", "RPC1"]. llama.cpp requires
    #   --device to appear AFTER --rpc.
    # tensor_split: relative weights per device. Leave empty to let llama.cpp
    #   split proportionally to each device's advertised free memory.
    rpc_servers: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    tensor_split: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_server_string_shapes(self) -> LlamaServerConfig:
        """Reject api_key and alias values llama.cpp would silently mangle.

        The key rule lives in the module-level :func:`_check_api_key`, which the
        MCP server and each remote-server entry share — the same minted string
        can end up in all three fields, so one of them accepting what another
        rejects is how a fleet becomes half-valid. That helper's docstring
        carries the per-character reasons.

        `alias` is checked here rather than there because it is llama-only, and
        it lives in this validator so both of the file's string rules are in one
        place — see the field comment above it.
        """
        _check_api_key(self.api_key, "model.server.api_key")
        if self.alias is not None and "," in self.alias:
            raise ValueError(
                "model.server.alias must not contain commas — llama.cpp parses "
                "--alias as a comma-separated list of aliases"
            )
        return self


class DraftModelConfig(BaseModel):
    """Speculative decoding: a small draft model proposing tokens for the target.

    SHIPPED DISABLED FLEET-WIDE. Both head templates set `enabled: false`, and
    configs/aureax-shard-head-35b.yaml's `draft:` block says why in detail: no
    --spec-draft-c is ever emitted, so the dense draft inherits the head's n_ctx
    and at context_size 131072 it OOMs the head's pod. Read that block before
    turning this on.

    Why the machinery exists at all: the draft proposes N tokens and the big
    model verifies all N in ONE forward pass, so the target's weights are read
    once per N tokens instead of once per token — the win scales with how much
    of the per-token byte budget is dense weights a batched verify reads once.

    It is also EXACTLY lossless. Rejection sampling reproduces the target
    model's output distribution bit for bit, which is what separates this from
    the other big lever (cutting `expert_used_count`) and why it survives a
    no-quality-loss constraint. On an MoE, though, do not expect linear scaling:
    verifying t tokens activates the union of experts across them, so a large
    `n_max` buys progressively less.

    Vocabulary must match the target EXACTLY or llama-server refuses to load.
    That is not the current guard: the draft in
    configs/aureax-network.example.yaml shares its vocabulary with both target
    models, so it loads fine — memory is what stops this, not vocabulary.
    """

    enabled: bool = False

    # Resolved against the same `model.local_path` as the target model, and
    # must exist in `model_registry.models` so download-model.sh can fetch it.
    filename: str = ""

    # --spec-draft-n-max / --spec-draft-n-min. Upstream default for n_max is 3.
    n_max: int = 4
    n_min: int = 1

    # --spec-draft-p-min: skip speculation when the draft's own confidence is
    # below this. Guards against burning target passes on drafts that will be
    # rejected anyway.
    p_min: float = 0.6

    # --spec-draft-ngl. MUST stay 0 on a sharded head: the target's layers live
    # on RPC devices, and letting the draft follow them there would ship the
    # draft's weights across the LAN and add hops to the very pass that is
    # supposed to be cheap. 0 keeps the draft entirely on the head's own cores.
    gpu_layers: int = 0

    # --spec-draft-threads. The head's own inference threads are otherwise idle
    # while workers compute, so the draft has cores to use.
    threads: int = 4

    @model_validator(mode="after")
    def _check_enabled_has_a_filename(self) -> DraftModelConfig:
        """`enabled` and `filename` are independent fields but the client needs
        both: client/config/parser.py emits the --spec-draft-* flags only for
        `enabled and filename`. So `draft: {enabled: true}` alone generates and
        deploys cleanly with speculation configured on and actually off, the
        only symptom being a throughput win that never arrives.
        """
        if self.enabled and not self.filename.strip():
            raise ValueError(
                "model.draft.enabled requires model.draft.filename — llama-server "
                "gets no --spec-draft-model and speculation is silently off"
            )
        return self


class RpcServerConfig(BaseModel):
    """llama.cpp ``rpc-server`` launch parameters (shard workers only).

    Kept separate from LlamaServerConfig because it configures a different
    binary with a different flag set.
    """

    # "auto" resolves to the pod's host IP via the AUREAX_HOST_IP downward-API
    # env var. Binding a specific interface rather than 0.0.0.0 keeps an
    # unauthenticated port off every other link the node happens to have.
    host: str = "auto"
    allowed_clients: list[str] = Field(default_factory=list)  # Generated from the group head.
    port: int = DEFAULT_RPC_PORT
    threads: int = -1          # -1 → omit the flag, matching parser.py convention
    cache: bool = True         # local tensor cache; avoids re-shipping weights on restart
    cache_dir: str = "/opt/aureax/rpc-cache"
    # No memory-advertisement setting: rpc-server in the pinned build accepts
    # only -h/-t/-d/-H/-p/-c, and an unknown option makes it exit at once. It
    # advertises real free memory to the head by itself.


class ModelConfig(BaseModel):
    """Model identity, file location, and server/inference settings."""

    source: str = "local"
    name: str
    filename: str
    quantization: str = "Q4_K_M"
    local_path: str = "/opt/aureax/models/"
    server: LlamaServerConfig = Field(default_factory=LlamaServerConfig)
    rpc_server: RpcServerConfig = Field(default_factory=RpcServerConfig)
    draft: DraftModelConfig = Field(default_factory=DraftModelConfig)


class AgentConfig(BaseModel):
    """LLM agent persona — system prompt and identity."""

    system_prompt: str = ""
    # RESERVED — only system_prompt is read (chat_service/planner). These two
    # exist for a future multi-agent identity story; ${group_id} resolves only
    # into agent_description today.
    agent_name: str = ""
    agent_description: str = ""


# RESERVED — ToolInputSchema/MCPTool and MCPServer.tools are declarative
# documentation only. The edge server registers its tools in code with
# @mcp.tool() decorators (client/mcp/server.py) and never reads these; a
# future registration-from-config or tool-filtering story would.
class ToolInputSchema(BaseModel):
    """JSON Schema for an MCP tool's input parameters."""

    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class MCPTool(BaseModel):
    """Definition of a single MCP tool exposed by the edge server."""

    name: str
    description: str = ""
    input_schema: ToolInputSchema = Field(default_factory=ToolInputSchema)


class MCPServer(BaseModel):
    """MCP server settings — transport, port, and registered tools."""

    enabled: bool = True
    # RESERVED — streamable-http is the only transport anything speaks.
    transport: str = "streamable-http"
    port: int = 9000

    # --- Bearer-token authentication ---------------------------------------
    # In a TEMPLATE this is `api_key: ${mcp_api_key}` — generation resolves it
    # to the node's own MCP key from ~/.aureax/keys/<node_id>.mcp.key, a
    # separate file from the llama key beside it. Absent/None = no auth, which
    # is what keeps the frozen 122B rollback pair and the shard-worker
    # templates valid: a worker sets enabled: false and runs no MCP server at
    # all, so a key there would be a credential nothing on the node reads.
    #
    # The key reaches the server through the RESOLVED CONFIG ONLY — never argv,
    # never an environment variable. llama-server needs LLAMA_API_KEY because
    # the supervisor spawns it as a separate process; the MCP server runs
    # in-process inside that same supervisor, which reads this field and hands
    # it straight to client/mcp/server.py, so there is no second copy to leak.
    # argv is world-readable under hostPID: true and is echoed into shipped
    # logs, which is exactly why the llama key never goes there either.
    #
    # /health stays exempt from the check. scripts/lib/common.sh's
    # mcp_health_wait and the kubelet's probes poll it with no credential, so
    # gating it would fail every deploy and crash-loop the pod.
    api_key: str | None = None

    # RESERVED — see the note above ToolInputSchema.
    tools: list[MCPTool] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_mcp_api_key(self) -> MCPServer:
        """Same alphabet as the llama key — see _check_api_key for why."""
        _check_api_key(self.api_key, "mcp.server.api_key")
        return self


class RemoteServer(BaseModel):
    """A remote MCP server the client connects to."""

    name: str
    url: str
    # RESERVED — streamable-http is the only transport anything speaks.
    transport: str = "streamable-http"

    # The PEER's MCP key, injected per entry by ${mcp_endpoints} at generate
    # time — present for a peer that enables MCP auth, absent for one that is
    # open, so a mixed fleet needs no flag. Never hand-written: templating reads
    # each peer's ~/.aureax/keys/<id>.mcp.key, and a literal pasted here goes
    # stale the moment that peer is rotated, which surfaces only as the
    # planner's remote turning 401 after an apply.
    api_key: str | None = None

    @model_validator(mode="after")
    def _check_remote_api_key(self) -> RemoteServer:
        _check_api_key(self.api_key, "mcp.client.remote_servers[].api_key")
        return self


class MCPClient(BaseModel):
    """MCP client settings — connects to remote MCP servers."""

    enabled: bool = False
    remote_servers: list[RemoteServer] = Field(default_factory=list)


class MCPConfig(BaseModel):
    """Combined MCP server and client config."""

    server: MCPServer = Field(default_factory=MCPServer)
    client: MCPClient = Field(default_factory=MCPClient)


class TelemetryCollect(BaseModel):
    """Which metric categories to collect on the edge node."""

    inference: list[str] = Field(default_factory=list)
    system: list[str] = Field(default_factory=list)
    agent: list[str] = Field(default_factory=list)


class TelemetryLocal(BaseModel):
    """Local telemetry buffering on the edge node."""

    db_path: str = "/opt/aureax/metrics/telemetry.db"
    # RESERVED — the collector's metric set is hardcoded (collect_all in
    # client/telemetry/collector.py); these lists do not select anything yet.
    collect: TelemetryCollect = Field(default_factory=TelemetryCollect)
    sample_interval_seconds: int = 60
    retention_days: int = 7
    # RESERVED — unenforced; retention_days (plus the 3x hard cutoff in
    # TelemetryStorage.rotate) is what actually bounds the buffer.
    max_db_size_mb: int = 50


class TelemetryRetry(BaseModel):
    """Retry policy for telemetry push failures."""

    max_attempts: int = 3
    backoff_base_seconds: int = 30
    backoff_max_seconds: int = 300


class TelemetryRemote(BaseModel):
    """Remote telemetry push settings — endpoint, batching, and retry."""

    # RESERVED — jsonl over HTTP is the only transport implemented.
    protocol: str = "jsonl_http"
    endpoint: str
    batch_size: int = 50
    push_interval_seconds: int = 60
    # RESERVED — batches are sent uncompressed.
    compress: bool = False
    retry: TelemetryRetry = Field(default_factory=TelemetryRetry)


class TelemetryConfig(BaseModel):
    """Edge node telemetry — local buffering and remote push."""

    enabled: bool = True
    local: TelemetryLocal = Field(default_factory=TelemetryLocal)
    remote: TelemetryRemote | None = None


# RESERVED — HealthCheck/ManagedService and ProcessConfig's service lists
# describe a declarative supervisor that nothing implements. The supervisor
# manages exactly one child, chosen by node.role in client/entrypoint.py, and
# reads only shutdown_timeout_seconds from this block.
class HealthCheck(BaseModel):
    """Health check probe for a managed service."""

    endpoint: str
    interval_seconds: int = 30
    timeout_seconds: int = 5
    unhealthy_threshold: int = 3


class ManagedService(BaseModel):
    """A child process managed by the edge client supervisor."""

    name: str
    type: str
    restart_policy: str = "always"
    max_restarts: int = 10
    restart_delay_seconds: int = 5
    health_check: HealthCheck | None = None
    conditional: str | None = None


class ProcessConfig(BaseModel):
    """Process supervisor settings — services, shutdown order, and timeouts."""

    # RESERVED — see the note above HealthCheck.
    managed_services: list[ManagedService] = Field(default_factory=list)
    # RESERVED — teardown order is fixed in AureaxClient.stop().
    shutdown_order: list[str] = Field(default_factory=list)
    shutdown_timeout_seconds: int = 30


# RESERVED — nothing reads any of these paths; every use site hardcodes its
# own. Kept as the shape a future platform-aware storage layout would fill in,
# for nodes whose filesystem conventions differ from the Linux ones below.
class StorageConfig(BaseModel):
    """Edge node filesystem paths for models, logs, queues, and metrics."""

    base_path: str = "/opt/aureax"
    models_path: str = "/opt/aureax/models"
    logs_path: str = "/opt/aureax/logs"
    queue_path: str = "/opt/aureax/mcp/queue"
    metrics_path: str = "/opt/aureax/metrics"
    tmp_path: str = "/tmp/aureax"


class LoggingConfig(BaseModel):
    """Edge node logging — level, format, rotation, and file paths."""

    # RESERVED, all six: the supervisor logs via logging.basicConfig and caps
    # the child log at a fixed size (client/entrypoint.py::_CappedLogFile) —
    # none of level/format/output/file_path/max_file_size_mb/max_files is read.
    level: str = "info"
    format: str = "json"
    output: str = "both"
    file_path: str = "/opt/aureax/logs/aureax-client.log"
    max_file_size_mb: int = 50
    max_files: int = 3
    llama_log_path: str = "/opt/aureax/logs/llama-server.log"

    # --- Remote log shipping -------------------------------------------------
    # Forwards node logs to the orchestrator over the same jsonl_http transport
    # telemetry already uses. This is the ONLY way node logs reach the
    # orchestrator:
    # `kubectl logs` proxies through each node's kubelet on :10250, which the
    # k3d server container cannot route to for a LAN-joined node, and a shard
    # worker runs no HTTP server of its own.
    ship_logs: bool = True

    # Where the local buffer lives, and the cursor marking what has shipped.
    log_db_path: str = "/opt/aureax/logs/system.db"

    # Levels forwarded as they happen. Everything below this is held in a ring
    # buffer and only persisted if the managed child dies — llama.cpp emits a
    # timing line every few tokens, so shipping "info" continuously would swamp
    # the orchestrator and put real traffic on a 1 GbE fleet that is already
    # latency-bound.
    ship_levels: list[str] = Field(default_factory=lambda: ["warn", "error"])

    # How many preceding lines to keep for a post-mortem. Buffered below
    # ship_levels and persisted only when the managed child dies, so a crash
    # ships with the output that explains it and a healthy node ships none of
    # it (client/log/capture.py::flush_crash_context).
    crash_context_lines: int = 80

    # The local buffer is hostPath-backed and survives pod restarts, so without
    # pruning it grows for the lifetime of the node. Mirrors
    # telemetry.local.retention_days; 0 disables pruning.
    retention_days: int = 7


class TLSPaths(BaseModel):
    """Where the node finds its TLS material. Stamped by the orchestrator.

    Not template-authored: templating.generate_edge_config assigns this block
    unconditionally, and the deployer installs the three files it names. A
    placeholder would be forgettable — nothing catches a template that simply
    lacks a block, and the result is one node that cannot start, discovered at
    deploy time — whereas an injected block cannot be omitted by a new or
    hand-edited template. It also leaves the frozen 122B rollback pair
    untouched, and stops advertising a per-node knob that does not exist: TLS
    is mandatory fleet-wide.

    The node-side filenames are fixed (`node.pem`, not `<node_id>.pem`) so that
    renaming a node in aureax-network.yaml cannot strand a file in
    /etc/aureax/tls that the config no longer names.
    """

    ca_file: str = "/etc/aureax/tls/ca.pem"
    cert_file: str = "/etc/aureax/tls/node.pem"
    key_file: str = "/etc/aureax/tls/node.key"


class DeploymentMetadata(BaseModel):
    """Stamped by the orchestrator during config generation."""

    deployed_at: str | None = None
    deployed_by: str | None = None
    config_version: str | None = None
    config_hash: str | None = None


class EdgeConfig(BaseModel):
    """Top-level schema for aureax-edge.yaml."""

    node: NodeConfig
    model: ModelConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    tls: TLSPaths = Field(default_factory=TLSPaths)
    deployment: DeploymentMetadata = Field(default_factory=DeploymentMetadata)
