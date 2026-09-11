"""Parse aureax-edge.yaml configuration on the client side."""

from __future__ import annotations

import os
from pathlib import Path

from ruamel.yaml import YAML

# The one orchestrator module the client may import: both Dockerfiles copy
# schema.py into the edge image (docker/Dockerfile.client), so the two sides
# share this walk instead of each keeping a copy that can drift.
from orchestrator.config.schema import DEFAULT_LLAMA_PORT, DEFAULT_RPC_PORT, to_plain_dict

yaml = YAML()


def load_edge_config_raw(path: str | Path = "/etc/aureax/aureax-edge.yaml") -> dict:
    """Load the edge config YAML. Returns raw dict (no Pydantic on client).

    Named ``_raw`` to distinguish it from ``orchestrator.config.loader``'s
    ``load_edge_config``, which returns a validated ``EdgeConfig`` model. Both
    are imported by orchestrator modules, and picking the wrong one silently
    swaps dict access for attribute access.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Edge config not found: {path}")
    # Explicit utf-8 rather than the locale default: this file is read inside
    # a container whose locale is POSIX, where the default is ASCII, and a
    # single non-ASCII character anywhere in the config would fail the read.
    with open(path, encoding="utf-8") as f:
        data = yaml.load(f)
    return to_plain_dict(data) if data else {}


_FLASH_ATTN_VALUES = ("on", "off", "auto")


def resolve_flash_attn(value: object) -> str:
    """Map a config `flash_attention` value onto llama-server's --flash-attn arg.

    Tri-state, and ALWAYS emitted by the caller — omitting the flag is not the
    same as `false`. llama-server's own default is LLAMA_FLASH_ATTN_TYPE_AUTO,
    and over RPC that probe always answers yes, because
    ggml_backend_rpc_device_supports_op is a bare `return true` with a TODO
    where the capability query should be. So on a sharded head an omitted flag
    silently means ON, whatever the config says.

    Which way it resolves is load-bearing at long context: without flash
    attention llama.cpp reserves an [n_kv x n_ubatch x n_head_q] F32 score
    tensor against the FULL cache, which is 4 GiB per worker at 131072 ctx,
    ubatch 512 and 16 query heads — on nodes with ~13 GiB free and no swap.

    An unrecognised string falls back to "auto" rather than "on": a typo should
    land on llama.cpp's own probe, not silently assert a setting nobody chose.
    """
    if value is None:
        return "auto"
    if isinstance(value, str):
        v = value.strip().lower()
        return v if v in _FLASH_ATTN_VALUES else "auto"
    return "on" if value else "off"


def get_llama_server_args(config: dict) -> list[str]:
    """Build llama-server command line args from config.

    model.server.api_key is DELIBERATELY not read here. argv is visible in
    `ps` to every process on the node — daemonset-edge.yaml sets hostPID:
    true — and _spawn_child logs the whole command line at INFO
    (client/entrypoint.py), which is teed to container stdout and therefore to
    `kubectl logs` and the orchestrator's shipped crash logs. The key reaches
    llama-server through the LLAMA_API_KEY environment variable, set in
    AureaxClient._start_llama_server; tests/test_llama_auth.py pins both halves
    of that contract.

    Raises:
        ValueError: when the config carries no ``tls.cert_file`` /
            ``tls.key_file``. There is no plaintext mode on this fleet.
    """
    model_cfg = config.get("model", {})
    server_cfg = model_cfg.get("server", {})
    model_path = Path(model_cfg.get("local_path", "/opt/aureax/models/")).expanduser()
    model_file = model_path / model_cfg.get("filename", "model.gguf")

    # `or` rather than a plain .get default: an explicit `threads:` with no value
    # parses as None, and comparing None > 0 below is a TypeError that crashes
    # the supervisor before llama-server is ever spawned. get_rpc_server_args
    # already guards the same case.
    threads = server_cfg.get("threads") or 4
    # threads_batch MIRRORS threads when unset — batch and generation want the
    # same core count on a node with one CPU pool. This is the behaviour of
    # record; LlamaServerConfig.threads_batch documents the same fallback, and
    # tests/test_rpc_args.py pins the two together.
    threads_batch = server_cfg.get("threads_batch") or threads

    # TLS is mandatory fleet-wide, so the flags are built here rather than
    # bolted on at the end: llama.cpp requires --device to follow --rpc, and
    # anything appended after that block would break the positional RPC0/RPC1
    # device naming a hand-written tensor_split depends on.
    #
    # Raising is the second half of failing loud. The alternative — emitting a
    # command line without them — starts a llama-server that serves plaintext
    # on :8080 and :30800 while every operator, every doc and every client in
    # this repo assumes HTTPS, and nothing in the stack would report it.
    tls_cfg = config.get("tls") or {}
    cert_file = tls_cfg.get("cert_file")
    key_file = tls_cfg.get("key_file")
    if not cert_file or not key_file:
        raise ValueError(
            "refusing to build a plaintext llama-server command line: the resolved "
            "config carries no tls.cert_file / tls.key_file. TLS is mandatory "
            "fleet-wide (ARCHITECTURE.md, 'TLS'); push this node's config with "
            "`aureax apply --node <id>`."
        )

    # `or` here too, for the same null-YAML reason as threads: a bare `host:`
    # yields None, which TypeErrors in _spawn_child's " ".join(cmd) before
    # llama-server is ever exec'd. The value is NOT normalised — argv must keep
    # the real bind address (0.0.0.0 is what the :30800 NodePort needs); only
    # the supervisor's own client-side URL maps wildcards to loopback.
    args = [
        "--model", str(model_file),
        "--host", str(server_cfg.get("host") or "127.0.0.1"),
        "--port", str(server_cfg.get("port", DEFAULT_LLAMA_PORT)),
        # PATHS, never key material: the private key stays in a root-only file
        # that llama-server opens itself. argv is world-readable under hostPID
        # and logged verbatim by _spawn_child, which is why the api_key takes
        # the environment instead (see this function's docstring).
        "--ssl-cert-file", str(cert_file),
        "--ssl-key-file", str(key_file),
        "--ctx-size", str(server_cfg.get("context_size", 4096)),
        "--batch-size", str(server_cfg.get("batch_size", 512)),
        "--n-gpu-layers", str(server_cfg.get("gpu_layers", 0)),
        "--parallel", str(server_cfg.get("parallel_slots", 1)),
    ]

    # Omit --threads when set to -1 (let llama-server auto-detect)
    if threads > 0:
        args += ["--threads", str(threads)]
    if threads_batch > 0:
        args += ["--threads-batch", str(threads_batch)]

    if server_cfg.get("use_mlock"):
        args.append("--mlock")
    if not server_cfg.get("use_mmap", True):
        args.append("--no-mmap")
    # Unconditional: see resolve_flash_attn. Omitting the flag hands the
    # decision to llama.cpp's AUTO probe, which over RPC always says yes.
    args += ["--flash-attn", resolve_flash_attn(server_cfg.get("flash_attention"))]
    if server_cfg.get("jinja"):
        args.append("--jinja")

    # Default TRUE, unlike every other bool here: the telemetry collector reads
    # tokens_per_second off this endpoint, so defaulting it off would silently
    # reinstate the blind spot it exists to fix.
    if server_cfg.get("metrics", True):
        args.append("--metrics")
    if server_cfg.get("backend_sampling"):
        args.append("--backend-sampling")

    # Omitted means the FLAG is omitted, which lands on llama.cpp's own f16 —
    # the same value LlamaServerConfig now defaults to, so the two agree. Do
    # not "restore" a default here: emitting one would silently change the KV
    # cache of every config that leaves the field out.
    cache_k = server_cfg.get("cache_type_k")
    cache_v = server_cfg.get("cache_type_v")
    if cache_k:
        args += ["-ctk", cache_k]
    if cache_v:
        args += ["-ctv", cache_v]

    # --- HTTP surface hardening ---------------------------------------------
    # Defaults re-declared from the schema (LlamaServerConfig), same as
    # `metrics`/`use_mmap` above: the client parses a raw dict with no Pydantic.
    #
    # /slots leaks the full prompt/response text of every in-flight request to
    # anyone who can reach the port, and nothing in this repo reads it: OFF
    # unless explicitly enabled.
    if not server_cfg.get("slots", False):
        args.append("--no-slots")

    # --no-webui, not --no-ui: both exist on the pinned build (b10375) but the
    # older spelling is the safer one across versions. The web UI's static
    # assets are EXEMPT from --api-key auth, so on an API-only node this flag
    # is what actually removes the surface.
    if not server_cfg.get("webui", True):
        args.append("--no-webui")

    cors = server_cfg.get("cors_origins")
    if cors:
        args += ["--cors-origins", str(cors)]

    # Friendly model id for /v1/models — without it llama reports the GGUF
    # path, which harness users then have to type verbatim.
    alias = server_cfg.get("alias")
    if alias:
        args += ["--alias", str(alias)]

    # --- Speculative decoding (draft model) ---------------------------------
    # Flag spellings verified against `llama-server --help` on the pinned build
    # (b10375), which uses the --spec-draft-* family; -md/--draft-p-min etc. are
    # aliases. Getting one wrong makes llama-server exit at startup, so the
    # canonical long forms are used here.
    #
    # Emitted only when explicitly enabled AND a filename is set: a draft model
    # whose vocabulary does not match the target is a hard load failure, so this
    # never turns itself on by inference.
    draft_cfg = model_cfg.get("draft") or {}
    if draft_cfg.get("enabled") and draft_cfg.get("filename"):
        draft_file = model_path / draft_cfg["filename"]
        args += [
            "--spec-draft-model", str(draft_file),
            "--spec-draft-n-max", str(draft_cfg.get("n_max", 4)),
            "--spec-draft-n-min", str(draft_cfg.get("n_min", 1)),
            "--spec-draft-p-min", str(draft_cfg.get("p_min", 0.6)),
            # Explicit, not defaulted: if the draft followed the target's
            # --n-gpu-layers onto the RPC devices, its weights would be shipped
            # over the LAN and every draft token would cost network hops.
            "--spec-draft-ngl", str(draft_cfg.get("gpu_layers", 0)),
        ]
        # Falls back to DraftModelConfig's own default of 4, not to "omit the
        # flag": schema defaults are never serialised into a deployed config, so
        # an omitting config would otherwise run llama.cpp's default while the
        # schema an author reads documents 4. -1 still means "omit and let
        # llama decide", matching --threads above.
        draft_threads = draft_cfg.get("threads") or 4
        if draft_threads > 0:
            args += ["--spec-draft-threads", str(draft_threads)]

    # --- Distributed inference (llama.cpp RPC backend) ----------------------
    # Appended last, and in this order, because llama.cpp requires --device to
    # come AFTER --rpc: remote devices are named RPC0, RPC1, ... in --rpc order
    # and --device resolves those names positionally.
    #
    # All three flags are omitted entirely when rpc_servers is empty, so a
    # standalone node's command line carries no RPC surface at all.
    rpc_servers = server_cfg.get("rpc_servers") or []
    if rpc_servers:
        args += ["--rpc", ",".join(str(s) for s in rpc_servers)]

        devices = server_cfg.get("devices") or []
        if devices:
            args += ["--device", ",".join(str(d) for d in devices)]

        tensor_split = server_cfg.get("tensor_split") or []
        if tensor_split:
            args += ["--tensor-split", ",".join(str(x) for x in tensor_split)]

    return args


def resolve_rpc_host(config: dict) -> str:
    """The address rpc-server will actually bind.

    Shared with the supervisor's readiness check: because "auto" binds one
    specific interface rather than 0.0.0.0, probing loopback would never see
    the listener even though it is up and serving.
    """
    rpc_cfg = (config.get("model") or {}).get("rpc_server", {})
    host = rpc_cfg.get("host", "auto")
    if host == "auto":
        # Bind the node's own LAN address rather than 0.0.0.0 — this port has
        # no authentication, so it should not be offered on every interface
        # the node happens to have. Falls back to all-interfaces only if the
        # downward-API env var is missing.
        return os.environ.get("AUREAX_HOST_IP") or "0.0.0.0"
    return str(host)


def resolve_rpc_port(config: dict) -> int:
    """The port rpc-server will actually bind.

    Shared with the supervisor's readiness check for the same reason
    :func:`resolve_rpc_host` is: the probe has to match the bind, and the two
    used to read the key separately with the default typed out on both sides.
    ``or`` rather than a plain default — a bare ``port:`` parses to None.
    """
    rpc_cfg = (config.get("model") or {}).get("rpc_server") or {}
    return int(rpc_cfg.get("port") or DEFAULT_RPC_PORT)


def get_rpc_server_args(config: dict) -> list[str]:
    """Build llama.cpp ``rpc-server`` args for a shard worker.

    The worker holds a share of the model's weights and executes tensor ops
    for the group's head. It serves no HTTP at all — readiness is a plain TCP
    accept on the RPC port.
    """
    rpc_cfg = (config.get("model") or {}).get("rpc_server", {})

    args = ["-H", resolve_rpc_host(config), "-p", str(resolve_rpc_port(config))]

    threads = rpc_cfg.get("threads", -1)
    if threads and threads > 0:
        args += ["-t", str(threads)]

    if rpc_cfg.get("cache", True):
        # Bare flag: the cache directory comes from $LLAMA_CACHE, which the
        # supervisor sets from rpc_server.cache_dir. Without the cache, every
        # pod restart re-pulls this node's multi-GiB share over the network.
        args.append("-c")

    # NOTE: no memory-advertisement flag. rpc-server in the pinned build
    # (b10375) accepts only -h/-t/-d/-H/-p/-c — verified against --help — and
    # passing an unknown option makes it exit immediately. It reports actual
    # free memory to the head on its own, which is what the default split uses.
    return args
