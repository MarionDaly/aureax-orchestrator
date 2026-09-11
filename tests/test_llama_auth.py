"""The llama-server argv contract, and where the API key is allowed to live.

The one rule this module exists for: the key NEVER appears on the command
line. daemonset-edge.yaml sets ``hostPID: true``, so every process on the node
— including any other container — can read the head's argv out of /proc, and
``_spawn_child`` logs the whole command line at INFO into both log sinks
(container stdout, hence ``kubectl logs`` and the orchestrator's shipped crash
logs). ``--api-key aureax_sk_...`` would therefore leak the fleet's keys into
files that get copied around casually. The key travels in the LLAMA_API_KEY
environment variable instead, which llama-server applies through the same
append handler as the flag.

The second rule is the allowlist: an argv the code builds by hand is only as
safe as the set of flags it can emit, so every ``-``-prefixed element produced
by any config in the matrix must be a flag someone listed here on purpose.

The third is TLS: the two ``--ssl-*-file`` flags must always be there, must
carry paths rather than PEM bytes (the same argv exposure the key avoids), and
a config that lacks them must make the builder RAISE — a llama-server started
without them serves plaintext on :8080 and :30800 with every probe green.
"""

from __future__ import annotations

import copy
import os

import pytest

from client.config.parser import get_llama_server_args

SECRET = "aureax_sk_secret123456789"

# The block orchestrator/config/templating.py stamps into every resolved
# config, and the deployer installs at these paths. Every entry below carries
# it because the builder refuses to produce a plaintext command line without
# it — that refusal is pinned by its own test at the bottom of this file.
TLS = {
    "ca_file": "/etc/aureax/tls/ca.pem",
    "cert_file": "/etc/aureax/tls/node.pem",
    "key_file": "/etc/aureax/tls/node.key",
}


def _cfg(**server) -> dict:
    """A config shaped like a real edge YAML, with `server` overrides applied."""
    return {
        "tls": TLS,
        "model": {
            "filename": "m.gguf",
            "local_path": "/opt/aureax/models/",
            "server": {"host": "0.0.0.0", "port": 8080, **server},
        },
    }


# Every shape of node the fleet actually runs. Anything asserted "for every
# config" below is asserted against all of them, so a flag that only appears
# for, say, a sharded head still has to be accounted for.
MATRIX: dict[str, dict] = {
    # The emptiest input the builder must still handle: get_llama_server_args
    # re-declares every default because the client parses raw YAML with no
    # Pydantic. Only the tls block is present, because that one is mandatory.
    "all-defaults": {"tls": TLS},
    "standalone": _cfg(),
    "head-with-rpc": _cfg(
        gpu_layers=99,
        rpc_servers=["10.77.0.2:50052", "10.77.0.3:50052"],
        devices=["RPC0", "RPC1"],
        tensor_split=[1, 1],
    ),
    "draft-enabled": {
        "tls": TLS,
        "model": {
            "filename": "m.gguf",
            "local_path": "/opt/aureax/models/",
            "draft": {"enabled": True, "filename": "d.gguf", "threads": 2},
            "server": {"host": "0.0.0.0", "port": 8080},
        },
    },
    "fully-hardened": _cfg(
        context_size=131072,
        batch_size=256,
        gpu_layers=99,
        parallel_slots=2,
        threads=8,
        threads_batch=8,
        use_mlock=True,
        use_mmap=False,
        flash_attention=True,
        jinja=True,
        metrics=True,
        backend_sampling=True,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        slots=False,
        webui=False,
        cors_origins="https://console.example.internal",
        alias="qwen3.5-35b",
    ),
}

MATRIX_IDS = sorted(MATRIX)


def _with_api_key(config: dict, key: str = SECRET) -> dict:
    """Copy of `config` with model.server.api_key set."""
    out = copy.deepcopy(config)
    out.setdefault("model", {}).setdefault("server", {})["api_key"] = key
    return out


# Every flag client/config/parser.py::get_llama_server_args is allowed to emit.
# Deliberately hand-maintained: adding a flag to the builder without adding it
# here fails the allowlist test, which is the point — a new llama-server flag
# is a decision about the node's exposed surface, not an implementation detail.
ALLOWED_FLAGS = {
    "--model", "--host", "--port", "--ctx-size", "--batch-size",
    "--n-gpu-layers", "--parallel", "--threads", "--threads-batch",
    "--mlock", "--no-mmap", "--flash-attn", "--jinja", "--metrics",
    "--backend-sampling", "-ctk", "-ctv",
    # HTTP surface hardening
    "--no-slots", "--no-webui", "--cors-origins", "--alias",
    # speculative decoding
    "--spec-draft-model", "--spec-draft-n-max", "--spec-draft-n-min",
    "--spec-draft-p-min", "--spec-draft-ngl", "--spec-draft-threads",
    # TLS (mandatory) — paths only, never key material; see the ssl test below
    "--ssl-cert-file", "--ssl-key-file",
    # distributed inference (always last)
    "--rpc", "--device", "--tensor-split",
}


# --- the key never reaches argv ----------------------------------------------


@pytest.mark.parametrize("name", MATRIX_IDS)
def test_api_key_is_never_passed_on_the_command_line(name) -> None:
    """argv is world-readable under hostPID and is logged verbatim.

    daemonset-edge.yaml runs the edge pods with hostPID: true, so `ps` on the
    node shows every argument to every process on it; _spawn_child then logs
    the command line into llama-server.log AND container stdout, which the
    orchestrator ships. A key on argv is a key in shipped logs.
    """
    args = get_llama_server_args(_with_api_key(MATRIX[name]))

    assert "--api-key" not in args
    assert "--api-key-file" not in args
    assert not any(SECRET in a for a in args)
    assert not any("aureax_sk_" in a for a in args), f"key prefix leaked into argv: {args}"


def test_api_key_does_not_perturb_the_rest_of_the_command_line() -> None:
    """Setting a key must change nothing else — it is an env-only concern."""
    plain = get_llama_server_args(MATRIX["fully-hardened"])
    keyed = get_llama_server_args(_with_api_key(MATRIX["fully-hardened"]))
    assert plain == keyed


# --- flags that must never be emitted ----------------------------------------


# Flags the builder must never emit, whatever the config says. None of them has
# a config field today, and none should grow one by accident.
NEVER_EMITTED = (
    # Each is the trigger for an unauthenticated remote memory-corruption bug
    # in llama-server: CVE-2026-21869 and CVE-2026-43631 respectively. Both are
    # OFF unless the flag turns them on.
    "--context-shift",
    "--sleep-idle-seconds",
    # The physical ubatch must stay at llama.cpp's 512 default. llama.cpp sizes
    # the attention score tensor [n_kv x n_ubatch x n_head_q], so at 131072
    # context a raised n_ubatch costs GiB per worker — and the shard workers'
    # memory limit is set between the flash-attention-on and -off peaks
    # deliberately, with no headroom for this. batch_size is the only lever
    # that is meant to exist, and it clamps n_ubatch downward only.
    "--ubatch-size",
    "-ub",
)


@pytest.mark.parametrize("name", MATRIX_IDS)
@pytest.mark.parametrize("flag", NEVER_EMITTED)
def test_deny_listed_flags_are_never_emitted(flag, name) -> None:
    """Flags with no config field, pinned so none quietly acquires one."""
    args = get_llama_server_args(_with_api_key(MATRIX[name]))
    assert flag not in args


@pytest.mark.parametrize("name", MATRIX_IDS)
def test_every_emitted_flag_is_on_the_allowlist(name) -> None:
    """No flag reaches llama-server from a MATRIX config without being listed here.

    llama-server takes ~200 flags, several of which widen the network surface
    (--host 0.0.0.0 variants, --path, --api-key-file, --slots-endpoint-path).
    An allowlist over the whole config matrix means a future flag has to be
    added consciously, in this file, next to the reason it is safe.

    Its reach is exactly the configs in MATRIX: a branch gated on a field no
    entry sets emits nothing here and passes unnoticed. That hole is what
    test_every_llama_server_config_field_is_exercised below closes, and it is
    why a new config field belongs in MATRIX at the same time.
    """
    args = get_llama_server_args(_with_api_key(MATRIX[name]))
    emitted = {a for a in args if a.startswith("-")}
    assert emitted <= ALLOWED_FLAGS, f"unlisted flag(s): {sorted(emitted - ALLOWED_FLAGS)}"


# Schema fields that are deliberately not argv inputs, so MATRIX owes them
# nothing: health_timeout_seconds is read by the supervisor's own health wait
# (client/entrypoint.py::_wait_for_llama_health), never by the arg builder.
NON_ARGV_SERVER_FIELDS = {"health_timeout_seconds"}


def test_every_llama_server_config_field_is_exercised() -> None:
    """Every model.server field must be set by some MATRIX entry.

    This is what gives the allowlist above its teeth. The allowlist can only
    see flags the matrix actually provokes, so a new config-gated branch is
    invisible to it until some entry sets the field that turns it on. Requiring
    the coverage here means adding a schema field forces a matrix entry, which
    in turn forces any flag it emits through the allowlist.

    api_key is covered by _with_api_key, which every argv assertion applies.
    """
    from orchestrator.config.schema import LlamaServerConfig

    exercised = {"api_key"}
    for config in MATRIX.values():
        exercised |= set((config.get("model") or {}).get("server") or {})

    missing = set(LlamaServerConfig.model_fields) - NON_ARGV_SERVER_FIELDS - exercised
    assert not missing, (
        f"model.server field(s) no MATRIX config sets: {sorted(missing)}. Add them to "
        "an entry (or to NON_ARGV_SERVER_FIELDS with a reason) — otherwise any flag "
        "they gate is never built here and the allowlist never sees it."
    )


# --- /slots, web UI, CORS, alias ---------------------------------------------


def test_slots_endpoint_is_off_by_default() -> None:
    """/slots returns the full prompt AND response text of in-flight requests
    to anyone who can reach the port, and nothing in this repo reads it."""
    assert "--no-slots" in get_llama_server_args(_cfg())
    assert "--no-slots" in get_llama_server_args(MATRIX["all-defaults"])


def test_metrics_defaults_on_and_can_be_turned_off() -> None:
    """The one bool here that defaults TRUE. client/telemetry/collector.py reads
    tokens_per_second off /metrics, so flipping the default would silently
    reinstate the throughput blind spot the endpoint exists to fix — with the
    whole suite still green."""
    assert "--metrics" in get_llama_server_args(_cfg())
    assert "--metrics" in get_llama_server_args(MATRIX["all-defaults"])
    assert "--metrics" not in get_llama_server_args(_cfg(metrics=False))


def test_kv_cache_type_is_omitted_rather_than_defaulted() -> None:
    """Omitting -ctk/-ctv lands on llama.cpp's own f16, which is what
    LlamaServerConfig defaults to — the two must not drift, because a schema
    default is never serialised into a deployed config, so a parser-side
    default of its own would silently change the KV cache of every config that
    leaves the field out."""
    from orchestrator.config.schema import LlamaServerConfig

    args = get_llama_server_args(MATRIX["all-defaults"])
    assert "-ctk" not in args and "-ctv" not in args
    assert LlamaServerConfig.model_fields["cache_type_k"].default == "f16"
    assert LlamaServerConfig.model_fields["cache_type_v"].default == "f16"

    keyed = get_llama_server_args(_cfg(cache_type_k="q8_0", cache_type_v="q8_0"))
    assert keyed[keyed.index("-ctk") + 1] == "q8_0"


def test_draft_threads_falls_back_to_the_schema_default() -> None:
    """A draft-enabled config that omits `threads` used to emit no
    --spec-draft-threads at all, so it got llama.cpp's default instead of the 4
    DraftModelConfig documents. Schema defaults never reach a node, so this
    fallback is the only thing that makes the documented value real."""
    from orchestrator.config.schema import DraftModelConfig

    default = DraftModelConfig.model_fields["threads"].default
    args = get_llama_server_args(
        {"tls": TLS, "model": {"draft": {"enabled": True, "filename": "d.gguf"}}}
    )
    assert args[args.index("--spec-draft-threads") + 1] == str(default)

    # -1 still means "omit and let llama-server decide", like --threads.
    negative = get_llama_server_args(
        {"tls": TLS, "model": {"draft": {"enabled": True, "filename": "d.gguf", "threads": -1}}}
    )
    assert "--spec-draft-threads" not in negative


def test_a_null_host_does_not_break_the_command_line() -> None:
    """`host:` with no value parses as None, which TypeErrors in _spawn_child's
    " ".join(cmd) before llama-server is ever exec'd — the same null-YAML
    hazard `threads` is guarded against."""
    args = get_llama_server_args({"tls": TLS, "model": {"server": {"host": None}}})
    assert args[args.index("--host") + 1] == "127.0.0.1"
    # A real wildcard bind is passed through untouched: llama-server must keep
    # listening on 0.0.0.0 for the :30800 NodePort.
    wild = get_llama_server_args(_cfg(host="0.0.0.0"))
    assert wild[wild.index("--host") + 1] == "0.0.0.0"


def test_slots_can_be_enabled_explicitly() -> None:
    assert "--no-slots" not in get_llama_server_args(_cfg(slots=True))


def test_webui_stays_on_by_default() -> None:
    """Turning the UI off for every node would be a behaviour change nobody asked
    for — the fleet's heads are reached through it during demos."""
    assert "--no-webui" not in get_llama_server_args(_cfg())
    assert "--no-webui" not in get_llama_server_args(MATRIX["all-defaults"])


def test_webui_false_emits_no_webui() -> None:
    """The web UI's static assets are EXEMPT from api_key auth by design, so on
    an API-only node this flag — not the key — is what removes the surface."""
    assert "--no-webui" in get_llama_server_args(_cfg(webui=False))


def test_cors_and_alias_are_absent_unless_set() -> None:
    """Both default to llama-server's own behaviour when the field is unset."""
    args = get_llama_server_args(_cfg())
    assert "--cors-origins" not in args
    assert "--alias" not in args


def test_cors_and_alias_carry_their_values() -> None:
    args = get_llama_server_args(
        _cfg(cors_origins="https://console.example.internal", alias="qwen3.5-35b")
    )
    assert args[args.index("--cors-origins") + 1] == "https://console.example.internal"
    assert args[args.index("--alias") + 1] == "qwen3.5-35b"


# --- ordering ----------------------------------------------------------------


def test_rpc_flags_still_come_last_with_every_hardening_flag_set() -> None:
    """llama.cpp resolves RPC0/RPC1 positionally from --rpc order.

    --device and --tensor-split must follow --rpc, and the hardening flags added
    since must not have been appended after the RPC block — inserting anything
    between --rpc and --device is what makes a hand-written tensor_split
    silently address the wrong machines.
    """
    cfg = copy.deepcopy(MATRIX["fully-hardened"])
    cfg["model"]["server"].update(
        rpc_servers=["10.77.0.2:50052", "10.77.0.3:50052"],
        devices=["RPC0", "RPC1"],
        tensor_split=[1, 1],
    )
    args = get_llama_server_args(cfg)

    rpc = args.index("--rpc")
    for flag in ("--no-webui", "--cors-origins", "--alias", "--flash-attn",
                 "--metrics", "--backend-sampling", "-ctk", "-ctv"):
        assert args.index(flag) < rpc, f"{flag} was emitted after --rpc"

    assert rpc < args.index("--device") < args.index("--tensor-split")
    assert args[args.index("--device") + 1] == "RPC0,RPC1"


# --- TLS is mandatory --------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "tls_block"),
    [
        ("no tls block at all", None),
        ("an empty tls block", {}),
        ("a block with no cert", {"ca_file": "/etc/aureax/tls/ca.pem",
                                  "key_file": "/etc/aureax/tls/node.key"}),
        ("a block with no key", {"ca_file": "/etc/aureax/tls/ca.pem",
                                 "cert_file": "/etc/aureax/tls/node.pem"}),
        ("an empty cert path", {**TLS, "cert_file": ""}),
    ],
)
def test_a_config_without_tls_paths_refuses_to_build_a_command_line(name, tls_block) -> None:
    """Emitting the command line anyway is the failure this raise prevents.

    llama-server without --ssl-cert-file serves plain HTTP on :8080 and, via
    the NodePort, on :30800 — while every client in this repo, every doc and
    the operator dial https. Nothing downstream would report it: the pod goes
    Ready, /health answers, and the only symptom is that the fleet is
    unencrypted. There is no plaintext mode, so the builder refuses instead.
    """
    cfg = copy.deepcopy(MATRIX["standalone"])
    if tls_block is None:
        cfg.pop("tls")
    else:
        cfg["tls"] = tls_block

    with pytest.raises(ValueError, match="refusing to build a plaintext"):
        get_llama_server_args(cfg)


@pytest.mark.parametrize("name", MATRIX_IDS)
def test_the_ssl_flags_carry_paths_and_never_key_material(name) -> None:
    """PEM bytes on argv would leak the node's private key exactly like the api_key.

    argv is world-readable under hostPID and _spawn_child logs it into both
    sinks, so the flags must name root-only FILES that llama-server opens
    itself. The paths are the ones the deployer installs.
    """
    args = get_llama_server_args(_with_api_key(MATRIX[name]))

    assert args[args.index("--ssl-cert-file") + 1] == TLS["cert_file"]
    assert args[args.index("--ssl-key-file") + 1] == TLS["key_file"]
    assert not any("BEGIN" in a for a in args), f"PEM material on argv: {args}"
    assert not any("PRIVATE KEY" in a for a in args)


def test_the_ssl_flags_precede_the_rpc_block() -> None:
    """--device resolves RPC0/RPC1 positionally from --rpc order.

    llama.cpp requires --device to follow --rpc, so anything appended after
    that block shifts the device naming a hand-written tensor_split depends on.
    The ssl flags go in with --port, at the head of the line.
    """
    args = get_llama_server_args(MATRIX["head-with-rpc"])

    rpc = args.index("--rpc")
    assert args.index("--ssl-cert-file") < rpc
    assert args.index("--ssl-key-file") < rpc
    # And directly after --port, where the raise and the flags are written.
    assert args[args.index("--port") + 2] == "--ssl-cert-file"


# --- the other half of the contract: LLAMA_API_KEY ---------------------------


@pytest.fixture
def clean_llama_env(monkeypatch):
    """Run with LLAMA_API_KEY unset, and restore the real environment after.

    _start_llama_server mutates os.environ directly (the child inherits it),
    so a test that leaves the variable behind hands a live key to whichever
    test runs next — and to every subprocess it spawns.

    The seed-then-delete dance is what makes the teardown unconditional.
    monkeypatch records an undo only for a name it observed, and delenv on a
    name that is already absent records nothing at all — so on a machine where
    LLAMA_API_KEY happens to be unset, the key production writes behind
    monkeypatch's back would survive the whole session. setenv first puts the
    original state (set or absent) in the ledger; the undo then removes that
    direct write whether or not any test touched the variable.
    """
    monkeypatch.setenv("LLAMA_API_KEY", "")
    monkeypatch.delenv("LLAMA_API_KEY")
    yield


def _client(config: dict, monkeypatch, bare_client, captured: list):
    """An AureaxClient whose _spawn_child records the child's environment."""
    from client.entrypoint import AureaxClient

    client = bare_client(config=config)

    def fake_spawn(_self, name, cmd, log_path):
        # Read the environment at SPAWN time — that is the moment the child
        # would inherit it, and the only moment the value matters.
        captured.append((name, cmd, os.environ.get("LLAMA_API_KEY")))

    monkeypatch.setattr(AureaxClient, "_spawn_child", fake_spawn)
    return client


def _keyed_config(api_key: str | None) -> dict:
    server = {"host": "127.0.0.1", "port": 8080}
    if api_key is not None:
        server["api_key"] = api_key
    return {
        "tls": TLS,
        "model": {"filename": "m.gguf", "local_path": "/opt/aureax/models/", "server": server},
        "logging": {"llama_log_path": "/tmp/aureax-test-llama.log"},
    }


def test_configured_key_is_handed_over_through_the_environment(
    monkeypatch, clean_llama_env, bare_client
) -> None:
    """The key must reach the child, and only through LLAMA_API_KEY."""
    captured: list = []
    client = _client(_keyed_config(SECRET), monkeypatch, bare_client, captured)

    client._start_llama_server()

    (name, cmd, env_key) = captured[0]
    assert name == "llama-server"
    assert env_key == SECRET, "the child would have inherited no key"
    assert not any("aureax_sk_" in a for a in cmd), f"key leaked into argv: {cmd}"


def test_a_stale_environment_key_is_popped_when_the_config_has_none(
    monkeypatch, clean_llama_env, bare_client
) -> None:
    """POPPED, not merely skipped.

    _spawn_child passes no env=, so the child inherits ours. A LLAMA_API_KEY
    from anywhere else (a DaemonSet env block, a debug shell, a previous
    attach) would keep auth ON for a node whose config says it is off — and the
    in-pod clients read the CONFIG, so they would send no header and 401
    against their own llama-server.
    """
    monkeypatch.setenv("LLAMA_API_KEY", "aureax_sk_left_over_from_somewhere_else")
    captured: list = []
    client = _client(_keyed_config(None), monkeypatch, bare_client, captured)

    client._start_llama_server()

    (_name, cmd, env_key) = captured[0]
    assert env_key is None, "a stale key stayed in the child's environment"
    assert "LLAMA_API_KEY" not in os.environ
    assert not any("aureax_sk_" in a for a in cmd)


def test_llama_api_key_helper_reads_the_config_not_the_environment(
    monkeypatch, clean_llama_env, bare_client
) -> None:
    """The per-node YAML pushed to /etc/aureax is authoritative for auth state.

    Every in-pod client (telemetry collector, chat client, harness) asks this
    helper what key to send, so if it read the environment instead they could
    drift apart from what llama-server is actually enforcing.
    """
    client = bare_client()

    monkeypatch.setenv("LLAMA_API_KEY", "aureax_sk_environment_value_only")
    client.config = _keyed_config(None)
    assert client._llama_api_key() == ""

    client.config = _keyed_config(SECRET)
    assert client._llama_api_key() == SECRET
