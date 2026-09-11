"""llama-server / rpc-server command-line construction for sharded groups."""

from __future__ import annotations

import pytest

from client.config.parser import (
    get_llama_server_args,
    get_rpc_server_args,
    resolve_flash_attn,
)

# The block orchestrator/config/templating.py stamps into every resolved
# config. get_llama_server_args REFUSES to build a command line without it —
# TLS is mandatory fleet-wide and a plaintext llama-server would serve :8080
# and :30800 unencrypted while every client here assumes HTTPS. The refusal
# itself is pinned in tests/test_llama_auth.py; every config below carries the
# block so these tests are about the flags they are named for.
TLS = {
    "ca_file": "/etc/aureax/tls/ca.pem",
    "cert_file": "/etc/aureax/tls/node.pem",
    "key_file": "/etc/aureax/tls/node.key",
}

BASE = {
    "tls": TLS,
    "model": {
        "filename": "m.gguf",
        "local_path": "/opt/aureax/models/",
        "server": {"host": "0.0.0.0", "port": 8080, "threads": 4, "threads_batch": 4},
    },
}


def _with_server(**server) -> dict:
    cfg = {
        "tls": TLS,
        "model": {**BASE["model"], "server": {**BASE["model"]["server"], **server}},
    }
    return cfg


def test_standalone_node_emits_no_rpc_flags() -> None:
    """A node outside a group must produce the pre-existing command line."""
    args = get_llama_server_args(BASE)
    assert "--rpc" not in args
    assert "--device" not in args
    assert "--tensor-split" not in args


def test_rpc_servers_are_comma_joined() -> None:
    args = get_llama_server_args(_with_server(rpc_servers=["10.77.0.2:50052", "10.77.0.3:50052"]))
    assert args[args.index("--rpc") + 1] == "10.77.0.2:50052,10.77.0.3:50052"


def test_device_flag_follows_rpc_flag() -> None:
    """llama.cpp resolves RPC0/RPC1 positionally and requires this order."""
    args = get_llama_server_args(
        _with_server(rpc_servers=["a:1", "b:2"], devices=["RPC0", "RPC1"])
    )
    assert args.index("--device") > args.index("--rpc")
    assert args[args.index("--device") + 1] == "RPC0,RPC1"


def test_tensor_split_is_emitted_after_rpc() -> None:
    args = get_llama_server_args(_with_server(rpc_servers=["a:1", "b:2"], tensor_split=[1, 2.5]))
    assert args.index("--tensor-split") > args.index("--rpc")
    assert args[args.index("--tensor-split") + 1] == "1,2.5"


def test_device_and_tensor_split_ignored_without_rpc_servers() -> None:
    """Both index RPC devices, so they are meaningless with no peers."""
    args = get_llama_server_args(_with_server(devices=["RPC0"], tensor_split=[1, 1]))
    assert "--device" not in args
    assert "--tensor-split" not in args


def test_gpu_layers_passes_through_for_sharding() -> None:
    """-ngl 99 is required or every layer stays local and workers idle."""
    args = get_llama_server_args(_with_server(gpu_layers=99, rpc_servers=["a:1"]))
    assert args[args.index("--n-gpu-layers") + 1] == "99"


# --- flash attention -------------------------------------------------------
#
# --flash-attn is ALWAYS emitted, for every value including `false`. Omitting
# it does not mean "off": llama-server's own default is
# LLAMA_FLASH_ATTN_TYPE_AUTO, and over RPC that probe always resolves to
# enabled, because ggml_backend_rpc_device_supports_op is a bare `return true`.
# So a flag emitted only when truthy makes `flash_attention: false` a silent
# no-op and the head runs WITH flash attention.
#
# It matters at long context: without flash attention llama.cpp reserves an
# [n_kv x n_ubatch x n_head_q] F32 score tensor, 4 GiB per worker at 131072 ctx
# on nodes with ~13 GiB free and no swap.


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (True, "on"),
        (False, "off"),
        ("auto", "auto"),
        ("on", "on"),
        ("off", "off"),
        ("AUTO", "auto"),
        (None, "auto"),
        ("nonsense", "auto"),
    ],
)
def test_flash_attn_resolves_tri_state(configured, expected) -> None:
    assert resolve_flash_attn(configured) == expected


@pytest.mark.parametrize("configured", [True, False, "auto"])
def test_flash_attn_flag_is_always_emitted(configured) -> None:
    """Omitting the flag silently hands the decision to llama.cpp's AUTO probe."""
    args = get_llama_server_args(_with_server(flash_attention=configured))
    assert "--flash-attn" in args


def test_flash_attn_false_actually_disables() -> None:
    """`false` must emit `--flash-attn off`, not nothing at all."""
    args = get_llama_server_args(_with_server(flash_attention=False))
    assert args[args.index("--flash-attn") + 1] == "off"


def test_flash_attn_defaults_to_auto_when_unset() -> None:
    """An unset field must keep llama.cpp's own behaviour, not newly force off."""
    args = get_llama_server_args(BASE)
    assert args[args.index("--flash-attn") + 1] == "auto"


def test_rpc_server_args_defaults(monkeypatch) -> None:
    monkeypatch.delenv("AUREAX_HOST_IP", raising=False)
    args = get_rpc_server_args({"model": {"rpc_server": {}}})
    # host "auto" with no downward-API value falls back to all interfaces.
    assert args[:4] == ["-H", "0.0.0.0", "-p", "50052"]
    assert "-c" in args


def test_rpc_server_host_auto_resolves_from_downward_api(monkeypatch) -> None:
    """Binding one interface keeps an unauthenticated port off the others."""
    monkeypatch.setenv("AUREAX_HOST_IP", "192.0.2.20")
    args = get_rpc_server_args({"model": {"rpc_server": {"host": "auto"}}})
    assert args[:2] == ["-H", "192.0.2.20"]


def test_rpc_server_explicit_host_wins(monkeypatch) -> None:
    monkeypatch.setenv("AUREAX_HOST_IP", "192.0.2.20")
    args = get_rpc_server_args({"model": {"rpc_server": {"host": "127.0.0.1"}}})
    assert args[:2] == ["-H", "127.0.0.1"]


def test_rpc_server_omits_threads_when_negative() -> None:
    """-1 means 'let llama.cpp decide', matching the llama-server convention."""
    assert "-t" not in get_rpc_server_args({"model": {"rpc_server": {"threads": -1}}})
    assert "-t" in get_rpc_server_args({"model": {"rpc_server": {"threads": 4}}})


def test_rpc_server_cache_flag_is_bare() -> None:
    """`-c` takes no value in the pinned build, and there is no -m/--mem."""
    args = get_rpc_server_args({"model": {"rpc_server": {"cache": False}}})
    assert "-c" not in args

    args = get_rpc_server_args({"model": {"rpc_server": {"cache": True}}})
    assert "-c" in args
    # An unknown option makes rpc-server exit immediately, so the emitted set
    # must stay within what --help actually lists.
    assert {a for a in args if a.startswith("-")} <= {"-H", "-p", "-t", "-c", "-d"}


def test_resolve_rpc_host_matches_the_bound_address(monkeypatch) -> None:
    """The supervisor's readiness probe must target what rpc-server binds.

    `host: "auto"` binds this node's LAN IP only, so a readiness check against
    127.0.0.1 never succeeds — the worker serves happily while the supervisor
    declares it dead and K8s crash-loops the pod.
    """
    from client.config.parser import resolve_rpc_host

    monkeypatch.setenv("AUREAX_HOST_IP", "192.0.2.20")
    cfg = {"model": {"rpc_server": {"host": "auto"}}}
    args = get_rpc_server_args(cfg)

    bound = args[args.index("-H") + 1]
    assert resolve_rpc_host(cfg) == bound == "192.0.2.20"

    # Explicit host, and the no-downward-API fallback, must agree too.
    cfg = {"model": {"rpc_server": {"host": "127.0.0.1"}}}
    assert resolve_rpc_host(cfg) == get_rpc_server_args(cfg)[1] == "127.0.0.1"

    monkeypatch.delenv("AUREAX_HOST_IP")
    cfg = {"model": {"rpc_server": {"host": "auto"}}}
    assert resolve_rpc_host(cfg) == get_rpc_server_args(cfg)[1] == "0.0.0.0"


# --- speculative decoding ----------------------------------------------------

def test_draft_flags_are_absent_unless_explicitly_enabled() -> None:
    """A draft model with a mismatched vocab is a hard load failure.

    So this must never switch itself on by inference: neither a filename alone
    nor `enabled` alone is sufficient.
    """
    assert "--spec-draft-model" not in get_llama_server_args({"tls": TLS, "model": {}})
    assert "--spec-draft-model" not in get_llama_server_args(
        {"tls": TLS, "model": {"draft": {"filename": "draft.gguf"}}}
    )
    assert "--spec-draft-model" not in get_llama_server_args(
        {"tls": TLS, "model": {"draft": {"enabled": True}}}
    )


def test_draft_model_path_is_resolved_against_local_path() -> None:
    args = get_llama_server_args(
        {
            "tls": TLS,
            "model": {
                "local_path": "/opt/aureax/models/",
                "draft": {"enabled": True, "filename": "Qwen3.5-0.8B-Q4_K_M.gguf"},
            },
        }
    )
    assert "/opt/aureax/models/Qwen3.5-0.8B-Q4_K_M.gguf" in args


def test_draft_never_offloads_to_rpc_devices() -> None:
    """--spec-draft-ngl must always be emitted, and default to 0.

    The target's layers live on RPC devices. If the draft inherited that
    offload, its weights would cross the LAN and every drafted token would cost
    network hops — defeating the point of drafting locally on the idle head.
    """
    args = get_llama_server_args(
        {
            "tls": TLS,
            "model": {
                "draft": {"enabled": True, "filename": "d.gguf"},
                "server": {"gpu_layers": 99, "rpc_servers": ["10.77.0.2:50052"]},
            },
        }
    )
    assert "--spec-draft-ngl" in args
    assert args[args.index("--spec-draft-ngl") + 1] == "0"


def test_draft_enabled_without_a_filename_is_refused_by_the_schema() -> None:
    """The client above must never infer this, which leaves the schema as the
    only place the mistake can be caught: `draft: {enabled: true}` alone
    generates and deploys cleanly with speculation configured on and silently
    off, the only symptom being a throughput win that never arrives."""
    import re

    from pydantic import ValidationError

    from orchestrator.config.schema import DraftModelConfig

    for bad in ({}, {"filename": "   "}):
        with pytest.raises(ValidationError, match=re.escape("draft.enabled requires")):
            DraftModelConfig(enabled=True, **bad)

    # Both shipped head configs are enabled:false with a filename present.
    assert DraftModelConfig(enabled=False, filename="d.gguf").enabled is False
    assert DraftModelConfig(enabled=True, filename="d.gguf").enabled is True


# --- KV cache type -----------------------------------------------------------

def test_kv_cache_defaults_state_what_an_omitting_config_runs() -> None:
    """-ctk/-ctv are emitted only for a present value, and generation writes the
    RAW resolved template — schema defaults never reach a node. So the schema
    default has to name llama.cpp's own default (f16), or it claims a 4x smaller
    KV cache than an omitting config actually gets at 131072 context.
    """
    from orchestrator.config.schema import LlamaServerConfig

    args = get_llama_server_args(BASE)
    assert "-ctk" not in args
    assert "-ctv" not in args
    assert LlamaServerConfig().cache_type_k == "f16"
    assert LlamaServerConfig().cache_type_v == "f16"


# --- threads / threads-batch -------------------------------------------------

def test_threads_batch_mirrors_threads_when_unset() -> None:
    """An omitted threads_batch runs with --threads-batch == --threads.

    Schema defaults are never serialised into a deployed config, so the parser
    is what a node actually runs while LlamaServerConfig is what a config author
    reads. The mirror is the behaviour of record: batch and generation want the
    same core count on a node with one CPU pool. Pinned here so the two homes
    cannot drift, the same way the -ctk/-ctv pair above is pinned.
    """
    args = get_llama_server_args(_with_server(threads=12, threads_batch=None))
    assert args[args.index("--threads") + 1] == "12"
    assert args[args.index("--threads-batch") + 1] == "12"

    # Neither set: both land on the parser's own default of 4.
    bare = get_llama_server_args({"tls": TLS, "model": {**BASE["model"], "server": {}}})
    assert bare[bare.index("--threads") + 1] == "4"
    assert bare[bare.index("--threads-batch") + 1] == "4"

    # An explicit threads_batch still wins over the mirror.
    split = get_llama_server_args(_with_server(threads=12, threads_batch=6))
    assert split[split.index("--threads") + 1] == "12"
    assert split[split.index("--threads-batch") + 1] == "6"
