"""Layer placement for a sharded group: which layers land on which device.

llama.cpp decides placement from two numbers the head is launched with,
``--n-gpu-layers`` and ``--tensor-split``, by a rule that is easy to get wrong
by hand (src/llama-model.cpp at b10375, ``get_layer_buft_list``; not re-read at
the current pin, b11429):

- ``i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0)``: layers below it stay on
  the head's own CPU.
- The remaining ``n_layer + 1 - i_gpu_start`` slots — every offloaded layer plus
  ONE extra slot for the output tensor, which is placed as if it were layer
  ``n_layer`` — are handed to the RPC devices in ``--rpc`` order by cumulative
  fraction of the normalised split: slot ``j`` goes to the first device whose
  cumulative fraction exceeds ``j / slots``.

Two consequences this module exists to make explicit. First, an even split is
not an even footprint: ``[1, 1, 1]`` over 40 layers is 14 / 14 / 12 + output,
and only some layers carry a KV cache. Second, INTEGER slot counts reproduce
themselves exactly under that rule, so a split written as counts — with the
last entry one larger, for the output tensor — places precisely the layers it
names. That is the form this module emits, and :func:`replay` is the rule
itself, used by validation and by the budget tests to see what a split does.

Policies: ``even`` is llama.cpp's own even split written as counts (byte for
byte the placement ``[1, 1, 1]`` gives today, so the default changes nothing);
``balanced`` reads the GGUF header for per-layer byte sizes and KV-cache
layers and minimises the heaviest device's footprint over all contiguous
partitions, which is what a head-resident layer range or an uneven fleet
needs. Neither policy needs anything beyond the standard library; the GGUF
reader is :mod:`orchestrator.config.gguf`.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path

# llama.cpp's own value: layers at or above this are "offload everything",
# and the head templates have always written it.
GPU_LAYERS_ALL = 99

# Bytes per KV-cache element for the cache types the fleet uses, as llama.cpp
# stores them (ggml type sizes: f16 2 B; q8_0 34 B per 32; q4_0 18 B per 32).
KV_BYTES_PER_ELEMENT = {"f16": 2.0, "bf16": 2.0, "f32": 4.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32}


@dataclass(frozen=True)
class LayerProfile:
    """What placement needs to know about a model, per layer.

    ``layer_bytes[il]`` is the weight bytes of block ``il``; ``kv_layer[il]``
    says whether the block keeps a KV cache (a full-attention layer) rather
    than a fixed recurrent state; ``kv_bytes_per_token`` is the per-token KV
    cost of ONE such layer at the configured cache type.
    """

    n_layer: int
    layer_bytes: list[int]
    kv_layer: list[bool]
    kv_bytes_per_token: float
    output_bytes: int = 0
    embedding_bytes: int = 0
    recurrent_bytes: int = 0  # per non-KV layer, per sequence; small, informational


@dataclass(frozen=True)
class Placement:
    """One group's computed launch values and the footprint they imply."""

    gpu_layers: int
    tensor_split: list[int]
    head_layers: int
    counts: list[int]              # slots per device; the last includes the output tensor
    device_bytes: list[int] = field(default_factory=list)  # weights + KV per device, when known
    head_bytes: int = 0            # weights + KV resident on the head, when known
    policy: str = "even"
    source: str = "none"           # "gguf", "registry" or "none" — where n_layer came from


# --- the rule itself -----------------------------------------------------------

def replay(n_layer: int, gpu_layers: int, tensor_split: list[float], n_devices: int) -> list[int]:
    """Where llama.cpp puts each of layers 0..n_layer (n_layer = the output tensor).

    Returns one entry per slot: ``-1`` for the head's CPU, else the RPC device
    index in ``--rpc`` order. A transcription of ``get_layer_buft_list`` at
    b10375 (not re-read at the current pin, b11429), including its float
    arithmetic, so that what this predicts is what the head does.
    """
    if n_devices <= 0 or gpu_layers <= 0:
        return [-1] * (n_layer + 1)
    weights = [float(x) for x in tensor_split] if tensor_split else [1.0] * n_devices
    if len(weights) != n_devices:
        raise ValueError(f"tensor_split has {len(weights)} entries for {n_devices} device(s)")
    total = sum(weights)
    if total <= 0:
        raise ValueError("tensor_split must sum to a positive number")
    cumulative: list[float] = []
    acc = 0.0
    for w in weights:
        acc += w
        cumulative.append(acc / total)
    i_gpu_start = max(n_layer + 1 - gpu_layers, 0)
    act_gpu_layers = min(gpu_layers, n_layer + 1)
    out: list[int] = []
    for il in range(n_layer + 1):
        if il < i_gpu_start or (il - i_gpu_start) >= act_gpu_layers:
            out.append(-1)
            continue
        frac = float(il - i_gpu_start) / act_gpu_layers
        out.append(min(bisect.bisect_right(cumulative, frac), n_devices - 1))
    return out


def even_counts(slots: int, n_devices: int) -> list[int]:
    """llama.cpp's even split, as the slot counts it actually produces.

    Device ``k`` gets the slots ``j`` with ``k/n <= j/slots < (k+1)/n``, so the
    first devices carry the remainder: 41 slots over 3 devices is 14/14/13
    (the last 13 being 12 layers plus the output tensor), exactly what a
    ``[1, 1, 1]`` split gives.
    """
    if n_devices <= 0:
        return []
    bounds = [math.ceil(k * slots / n_devices) for k in range(n_devices + 1)]
    return [bounds[k + 1] - bounds[k] for k in range(n_devices)]


def balanced_counts(profile: LayerProfile, head_layers: int, n_devices: int,
                    ctx_tokens: int, n_seq: int = 1) -> tuple[list[int], list[int]]:
    """Contiguous partition of the offloaded slots minimising the heaviest device.

    Slot bytes are weights plus the KV cache the layer will hold at ``ctx_tokens``
    (KV is allocated on the device that owns the layer), plus the output tensor
    on the last slot. Returns ``(counts, device_bytes)``. Dynamic programme over
    prefixes: ``n_layer x n_devices x n_layer`` is microseconds here.
    """
    if n_devices <= 0:
        return [], []
    slots: list[int] = []
    for il in range(head_layers, profile.n_layer):
        kv = int(profile.kv_bytes_per_token * ctx_tokens) if profile.kv_layer[il] else profile.recurrent_bytes * n_seq
        slots.append(profile.layer_bytes[il] + kv)
    slots.append(profile.output_bytes)
    n = len(slots)
    if n < n_devices:
        raise ValueError(
            f"{n} offloaded slot(s) cannot be spread over {n_devices} device(s) — "
            "lower head_layers or use fewer workers"
        )
    prefix = [0]
    for b in slots:
        prefix.append(prefix[-1] + b)

    # best[d][i]: minimal max-load when the first i slots are spread over d devices
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(n_devices + 1)]
    cut = [[0] * (n + 1) for _ in range(n_devices + 1)]
    best[0][0] = 0
    for d in range(1, n_devices + 1):
        for i in range(d, n + 1):
            for j in range(d - 1, i):
                load = max(best[d - 1][j], prefix[i] - prefix[j])
                if load < best[d][i]:
                    best[d][i] = load
                    cut[d][i] = j
    counts: list[int] = []
    i = n
    for d in range(n_devices, 0, -1):
        j = cut[d][i]
        counts.append(i - j)
        i = j
    counts.reverse()
    device_bytes: list[int] = []
    start = 0
    for c in counts:
        device_bytes.append(prefix[start + c] - prefix[start])
        start += c
    return counts, device_bytes


# --- reading a model -------------------------------------------------------------

def profile_from_gguf(path: Path, cache_type_k: str = "f16", cache_type_v: str = "f16") -> LayerProfile:
    """Per-layer sizes and KV layers from a GGUF header (data is never read)."""
    from orchestrator.config import gguf

    header = gguf.read_header(path)
    arch = gguf.architecture(header)
    meta = header.metadata
    n_layer = int(gguf.block_count(header))
    layer_bytes = [0] * n_layer
    has_k = [False] * n_layer
    has_qkv = [False] * n_layer
    has_ssm = [False] * n_layer
    output_bytes = 0
    embedding_bytes = 0
    for t in header.tensors:
        nbytes = gguf.tensor_nbytes(t)
        il = gguf.layer_index(t.name)
        if il is None:
            if t.name.startswith("output.") or t.name == "output.weight":
                output_bytes += nbytes
            elif t.name.startswith("token_embd"):
                embedding_bytes += nbytes
            continue
        if il >= n_layer:
            # An MTP/nextn block beyond block_count: not a decode layer.
            continue
        layer_bytes[il] += nbytes
        rest = t.name.split(".", 2)[2] if t.name.count(".") >= 2 else ""
        if rest.startswith("attn_k."):
            has_k[il] = True
        if rest.startswith("attn_qkv."):
            has_qkv[il] = True
        if rest.startswith("ssm_"):
            has_ssm[il] = True
    # A layer keeps a KV cache when it has a separate K projection, or a fused
    # QKV projection without recurrent (ssm_*) state. Hybrid Gated-DeltaNet
    # layers carry ssm_* tensors and no attn_k, so they fall out as recurrent.
    kv_layer = [has_k[il] or (has_qkv[il] and not has_ssm[il]) for il in range(n_layer)]

    n_head = int(meta.get(f"{arch}.attention.head_count", 0) or 0)
    n_head_kv = int(meta.get(f"{arch}.attention.head_count_kv", n_head) or n_head)
    n_embd = int(meta.get(f"{arch}.embedding_length", 0) or 0)
    head_dim = n_embd // n_head if n_head else 0
    key_len = int(meta.get(f"{arch}.attention.key_length", head_dim) or head_dim)
    value_len = int(meta.get(f"{arch}.attention.value_length", head_dim) or head_dim)
    k_bytes = KV_BYTES_PER_ELEMENT.get(str(cache_type_k).lower(), 2.0)
    v_bytes = KV_BYTES_PER_ELEMENT.get(str(cache_type_v).lower(), 2.0)
    kv_bytes_per_token = n_head_kv * (key_len * k_bytes + value_len * v_bytes)

    # Recurrent state per sequence for ssm layers (f32): d_state x d_inner plus
    # the conv window (d_conv - 1) x (d_inner + 2 x n_group x d_state).
    d_state = int(meta.get(f"{arch}.ssm.state_size", 0) or 0)
    d_inner = int(meta.get(f"{arch}.ssm.inner_size", 0) or 0)
    d_conv = int(meta.get(f"{arch}.ssm.conv_kernel", 0) or 0)
    n_group = int(meta.get(f"{arch}.ssm.group_count", 0) or 0)
    recurrent = 0
    if d_state and d_inner:
        recurrent = 4 * (d_state * d_inner + max(d_conv - 1, 0) * (d_inner + 2 * n_group * d_state))

    return LayerProfile(
        n_layer=n_layer, layer_bytes=layer_bytes, kv_layer=kv_layer,
        kv_bytes_per_token=kv_bytes_per_token, output_bytes=output_bytes,
        embedding_bytes=embedding_bytes, recurrent_bytes=recurrent,
    )


# --- the group-level decision ------------------------------------------------------

def plan(*, n_devices: int, head_layers: int = 0, policy: str = "even",
         profile: LayerProfile | None = None, n_layer: int | None = None,
         ctx_tokens: int = 0, n_seq: int = 1, source: str = "none") -> Placement:
    """Compute ``gpu_layers`` and an integer ``tensor_split`` for one group.

    With ``head_layers == 0`` and policy ``even`` the result is llama.cpp's own
    even split — written as ``[1, 1, ...]`` precisely so a fleet that never
    asked for placement keeps the command line it has today. Any head-resident
    layers need the layer count (from the GGUF profile or the registry's
    ``layers`` hint); ``balanced`` needs the profile.
    """
    if n_devices <= 0:
        raise ValueError("a sharded group needs at least one RPC device")
    if head_layers < 0:
        raise ValueError("head_layers must be 0 or more")
    if profile is not None:
        n_layer = profile.n_layer
        source = "gguf"

    if head_layers == 0 and policy == "even":
        return Placement(gpu_layers=GPU_LAYERS_ALL, tensor_split=[1] * n_devices,
                         head_layers=0, counts=even_counts((n_layer + 1) if n_layer else 0, n_devices) if n_layer else [],
                         policy="even", source=source)

    if n_layer is None:
        raise ValueError(
            "head-resident layers or balanced placement need the model's layer count: "
            "put the GGUF in models/ (download-model.sh) or set `layers:` on its "
            "model_registry entry"
        )
    if head_layers >= n_layer:
        raise ValueError(f"head_layers {head_layers} leaves no layer for {n_devices} RPC device(s) of a {n_layer}-layer model")
    slots = n_layer + 1 - head_layers
    gpu_layers = slots  # i_gpu_start = n_layer + 1 - gpu_layers = head_layers

    if policy == "balanced":
        if profile is None:
            raise ValueError("balanced placement needs the GGUF in models/ to read per-layer sizes")
        counts, device_bytes = balanced_counts(profile, head_layers, n_devices, ctx_tokens, n_seq)
    elif policy == "even":
        counts, device_bytes = even_counts(slots, n_devices), []
        if profile is not None:
            device_bytes = _device_bytes(profile, head_layers, counts, ctx_tokens, n_seq)
    else:
        raise ValueError(f"unknown placement policy {policy!r} (even or balanced)")

    head_bytes = 0
    if profile is not None:
        head_bytes = profile.embedding_bytes + sum(
            profile.layer_bytes[il] + (int(profile.kv_bytes_per_token * ctx_tokens) if profile.kv_layer[il] else profile.recurrent_bytes * n_seq)
            for il in range(head_layers)
        )
    placed = replay(n_layer, gpu_layers, counts, n_devices)
    assert placed[:head_layers] == [-1] * head_layers and placed[-1] == n_devices - 1, placed
    return Placement(gpu_layers=gpu_layers, tensor_split=list(counts), head_layers=head_layers,
                     counts=counts, device_bytes=device_bytes, head_bytes=head_bytes,
                     policy=policy, source=source)


def _device_bytes(profile: LayerProfile, head_layers: int, counts: list[int], ctx_tokens: int, n_seq: int) -> list[int]:
    out: list[int] = []
    il = head_layers
    for c in counts:
        total = 0
        for _ in range(c):
            if il < profile.n_layer:
                kv = int(profile.kv_bytes_per_token * ctx_tokens) if profile.kv_layer[il] else profile.recurrent_bytes * n_seq
                total += profile.layer_bytes[il] + kv
            else:
                total += profile.output_bytes
            il += 1
        out.append(total)
    return out
