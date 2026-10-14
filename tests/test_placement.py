"""orchestrator/config/placement.py: llama.cpp's layer-placement rule and the
counts this repo derives from it.

Every assertion about where a layer lands is checked against ``replay``, a
transcription of ``get_layer_buft_list`` at b10375 (not re-read at the current
pin, b11429), so the
numbers here are the ones the head would actually produce — not a reading of
the comments in the head template.
"""

from __future__ import annotations

import pytest

from orchestrator.config import placement as p


def _counts(placed: list[int], n_devices: int) -> list[int]:
    return [placed.count(k) for k in range(n_devices)]


def test_an_even_split_is_not_an_even_footprint() -> None:
    """[1, 1, 1] over 40 layers is 14 / 14 / 12 + output: llama.cpp divides the
    41 slots (40 layers plus the output tensor) by cumulative fraction, so the
    first devices carry the remainder. The template comment says the same;
    this pins it to the rule."""
    placed = p.replay(40, 99, [1, 1, 1], 3)
    assert placed.count(-1) == 0
    assert _counts(placed, 3) == [14, 14, 13]
    assert placed[-1] == 2, "the output tensor is placed as layer n_layer, on the last device"
    assert p.even_counts(41, 3) == [14, 14, 13]


def test_integer_counts_reproduce_themselves() -> None:
    """A split written as slot counts places exactly those slots — the form the
    generator emits, so what the manifest says is what the head does."""
    for counts in ([14, 14, 13], [10, 10, 11], [20, 21], [1, 40]):
        n = len(counts)
        placed = p.replay(40, sum(counts), counts, n)
        assert _counts(placed, n) == counts, counts


def test_head_resident_layers_come_off_the_front() -> None:
    """gpu_layers 31 on a 40-layer model keeps layers 0-9 on the head CPU, and
    [10, 10, 11] then gives each worker ten layers with the output on the last."""
    placed = p.replay(40, 31, [10, 10, 11], 3)
    assert placed[:10] == [-1] * 10
    assert _counts(placed, 3) == [10, 10, 11]


def test_plan_default_is_todays_command_line() -> None:
    """head_layers 0 + even = [1, 1, 1] and 99, byte for byte what the templates
    have always carried — the default must change nothing on a deployed fleet."""
    placed = p.plan(n_devices=3)
    assert (placed.gpu_layers, placed.tensor_split) == (99, [1, 1, 1])
    assert placed.head_layers == 0 and placed.policy == "even"


def test_plan_with_head_layers_needs_a_layer_count() -> None:
    with pytest.raises(ValueError, match="layer count"):
        p.plan(n_devices=3, head_layers=10)
    placed = p.plan(n_devices=3, head_layers=10, n_layer=40, source="registry")
    assert placed.gpu_layers == 31
    assert sum(placed.tensor_split) == 31
    replayed = p.replay(40, placed.gpu_layers, placed.tensor_split, 3)
    assert replayed[:10] == [-1] * 10 and replayed[-1] == 2


def test_plan_refuses_a_head_that_keeps_everything() -> None:
    with pytest.raises(ValueError, match="leaves no layer"):
        p.plan(n_devices=3, head_layers=40, n_layer=40)


def _profile(layer_bytes, kv_layer, output=1000, kv_per_token=2.0):
    return p.LayerProfile(
        n_layer=len(layer_bytes), layer_bytes=list(layer_bytes), kv_layer=list(kv_layer),
        kv_bytes_per_token=kv_per_token, output_bytes=output,
    )


def test_balanced_minimises_the_heaviest_device() -> None:
    """Six layers of 5,5,5,1,1,1 units plus a 1-unit output over two devices:
    the even cut (3 + 3) loads 15 vs 4; the balanced cut (2 + 5) loads 10 vs 9."""
    prof = _profile([5, 5, 5, 1, 1, 1], [False] * 6, output=1, kv_per_token=0)
    counts, loads = p.balanced_counts(prof, 0, 2, ctx_tokens=0)
    assert counts == [2, 5] and loads == [10, 9]
    placed = p.plan(n_devices=2, policy="balanced", profile=prof)
    assert placed.tensor_split == [2, 5] and placed.gpu_layers == 7
    assert _counts(p.replay(6, 7, [2, 5], 2), 2) == [2, 5]


def test_balanced_counts_kv_on_the_device_that_owns_the_layer() -> None:
    """A KV layer at a long context outweighs its weights; the split must see it."""
    prof = _profile([10, 10, 10, 10], [False, False, True, False], output=0, kv_per_token=1.0)
    counts, loads = p.balanced_counts(prof, 0, 2, ctx_tokens=30)
    # Slots: 10, 10, 40 (10 + 30 KV), 10, output 0 -> best cut is [2, 3]: 20 vs 50
    # beats [3, 2]: 60 vs 10.
    assert counts == [2, 3] and loads == [20, 50]


def test_balanced_needs_the_profile() -> None:
    with pytest.raises(ValueError, match="needs the GGUF"):
        p.plan(n_devices=2, policy="balanced", n_layer=6)


def test_plan_reports_head_and_device_footprints() -> None:
    prof = _profile([4, 4, 4, 4], [True, False, False, True], output=2, kv_per_token=1.0)
    prof = p.LayerProfile(**{**prof.__dict__, "embedding_bytes": 3})
    placed = p.plan(n_devices=2, head_layers=1, policy="even", profile=prof, ctx_tokens=10)
    # Head: embedding 3 + layer 0 (4 + 10 KV) = 17. Devices: slots are layers 1..3 + output
    # = [4, 4, 14, 2] over 2 devices evenly -> [2, 2]: 8 and 16.
    assert placed.head_bytes == 17
    assert placed.counts == [2, 2] and placed.device_bytes == [8, 16]
    assert placed.gpu_layers == 4 and placed.tensor_split == [2, 2]
