"""The GGUF header reader and the rpc-server tensor-cache seeder.

``orchestrator/config/gguf.py`` reproduces one contract: the cache key
rpc-server computes for a tensor is the FNV-1a-64 of the GGUF byte range
``[data_offset + offset, + nbytes)``, for every tensor STRICTLY over 10 MiB.
Get the data offset, a tensor's size or the threshold wrong by one and every
key is wrong — and a wrong key is silent: rpc-server just misses the cache and
ships the tensor, so the seeded directory looks fine and does nothing. That is
why the parser is checked against hand-built files with known layouts, the
hash against published vectors, and the compiled fast path against the Python
reference byte for byte.

``scripts/seed-rpc-cache.py`` is exercised end to end through a subprocess, the
way an operator runs it, so the sys.path shim is under test too. Every run
points ``$HOME`` at ``tmp_path`` (``tmp_aureax_home``): the compiled hasher is
built into ``~/.aureax/bin``, which must never be the real one from a test.
The one test that reads a real model file is opt-in by presence and skips on a
fresh clone.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from orchestrator.config import gguf

from .conftest import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "seed-rpc-cache.py"
REAL_MODEL = PROJECT_ROOT / "models" / "Qwen3.5-35B-A3B-Q4_K_M.gguf"
MIB = 1024 * 1024
BIG_NBYTES = 10 * MIB + 4096

needs_cc = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler: the fast path cannot be built")


# --- Building GGUF files by hand ---------------------------------------------


def _str(value: str | bytes) -> bytes:
    raw = value.encode() if isinstance(value, str) else value
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _str(key) + struct.pack("<I", vtype) + payload


def _u32(key: str, value: int) -> bytes:
    return _kv(key, gguf.GGUF_TYPE_UINT32, struct.pack("<I", value))


def _string_kv(key: str, value: str) -> bytes:
    return _kv(key, gguf.GGUF_TYPE_STRING, _str(value))


def _string_array(values: list[str]) -> bytes:
    return struct.pack("<IQ", gguf.GGUF_TYPE_STRING, len(values)) + b"".join(_str(v) for v in values)


@dataclass
class _Tensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    data: bytes


def _pad(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _write_gguf(path: Path, kvs: list[bytes], tensors: list[_Tensor], alignment: int = 32) -> tuple[int, list[int]]:
    """Write a v3 GGUF exactly as gguf.cpp lays one out; return (data_offset, offsets).

    ``alignment`` must agree with any ``general.alignment`` in ``kvs`` — the
    point of passing it separately is that the test, not the parser, decides
    where the bytes go.
    """
    infos = b""
    offsets = []
    cursor = 0
    for t in tensors:
        offsets.append(cursor)
        infos += _str(t.name) + struct.pack("<I", len(t.dims)) + struct.pack(f"<{len(t.dims)}Q", *t.dims)
        infos += struct.pack("<IQ", t.ggml_type, cursor)
        cursor = _pad(cursor + len(t.data), alignment)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + infos
    data_offset = _pad(len(head), alignment)
    body = bytearray()
    for t, offset in zip(tensors, offsets, strict=True):
        body += b"\0" * (offset - len(body))
        body += t.data
    path.write_bytes(head + b"\0" * (data_offset - len(head)) + bytes(body))
    return data_offset, offsets


def _small_model(path: Path) -> tuple[int, list[int], list[_Tensor]]:
    """Three small tensors of two types, a non-default alignment and every common KV shape."""
    tensors = [
        _Tensor("blk.0.attn_q.weight", (8, 2), 0, bytes(range(64))),                    # F32: 16 x 4
        _Tensor("blk.1.ffn_down.weight", (64, 3), 8, bytes(i % 251 for i in range(204))),  # Q8_0: 2 x 34 x 3
        _Tensor("output_norm.weight", (5,), 0, b"\x01" * 20),                             # F32: 5 x 4
    ]
    nested = struct.pack("<IQ", gguf.GGUF_TYPE_ARRAY, 2)
    nested += _string_array(["x", "y"]) + struct.pack("<IQ", gguf.GGUF_TYPE_INT32, 2) + struct.pack("<2i", -1, 7)
    kvs = [
        _string_kv("general.architecture", "testarch"),
        _u32("testarch.block_count", 2),
        _u32("general.alignment", 64),
        _kv("tokenizer.ggml.tokens", gguf.GGUF_TYPE_ARRAY, _string_array(["<s>", "héllo", ""])),
        _kv("tokenizer.ggml.token_type", gguf.GGUF_TYPE_ARRAY,
            struct.pack("<IQ", gguf.GGUF_TYPE_INT32, 3) + struct.pack("<3i", 1, 2, 3)),
        _kv("test.nested", gguf.GGUF_TYPE_ARRAY, nested),
        _kv("test.rope_theta", gguf.GGUF_TYPE_FLOAT32, struct.pack("<f", 0.5)),
        _kv("test.flag", gguf.GGUF_TYPE_BOOL, b"\x01"),
        _kv("test.big", gguf.GGUF_TYPE_UINT64, struct.pack("<Q", 2**40)),
    ]
    data_offset, offsets = _write_gguf(path, kvs, tensors, alignment=64)
    return data_offset, offsets, tensors


# --- The header ----------------------------------------------------------------


def test_the_parser_round_trips_a_hand_built_file(tmp_path) -> None:
    path = tmp_path / "small.gguf"
    data_offset, offsets, tensors = _small_model(path)

    header = gguf.read_header(path)

    assert header.version == 3
    assert header.alignment == 64
    assert header.data_offset == data_offset
    assert header.data_offset % 64 == 0
    assert [t.name for t in header.tensors] == [t.name for t in tensors]
    assert [t.dims for t in header.tensors] == [t.dims for t in tensors]
    assert [t.n_dims for t in header.tensors] == [2, 2, 1]
    assert [t.ggml_type for t in header.tensors] == [0, 8, 0]
    assert [t.offset for t in header.tensors] == offsets == [0, 64, 320]
    assert [gguf.tensor_nbytes(t) for t in header.tensors] == [len(t.data) for t in tensors] == [64, 204, 20]

    # The offsets are not just self-consistent: they point at the bytes.
    raw = path.read_bytes()
    for info, tensor in zip(header.tensors, tensors, strict=True):
        start = header.data_offset + info.offset
        assert raw[start:start + gguf.tensor_nbytes(info)] == tensor.data


def test_metadata_values_of_every_shape_come_back(tmp_path) -> None:
    path = tmp_path / "small.gguf"
    _small_model(path)

    meta = gguf.read_header(path).metadata

    assert meta["general.architecture"] == "testarch"
    assert meta["tokenizer.ggml.tokens"] == ["<s>", "héllo", ""]
    assert meta["tokenizer.ggml.token_type"] == [1, 2, 3]
    assert meta["test.nested"] == [["x", "y"], [-1, 7]]
    assert meta["test.rope_theta"] == 0.5
    assert meta["test.flag"] is True
    assert meta["test.big"] == 2**40


def test_architecture_block_count_and_layer_index(tmp_path) -> None:
    path = tmp_path / "small.gguf"
    _small_model(path)
    header = gguf.read_header(path)

    assert gguf.architecture(header) == "testarch"
    assert gguf.block_count(header) == 2
    assert [gguf.layer_index(t.name) for t in header.tensors] == [0, 1, None]
    assert gguf.layer_index("blk.12.ffn_up_exps.weight") == 12
    assert gguf.layer_index("blk.x.attn_q.weight") is None
    assert gguf.layer_index("blk.3") is None
    assert gguf.layer_index("token_embd.weight") is None


def test_alignment_defaults_to_32_and_the_data_section_starts_on_it(tmp_path) -> None:
    path = tmp_path / "default.gguf"
    data_offset, _ = _write_gguf(
        path, [_string_kv("general.architecture", "a")], [_Tensor("t", (3,), 0, b"\x00" * 12)]
    )

    header = gguf.read_header(path)

    assert header.alignment == gguf.DEFAULT_ALIGNMENT == 32
    assert header.data_offset == data_offset
    assert header.data_offset % 32 == 0


def test_the_type_table_matches_ggml_at_the_pinned_commit() -> None:
    """Spot checks against ggml-common.h's static_asserts; a wrong size is a wrong key."""
    expected = {
        0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
        9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
        15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
        21: (256, 110), 22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4),
        27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
        39: (32, 17),
    }
    for type_id, traits in expected.items():
        assert gguf.GGML_TYPE_TRAITS[type_id] == traits, gguf.type_name(type_id)
    for removed in (4, 5, 31, 32, 33, 36, 37, 38):
        assert removed not in gguf.GGML_TYPE_TRAITS


def test_a_row_that_is_not_whole_blocks_is_refused() -> None:
    with pytest.raises(gguf.GGUFError, match="multiple of 32"):
        gguf.tensor_nbytes(gguf.TensorInfo("bad", (33, 2), 8, 0))
    with pytest.raises(gguf.GGUFError, match="unknown or removed ggml type"):
        gguf.tensor_nbytes(gguf.TensorInfo("gone", (32,), 31, 0))
    assert gguf.tensor_nbytes(gguf.TensorInfo("q4k", (512, 4), 12, 0)) == 2 * 144 * 4


# --- Corrupt input fails fast and by name --------------------------------------


def _prefix(n_tensors: int = 0, n_kv: int = 0, version: int = 3) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, n_tensors, n_kv)


@pytest.mark.parametrize(
    ("blob", "message"),
    [
        (b"GGML" + b"\0" * 20, "not a GGUF file"),
        (_prefix(version=1), "v1"),
        (_prefix(version=0x03000000), "byte-swapped"),
        (_prefix(version=4), "version 4"),
        (_prefix(n_kv=2**60), "metadata"),
        (_prefix(n_tensors=2**50), "tensor infos"),
        # A key claiming a string just over the cap: refused before any allocation.
        (_prefix(n_kv=1) + struct.pack("<Q", gguf.MAX_STRING_BYTES + 1) + b"\0" * 64, "cap"),
        # An array whose count cannot fit in what is left of the file.
        (_prefix(n_kv=1) + _str("k") + struct.pack("<IIQ", gguf.GGUF_TYPE_ARRAY, gguf.GGUF_TYPE_UINT32, 2**40),
         "truncated"),
        (_prefix(n_kv=1) + _str("k") + struct.pack("<I", 13) + b"\0" * 8, "unknown GGUF type 13"),
        (_prefix(n_kv=2) + _u32("k", 1) + _u32("k", 2), "duplicate metadata key"),
        (_prefix(n_kv=1) + _kv("general.alignment", gguf.GGUF_TYPE_UINT32, struct.pack("<I", 48)),
         "power of two"),
        (_prefix(n_kv=1) + _kv("general.alignment", gguf.GGUF_TYPE_UINT64, struct.pack("<Q", 32)),
         "must be a uint32"),
        (_prefix(n_kv=1) + _u32("k", 1)[:-2], "truncated"),
    ],
)
def test_corrupt_headers_are_refused(tmp_path, blob: bytes, message: str) -> None:
    path = tmp_path / "corrupt.gguf"
    path.write_bytes(blob)

    with pytest.raises(gguf.GGUFError, match=message):
        gguf.read_header(path)


def test_a_truncated_download_is_refused_rather_than_listed(tmp_path) -> None:
    path = tmp_path / "small.gguf"
    _small_model(path)
    path.write_bytes(path.read_bytes()[:-10])

    with pytest.raises(gguf.GGUFError, match="truncated download"):
        gguf.read_header(path)


def test_a_tensor_out_of_packing_order_is_refused(tmp_path) -> None:
    """gguf.cpp requires each offset to be the running padded total; so does the reader."""
    path = tmp_path / "gap.gguf"
    infos = _str("a") + struct.pack("<IQ", 1, 4) + struct.pack("<IQ", 0, 32)  # should be at 0
    head = _prefix(n_tensors=1) + infos
    path.write_bytes(head + b"\0" * (_pad(len(head), 32) - len(head)) + b"\0" * 64)

    with pytest.raises(gguf.GGUFError, match="expected 0"):
        gguf.read_header(path)


# --- The hash ------------------------------------------------------------------


def test_fnv1a64_matches_the_published_vectors() -> None:
    assert gguf.fnv1a64(b"") == 0xCBF29CE484222325
    assert gguf.fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert gguf.fnv1a64(b"foobar") == 0x85944171F73967E8
    assert gguf.fnv1a64(memoryview(b"foobar")) == 0x85944171F73967E8
    # Continuing from a running state is the same as hashing in one go.
    assert gguf.fnv1a64(b"bar", gguf.fnv1a64(b"foo")) == gguf.fnv1a64(b"foobar")


def test_cache_file_name_is_rpc_servers_016x() -> None:
    assert gguf.cache_file_name(0x85944171F73967E8) == "85944171f73967e8"
    assert gguf.cache_file_name(1) == "0000000000000001"
    with pytest.raises(ValueError):
        gguf.cache_file_name(1 << 64)


def test_a_file_range_hashes_like_its_slice_across_chunk_boundaries(tmp_path) -> None:
    data = random.Random(7).randbytes(5000)
    path = tmp_path / "blob.bin"
    path.write_bytes(data)

    for offset, nbytes in ((0, 0), (0, 5000), (13, 977), (4999, 1)):
        assert gguf.fnv1a64_file_range(path, offset, nbytes, chunk=64) == gguf.fnv1a64(data[offset:offset + nbytes])

    seen: list[int] = []
    gguf.fnv1a64_file_range(path, 0, 5000, chunk=1024, progress=seen.append)
    assert seen == [1024, 1024, 1024, 1024, 904]


def test_a_file_range_past_the_end_raises_instead_of_hashing_a_prefix(tmp_path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"0123456789")

    with pytest.raises(gguf.GGUFError, match="past the end"):
        gguf.fnv1a64_file_range(path, 5, 6)


# --- Which tensors are cached ---------------------------------------------------


def test_cache_entries_take_only_tensors_strictly_over_10_mib(tmp_path) -> None:
    """HASH_THRESHOLD is `size > 10 MiB` in ggml-rpc.cpp: exactly 10 MiB is always shipped."""
    exact = gguf.TensorInfo("blk.0.exact.weight", (10 * MIB,), 24, 0)          # I8: 10 MiB
    over = gguf.TensorInfo("blk.0.over.weight", (10 * MIB + 1,), 24, 10 * MIB)  # I8: 10 MiB + 1
    small = gguf.TensorInfo("norm.weight", (4,), 0, 20 * MIB + 32)
    header = gguf.GGUFHeader(
        path=tmp_path / "fake.gguf", version=3, metadata={}, tensors=[exact, over, small],
        data_offset=4096, alignment=32,
    )

    entries = gguf.cache_entries(header)

    assert entries == [gguf.CacheEntry("blk.0.over.weight", 4096 + 10 * MIB, 10 * MIB + 1)]
    assert gguf.CACHE_THRESHOLD == 10 * MIB


# --- The seeding script, end to end --------------------------------------------


@pytest.fixture(scope="module")
def big_model(tmp_path_factory) -> dict:
    """A GGUF with one tensor over the threshold (10 MiB + 4096 random bytes) and one under."""
    path = tmp_path_factory.mktemp("model") / "big.gguf"
    payload = random.Random(1234).randbytes(BIG_NBYTES)
    tensors = [
        _Tensor("output_norm.weight", (16,), 0, b"\x02" * 64),
        _Tensor("blk.0.ffn_up_exps.weight", (BIG_NBYTES,), 24, payload),
    ]
    data_offset, offsets = _write_gguf(path, [_string_kv("general.architecture", "testarch")], tensors)
    return {
        "path": path,
        "payload": payload,
        "offset": data_offset + offsets[1],
        "key": gguf.cache_file_name(gguf.fnv1a64(payload)),
    }


def _run(*args, pure: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if pure:
        env["AUREAX_FNV_PURE_PYTHON"] = "1"
    else:
        env.pop("AUREAX_FNV_PURE_PYTHON", None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        check=False,
    )


def _manifest(model: dict, out: Path, *extra, pure: bool = True) -> dict:
    proc = _run("manifest", model["path"], "--out", out, *extra, pure=pure)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def _load_script_module():
    spec = importlib.util.spec_from_file_location("seed_rpc_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_lists_the_big_tensor_with_its_absolute_range(big_model, tmp_aureax_home) -> None:
    doc = _manifest(big_model, tmp_aureax_home / "m.json")

    assert set(doc) == {"gguf", "size_bytes", "sha_note", "threshold", "entries"}
    assert doc["gguf"] == "big.gguf"
    assert doc["size_bytes"] == big_model["path"].stat().st_size
    assert doc["threshold"] == 10485760
    assert doc["entries"] == [
        {"name": "blk.0.ffn_up_exps.weight", "offset": big_model["offset"], "nbytes": BIG_NBYTES, "hash": None}
    ]


def test_manifest_defaults_to_a_file_next_to_the_gguf(tmp_path, tmp_aureax_home, big_model) -> None:
    model = dict(big_model, path=tmp_path / "copy.gguf")
    shutil.copyfile(big_model["path"], model["path"])

    proc = _run("manifest", model["path"])

    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "copy.gguf.rpc-manifest.json").is_file()


def _round_trip(model: dict, home: Path, pure: bool) -> str:
    """manifest --hash, export, export again, verify: the operator's whole loop."""
    doc = _manifest(model, home / "m.json", "--hash", pure=pure)
    key = doc["entries"][0]["hash"]
    dest = home / "cache" / "rpc"

    proc = _run("export", model["path"], "--manifest", home / "m.json", "--dest", dest, pure=pure)
    assert proc.returncode == 0, proc.stderr
    assert "1 written" in proc.stdout
    assert (dest / key).read_bytes() == model["payload"]
    assert sorted(p.name for p in dest.iterdir()) == [key], "left a temporary behind"

    again = _run("export", model["path"], "--manifest", home / "m.json", "--dest", dest, pure=pure)
    assert again.returncode == 0, again.stderr
    assert "0 written" in again.stdout and "1 already present" in again.stdout

    verify = _run("verify", dest, "--manifest", home / "m.json", pure=pure)
    assert verify.returncode == 0, verify.stdout + verify.stderr
    assert "1 ok, 0 mismatched, 0 unknown" in verify.stdout
    return key


def test_round_trip_with_the_pure_python_hasher(big_model, tmp_aureax_home) -> None:
    assert _round_trip(big_model, tmp_aureax_home, pure=True) == big_model["key"]
    assert not (tmp_aureax_home / ".aureax" / "bin" / "fnv1a64").exists(), "forced pure Python still built cc"


@needs_cc
def test_round_trip_with_the_compiled_hasher_gives_identical_keys(big_model, tmp_aureax_home) -> None:
    assert _round_trip(big_model, tmp_aureax_home, pure=False) == big_model["key"]
    assert (tmp_aureax_home / ".aureax" / "bin" / "fnv1a64").is_file()


@needs_cc
def test_the_compiled_and_python_hashers_agree_on_odd_ranges(big_model, tmp_aureax_home) -> None:
    """Including ranges larger than the C program's 4 MiB buffer, so its chunk loop runs."""
    module = _load_script_module()
    native = module.Hasher(pure_python=False)
    python = module.Hasher(pure_python=True)
    assert native.binary is not None and python.binary is None

    path = big_model["path"]
    head = path.read_bytes()[:5000]
    for offset, nbytes in ((0, 0), (0, 5000), (17, 4093)):
        expected = gguf.cache_file_name(gguf.fnv1a64(head[offset:offset + nbytes]))
        assert native.range(path, offset, nbytes) == python.range(path, offset, nbytes) == expected
    assert native.range(path, big_model["offset"], BIG_NBYTES) == big_model["key"]


@needs_cc
def test_a_stale_compiled_hasher_is_rebuilt_rather_than_trusted(big_model, tmp_aureax_home) -> None:
    binary = tmp_aureax_home / ".aureax" / "bin" / "fnv1a64"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 0000000000000000\n")
    binary.chmod(0o700)

    module = _load_script_module()
    hasher = module.Hasher(pure_python=False)

    assert hasher.binary == binary
    assert not binary.read_bytes().startswith(b"#!"), "the impostor was trusted instead of replaced"
    assert hasher.range(big_model["path"], 0, 0) == "cbf29ce484222325"


def test_verify_flags_a_truncated_file_and_strays_and_deletes_only_cache_names(big_model, tmp_aureax_home) -> None:
    home = tmp_aureax_home
    key = _round_trip(big_model, home, pure=True)
    dest = home / "cache" / "rpc"
    manifest = home / "m.json"

    # The truncated-write hazard: the right name, the wrong length.
    with (dest / key).open("r+b") as f:
        f.truncate(BIG_NBYTES // 2)
    stray = "0123456789abcdef"
    (dest / stray).write_bytes(b"\0" * 100)          # e.g. a cached prefill mask
    (dest / f".{key}.tmp").write_bytes(b"\0" * 10)   # an interrupted export
    (dest / "notes.txt").write_text("not ours")      # not a name rpc-server writes

    proc = _run("verify", dest, "--manifest", manifest)
    assert proc.returncode == 1
    assert f"MISMATCHED {key}: {BIG_NBYTES // 2} bytes" in proc.stdout
    assert f"UNKNOWN    {stray}" in proc.stdout
    assert "0 ok, 1 mismatched, 3 unknown" in proc.stdout

    proc = _run("verify", dest, "--manifest", manifest, "--delete-unknown", "--delete-mismatched")
    assert proc.returncode == 1, "notes.txt is kept, so an unknown file remains"
    assert sorted(p.name for p in dest.iterdir()) == ["notes.txt"]
    assert "kept    notes.txt" in proc.stdout and "the cache directory?" in proc.stdout

    (dest / "notes.txt").unlink()
    proc = _run("verify", dest, "--manifest", manifest)
    assert proc.returncode == 0, proc.stdout
    assert "1 manifest entries not present" in proc.stdout


def test_verify_without_hashes_matches_by_size_and_says_so(big_model, tmp_aureax_home) -> None:
    home = tmp_aureax_home
    _manifest(big_model, home / "plain.json")
    dest = home / "rpc"
    proc = _run("export", big_model["path"], "--manifest", home / "plain.json", "--dest", dest)
    assert proc.returncode == 0, proc.stderr
    assert (dest / big_model["key"]).is_file(), "export computes a key the manifest lacks"

    proc = _run("verify", dest, "--manifest", home / "plain.json")
    assert proc.returncode == 0, proc.stdout
    assert "carries no hashes" in proc.stdout and "SIZE only" in proc.stdout
    assert "1 matched by size only" in proc.stdout

    (dest / "fedcba9876543210").write_bytes(b"\0" * 7)
    proc = _run("verify", dest, "--manifest", home / "plain.json")
    assert proc.returncode == 1
    assert "UNKNOWN    fedcba9876543210" in proc.stdout


def test_export_refuses_a_manifest_hashed_from_other_bytes(big_model, tmp_aureax_home) -> None:
    """Same layout, different bytes: publishing them under the old key would poison the cache."""
    home = tmp_aureax_home
    doc = _manifest(big_model, home / "m.json", "--hash")
    doc["entries"][0]["hash"] = "00000000deadbeef"
    (home / "m.json").write_text(json.dumps(doc))
    dest = home / "rpc"

    proc = _run("export", big_model["path"], "--manifest", home / "m.json", "--dest", dest)

    assert proc.returncode == 1
    assert "hashed from different bytes" in proc.stderr
    assert list(dest.iterdir()) == [], "nothing may be left under a name it does not hash to"


def test_export_refuses_a_manifest_written_for_another_file(big_model, tmp_aureax_home, tmp_path) -> None:
    home = tmp_aureax_home
    _manifest(big_model, home / "m.json")
    other = tmp_path / "other.gguf"
    _small_model(other)

    proc = _run("export", other, "--manifest", home / "m.json", "--dest", home / "rpc")

    assert proc.returncode == 1
    assert "a different file" in proc.stderr


def test_export_only_hashed_refuses_a_manifest_without_hashes(big_model, tmp_aureax_home) -> None:
    home = tmp_aureax_home
    _manifest(big_model, home / "m.json")

    proc = _run("export", big_model["path"], "--manifest", home / "m.json", "--dest", home / "rpc", "--only-hashed")

    assert proc.returncode == 1
    assert "manifest --hash" in proc.stderr


# --- The real model, when this machine has it -----------------------------------


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason=f"{REAL_MODEL.name} is not present")
def test_the_deployed_35b_header_reads_as_expected() -> None:
    header = gguf.read_header(REAL_MODEL)

    assert gguf.architecture(header) == "qwen35moe"
    assert len(header.tensors) == 733
    assert gguf.block_count(header) == 40
    cached = sum(e.nbytes for e in gguf.cache_entries(header))
    assert 19 * 1024**3 <= cached <= 21 * 1024**3


# --- push plan ------------------------------------------------------------------


def test_push_plan_targets_each_enabled_worker_cache_over_sudo_rsync(tmp_path, monkeypatch) -> None:
    """One rsync per worker, into <cache_dir>/rpc/ on the node, size-only and
    through the fleet's known_hosts store, with the remote side under sudo
    because the pod created the cache as root."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("seed_rpc_cache", "scripts/seed-rpc-cache.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setenv("HOME", str(tmp_path))
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "w.yaml").write_text("model:\n  rpc_server: {cache_dir: /data/rpc-cache}\n")
    net = configs / "aureax-network.yaml"
    net.write_text(
        "network: {name: t, version: '0.1.0'}\n"
        "model_registry: {models: []}\n"
        "ssh:\n  targets:\n"
        "    - {node_id: h, host: 10.77.0.1, config_file: w.yaml}\n"
        "    - {node_id: w1, host: 10.77.0.2, user: cpedge, port: 2222, key_file: /k/id, config_file: w.yaml}\n"
        "    - {node_id: solo, host: 10.77.0.3, config_file: w.yaml}\n"
        "node_groups:\n  - {name: g, head: h, workers: [w1]}\n"
    )
    plan = mod.push_plan(net, "g", tmp_path / "export")

    assert [item["node_id"] for item in plan] == ["w1"], "only the group's workers, never the head or a standalone node"
    argv = plan[0]["argv"]
    assert argv[0] == "rsync" and "--size-only" in argv
    assert argv[argv.index("--rsync-path") + 1] == "sudo rsync"
    rsh = argv[argv.index("-e") + 1]
    assert "-p 2222" in rsh and "-i /k/id" in rsh and "known_hosts" in rsh and "BatchMode=yes" in rsh
    assert argv[-1] == "cpedge@10.77.0.2:/data/rpc-cache/rpc/"
    assert argv[-2].endswith("/export/")

    with pytest.raises(mod.Refusal, match="no node_groups entry"):
        mod.push_plan(net, "nope", tmp_path / "export")
