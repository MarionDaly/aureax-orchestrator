"""Read a GGUF header, and name the tensors llama.cpp's rpc-server caches on disk.

A sharded group's cold start is the head streaming every tensor to its workers
before ``/health`` answers; a warm start is the same load with each worker
answering "already have it" from its own tensor cache (ARCHITECTURE.md,
"Measurements from one test bench": 280 s cold against ~107 s warm on that
bench). The cache is filled only as a side effect of a cold load, so a fresh
worker, a re-imaged one, or one that a resize handed different tensors always
pays the cold price first. This module is the offline half of making the warm
case deliberate: it reads which tensors a GGUF holds and where their bytes sit,
which is everything needed to compute the keys rpc-server would compute and to
place the files before the first load (``scripts/seed-rpc-cache.py``).

What rpc-server does at llama.cpp build b10375 (commit
``ba360efe1f574ebae727aad64112d18ecedca85a``, ``ggml/src/ggml-rpc/ggml-rpc.cpp``
and ``tools/rpc/rpc-server.cpp``), where this contract was read. The pin has
since moved to b11429, where the cache applies to WEIGHTS only (upstream
restricted it in b10985, so runtime tensors such as prefill masks are no longer
written); the directory, threshold and key below were not re-read there:

* The cache directory is ``$LLAMA_CACHE/rpc/``. The supervisor sets
  ``LLAMA_CACHE`` from ``model.rpc_server.cache_dir`` (default
  ``/opt/aureax/rpc-cache``), so on a worker it is ``/opt/aureax/rpc-cache/rpc/``.
* Only a ``set_tensor`` whose size is STRICTLY greater than ``HASH_THRESHOLD``
  (10 MiB) goes through the cache — :data:`CACHE_THRESHOLD`. Exactly 10 MiB is
  always shipped, never cached.
* The file name is the 64-bit FNV-1a of the bytes as ``%016x``, and the file is
  those bytes with no header. The key therefore depends on CONTENT only — not on
  the tensor's name, the worker, the group or ``tensor_split`` — which is what
  makes pre-seeding possible at all: the topology decides which keys a worker
  is asked for, never what the keys are.
* llama.cpp's model loader sends each tensor bound for an RPC buffer whole,
  from offset 0, so the bytes the head hashes are exactly the GGUF byte range
  ``[data_offset + tensor.offset, + nbytes)``. That is the whole contract this
  module reproduces; :func:`cache_entries` returns those ranges.

The header reader mirrors ``ggml/src/gguf.cpp`` at the same commit where it
matters for that contract — v2/v3 only, the alignment rule, tensors packed in
header order at padded offsets, ``ne[0]`` a multiple of the block size — so a
file this reader accepts is laid out the way llama.cpp will read it. Two
deliberate differences: nested arrays are parsed (gguf-py can write them,
llama.cpp refuses them; an inventory tool has no reason to reject a file over
metadata it never uses), and every length is capped before anything is
allocated, because a corrupt or hostile file must fail in milliseconds rather
than ask for gigabytes — see :data:`MAX_STRING_BYTES` and
:data:`MAX_HEADER_BYTES`.

Standard library only, and no ``orchestrator`` imports: the script that uses it
must run under a bare ``python3`` with no project dependencies installed, and
the edge client should be able to copy this file as-is.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"
# v1 used 32-bit counts and llama.cpp has refused it for years; 3 is
# GGUF_VERSION at the pinned commit.
GGUF_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32  # GGUF_DEFAULT_ALIGNMENT, used when general.alignment is absent
ALIGNMENT_KEY = "general.alignment"
GGML_MAX_DIMS = 4
GGML_MAX_NAME = 64  # gguf.cpp refuses a tensor name of this many bytes or more
_INT64_MAX = (1 << 63) - 1

# rpc-server's HASH_THRESHOLD. A tensor is cached only when STRICTLY larger.
CACHE_THRESHOLD = 10 * 1024 * 1024

FNV64_OFFSET_BASIS = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF

# Hard caps for corrupt input. The largest legitimate strings in a GGUF are
# chat templates (tens of KiB); the largest legitimate header is a big vocab
# (~10 MiB for a 250k-token tokenizer with merges). Both caps are far above
# that and far below "allocate the file size".
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_HEADER_BYTES = 256 * 1024 * 1024
MAX_ARRAY_NESTING = 4
_READ_CHUNK = 256 * 1024

# GGUF metadata value types (gguf.h, enum gguf_type).
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "B",
    GGUF_TYPE_INT8: "b",
    GGUF_TYPE_UINT16: "H",
    GGUF_TYPE_INT16: "h",
    GGUF_TYPE_UINT32: "I",
    GGUF_TYPE_INT32: "i",
    GGUF_TYPE_FLOAT32: "f",
    GGUF_TYPE_BOOL: "?",
    GGUF_TYPE_UINT64: "Q",
    GGUF_TYPE_INT64: "q",
    GGUF_TYPE_FLOAT64: "d",
}
_SCALARS = {vtype: struct.Struct("<" + fmt) for vtype, fmt in _SCALAR_FORMATS.items()}
_U32 = _SCALARS[GGUF_TYPE_UINT32]
_U64 = _SCALARS[GGUF_TYPE_UINT64]

# Smallest possible encodings, used to reject a count that cannot fit in what
# is left of the file before looping over it. A KV pair is at least a length,
# a one-byte key (gguf.cpp refuses an empty one), a type and a one-byte value;
# a tensor info is at least a name length, n_dims, a type and an offset.
_MIN_KV_BYTES = 8 + 1 + 4 + 1
_MIN_TENSOR_INFO_BYTES = 8 + 4 + 4 + 8
_MIN_ARRAY_HEADER_BYTES = 4 + 8

# ggml type id -> (name, block_size, type_size), from ``type_traits`` in
# ggml/src/ggml.c and the block structs in ggml/src/ggml-common.h at the pinned
# commit (each size is the struct's own static_assert). Ids 4, 5, 31-33 and
# 36-38 are types upstream removed; their traits are zero there, and gguf.cpp
# refuses a tensor that uses one, so they are absent here too.
_GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    # Newer than MXFP4 and present at the pinned commit; kept so a GGUF using
    # them reads rather than failing as an "unknown type".
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
}
GGML_TYPE_TRAITS: dict[int, tuple[int, int]] = {tid: (b, s) for tid, (_, b, s) in _GGML_TYPES.items()}
GGML_TYPE_NAMES: dict[int, str] = {tid: name for tid, (name, _, _) in _GGML_TYPES.items()}


class GGUFError(ValueError):
    """The file is not a GGUF this reader can vouch for, or not the bytes its header claims."""


@dataclass(frozen=True)
class TensorInfo:
    """One tensor-info record. ``offset`` is relative to the data section, as stored."""

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def n_dims(self) -> int:
        return len(self.dims)


@dataclass
class GGUFHeader:
    """Everything before the data section. ``data_offset`` is absolute in the file."""

    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    data_offset: int
    alignment: int


@dataclass(frozen=True)
class CacheEntry:
    """A tensor rpc-server caches: its bytes are ``file[offset_in_file, + nbytes)``."""

    name: str
    offset_in_file: int
    nbytes: int


def _pad(value: int, alignment: int) -> int:
    """GGML_PAD: the first multiple of ``alignment`` at or after ``value``."""
    return (value + alignment - 1) // alignment * alignment


def type_name(ggml_type: int) -> str:
    """Human name for a ggml type id, for messages."""
    return GGML_TYPE_NAMES.get(ggml_type, f"type {ggml_type}")


def tensor_nbytes(info: TensorInfo) -> int:
    """Bytes the tensor occupies: ``product(dims) / block_size * type_size``.

    The same arithmetic as ``ggml_nbytes`` for a contiguous tensor. A row length
    that is not a whole number of blocks cannot be laid out at all, and gguf.cpp
    refuses such a file, so it raises rather than rounding.
    """
    traits = GGML_TYPE_TRAITS.get(info.ggml_type)
    if traits is None:
        raise GGUFError(f"tensor {info.name!r}: unknown or removed ggml type {info.ggml_type}")
    block_size, type_size = traits
    row = info.dims[0] if info.dims else 1
    if row % block_size:
        raise GGUFError(
            f"tensor {info.name!r}: {type_name(info.ggml_type)} needs a row length that is a "
            f"multiple of {block_size}, got {row}"
        )
    elements = 1
    for dim in info.dims:
        elements *= dim
    return elements // block_size * type_size


class _Reader:
    """Sequential little-endian reads with every length checked BEFORE it is read.

    Buffered in small chunks so a vocabulary of a quarter of a million strings
    does not cost a syscall each, and bounded by :data:`MAX_HEADER_BYTES` so a
    corrupt count cannot walk the reader through gigabytes of tensor data. At
    most one read-ahead chunk past the end of the header is ever read.
    """

    def __init__(self, f, file_size: int, path: Path) -> None:
        self._f = f
        self._path = path
        self._buf = b""
        self._pos = 0
        self._base = 0
        self.file_size = file_size
        self._limit = min(file_size, MAX_HEADER_BYTES)

    def tell(self) -> int:
        return self._base + self._pos

    def budget(self) -> int:
        """Header bytes that may still be read."""
        return self._limit - self.tell()

    def _refuse_overrun(self, needed: int, what: str) -> GGUFError:
        if self._limit < self.file_size:
            return GGUFError(
                f"{self._path}: {what} would run the header past {MAX_HEADER_BYTES} bytes — "
                "corrupt, or not a model header"
            )
        return GGUFError(
            f"{self._path}: truncated — {what} needs {needed} bytes at offset {self.tell()}, "
            f"the file has {self.budget()} left"
        )

    def _fill(self, n: int, what: str) -> None:
        if len(self._buf) - self._pos >= n:
            return
        if n > self.budget():
            raise self._refuse_overrun(n, what)
        keep = self._buf[self._pos:]
        more = self._f.read(max(n - len(keep), _READ_CHUNK))
        self._base += self._pos
        self._buf = keep + more
        self._pos = 0
        if len(self._buf) < n:  # the file shrank under us
            raise self._refuse_overrun(n, what)

    def take(self, n: int, what: str) -> bytes:
        self._fill(n, what)
        start = self._pos
        self._pos += n
        return self._buf[start:self._pos]

    def unpack(self, st: struct.Struct, what: str) -> tuple:
        self._fill(st.size, what)
        values = st.unpack_from(self._buf, self._pos)
        self._pos += st.size
        return values

    def check_count(self, count: int, min_size: int, what: str) -> None:
        """Refuse a count whose smallest possible encoding cannot fit."""
        if count * min_size > self.budget():
            raise self._refuse_overrun(count * min_size, f"{what} ({count} elements)")

    def raw_string(self, what: str) -> bytes:
        (length,) = self.unpack(_U64, what)
        if length > MAX_STRING_BYTES:
            raise GGUFError(
                f"{self._path}: {what} claims a {length}-byte string, over the "
                f"{MAX_STRING_BYTES}-byte cap — corrupt header"
            )
        return self.take(length, what)

    def string(self, what: str) -> str:
        # Values are informational, so undecodable bytes degrade to U+FFFD
        # rather than refusing a model over its tokenizer's metadata.
        return self.raw_string(what).decode("utf-8", errors="replace")

    def name(self, what: str) -> str:
        # Keys and tensor names must round-trip exactly — a manifest is keyed by
        # them — so these decode strictly.
        raw = self.raw_string(what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError(f"{self._path}: {what} is not valid UTF-8 ({exc})") from None


def _read_array(r: _Reader, etype: int, count: int, what: str, depth: int) -> list:
    if depth > MAX_ARRAY_NESTING:
        raise GGUFError(f"{r._path}: {what} nests arrays deeper than {MAX_ARRAY_NESTING}")
    scalar = _SCALARS.get(etype)
    if scalar is not None:
        r.check_count(count, scalar.size, what)
        raw = r.take(count * scalar.size, what)
        return list(struct.unpack(f"<{count}{_SCALAR_FORMATS[etype]}", raw))
    if etype == GGUF_TYPE_STRING:
        r.check_count(count, _U64.size, what)
        return [r.string(what) for _ in range(count)]
    if etype == GGUF_TYPE_ARRAY:
        r.check_count(count, _MIN_ARRAY_HEADER_BYTES, what)
        out = []
        for _ in range(count):
            (inner_type,) = r.unpack(_U32, what)
            (inner_count,) = r.unpack(_U64, what)
            out.append(_read_array(r, inner_type, inner_count, what, depth + 1))
        return out
    raise GGUFError(f"{r._path}: {what} is an array of unknown GGUF type {etype}")


def _read_value(r: _Reader, vtype: int, what: str) -> Any:
    scalar = _SCALARS.get(vtype)
    if scalar is not None:
        return r.unpack(scalar, what)[0]
    if vtype == GGUF_TYPE_STRING:
        return r.string(what)
    if vtype == GGUF_TYPE_ARRAY:
        (etype,) = r.unpack(_U32, what)
        (count,) = r.unpack(_U64, what)
        return _read_array(r, etype, count, what, depth=1)
    raise GGUFError(f"{r._path}: {what} has unknown GGUF type {vtype}")


def _read_version(r: _Reader, path: Path) -> int:
    (version,) = r.unpack(_U32, "version")
    if version and version & 0xFFFF == 0:
        # A v3 file written big-endian reads as 0x03000000 here; gguf.cpp
        # reports the same case the same way.
        raise GGUFError(f"{path}: version {version:#x} looks byte-swapped — a big-endian GGUF, not supported")
    if version == 1:
        raise GGUFError(f"{path}: GGUF v1 (32-bit counts) is not supported, and llama.cpp refuses it too")
    if version not in GGUF_VERSIONS:
        raise GGUFError(f"{path}: GGUF version {version} is not one of {GGUF_VERSIONS}")
    return version


def _read_tensor_info(r: _Reader, index: int) -> TensorInfo:
    what = f"tensor {index}"
    name = r.name(f"{what} name")
    if len(name.encode("utf-8")) >= GGML_MAX_NAME:
        raise GGUFError(f"{r._path}: {what} name {name!r} is {GGML_MAX_NAME} bytes or longer")
    (n_dims,) = r.unpack(_U32, f"{what} n_dims")
    if n_dims > GGML_MAX_DIMS:
        raise GGUFError(f"{r._path}: tensor {name!r} has {n_dims} dimensions, more than {GGML_MAX_DIMS}")
    dims = r.unpack(struct.Struct(f"<{n_dims}Q"), f"{what} dims") if n_dims else ()
    if any(dim > _INT64_MAX for dim in dims):
        # gguf.cpp reads these as int64 and refuses a negative one.
        raise GGUFError(f"{r._path}: tensor {name!r} has a dimension over INT64_MAX: {dims}")
    (ggml_type,) = r.unpack(_U32, f"{what} type")
    (offset,) = r.unpack(_U64, f"{what} offset")
    info = TensorInfo(name=name, dims=tuple(dims), ggml_type=ggml_type, offset=offset)
    tensor_nbytes(info)  # unknown type, or a row that is not whole blocks
    return info


def read_header(path: str | os.PathLike) -> GGUFHeader:
    """Parse everything up to the data section. Never reads tensor data.

    Beyond parsing, it checks what a cache seeder depends on: every tensor sits
    at the padded offset gguf.cpp expects, and every tensor's bytes lie inside
    the file — so a truncated download is refused here, by name, instead of
    yielding a manifest whose last entries point past the end.
    """
    path = Path(path)
    with path.open("rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        r = _Reader(f, file_size, path)
        magic = r.take(4, "magic")
        if magic != GGUF_MAGIC:
            raise GGUFError(f"{path}: not a GGUF file (magic {magic!r})")
        version = _read_version(r, path)
        (n_tensors,) = r.unpack(_U64, "tensor count")
        (n_kv,) = r.unpack(_U64, "metadata count")
        r.check_count(n_kv, _MIN_KV_BYTES, "metadata")
        r.check_count(n_tensors, _MIN_TENSOR_INFO_BYTES, "tensor infos")

        metadata: dict[str, Any] = {}
        alignment_type = None
        for i in range(n_kv):
            key = r.name(f"metadata key {i}")
            if not key:
                raise GGUFError(f"{path}: metadata key {i} is empty")
            if key in metadata:
                raise GGUFError(f"{path}: duplicate metadata key {key!r}")
            (vtype,) = r.unpack(_U32, f"type of {key!r}")
            metadata[key] = _read_value(r, vtype, repr(key))
            if key == ALIGNMENT_KEY:
                alignment_type = vtype

        alignment = DEFAULT_ALIGNMENT
        if alignment_type is not None:
            # gguf.cpp reads it with gguf_get_val_u32, which aborts on any
            # other type, and then requires a power of two.
            alignment = metadata[ALIGNMENT_KEY]
            if alignment_type != GGUF_TYPE_UINT32:
                raise GGUFError(f"{path}: {ALIGNMENT_KEY} must be a uint32, got GGUF type {alignment_type}")
            if alignment == 0 or alignment & (alignment - 1):
                raise GGUFError(f"{path}: {ALIGNMENT_KEY} = {alignment} is not a power of two")

        tensors: list[TensorInfo] = []
        seen: set[str] = set()
        for i in range(n_tensors):
            info = _read_tensor_info(r, i)
            if info.name in seen:
                raise GGUFError(f"{path}: duplicate tensor name {info.name!r}")
            seen.add(info.name)
            tensors.append(info)

        # gguf.cpp seeks to the padded position only when there is tensor data.
        data_offset = _pad(r.tell(), alignment) if tensors else r.tell()

    expected = 0
    for info in tensors:
        nbytes = tensor_nbytes(info)
        if info.offset != expected:
            raise GGUFError(
                f"{path}: tensor {info.name!r} is at data offset {info.offset}, expected {expected} "
                "— llama.cpp requires tensors packed in header order"
            )
        end = data_offset + info.offset + nbytes
        if end > file_size:
            raise GGUFError(
                f"{path}: tensor {info.name!r} ends at byte {end} but the file is {file_size} bytes "
                "— truncated download?"
            )
        expected += _pad(nbytes, alignment)

    return GGUFHeader(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tensors,
        data_offset=data_offset,
        alignment=alignment,
    )


def architecture(header: GGUFHeader) -> str | None:
    """``general.architecture``, the prefix every model-specific key carries."""
    value = header.metadata.get("general.architecture")
    return value if isinstance(value, str) else None


def block_count(header: GGUFHeader) -> int | None:
    """``<arch>.block_count``: the number of ``blk.N.`` layers."""
    arch = architecture(header)
    if arch is None:
        return None
    value = header.metadata.get(f"{arch}.block_count")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def layer_index(name: str) -> int | None:
    """N for a ``blk.N.<rest>`` tensor name; None for embeddings, output and the like."""
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "blk" and parts[1].isascii() and parts[1].isdigit():
        return int(parts[1])
    return None


def fnv1a64(data: bytes | bytearray | memoryview, state: int = FNV64_OFFSET_BASIS) -> int:
    """64-bit FNV-1a, byte for byte the ``fnv_hash`` rpc-server keys its cache with.

    ``state`` continues a running hash, so a range can be hashed in chunks.
    Byte-serial by definition — every step depends on the last — which is why
    this runs at only tens of MB/s in Python and the seeding script prefers a
    compiled equivalent for whole models.
    """
    h = state
    prime = FNV64_PRIME
    mask = _MASK64
    for byte in bytes(data):
        h = ((h ^ byte) * prime) & mask
    return h


def fnv1a64_file_range(
    path: str | os.PathLike,
    offset: int,
    nbytes: int,
    chunk: int = 16 * 1024 * 1024,
    *,
    progress: Callable[[int], None] | None = None,
) -> int:
    """FNV-1a-64 of ``path[offset, offset + nbytes)``, read ``chunk`` bytes at a time.

    A range running past the end of the file raises rather than hashing the
    shorter prefix: a short hash is a valid-looking key for bytes nobody has.
    ``progress`` is called with each chunk's length.
    """
    if offset < 0 or nbytes < 0 or chunk <= 0:
        raise ValueError(f"bad range: offset={offset} nbytes={nbytes} chunk={chunk}")
    h = FNV64_OFFSET_BASIS
    left = nbytes
    with Path(path).open("rb") as f:
        f.seek(offset)
        while left:
            block = f.read(min(chunk, left))
            if not block:
                raise GGUFError(f"{path}: range [{offset}, +{nbytes}) runs {left} bytes past the end of the file")
            h = fnv1a64(block, h)
            left -= len(block)
            if progress is not None:
                progress(len(block))
    return h


def cache_entries(header: GGUFHeader) -> list[CacheEntry]:
    """The tensors rpc-server would cache, in header order, with absolute byte ranges."""
    entries = []
    for info in header.tensors:
        nbytes = tensor_nbytes(info)
        if nbytes > CACHE_THRESHOLD:
            entries.append(CacheEntry(info.name, header.data_offset + info.offset, nbytes))
    return entries


def cache_file_name(hash_value: int) -> str:
    """rpc-server's file name for a key: ``%016x``, lowercase, zero-padded."""
    if not 0 <= hash_value <= _MASK64:
        raise ValueError(f"not a 64-bit hash: {hash_value!r}")
    return f"{hash_value:016x}"
