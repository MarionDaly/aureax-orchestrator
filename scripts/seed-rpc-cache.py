#!/usr/bin/env python3
"""Pre-seed and audit llama.cpp rpc-server's tensor cache from a GGUF.

Usage:
    python3 scripts/seed-rpc-cache.py manifest <gguf> [--out PATH] [--hash]
    python3 scripts/seed-rpc-cache.py export <gguf> --manifest PATH --dest DIR [--only-hashed]
    python3 scripts/seed-rpc-cache.py verify DIR --manifest PATH [--delete-unknown] [--delete-mismatched]

Every subcommand also takes --pure-python (or AUREAX_FNV_PURE_PYTHON=1) to skip
the compiled hasher.

Why this exists: a worker's rpc-server cache turns a sharded head's cold load
into a warm one (ARCHITECTURE.md, "Measurements from one test bench"), but the
cache only ever fills as a side effect of a cold load, so every fresh,
re-imaged or resized worker pays the cold price first. rpc-server keys each
cached tensor by the 64-bit FNV-1a of its bytes and nothing else, so the files
can be produced offline from the GGUF and placed in the worker's
``$LLAMA_CACHE/rpc/`` (``/opt/aureax/rpc-cache/rpc/`` on our workers) before the
first load. ``orchestrator/config/gguf.py`` documents the rules this
reproduces.

* ``manifest`` lists every tensor over the 10 MiB threshold with its byte range
  in the file, and with ``--hash`` its cache key. Written next to the GGUF as
  ``<gguf>.rpc-manifest.json`` unless ``--out`` says otherwise. A split GGUF is
  one manifest per part: each part is a complete GGUF of its own.
* ``export`` writes each entry's bytes to ``DIR/<key>``, atomically, and
  re-hashes every file it wrote before giving it its name: a file whose name is
  not the hash of its content is worse than no file, because rpc-server trusts
  the name and would load the wrong weights without a word.
* ``verify`` audits a cache directory against a manifest: a known name at the
  wrong size is the truncated-write hazard (rpc-server's own write is not
  atomic, and it loads whatever is there), and a name the manifest does not know
  is a stray — another model's tensor, or a large runtime input such as a
  prefill mask that b10375 cached the same way (the current pin, b11429,
  caches weights only, so new strays of that kind come only from older images).

Hashing is byte-serial (each step depends on the last), so pure Python runs at
roughly 15 MB/s — tens of minutes for a 20 GB model. When ``cc`` exists, a
40-line C equivalent is compiled once into ``~/.aureax/bin/fnv1a64`` and
self-tested against the Python reference on every run before it is trusted; the
Python loop stays as the fallback and gives the same answers.

Standard library only, and the only project import is
``orchestrator.config.gguf`` (itself stdlib-only), so this runs under a bare
``python3`` on the Mac and can run inside the edge image later.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Prefer the checkout this script sits in over any installed copy, so a script
# run from a working tree always reads that tree's parser; fall back to the
# installed package when the script has been copied somewhere on its own.
_CHECKOUT = Path(__file__).resolve().parent.parent
if (_CHECKOUT / "orchestrator" / "config" / "gguf.py").is_file():
    sys.path.insert(0, str(_CHECKOUT))

from orchestrator.config import gguf  # noqa: E402 — needs the path shim above

PURE_PYTHON_ENV = "AUREAX_FNV_PURE_PYTHON"
MANIFEST_SUFFIX = ".rpc-manifest.json"
PROGRESS_STEP = 2 * 1024**3
COPY_CHUNK = 16 * 1024 * 1024

SHA_NOTE = (
    "Each entry's 'hash' is the 64-bit FNV-1a of that tensor's raw bytes, rpc-server's "
    "cache key and file name (%016x), not a cryptographic digest. Nothing here is a sha256; "
    "the model registry's sha256 / sha256_parts verify the GGUF itself."
)

# The fast path. Same algorithm and output format as gguf.fnv1a64_file_range;
# the 4 MiB buffer means any range over 4 MiB crosses a chunk boundary, which
# the test suite exercises against the Python reference.
FNV_C_SOURCE = r"""
/* fnv1a64 FILE OFFSET NBYTES: print the 64-bit FNV-1a of FILE[OFFSET, OFFSET+NBYTES)
 * as 16 lowercase hex digits, rpc-server's tensor-cache key. Built and self-tested
 * by scripts/seed-rpc-cache.py. Exit 1 on an I/O error or a short file, 2 on usage. */
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

static unsigned char buf[1 << 22];

static int parse(const char *s, unsigned long long *out) {
    char *end;
    if (*s < '0' || *s > '9') return -1;
    errno = 0;
    *out = strtoull(s, &end, 10);
    return (errno != 0 || *end != '\0') ? -1 : 0;
}

int main(int argc, char **argv) {
    unsigned long long offset, left;
    if (argc != 4 || parse(argv[2], &offset) || parse(argv[3], &left)) {
        fprintf(stderr, "usage: fnv1a64 FILE OFFSET NBYTES\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (f == NULL) { perror(argv[1]); return 1; }
    if (fseeko(f, (off_t)offset, SEEK_SET) != 0) { perror("fseeko"); return 1; }
    uint64_t h = 0xcbf29ce484222325ULL;
    while (left > 0) {
        size_t want = left < sizeof buf ? (size_t)left : sizeof buf;
        size_t got = fread(buf, 1, want, f);
        if (got == 0) {
            if (ferror(f)) perror(argv[1]);
            else fprintf(stderr, "%s: range runs %llu bytes past the end of the file\n", argv[1], left);
            return 1;
        }
        for (size_t i = 0; i < got; i++) {
            h ^= buf[i];
            h *= 0x100000001b3ULL;
        }
        left -= got;
    }
    printf("%016" PRIx64 "\n", h);
    return 0;
}
"""


class Refusal(Exception):
    """An operator-facing failure: printed as one message, exit 1."""


def _note(message: str) -> None:
    print(message, file=sys.stderr)


def _gib(nbytes: int) -> str:
    return f"{nbytes / 1024**3:.2f} GiB"


class Progress:
    """One stderr line per PROGRESS_STEP bytes, so a long hash is visibly alive."""

    def __init__(self, total: int, verb: str) -> None:
        self.total = total
        self.verb = verb
        self.done = 0
        self._next = PROGRESS_STEP
        self._start = time.monotonic()

    def __call__(self, nbytes: int) -> None:
        self.done += nbytes
        if self.done < self._next:
            return
        while self._next <= self.done:
            self._next += PROGRESS_STEP
        elapsed = max(time.monotonic() - self._start, 1e-6)
        _note(f"  {self.verb} {_gib(self.done)} of {_gib(self.total)} ({self.done / elapsed / 1e6:.0f} MB/s)")


def _ignore_progress(_nbytes: int) -> None:
    pass


# --- Hashing -----------------------------------------------------------------


def _hasher_path() -> Path:
    return Path.home() / ".aureax" / "bin" / "fnv1a64"


def _run_native(binary: Path, path: Path, offset: int, nbytes: int) -> int:
    proc = subprocess.run(
        [str(binary), str(path), str(offset), str(nbytes)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise Refusal(f"{binary} failed on {path} [{offset}, +{nbytes}): {proc.stderr.strip()}")
    try:
        return int(proc.stdout.strip(), 16)
    except ValueError:
        raise Refusal(f"{binary} printed {proc.stdout.strip()!r}, not a hash") from None


def _self_test(binary: Path) -> bool:
    """Trust a compiled hasher only after it agrees with the Python reference.

    Runs every time, not just after a build: the binary lives in $HOME and
    outlives this script, so a stale build from an older source, a binary
    copied from another architecture, or a damaged file all have to fail here
    rather than mint wrong keys.
    """
    sample = b"rpc-server cache key " + bytes(range(256)) * 3 + b"foobar"
    cases = [(0, 0), (0, len(sample)), (len(sample) - 6, 6), (5, 300)]
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.bin"
        probe.write_bytes(sample)
        try:
            for offset, nbytes in cases:
                if _run_native(binary, probe, offset, nbytes) != gguf.fnv1a64(sample[offset:offset + nbytes]):
                    return False
        except (OSError, ValueError, Refusal):
            return False
    return True


def _build(cc: str, binary: Path) -> None:
    """Compile FNV_C_SOURCE and move it into place atomically."""
    aureax = binary.parent.parent
    for directory in (aureax, binary.parent):
        directory.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=binary.parent) as tmp:
        source = Path(tmp) / "fnv1a64.c"
        source.write_text(FNV_C_SOURCE)
        built = Path(tmp) / "fnv1a64"
        proc = subprocess.run(
            [cc, "-O2", "-o", str(built), str(source)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0 or not built.is_file():
            lines = proc.stderr.strip().splitlines()
            raise Refusal(lines[-1] if lines else f"{cc} exited {proc.returncode}")
        os.chmod(built, 0o700)
        os.replace(built, binary)


def _native_hasher() -> Path | None:
    binary = _hasher_path()
    if binary.is_file() and _self_test(binary):
        return binary
    cc = shutil.which("cc")
    if cc is None:
        _note("No C compiler (cc) on PATH: hashing in pure Python, about 15 MB/s.")
        return None
    try:
        _build(cc, binary)
    except (OSError, subprocess.SubprocessError, Refusal) as exc:
        _note(f"Could not build {binary} ({exc}): hashing in pure Python, about 15 MB/s.")
        return None
    if _self_test(binary):
        _note(f"Built {binary}.")
        return binary
    _note(f"{binary} disagrees with the Python reference: hashing in pure Python, about 15 MB/s.")
    return None


class Hasher:
    """FNV-1a-64 over a file range: the compiled helper when trusted, else Python."""

    def __init__(self, pure_python: bool) -> None:
        forced = pure_python or os.environ.get(PURE_PYTHON_ENV, "") not in ("", "0")
        self.binary = None if forced else _native_hasher()

    @property
    def kind(self) -> str:
        return "native" if self.binary else "python"

    def range(self, path: Path, offset: int, nbytes: int, progress=_ignore_progress) -> str:
        """The cache file name for ``path[offset, + nbytes)``."""
        if self.binary is not None:
            value = _run_native(self.binary, path, offset, nbytes)
            progress(nbytes)
        else:
            value = gguf.fnv1a64_file_range(path, offset, nbytes, progress=progress)
        return gguf.cache_file_name(value)


# --- Files -------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    """Persist renames, best effort: not every filesystem allows a directory fsync."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def _copy_range(src: Path, offset: int, nbytes: int, dest: Path) -> None:
    """``src[offset, + nbytes)`` into ``dest``, fsynced. Leaves nothing on failure."""
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "wb") as out, src.open("rb") as f:
            f.seek(offset)
            left = nbytes
            while left:
                block = f.read(min(COPY_CHUNK, left))
                if not block:
                    raise Refusal(f"{src}: range [{offset}, +{nbytes}) runs past the end of the file")
                out.write(block)
                left -= len(block)
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        dest.unlink(missing_ok=True)
        raise


def _has_size(path: Path, nbytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size == nbytes
    except OSError:
        return False


def _is_cache_name(name: str) -> bool:
    """A name rpc-server could have written: exactly 16 lowercase hex digits."""
    return len(name) == 16 and all(c in "0123456789abcdef" for c in name)


def _is_export_leftover(name: str) -> bool:
    """A ``.<key>.tmp`` an interrupted export left behind."""
    return name.startswith(".") and name.endswith(".tmp") and _is_cache_name(name[1:-4])


# --- Manifest ----------------------------------------------------------------


def default_manifest_path(gguf_path: Path) -> Path:
    return gguf_path.with_name(gguf_path.name + MANIFEST_SUFFIX)


def load_manifest(path: Path) -> dict:
    """Read a manifest and refuse one whose shape this tool did not write."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"cannot read manifest {path}: {exc}") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
        raise Refusal(f"{path}: not a seed-rpc-cache manifest (no 'entries' list)")
    if doc.get("threshold") != gguf.CACHE_THRESHOLD:
        raise Refusal(
            f"{path}: written for a threshold of {doc.get('threshold')!r}; the pinned rpc-server "
            f"caches tensors over {gguf.CACHE_THRESHOLD} bytes"
        )
    if not isinstance(doc.get("size_bytes"), int):
        raise Refusal(f"{path}: 'size_bytes' is missing or not an integer")
    for i, entry in enumerate(doc["entries"]):
        ok = (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and isinstance(entry.get("offset"), int) and entry["offset"] >= 0
            and isinstance(entry.get("nbytes"), int) and entry["nbytes"] > gguf.CACHE_THRESHOLD
            and (entry.get("hash") is None or (isinstance(entry["hash"], str) and _is_cache_name(entry["hash"])))
        )
        if not ok:
            raise Refusal(f"{path}: entry {i} is malformed: {entry!r}")
    return doc


def cmd_manifest(args: argparse.Namespace) -> int:
    source = Path(args.gguf)
    header = gguf.read_header(source)
    entries = gguf.cache_entries(header)
    total = sum(e.nbytes for e in entries)

    hashes: list[str | None] = [None] * len(entries)
    hasher = None
    if args.hash:
        hasher = Hasher(args.pure_python)
        _note(f"Hashing {len(entries)} tensors, {_gib(total)}, with the {hasher.kind} hasher.")
        progress = Progress(total, "hashed")
        hashes = [hasher.range(source, e.offset_in_file, e.nbytes, progress) for e in entries]

    doc = {
        "gguf": source.name,
        "size_bytes": source.stat().st_size,
        "sha_note": SHA_NOTE,
        "threshold": gguf.CACHE_THRESHOLD,
        "entries": [
            {"name": e.name, "offset": e.offset_in_file, "nbytes": e.nbytes, "hash": hashes[i]}
            for i, e in enumerate(entries)
        ],
    }
    out = Path(args.out) if args.out else default_manifest_path(source)
    _write_text_atomic(out, json.dumps(doc, indent=2) + "\n")

    arch = gguf.architecture(header) or "unknown architecture"
    print(f"Wrote {out}")
    print(f"  {arch}: {len(header.tensors)} tensors, {len(entries)} over the "
          f"{gguf.CACHE_THRESHOLD}-byte threshold, {total} bytes ({_gib(total)}) cacheable")
    if hasher is not None:
        distinct = len(set(hashes))
        dup = f" ({len(entries) - distinct} share a key with another tensor)" if distinct != len(entries) else ""
        print(f"  hashes: {distinct} distinct cache keys, {hasher.kind} hasher{dup}")
    else:
        print("  hashes: not computed — export computes them; verify can match by size only. Rerun with --hash.")
    return 0


# --- Export ------------------------------------------------------------------


def _check_manifest_matches(doc: dict, source: Path, manifest_path: Path) -> None:
    """Refuse a manifest that does not describe this GGUF's layout."""
    size = source.stat().st_size
    if size != doc["size_bytes"]:
        raise Refusal(
            f"{source} is {size} bytes but {manifest_path} was written for a {doc['size_bytes']}-byte "
            f"file ({doc.get('gguf')!r}) — a different file, or an incomplete download"
        )
    if source.name != doc.get("gguf"):
        _note(f"Note: {manifest_path} names {doc.get('gguf')!r}; exporting from {source.name} (same size).")
    derived = [(e.name, e.offset_in_file, e.nbytes) for e in gguf.cache_entries(gguf.read_header(source))]
    listed = [(e["name"], e["offset"], e["nbytes"]) for e in doc["entries"]]
    if derived != listed:
        raise Refusal(f"{manifest_path} does not list the tensors {source} holds at those offsets — regenerate it")


def cmd_export(args: argparse.Namespace) -> int:
    source = Path(args.gguf)
    manifest_path = Path(args.manifest)
    doc = load_manifest(manifest_path)
    _check_manifest_matches(doc, source, manifest_path)

    entries = doc["entries"]
    selected = [e for e in entries if e["hash"] or not args.only_hashed]
    unhashed_skipped = len(entries) - len(selected)
    if not selected and unhashed_skipped:
        raise Refusal(f"--only-hashed, but {manifest_path} carries no hashes — run `manifest --hash` first")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    pending = [e for e in selected if not (e["hash"] and _has_size(dest / e["hash"], e["nbytes"]))]
    needed = sum(e["nbytes"] for e in pending)
    free = shutil.disk_usage(dest).free
    if needed > free:
        raise Refusal(f"{dest} has {_gib(free)} free; exporting needs up to {_gib(needed)}")

    hasher = Hasher(args.pure_python)
    _note(f"Exporting up to {len(pending)} tensors, {_gib(needed)}, into {dest} ({hasher.kind} hasher).")
    progress = Progress(needed, "exported")
    written = present = 0
    written_bytes = 0
    for entry in selected:
        key = entry["hash"] or hasher.range(source, entry["offset"], entry["nbytes"])
        final = dest / key
        if _has_size(final, entry["nbytes"]):
            present += 1
            continue
        if final.exists():
            _note(f"Replacing {final}: {final.stat().st_size} bytes on disk, {entry['nbytes']} expected.")
        tmp = dest / f".{key}.tmp"
        _copy_range(source, entry["offset"], entry["nbytes"], tmp)
        # Re-hash what was written before it gets a name rpc-server will trust.
        # Same size and same layout is not same bytes: a re-uploaded GGUF with
        # identical shapes would otherwise publish its weights under the old keys.
        actual = hasher.range(tmp, 0, entry["nbytes"])
        if actual != key:
            tmp.unlink(missing_ok=True)
            raise Refusal(
                f"{entry['name']}: the bytes in {source} hash to {actual}, but {manifest_path} says {key} — "
                "the manifest was hashed from different bytes. Rerun `manifest --hash` on this file."
            )
        os.replace(tmp, final)
        written += 1
        written_bytes += entry["nbytes"]
        progress(entry["nbytes"])
    _fsync_dir(dest)

    print(f"Exported into {dest}: {written} written ({_gib(written_bytes)}), {present} already present")
    if unhashed_skipped:
        print(f"  {unhashed_skipped} entries skipped: no hash in the manifest (--only-hashed)")
    return 0


# --- Verify ------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    doc = load_manifest(Path(args.manifest))
    if not directory.is_dir():
        raise Refusal(f"{directory} is not a directory")

    entries = doc["entries"]
    expected = {e["hash"]: e["nbytes"] for e in entries if e["hash"]}
    unhashed_sizes = {e["nbytes"] for e in entries if not e["hash"]}
    size_only_mode = not expected

    ok: list[str] = []
    by_size: list[str] = []
    mismatched: list[tuple[str, int, int]] = []
    unknown: list[tuple[str, int | None]] = []
    skipped_dirs: list[str] = []
    for item in sorted(os.scandir(directory), key=lambda d: d.name):
        if item.is_dir():
            skipped_dirs.append(item.name)
            continue
        size = item.stat().st_size if item.is_file() else None  # None: a dangling link
        if item.name in expected:
            if size == expected[item.name]:
                ok.append(item.name)
            else:
                mismatched.append((item.name, -1 if size is None else size, expected[item.name]))
        elif _is_cache_name(item.name) and size in unhashed_sizes:
            by_size.append(item.name)
        else:
            unknown.append((item.name, size))

    if size_only_mode:
        print(f"{args.manifest} carries no hashes: files were matched by SIZE only, so a right-sized file "
              "under the wrong name passes. Run `manifest --hash` for a name check.")
    for name, size, want in mismatched:
        print(f"MISMATCHED {name}: {size} bytes, the manifest says {want} — rpc-server would load it as-is")
    for name, size in unknown:
        shown = "dangling link" if size is None else f"{size} bytes"
        print(f"UNKNOWN    {name}: {shown}, not in the manifest")

    removed = 0
    if args.delete_mismatched:
        for name, _, _ in mismatched:
            (directory / name).unlink()
            print(f"removed {name}")
            removed += 1
        mismatched = []
    if args.delete_unknown:
        kept = []
        for name, size in unknown:
            # Only names rpc-server (or an interrupted export) could have made.
            # Anything else here is more likely a sign that DIR is the wrong
            # directory than a stray cache file, so it is reported, never deleted.
            if _is_cache_name(name) or _is_export_leftover(name):
                (directory / name).unlink()
                print(f"removed {name}")
                removed += 1
            else:
                kept.append((name, size))
                print(f"kept    {name}: not a name rpc-server writes — is {directory} the cache directory?")
        unknown = kept
    if removed:
        _fsync_dir(directory)

    present = set(ok)
    absent = [e for e in entries if e["hash"] and e["hash"] not in present]
    summary = f"{directory}: {len(ok)} ok, {len(mismatched)} mismatched, {len(unknown)} unknown"
    if by_size:
        summary += f", {len(by_size)} matched by size only"
    if removed:
        summary += f" ({removed} removed)"
    print(summary)
    if absent:
        print(f"  {len(absent)} manifest entries not present ({_gib(sum(e['nbytes'] for e in absent))}) — "
              "expected on a worker that holds only its share")
    if skipped_dirs:
        print(f"  ignored subdirectories: {', '.join(skipped_dirs)}")
        if "rpc" in skipped_dirs and not (ok or by_size or mismatched):
            print(f"  rpc-server caches into $LLAMA_CACHE/rpc/ — did you mean {directory / 'rpc'}?")
    return 1 if mismatched or unknown else 0


# --- CLI ---------------------------------------------------------------------


# --- push: an exported cache onto a group's workers --------------------------------

SSH_OPTS = (
    "-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
    "-o ServerAliveInterval=15 -o ServerAliveCountMax=4"
)


def push_plan(network_config_path: Path, group_name: str, export_dir: Path) -> list[dict]:
    """One rsync invocation per worker of the group, as data.

    Reads the fleet file through the orchestrator's own loader (so this needs
    the project environment: run it with ``uv run``), resolves each worker's
    user, port and key the way scripts/lib/common.sh does, and targets the
    cache directory the worker's template names (``model.rpc_server.cache_dir``,
    default /opt/aureax/rpc-cache) plus the ``rpc/`` level rpc-server reads.
    The cache is root-owned on the node — the supervisor creates it inside the
    pod — so the remote side runs rsync under sudo, exactly as the deploy
    scripts reach every other root-owned path.
    """
    from orchestrator.config.loader import load_network_config
    from orchestrator.deploy.ssh import DEFAULT_KEY_STORE, resolve_key_path

    cfg = load_network_config(network_config_path)
    group = next((g for g in cfg.node_groups if g.name == group_name), None)
    if group is None:
        raise Refusal(f"no node_groups entry named {group_name!r} in {network_config_path}")
    by_id = {t.node_id: t for t in cfg.ssh.targets}
    key_store = getattr(cfg.ssh, "key_store", DEFAULT_KEY_STORE)
    plan: list[dict] = []
    for worker in group.workers:
        target = by_id.get(worker)
        if target is None:
            raise Refusal(f"group {group_name!r} names worker {worker!r}, which has no ssh.targets entry")
        if not target.enabled:
            continue
        key = resolve_key_path(target.key_file, key_store) or str(Path("~/.ssh/id_ed25519").expanduser())
        cache_dir = "/opt/aureax/rpc-cache"
        if target.config_file:
            try:
                from client.config.parser import load_edge_config_raw

                raw = load_edge_config_raw(network_config_path.parent / target.config_file)
                cache_dir = str(((raw.get("model") or {}).get("rpc_server") or {}).get("cache_dir") or cache_dir)
            except Exception:
                pass
        rsh = f"ssh -p {target.port} {SSH_OPTS} -o UserKnownHostsFile={Path.home() / '.aureax' / 'known_hosts'} -i {key}"
        plan.append({
            "node_id": worker,
            "host": target.host,
            "argv": [
                "rsync", "-a", "--size-only", "--chmod=F0644,D0755", "--rsync-path", "sudo rsync",
                "-e", rsh,
                f"{export_dir.rstrip('/') if isinstance(export_dir, str) else str(export_dir).rstrip('/')}/",
                f"{target.user}@{target.host}:{cache_dir.rstrip('/')}/rpc/",
            ],
        })
    return plan


def cmd_push(args: argparse.Namespace) -> int:
    """rsync an exported cache directory onto every worker of one group, in parallel.

    ``--size-only`` is deliberate: the files are named by their content hash
    and have a fixed size, so size is the right identity and a byte-wise
    checksum of 20 GB on both ends would be pure cost. rsync writes each file
    under a temporary dot-name and renames it into place, which is what keeps
    an interrupted transfer out of rpc-server's way (the worker's audit also
    removes dotfiles at start). Workers are pushed concurrently because each
    has its own link to the switch; the orchestrator's uplink is the limit.
    """
    import concurrent.futures

    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        raise Refusal(f"{export_dir} is not a directory — run `export` first")
    if not shutil.which("rsync"):
        raise Refusal("rsync is not on PATH on this machine; the push uses it")
    plan = push_plan(Path(args.network_config), args.group, export_dir)
    if not plan:
        raise Refusal(f"group {args.group!r} has no enabled workers")
    if args.print_only:
        for item in plan:
            print(f"{item['node_id']} ({item['host']}): " + " ".join(_quote(a) for a in item["argv"]))
        return 0

    def run(item: dict) -> tuple[str, int, str]:
        proc = subprocess.run(item["argv"], capture_output=True, text=True)
        return item["node_id"], proc.returncode, (proc.stderr or proc.stdout).strip()

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for node_id, rc, tail in pool.map(run, plan):
            if rc == 0:
                _note(f"{node_id}: pushed")
            else:
                failures += 1
                _note(f"{node_id}: rsync exited {rc}: {tail[-400:]}")
    return 1 if failures else 0


def _quote(word: str) -> str:
    import shlex

    return shlex.quote(word)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--pure-python",
        action="store_true",
        help=f"hash in Python even when cc is available (also: {PURE_PYTHON_ENV}=1)",
    )
    parser = argparse.ArgumentParser(
        prog="seed-rpc-cache.py",
        description="Pre-seed and audit llama.cpp rpc-server's tensor cache from a GGUF.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifest", parents=[common], help="list the tensors rpc-server would cache")
    p.add_argument("gguf")
    p.add_argument("--out", help=f"output path (default: <gguf>{MANIFEST_SUFFIX})")
    p.add_argument("--hash", action="store_true", help="compute each entry's cache key")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("export", parents=[common], help="write cache files for a manifest's entries")
    p.add_argument("gguf")
    p.add_argument("--manifest", required=True)
    p.add_argument("--dest", required=True, help="the cache directory, i.e. $LLAMA_CACHE/rpc")
    p.add_argument("--only-hashed", action="store_true", help="skip entries the manifest has no hash for")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify", parents=[common], help="audit a cache directory against a manifest")
    p.add_argument("dir")
    p.add_argument("--manifest", required=True)
    p.add_argument("--delete-unknown", action="store_true", help="remove cache-named files the manifest does not list")
    p.add_argument("--delete-mismatched", action="store_true", help="remove known files of the wrong size")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("push", parents=[common], help="rsync an exported cache onto a group's workers")
    p.add_argument("--export-dir", required=True, help="the directory `export --dest` wrote")
    p.add_argument("--group", required=True, help="node_groups entry whose workers receive it")
    p.add_argument("--network-config", default="configs/aureax-network.yaml")
    p.add_argument("--jobs", type=int, default=8, help="workers pushed concurrently")
    p.add_argument("--print-only", action="store_true", help="print the rsync commands and exit")
    p.set_defaults(func=cmd_push)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (Refusal, gguf.GGUFError, OSError) as exc:
        print(f"seed-rpc-cache: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
