"""Do a group's members really run the byte-identical llama.cpp binaries?

The llama.cpp RPC wire format is version-locked: a head that dials a peer built
from another commit connects and then HANGS at the handshake, with no error on
either side. The only other symptom is an edge pod sitting ``0/1 Running``
until its startupProbe budget (3015s) expires, which looks exactly like a slow
model load.

**Why the binaries and not the image digest.** Comparing each member's pod
``containerStatuses[].imageID`` reports a mismatch on every start of a perfectly
healthy fleet::

    group 'qwen35b-a': the worker shard-w1 on node-a runs image
    sha256:1b1b... but worker shard-w2 on node-b runs sha256:4644...

— on members whose binaries are identical (``scripts/test-all.sh`` §6 compares
``rpc-server``'s sha256 across members over SSH, and agrees). Every
node builds ``aureax-client:dev`` NATIVELY — ``scripts/build-on-node.sh``, once
per node, from ``setup-all.sh`` and ``update-all.sh`` — so the image is built
independently N times from the same pinned ``LLAMA_CPP_VERSION``. Layer
timestamps, apt mirror state and base-image pulls all differ, so the digests
differ *by construction* while the compiled binaries do not. An image digest is
the wrong observable for this fleet; the thing the RPC handshake actually cares
about is the bytes of ``llama-server`` and ``rpc-server``, so that is what this
module compares — the same question ``test-all.sh`` §6 asks, in the same shape.

One SSH round trip per member, one container start per round trip. Shared by
the pre-start gate (``lifecycle._assert_group_builds_match``) and
``aureax group doctor`` (``cli._group_problems``) so both ask it identically:
``BUILD_PROBE`` is spliced into ``cli._DOCTOR_PROBE`` rather than duplicated,
and both paths read the same ``key=value`` fact names back.

Nothing here logs or exits. ``compare_group`` is pure over fact dicts, so every
verdict is unit-testable with no SSH, and the two callers report through
different channels (a SystemLogger that raises, a CLI that colours and sets an
exit code).
"""

from __future__ import annotations

import re

from orchestrator.deploy.ssh import DEFAULT_KEY_STORE

# The two binaries the RPC wire format is version-locked across. Both
# Dockerfiles install them with `COPY --from=llama-builder /out/bin/
# /usr/local/bin/` (docker/Dockerfile.client, docker/Dockerfile.client.cuda)
# and the supervisor launches one of the two
# depending on node.role, so a group is coherent only if BOTH match: a head
# runs llama-server against peers running rpc-server.
LLAMA_SERVER_PATH = "/usr/local/bin/llama-server"
RPC_SERVER_PATH = "/usr/local/bin/rpc-server"

CLIENT_IMAGE = "aureax-client:dev"

# Fact keys the probe prints. Named here because `cli._parse_doctor_facts`
# parses the same lines out of the doctor's larger probe.
LLAMA_SHA_KEY = "llama_server_sha"
RPC_SHA_KEY = "rpc_server_sha"
BUILD_KEY = "llama_build_docker"
ERROR_KEY = "build_probe_error"

# Injected on this side, not reported by the node — the remedy line has to name
# an address the operator can paste into ./scripts/build-on-node.sh, and the
# node cannot know which address we reached it on.
HOST_KEY = "ssh_host"

BUILD_UNAVAILABLE = "unavailable (image missing, or docker/sudo failed)"

# Seconds for one member's round trip. A ceiling, not a tuned value: the
# container start plus two sha256sums is seconds on an edge node.
PROBE_TIMEOUT = 60

# ONE container start for all three facts. Notes on the shape, each of which
# was a real failure somewhere in this repo:
#   - `sudo -n`: no remote tty is allocated, so a plain `sudo` on a node whose
#     sudoers wants a password prompts into a discarded stderr and then reads
#     our stdin — a hang, on exactly the node being diagnosed.
#   - `--pull=never`: `docker run` defaults to `--pull=missing`, so on a node
#     whose image is genuinely absent it reaches for docker.io and blocks past
#     the budget instead of failing.
#   - `timeout 20`: run_command DISCARDS the stdout it already collected the
#     moment its own budget expires, so an unbounded probe against a wedged
#     docker daemon takes every other fact in the same round trip down with it.
#   - `2>&1`: the remote's own error text is the difference between "rebuild
#     this node" and "fix sudo on this node", so it has to survive the trip.
#     That is also why a value is only trusted when it is 64 hex digits
#     (`checksums` below) — an error message must never read as a mismatch.
BUILD_PROBE = rf"""
_AX_OUT=$(timeout 20 sudo -n docker run --rm --pull=never --entrypoint sh {CLIENT_IMAGE} \
  -c 'sha256sum {LLAMA_SERVER_PATH} {RPC_SERVER_PATH}; cat /usr/local/share/aureax-llama-build.json' 2>&1)
echo "{LLAMA_SHA_KEY}=$(printf '%s\n' "$_AX_OUT" | awk '$2 ~ /llama-server$/ {{print $1; exit}}')"
echo "{RPC_SHA_KEY}=$(printf '%s\n' "$_AX_OUT" | awk '$2 ~ /rpc-server$/ {{print $1; exit}}')"
echo "{BUILD_KEY}=$(printf '%s\n' "$_AX_OUT" | grep -m1 '^{{' || echo '{BUILD_UNAVAILABLE}')"
echo "{ERROR_KEY}=$(printf '%s\n' "$_AX_OUT" | grep -v '^[0-9a-f]\{{64\}} ' | grep -v '^{{' \
  | tr '\n' ' ' | cut -c1-200)"
"""

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_probe(out: str) -> dict[str, str]:
    """``key=value`` lines from :data:`BUILD_PROBE`'s stdout.

    Deliberately the same rule ``cli._parse_doctor_facts`` follows — split on
    the FIRST ``=`` only — because the doctor parses these very lines out of
    its own larger probe and the two must not disagree about a value containing
    an ``=`` (the error text can).
    """
    facts: dict[str, str] = {}
    for raw in (out or "").splitlines():
        key, sep, value = raw.strip().partition("=")
        if sep and key.strip():
            facts[key.strip()] = value.strip()
    return facts


def probe_member(target, *, key_store: str = DEFAULT_KEY_STORE,
                 timeout: int = PROBE_TIMEOUT) -> dict[str, str]:
    """One member's build facts. Never raises.

    Returns the parsed probe facts plus ``ssh_host``, or — when the node could
    not be reached or the probe itself failed — a dict carrying ``error``.
    ``compare_group`` turns that into a note rather than a problem: a node we
    could not ask has an UNKNOWN build, and refusing to start a fleet over an
    SSH blip would be a worse bug than the one this closes.

    (The build id comes back under ``llama_build_docker``, the key the doctor's
    own verdict already reads: it is the build of the DOCKER copy of the image,
    which a stale containerd import can lag — a distinction the doctor makes
    beside this one.)
    """
    from orchestrator.deploy.ssh import conn_kwargs, run_command

    facts: dict[str, str] = {HOST_KEY: target.host}
    try:
        rc, out, err = run_command(
            target.host, BUILD_PROBE, **conn_kwargs(target, key_store, timeout=timeout)
        )
    except Exception as e:
        facts["error"] = f"unreachable over SSH ({type(e).__name__}: {e})"
        return facts
    if rc == 124:
        # run_command's timeout convention: it returns no stdout at all, so
        # there is nothing to parse and "no facts" must not read as "no image".
        facts["error"] = f"the build probe timed out after {timeout}s"
        return facts
    facts.update(parse_probe(out))
    if not checksums(facts):
        detail = facts.get(ERROR_KEY) or (err or "").strip() or f"exit {rc}"
        facts["error"] = f"could not read the binaries' checksums ({detail[:200]})"
    return facts


def probe_group(targets, *, key_store: str = DEFAULT_KEY_STORE,
                timeout: int = PROBE_TIMEOUT) -> dict[str, dict[str, str]]:
    """Probe every target concurrently. Returns node_id -> facts.

    Concurrent for the reason ``cli._fan_out`` is: the cost here is latency,
    and run serially one powered-off member charges its whole ``timeout`` to
    every member queued behind it — eight members would be eight minutes on a
    path that gates a start.
    """
    from concurrent.futures import ThreadPoolExecutor

    targets = list(targets)
    if not targets:
        return {}
    with ThreadPoolExecutor(max_workers=min(len(targets), 16)) as pool:
        results = list(pool.map(
            lambda t: probe_member(t, key_store=key_store, timeout=timeout), targets
        ))
    return {t.node_id: facts for t, facts in zip(targets, results, strict=True)}


def checksums(facts: dict[str, str] | None) -> tuple[str, str] | None:
    """``(llama_server_sha, rpc_server_sha)``, or None when either is unreadable.

    Only 64 hex digits count. The probe merges the remote's stderr into its own
    output so a failure's reason survives the trip, which means the answer has
    to be RECOGNISED rather than assumed: a sudo refusal read as a checksum
    would send the operator rebuilding images to fix an SSH problem. Same
    contract as ``scripts/test-all.sh``'s ``sha_of``.
    """
    llama = (facts or {}).get(LLAMA_SHA_KEY, "")
    rpc = (facts or {}).get(RPC_SHA_KEY, "")
    if _SHA_RE.match(llama) and _SHA_RE.match(rpc):
        return llama, rpc
    return None


def _uncovered_note(group, node_id: str, facts: dict[str, str]) -> str:
    """The SKIP line for a member whose binaries could not be read."""
    reason = (
        facts.get("error")
        or facts.get(ERROR_KEY)
        or "it reported no checksum for llama-server, rpc-server, or both"
    )
    return (
        f"SKIP {node_id}: the build check did not cover this node — {reason}. "
        f"Group {group.name!r}'s build consistency is UNKNOWN for it; the members "
        f"that did answer were still compared."
    )


def _mismatch_problem(group, head_id: str, head_sums: tuple[str, str],
                      node_id: str, sums: tuple[str, str], host: str) -> str:
    """The FAIL line for a member whose binaries differ from the head's."""
    return (
        f"FAIL {node_id}: llama-server/rpc-server do not match head {head_id} in group "
        f"{group.name!r}.\n"
        f"       {node_id:<14} llama-server {sums[0][:12]}  rpc-server {sums[1][:12]}\n"
        f"       {head_id:<14} llama-server {head_sums[0][:12]}  rpc-server {head_sums[1][:12]}\n"
        f"       The llama.cpp RPC wire format is version-locked: the head connects and then\n"
        f"       HANGS at the handshake with no error on either side, and the pod sits 0/1\n"
        f"       Running until the startupProbe budget (3015s) expires. Rebuild the odd node\n"
        f"       out: ./scripts/build-on-node.sh {host or node_id}  (or ./scripts/update-all.sh\n"
        f"       for the whole fleet), then re-run start."
    )


def compare_group(group, facts_by_node: dict[str, dict[str, str]]) -> tuple[list[str], list[str]]:
    """One group's build verdict: ``(problems, notes)``.

    ``problems`` are FAIL lines — an empty list means every member that could
    be read runs the same binaries, and that is what gates the start and the
    doctor's exit code. ``notes`` are the OK/SKIP lines, which exist because
    "checked, all good" and "not checked at all" look identical otherwise.

    Fails OPEN on a member that could not be probed and CLOSED on two
    checksums that differ. An unreadable checksum is not a mismatch: the image
    digests this replaced failed the opposite way and refused every start on a
    fleet whose binaries were identical (see the module docstring), so a check
    that cannot see something says so instead of guessing.

    The head is the reference because it is the member the operator looks at
    first — and when the head itself could not be read there is nothing to
    compare against, so the whole group is a note.
    """
    problems: list[str] = []
    notes: list[str] = []
    facts_of = {nid: (facts_by_node.get(nid) or {}) for nid in group.members}
    sums = {nid: s for nid, f in facts_of.items() if (s := checksums(f))}

    for nid in group.members:
        if nid not in sums:
            notes.append(_uncovered_note(group, nid, facts_of[nid]))

    head_sums = sums.get(group.head)
    if head_sums is None:
        if sums:
            notes.append(
                f"SKIP group {group.name!r}: head {group.head} could not be read, so the "
                f"{len(sums)} member(s) that answered were not compared — the head is the "
                f"reference every worker is checked against."
            )
        return problems, notes

    for nid, member_sums in sums.items():
        if nid == group.head or member_sums == head_sums:
            continue
        problems.append(_mismatch_problem(
            group, group.head, head_sums, nid, member_sums, facts_of[nid].get(HOST_KEY, "")
        ))

    if not problems and len(sums) > 1:
        notes.append(
            f"OK   llama-server/rpc-server checksums identical on every member "
            f"({len(sums)}/{len(group.members)} read)"
        )
    return problems, notes
