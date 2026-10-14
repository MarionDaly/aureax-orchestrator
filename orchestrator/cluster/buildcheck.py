"""Do a group's members really run the same llama.cpp?

The llama.cpp RPC wire format is version-locked: a head that dials a peer built
from another commit connects and then HANGS at the handshake, with no error on
either side. The only other symptom is an edge pod sitting ``0/1 Running``
until its startupProbe budget (3015s) expires, which looks exactly like a slow
model load.

**Two facts, two scopes.** What the handshake requires is the same RPC protocol
version and the same GGML op table, and both are properties of the llama.cpp
COMMIT: the messages themselves are packed little-endian fixed-width structs,
so the protocol is architecture-neutral. Hence two checks, and the first wins
whenever they disagree:

- the llama.cpp ``commit`` each image records at build time
  (``BUILD_JSON_PATH``, written by both Dockerfiles once they have verified the
  clone against ``LLAMA_CPP_COMMIT``) must be the head's on every member that
  reports one. That is the version lock itself.
- the sha256 of ``llama-server`` and ``rpc-server`` must be the head's on every
  member of the head's ARCHITECTURE (``uname -m`` inside the image). An aarch64
  image and an x86_64 CUDA image can never be byte-identical, even built from
  one commit, so a member of another architecture is compared by commit alone
  and the verdict says so on a line of its own. Within one architecture the
  bytes stay the stronger check: a commit is what the build said it cloned,
  while the binaries also catch a different toolchain, different flags or a
  patched tree.

An architecture that could not be read never relaxes anything: such a pair is
held to identical binaries, exactly as before this module knew about
architectures. Only a POSITIVE "these are two different architectures" moves a
member onto the commit-only comparison.

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
about is what ``llama-server`` and ``rpc-server`` were built from, so that is
what this module compares — the same question ``test-all.sh`` §6 asks, in the
same shape.

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

import json
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

# {"version": "<tag>", "commit": "<sha>"}, written by both Dockerfiles after
# they check the clone against LLAMA_CPP_COMMIT, so the commit in it is the one
# that was actually compiled rather than the one the tag pointed at that day.
BUILD_JSON_PATH = "/usr/local/share/aureax-llama-build.json"

CLIENT_IMAGE = "aureax-client:dev"

# Fact keys the probe prints. Named here because `cli._parse_doctor_facts`
# parses the same lines out of the doctor's larger probe.
LLAMA_SHA_KEY = "llama_server_sha"
RPC_SHA_KEY = "rpc_server_sha"
BUILD_KEY = "llama_build_docker"
# `uname -m` INSIDE the image's container. Deliberately not `arch`: the
# doctor's own probe already prints `arch=<host> cores=... mem=...` as its first
# line, and this probe is spliced in after it, so a second `arch=` line would
# silently replace that fact in the doctor's parsed dict.
ARCH_KEY = "image_arch"
ERROR_KEY = "build_probe_error"

# Injected on this side, not reported by the node — the remedy line has to name
# an address the operator can paste into ./scripts/build-on-node.sh, and the
# node cannot know which address we reached it on.
HOST_KEY = "ssh_host"

BUILD_UNAVAILABLE = "unavailable (image missing, or docker/sudo failed)"

# Seconds for one member's round trip. A ceiling, not a tuned value: the
# container start plus two sha256sums is seconds on an edge node.
PROBE_TIMEOUT = 60

# ONE container start for all four facts. Notes on the shape, each of which
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
#     That is also why a value is only trusted when it is recognised
#     (`checksums`, `llama_commit`, `image_arch` below) — an error message must
#     never read as a mismatch.
#   - the architecture line is tagged INSIDE the container (`image_arch=...`)
#     and picked out by that tag, because a bare `uname -m` line would be
#     indistinguishable from one line of an error message.
BUILD_PROBE = rf"""
_AX_OUT=$(timeout 20 sudo -n docker run --rm --pull=never --entrypoint sh {CLIENT_IMAGE} \
  -c 'sha256sum {LLAMA_SERVER_PATH} {RPC_SERVER_PATH}; cat {BUILD_JSON_PATH}; echo "{ARCH_KEY}=$(uname -m)"' 2>&1)
echo "{LLAMA_SHA_KEY}=$(printf '%s\n' "$_AX_OUT" | awk '$2 ~ /llama-server$/ {{print $1; exit}}')"
echo "{RPC_SHA_KEY}=$(printf '%s\n' "$_AX_OUT" | awk '$2 ~ /rpc-server$/ {{print $1; exit}}')"
echo "{BUILD_KEY}=$(printf '%s\n' "$_AX_OUT" | grep -m1 '^{{' || echo '{BUILD_UNAVAILABLE}')"
echo "{ARCH_KEY}=$(printf '%s\n' "$_AX_OUT" | awk -F= '$1 == "{ARCH_KEY}" {{print $2; exit}}')"
echo "{ERROR_KEY}=$(printf '%s\n' "$_AX_OUT" | grep -v '^[0-9a-f]\{{64\}} ' | grep -v '^{{' \
  | grep -v '^{ARCH_KEY}=' | tr '\n' ' ' | cut -c1-200)"
"""

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
# 40 hex digits from `git rev-parse HEAD` (64 in a SHA-256 repository); 7 is
# git's shortest abbreviation. Anything else is not a commit.
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
# What `uname -m` prints on Linux: aarch64, x86_64, armv7l, riscv64, ppc64le...
_ARCH_RE = re.compile(r"^[A-Za-z0-9_]{1,32}$")

# How many characters of a checksum or a commit a verdict line shows.
_SHORT = 12


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

    (The build record comes back under ``llama_build_docker``, the key the
    doctor's own verdict already reads: it is the build of the DOCKER copy of
    the image, which a stale containerd import can lag — a distinction the
    doctor makes beside this one.)
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


def llama_commit(facts: dict[str, str] | None) -> str | None:
    """The llama.cpp commit the image recorded at build time, or None.

    Recognised rather than assumed, for the reason ``checksums`` is: the probe
    prints ``BUILD_UNAVAILABLE`` when the record could not be read, and a record
    that carries a ``version`` but no ``commit`` says nothing about the commit.
    Both are an unreadable commit — a note — never a mismatch.
    """
    raw = (facts or {}).get(BUILD_KEY, "")
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        return None
    value = record.get("commit") if isinstance(record, dict) else None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if _COMMIT_RE.match(value) else None


def image_arch(facts: dict[str, str] | None) -> str | None:
    """``uname -m`` inside the member's image, or None when unreadable."""
    value = (facts or {}).get(ARCH_KEY, "")
    return value if _ARCH_RE.match(value) else None


def _arch_label(arch: str | None) -> str:
    return arch or "architecture unreported"


def _uncovered_note(group, node_id: str, facts: dict[str, str]) -> str:
    """The SKIP line for a member that reported neither binaries nor a commit."""
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


def _binaries_unread_note(node_id: str, facts: dict[str, str], commit: str) -> str:
    """The SKIP line for a member whose commit was read but whose binaries were not."""
    reason = (
        facts.get("error")
        or facts.get(ERROR_KEY)
        or "it reported no checksum for llama-server, rpc-server, or both"
    )
    return (
        f"SKIP {node_id}: the build check did not cover this node's binaries — {reason}. "
        f"Its llama.cpp commit {commit[:_SHORT]} was still read."
    )


def _commit_unread_note(node_id: str, facts: dict[str, str]) -> str:
    """The SKIP line for a member whose binaries were read but whose commit was not."""
    raw = facts.get(BUILD_KEY, "")
    return (
        f"SKIP {node_id}: its llama.cpp commit could not be read — the build record said "
        f"{raw or '(nothing)'!r}. Its binaries were still read."
    )


def _hang_and_remedy(host: str) -> str:
    """The consequence and the two remedies, shared by both FAIL shapes."""
    return (
        f"       The llama.cpp RPC wire format is version-locked: the head connects and then\n"
        f"       HANGS at the handshake with no error on either side, and the pod sits 0/1\n"
        f"       Running until the startupProbe budget (3015s) expires. Rebuild the odd node\n"
        f"       out: ./scripts/build-on-node.sh {host}  (or ./scripts/update-all.sh\n"
        f"       for the whole fleet), then re-run start."
    )


def _commit_problem(group, head_id: str, head_commit: str, head_arch: str | None,
                    node_id: str, commit: str, arch: str | None, host: str) -> str:
    """The FAIL line for a member built from another llama.cpp commit than the head."""
    return (
        f"FAIL {node_id}: llama.cpp commit {commit[:_SHORT]} does not match head {head_id}'s "
        f"{head_commit[:_SHORT]} in group {group.name!r}.\n"
        f"       {node_id:<14} commit {commit[:_SHORT]:<12}  {_arch_label(arch)}\n"
        f"       {head_id:<14} commit {head_commit[:_SHORT]:<12}  {_arch_label(head_arch)}\n"
        + _hang_and_remedy(host or node_id)
    )


def _mismatch_problem(group, head_id: str, head_sums: tuple[str, str], head_arch: str | None,
                      head_commit: str | None, node_id: str, sums: tuple[str, str],
                      arch: str | None, commit: str | None, host: str) -> str:
    """The FAIL line for a same-architecture member whose binaries differ from the head's."""
    context = ""
    if arch is None or head_arch is None:
        unread = " and ".join(n for n, a in ((node_id, arch), (head_id, head_arch)) if a is None)
        context += (
            f"       Image architecture unreported for {unread}, so the two were held to\n"
            f"       identical binaries (only a known other architecture is compared by commit).\n"
        )
    if commit and head_commit:
        # Reaching here means the commits matched: a commit mismatch is its own
        # FAIL and wins. Saying so points at the toolchain, not the version.
        context += (
            f"       Both report llama.cpp commit {commit[:_SHORT]}, so the difference is the\n"
            f"       toolchain, the build flags or the source tree.\n"
        )
    return (
        f"FAIL {node_id}: llama-server/rpc-server do not match head {head_id} in group "
        f"{group.name!r}.\n"
        f"       {node_id:<14} llama-server {sums[0][:_SHORT]}  rpc-server {sums[1][:_SHORT]}  "
        f"{_arch_label(arch)}\n"
        f"       {head_id:<14} llama-server {head_sums[0][:_SHORT]}  "
        f"rpc-server {head_sums[1][:_SHORT]}  {_arch_label(head_arch)}\n"
        + context
        + _hang_and_remedy(host or node_id)
    )


def _not_compared_note(group, node_id: str, *, arch: str | None, head_arch: str | None,
                       has_sums: bool, has_commit: bool, head_has_sums: bool,
                       head_has_commit: bool, cross_arch: bool) -> str:
    """The SKIP line for a member that reported something, but nothing the head did."""
    head_id = group.head
    if cross_arch:
        if not has_commit and not head_has_commit:
            missing = f"neither it nor head {head_id} reported one"
        elif not has_commit:
            missing = "it reported none"
        else:
            missing = f"head {head_id} reported none"
        why = (
            f"it is {arch} and head {head_id} is {head_arch}, so only the llama.cpp "
            f"commit is comparable, and {missing}"
        )
    else:
        mine = "binaries" if has_sums else "llama.cpp commit"
        theirs = "binaries" if head_has_sums else "llama.cpp commit"
        why = f"it reported only its {mine} and head {head_id} only its {theirs}"
    return (
        f"SKIP {node_id}: not compared — {why}. Group {group.name!r}'s build consistency "
        f"is UNKNOWN for it."
    )


def compare_group(group, facts_by_node: dict[str, dict[str, str]]) -> tuple[list[str], list[str]]:
    """One group's build verdict: ``(problems, notes)``.

    ``problems`` are FAIL lines — an empty list means every member that could
    be read was built from the head's llama.cpp commit and, within the head's
    architecture, runs the head's very binaries. That is what gates the start
    and the doctor's exit code. ``notes`` are the OK/SKIP lines, which exist
    because "checked, all good" and "not checked at all" look identical
    otherwise.

    Per member, against the head:

    1. commit — compared whenever both sides recorded one, whatever the
       architecture. A different commit is a FAIL and is the member's ONLY
       FAIL: it is the root cause, and the binaries of two commits differ as a
       matter of course.
    2. binaries — compared whenever both sides were read and the member is not
       KNOWN to be another architecture. An unreported architecture is held to
       identical binaries, so a lost ``uname -m`` can never relax the check.
    3. a member of another architecture that passed (1) gets one OK line saying
       it was compared by commit; a member that reported something but shares no
       readable fact with the head gets a SKIP saying it was not compared.

    Fails OPEN on what could not be read and CLOSED on two values that differ.
    An unreadable value is not a mismatch: the image digests this replaced
    failed the opposite way and refused every start on a fleet whose binaries
    were identical (see the module docstring), so a check that cannot see
    something says so instead of guessing.

    The head is the reference because it is the member the operator looks at
    first — and when the head itself could not be read there is nothing to
    compare against, so the whole group is a note.
    """
    problems: list[str] = []
    notes: list[str] = []
    head = group.head
    facts_of = {nid: (facts_by_node.get(nid) or {}) for nid in group.members}
    sums = {nid: s for nid, f in facts_of.items() if (s := checksums(f))}
    commits = {nid: c for nid, f in facts_of.items() if (c := llama_commit(f))}
    arches = {nid: a for nid, f in facts_of.items() if (a := image_arch(f))}

    # What each member could not report — facts about the member alone, so
    # they come first and in member order, head included.
    for nid in group.members:
        if nid not in sums and nid not in commits:
            notes.append(_uncovered_note(group, nid, facts_of[nid]))
        elif nid not in sums:
            notes.append(_binaries_unread_note(nid, facts_of[nid], commits[nid]))
        elif nid not in commits:
            notes.append(_commit_unread_note(nid, facts_of[nid]))

    head_sums, head_commit, head_arch = sums.get(head), commits.get(head), arches.get(head)
    answered = (set(sums) | set(commits)) - {head}
    if head_sums is None and head_commit is None:
        if answered:
            notes.append(
                f"SKIP group {group.name!r}: head {head} could not be read, so the "
                f"{len(answered)} member(s) that answered were not compared — the head is the "
                f"reference every worker is checked against."
            )
        return problems, notes
    if head_sums is None:
        notes.append(
            f"SKIP group {group.name!r}: head {head}'s binaries could not be read, so every "
            f"member was compared by llama.cpp commit alone."
        )
    elif head_commit is None:
        notes.append(
            f"SKIP group {group.name!r}: head {head}'s llama.cpp commit could not be read, so "
            f"members were compared by their binaries alone — which a member of another "
            f"architecture than the head cannot be."
        )

    # Counts include the head, which agrees with itself; an OK line needs a
    # second member, or it would claim a comparison that never happened.
    commits_matched = 1 if head_commit else 0
    sums_matched = 1 if head_sums else 0
    commit_failed = False
    for nid in group.members:
        if nid == head:
            continue
        facts = facts_of[nid]
        host = facts.get(HOST_KEY, "")
        commit, member_sums, arch = commits.get(nid), sums.get(nid), arches.get(nid)

        if commit and head_commit:
            if commit != head_commit:
                problems.append(_commit_problem(
                    group, head, head_commit, head_arch, nid, commit, arch, host
                ))
                commit_failed = True
                continue
            commits_matched += 1

        cross_arch = arch is not None and head_arch is not None and arch != head_arch
        if member_sums and head_sums and not cross_arch:
            if member_sums != head_sums:
                problems.append(_mismatch_problem(
                    group, head, head_sums, head_arch, head_commit,
                    nid, member_sums, arch, commit, host,
                ))
            else:
                sums_matched += 1
            continue

        if commit and head_commit:
            if cross_arch:
                notes.append(
                    f"OK   {nid}: {arch} member compared by commit: same commit "
                    f"{commit[:_SHORT]} as head {head} ({head_arch})"
                )
            # Otherwise a member (or head) whose binaries were unreadable, held
            # to the commit alone — its own SKIP line above already says so.
            continue

        if member_sums or commit:
            notes.append(_not_compared_note(
                group, nid, arch=arch, head_arch=head_arch,
                has_sums=bool(member_sums), has_commit=bool(commit),
                head_has_sums=bool(head_sums), head_has_commit=bool(head_commit),
                cross_arch=cross_arch,
            ))

    if head_commit and not commit_failed and commits_matched > 1:
        notes.append(
            f"OK   llama.cpp commit {head_commit[:_SHORT]} identical on every member "
            f"({commits_matched}/{len(group.members)} read)"
        )
    if not problems and sums_matched > 1:
        other_arch = any(
            a != head_arch for nid, a in arches.items() if nid != head and head_arch
        )
        scope = f"every {head_arch} member" if other_arch else "every member"
        notes.append(
            f"OK   llama-server/rpc-server checksums identical on {scope} "
            f"({sums_matched}/{len(group.members)} read)"
        )
    return problems, notes
