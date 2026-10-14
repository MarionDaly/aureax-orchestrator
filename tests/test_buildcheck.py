"""The group build check: are these machines running the same llama.cpp?

Every case here is one where the answer decides whether a start proceeds. The
gate this replaced compared pod image digests and refused EVERY start on the
real fleet — each node builds `aureax-client:dev` natively, so the digests
differ by construction while the binaries do not. So two properties are pinned
as hard as the mismatch itself: identical binaries pass, and a member that
could not be READ is a note rather than a refusal.

Two scopes since a CUDA x86_64 member may join an aarch64 group: the llama.cpp
commit must match the head's on every member (the RPC protocol is a property
of the commit, and architecture-neutral), and the binaries must match the
head's only within the head's architecture — the two images can never be
byte-identical.
"""

from __future__ import annotations

from orchestrator.cluster import buildcheck
from tests.conftest import make_network_config, make_targets

_SHA_LLAMA = "a" * 64
_SHA_RPC = "b" * 64
_SHA_OTHER_LLAMA = "c" * 64
_SHA_OTHER_RPC = "d" * 64
# The llama.cpp commit every fixture member was built from unless a test says
# otherwise, and one it was not.
_COMMIT = "0f1e2d3c"
_COMMIT_OTHER = "9a8b7c6d"


def _group(*workers: str, name: str = "g", head: str = "h"):
    cfg = make_network_config(
        make_targets(head, *workers),
        [{"name": name, "head": head, "workers": list(workers)}],
    )
    return cfg.node_groups[0], cfg


def _facts(llama: str = _SHA_LLAMA, rpc: str = _SHA_RPC, host: str = "10.77.0.1",
           commit: str | None = _COMMIT, arch: str = "aarch64") -> dict:
    """One member's probe facts. ``commit=None`` is an image whose build record
    could not be read — the probe's own fallback text, verbatim."""
    return {
        buildcheck.HOST_KEY: host,
        buildcheck.LLAMA_SHA_KEY: llama,
        buildcheck.RPC_SHA_KEY: rpc,
        buildcheck.BUILD_KEY: (
            f'{{"version":"b10375","commit":"{commit}"}}' if commit is not None
            else buildcheck.BUILD_UNAVAILABLE
        ),
        buildcheck.ARCH_KEY: arch,
        buildcheck.ERROR_KEY: "",
    }


# --- compare_group -----------------------------------------------------------

def test_identical_binaries_are_no_problem_and_say_so() -> None:
    group, _cfg = _group("w1", "w2")

    problems, notes = buildcheck.compare_group(
        group, {"h": _facts(), "w1": _facts(), "w2": _facts()}
    )

    assert problems == []
    assert any(n.startswith("OK   llama-server/rpc-server checksums identical") for n in notes), notes


def test_a_worker_with_a_different_rpc_server_is_a_problem() -> None:
    """The failure this whole module exists for: the head connects to a peer
    built from another commit and then HANGS at the handshake, with no error on
    either side and a pod stuck 0/1 Running for the whole probe budget."""
    group, _cfg = _group("w1", "w2")

    problems, _notes = buildcheck.compare_group(
        group,
        {
            "h": _facts(host="10.77.0.1"),
            "w1": _facts(host="10.77.0.2"),
            "w2": _facts(rpc=_SHA_OTHER_RPC, host="10.77.0.3"),
        },
    )

    assert len(problems) == 1, problems
    problem = problems[0]
    assert problem.startswith("FAIL w2:"), problem
    assert "w1" not in problem, "named a member that matches the head"
    assert "h" in problem and _SHA_RPC[:12] in problem and _SHA_OTHER_RPC[:12] in problem
    assert "./scripts/build-on-node.sh 10.77.0.3" in problem, "the remedy needs the odd node's address"
    assert "./scripts/update-all.sh" in problem


def test_a_worker_whose_llama_server_differs_is_caught_too() -> None:
    """Both binaries matter: a head runs llama-server against peers running
    rpc-server, so a group is coherent only when both halves match."""
    group, _cfg = _group("w1")

    problems, _notes = buildcheck.compare_group(
        group, {"h": _facts(), "w1": _facts(llama=_SHA_OTHER_LLAMA)}
    )

    assert len(problems) == 1 and problems[0].startswith("FAIL w1:")


def test_an_unreadable_member_is_a_note_not_a_problem() -> None:
    """Fail OPEN on a member we could not ask. Refusing a start over an SSH
    blip is a worse bug than the one this gate closes — and the digest gate
    that failed closed refused every start on a coherent fleet."""
    group, _cfg = _group("w1", "w2")

    problems, notes = buildcheck.compare_group(
        group,
        {
            "h": _facts(),
            "w1": _facts(),
            "w2": {buildcheck.HOST_KEY: "10.77.0.3", "error": "unreachable over SSH (SSHException: x)"},
        },
    )

    assert problems == []
    uncovered = [n for n in notes if n.startswith("SKIP w2:")]
    assert uncovered, notes
    assert "did not cover this node" in uncovered[0]
    assert "unreachable over SSH" in uncovered[0], "the reason has to survive into the note"
    assert any(n.startswith("OK ") for n in notes), "the members that did answer were still compared"


def test_a_sudo_failure_reads_as_unreadable_rather_than_as_a_mismatch() -> None:
    """The probe merges the remote's stderr into its output so the reason
    survives, which means the answer has to be recognised: a sudo refusal read
    as a checksum sends the operator rebuilding images to fix sudo."""
    group, _cfg = _group("w1")
    facts = _facts(llama="", rpc="")
    facts[buildcheck.ERROR_KEY] = "sudo: a password is required"

    problems, notes = buildcheck.compare_group(group, {"h": _facts(), "w1": facts})

    assert problems == []
    assert any("sudo: a password is required" in n for n in notes), notes


def test_an_unreadable_head_leaves_the_group_uncompared() -> None:
    """The head is the reference; without it there is nothing to compare
    against, and picking a worker instead would report the majority as odd."""
    group, _cfg = _group("w1", "w2")

    problems, notes = buildcheck.compare_group(
        group,
        {
            "h": {buildcheck.HOST_KEY: "10.77.0.1", "error": "unreachable over SSH"},
            "w1": _facts(),
            "w2": _facts(rpc=_SHA_OTHER_RPC),
        },
    )

    assert problems == [], "two workers disagreeing is not a verdict without the head"
    assert any("head h could not be read" in n for n in notes), notes
    assert not any(n.startswith("OK ") for n in notes)


def test_a_group_nobody_answered_reports_each_member_once() -> None:
    group, _cfg = _group("w1")

    problems, notes = buildcheck.compare_group(group, {})

    assert problems == []
    assert [n.split(":")[0] for n in notes] == ["SKIP h", "SKIP w1"], notes


def test_a_single_member_that_answered_is_not_called_consistent() -> None:
    """One checksum agrees with itself. Printing OK there would claim a check
    that never happened."""
    group, _cfg = _group("w1")

    problems, notes = buildcheck.compare_group(
        group, {"h": _facts(), "w1": {buildcheck.HOST_KEY: "10.77.0.2", "error": "unreachable"}}
    )

    assert problems == []
    assert not any(n.startswith("OK ") for n in notes), notes


# --- commit fleet-wide, binaries within one architecture --------------------
# A CUDA x86_64 member may join an aarch64 group: the RPC protocol is a property
# of the llama.cpp commit and its structs are packed little-endian, so it is
# architecture-neutral — but two architectures' binaries can never be
# byte-identical, so across them the commit is the whole comparison.

def test_a_same_commit_member_of_another_architecture_passes_on_the_commit() -> None:
    """The case this split exists for. Comparing bytes across architectures
    would refuse every mixed group, however carefully it was built."""
    group, _cfg = _group("w1", "gpu")

    problems, notes = buildcheck.compare_group(
        group,
        {
            "h": _facts(),
            "w1": _facts(),
            "gpu": _facts(llama=_SHA_OTHER_LLAMA, rpc=_SHA_OTHER_RPC, arch="x86_64"),
        },
    )

    assert problems == [], problems
    assert (
        f"OK   gpu: x86_64 member compared by commit: same commit {_COMMIT} as head h (aarch64)"
        in notes
    ), notes
    assert any(
        n.startswith("OK   llama-server/rpc-server checksums identical on every aarch64 member (2/3")
        for n in notes
    ), "the bytes were still compared among the head's architecture, and the line says whose"
    assert any(n.startswith(f"OK   llama.cpp commit {_COMMIT} identical on every member (3/3")
               for n in notes), notes


def test_a_member_of_another_architecture_on_another_commit_fails_naming_both() -> None:
    """Architecture moves a member onto the commit-only comparison; it never
    exempts it from the version lock."""
    group, _cfg = _group("w1", "gpu")

    problems, _notes = buildcheck.compare_group(
        group,
        {
            "h": _facts(host="10.77.0.1"),
            "w1": _facts(host="10.77.0.2"),
            "gpu": _facts(llama=_SHA_OTHER_LLAMA, rpc=_SHA_OTHER_RPC, host="10.77.0.3",
                          commit=_COMMIT_OTHER, arch="x86_64"),
        },
    )

    assert len(problems) == 1, problems
    problem = problems[0]
    assert problem.startswith(f"FAIL gpu: llama.cpp commit {_COMMIT_OTHER} does not match head h's "
                              f"{_COMMIT}"), problem
    assert "x86_64" in problem and "aarch64" in problem
    assert "HANGS at the handshake" in problem
    assert "./scripts/build-on-node.sh 10.77.0.3" in problem
    assert "./scripts/update-all.sh" in problem


def test_a_same_architecture_member_with_different_binaries_still_fails() -> None:
    """Within one architecture the bytes stay the check, commit or no commit: a
    matching commit with different binaries is a different toolchain, flags or
    tree, and the message says that rather than blaming the version."""
    group, _cfg = _group("w1")

    problems, _notes = buildcheck.compare_group(
        group, {"h": _facts(), "w1": _facts(rpc=_SHA_OTHER_RPC, host="10.77.0.2")}
    )

    assert len(problems) == 1, problems
    problem = problems[0]
    assert problem.startswith("FAIL w1: llama-server/rpc-server do not match head h"), problem
    assert _SHA_RPC[:12] in problem and _SHA_OTHER_RPC[:12] in problem
    assert f"Both report llama.cpp commit {_COMMIT}" in problem
    assert "./scripts/build-on-node.sh 10.77.0.2" in problem and "./scripts/update-all.sh" in problem


def test_a_member_with_no_build_record_but_matching_binaries_passes_with_a_note() -> None:
    """An unreadable build record is a note, exactly as an unreadable checksum
    is — and the binaries it did report were still compared."""
    group, _cfg = _group("w1")

    problems, notes = buildcheck.compare_group(
        group, {"h": _facts(), "w1": _facts(commit=None)}
    )

    assert problems == [], problems
    skip = [n for n in notes if n.startswith("SKIP w1:")]
    assert len(skip) == 1, notes
    assert "llama.cpp commit could not be read" in skip[0]
    assert buildcheck.BUILD_UNAVAILABLE in skip[0], "the probe's own words reach the note"
    assert any(n.startswith("OK   llama-server/rpc-server checksums identical on every member")
               for n in notes), notes


def test_a_different_commit_with_identical_binaries_fails_on_the_commit() -> None:
    """Impossible in practice, but it pins which check wins: the commit is the
    version lock, so a disagreement there is the member's one FAIL, and the
    matching bytes do not argue it away."""
    group, _cfg = _group("w1")

    problems, notes = buildcheck.compare_group(
        group, {"h": _facts(), "w1": _facts(commit=_COMMIT_OTHER)}
    )

    assert len(problems) == 1, problems
    assert problems[0].startswith("FAIL w1: llama.cpp commit"), problems[0]
    assert "do not match head" not in problems[0], "the binaries check must not be the one reported"
    assert not any(n.startswith("OK ") for n in notes), notes


def test_a_member_with_a_commit_but_no_checksums_is_held_to_the_commit() -> None:
    """Readable commit, unreadable binaries: compared by commit alone, so a
    commit mismatch still fails and a matching one is a note, not a refusal."""
    group, _cfg = _group("w1", "w2")
    unread = _facts(llama="", rpc="", commit=_COMMIT)
    unread[buildcheck.ERROR_KEY] = "sha256sum: /usr/local/bin/rpc-server: No such file"
    odd = _facts(llama="", rpc="", commit=_COMMIT_OTHER, host="10.77.0.3")

    problems, notes = buildcheck.compare_group(group, {"h": _facts(), "w1": unread, "w2": odd})

    assert len(problems) == 1 and problems[0].startswith("FAIL w2: llama.cpp commit"), problems
    w1_notes = [n for n in notes if n.startswith("SKIP w1:")]
    assert len(w1_notes) == 1, notes
    assert "did not cover this node's binaries" in w1_notes[0]
    assert "No such file" in w1_notes[0], "the reason has to survive into the note"


def test_an_unreported_architecture_never_relaxes_the_binary_check() -> None:
    """Only a KNOWN other architecture moves a member onto the commit-only
    comparison. A lost `uname -m` must not turn a same-architecture mismatch
    into a pass — the check this change must not weaken."""
    group, _cfg = _group("w1")

    problems, _notes = buildcheck.compare_group(
        group, {"h": _facts(), "w1": _facts(rpc=_SHA_OTHER_RPC, arch="")}
    )

    assert len(problems) == 1, problems
    assert problems[0].startswith("FAIL w1: llama-server/rpc-server do not match head h")
    assert "Image architecture unreported for w1" in problems[0]


def test_another_architecture_with_no_commit_is_uncompared_not_failed() -> None:
    """Across architectures the commit is the only comparable fact. Without it
    the member is UNKNOWN — said out loud, never a pass and never a refusal."""
    group, _cfg = _group("gpu")

    problems, notes = buildcheck.compare_group(
        group,
        {"h": _facts(), "gpu": _facts(llama=_SHA_OTHER_LLAMA, commit=None, arch="x86_64")},
    )

    assert problems == [], problems
    assert any(
        n.startswith("SKIP gpu: not compared — it is x86_64 and head h is aarch64")
        and "UNKNOWN" in n
        for n in notes
    ), notes
    assert not any(n.startswith("OK ") for n in notes), notes


def test_a_head_whose_binaries_were_unreadable_still_anchors_the_commit() -> None:
    """The head is the reference for both checks separately: losing its bytes
    leaves the commit comparison standing."""
    group, _cfg = _group("w1")
    head = _facts(llama="", rpc="")
    head[buildcheck.ERROR_KEY] = "sudo: a password is required"

    problems, notes = buildcheck.compare_group(
        group, {"h": head, "w1": _facts(commit=_COMMIT_OTHER, host="10.77.0.2")}
    )

    assert len(problems) == 1 and problems[0].startswith("FAIL w1: llama.cpp commit"), problems
    assert any("head h's binaries could not be read" in n for n in notes), notes


# --- checksums ---------------------------------------------------------------

def test_only_sixty_four_hex_digits_count_as_a_checksum() -> None:
    assert buildcheck.checksums(_facts()) == (_SHA_LLAMA, _SHA_RPC)
    assert buildcheck.checksums(_facts(llama="")) is None
    assert buildcheck.checksums(_facts(rpc="sudo: a password is required")) is None
    assert buildcheck.checksums(_facts(llama="a" * 63)) is None
    assert buildcheck.checksums(_facts(llama=("A" * 64))) is None, "sha256sum prints lower case"
    assert buildcheck.checksums({}) is None
    assert buildcheck.checksums(None) is None


def test_only_a_recorded_commit_counts_as_a_commit() -> None:
    """The fallback text and a record with no commit are unreadable, never a
    commit — a mismatch built from either would send the operator rebuilding
    an image to fix a probe."""
    assert buildcheck.llama_commit(_facts()) == _COMMIT
    assert buildcheck.llama_commit(_facts(commit=None)) is None
    assert buildcheck.llama_commit({buildcheck.BUILD_KEY: '{"version":"b10375"}'}) is None
    assert buildcheck.llama_commit({buildcheck.BUILD_KEY: '{"commit":"not-a-sha"}'}) is None
    assert buildcheck.llama_commit({buildcheck.BUILD_KEY: '["0f1e2d3c"]'}) is None
    assert buildcheck.llama_commit({buildcheck.BUILD_KEY: '{"commit":"0F1E2D3C"}'}) == "0f1e2d3c"
    # Extra fields are ignored: a record that also carries rpc_proto still yields its commit.
    assert buildcheck.llama_commit(
        {buildcheck.BUILD_KEY: '{"version":"b10375","commit":"0f1e2d3c","rpc_proto":"3.0.0"}'}
    ) == "0f1e2d3c"
    assert buildcheck.llama_commit({}) is None
    assert buildcheck.llama_commit(None) is None


def test_only_a_uname_shaped_value_counts_as_an_architecture() -> None:
    assert buildcheck.image_arch(_facts(arch="x86_64")) == "x86_64"
    assert buildcheck.image_arch(_facts(arch="aarch64")) == "aarch64"
    assert buildcheck.image_arch(_facts(arch="")) is None
    assert buildcheck.image_arch(_facts(arch="sh: uname: not found")) is None
    assert buildcheck.image_arch(None) is None


# --- probe_member / probe_group ---------------------------------------------

class _Target:
    """The two SSHTarget attributes the probe path touches."""

    def __init__(self, node_id: str, host: str) -> None:
        self.node_id = node_id
        self.host = host


def _fake_run_command(monkeypatch, answer):
    """Answer probe_member's one SSH round trip. ``answer`` is a
    ``(rc, stdout, stderr)`` tuple, an exception to raise, or a callable."""
    import orchestrator.deploy.ssh as ssh_mod

    seen: list[tuple[str, str]] = []

    def run_command(host, command, **_kwargs):
        seen.append((host, command))
        if isinstance(answer, Exception):
            raise answer
        return answer(host) if callable(answer) else answer

    monkeypatch.setattr(ssh_mod, "run_command", run_command)
    monkeypatch.setattr(ssh_mod, "conn_kwargs", lambda *_a, **_k: {})
    return seen


def _probe_stdout(llama: str = _SHA_LLAMA, rpc: str = _SHA_RPC, error: str = "",
                  arch: str = "aarch64") -> str:
    return (
        f"{buildcheck.LLAMA_SHA_KEY}={llama}\n"
        f"{buildcheck.RPC_SHA_KEY}={rpc}\n"
        f'{buildcheck.BUILD_KEY}={{"version":"b10375","commit":"0f1e2d3c"}}\n'
        f"{buildcheck.ARCH_KEY}={arch}\n"
        f"{buildcheck.ERROR_KEY}={error}\n"
    )


def test_probe_member_parses_the_two_checksums_and_the_build() -> None:
    facts = buildcheck.parse_probe(_probe_stdout())

    assert buildcheck.checksums(facts) == (_SHA_LLAMA, _SHA_RPC)
    assert facts[buildcheck.BUILD_KEY] == '{"version":"b10375","commit":"0f1e2d3c"}'


def test_probe_member_parses_the_image_architecture() -> None:
    facts = buildcheck.parse_probe(_probe_stdout(arch="x86_64"))

    assert buildcheck.image_arch(facts) == "x86_64"
    assert buildcheck.llama_commit(facts) == _COMMIT


def test_the_architecture_is_read_in_the_same_container_start() -> None:
    """One container start per member, still: `uname -m` runs INSIDE the image
    beside the two sha256sums and the build record, and comes back tagged so an
    error line can never read as an architecture. Not under the key `arch` —
    the doctor's probe already prints the HOST's architecture under it, and the
    spliced-in line would replace that fact."""
    probe = buildcheck.BUILD_PROBE

    assert probe.count("docker run") == 1
    container_script = probe.split(" -c '", 1)[1].split("'", 1)[0]
    assert "sha256sum /usr/local/bin/llama-server /usr/local/bin/rpc-server" in container_script
    assert f"cat {buildcheck.BUILD_JSON_PATH}" in container_script
    assert f'echo "{buildcheck.ARCH_KEY}=$(uname -m)"' in container_script
    assert buildcheck.ARCH_KEY != "arch"


def test_probe_member_reports_the_nodes_own_failure_text(monkeypatch) -> None:
    """`sudo -n` on a node whose sudoers wants a password fails immediately
    rather than hanging, and the message is the difference between "rebuild
    this node" and "fix sudo on this node"."""
    out = _probe_stdout(llama="", rpc="", error="sudo: a password is required")
    _fake_run_command(monkeypatch, (0, out, ""))

    facts = buildcheck.probe_member(_Target("w1", "10.77.0.2"))

    assert buildcheck.checksums(facts) is None
    assert "sudo: a password is required" in facts["error"]
    assert facts[buildcheck.HOST_KEY] == "10.77.0.2"


def test_probe_member_survives_an_unreachable_node(monkeypatch) -> None:
    """It must never raise: one dead member would otherwise abort the start it
    is only supposed to leave uncovered."""
    _fake_run_command(monkeypatch, OSError("no route to host"))

    facts = buildcheck.probe_member(_Target("w1", "10.77.0.2"))

    assert "unreachable over SSH" in facts["error"] and "no route to host" in facts["error"]
    assert buildcheck.checksums(facts) is None


def test_probe_member_names_a_timeout_as_a_timeout(monkeypatch) -> None:
    """run_command returns 124 with NO stdout, so "no facts" must not read as
    "no image" — a wedged docker daemon is the state the probe diagnoses."""
    _fake_run_command(monkeypatch, (124, "", "command timed out after 60s: _AX_OUT=$(timeout"))

    facts = buildcheck.probe_member(_Target("w1", "10.77.0.2"), timeout=60)

    assert facts["error"] == "the build probe timed out after 60s"


def test_probe_member_asks_for_both_binaries_inside_the_image(monkeypatch) -> None:
    """The command shape is the contract with scripts/test-all.sh §6: the
    checksum is taken INSIDE the container, of the binaries the RPC handshake
    is version-locked across."""
    seen = _fake_run_command(monkeypatch, (0, _probe_stdout(), ""))

    buildcheck.probe_member(_Target("w1", "10.77.0.2"))

    _host, command = seen[0]
    assert "sha256sum /usr/local/bin/llama-server /usr/local/bin/rpc-server" in command
    assert "aureax-client:dev" in command
    assert "sudo -n" in command, "no remote tty: a password prompt would hang the probe"
    assert "--pull=never" in command, "an absent image must fail, not reach for docker.io"


def test_probe_group_keys_every_member_and_never_raises(monkeypatch) -> None:
    _fake_run_command(
        monkeypatch,
        lambda host: (0, _probe_stdout(rpc=_SHA_OTHER_RPC if host == "10.77.0.3" else _SHA_RPC), ""),
    )
    targets = [_Target("h", "10.77.0.1"), _Target("w1", "10.77.0.2"), _Target("w2", "10.77.0.3")]

    facts = buildcheck.probe_group(targets, timeout=5)

    assert set(facts) == {"h", "w1", "w2"}
    group, _cfg = _group("w1", "w2")
    problems, _notes = buildcheck.compare_group(group, facts)
    assert len(problems) == 1 and problems[0].startswith("FAIL w2:")


def test_probe_group_with_no_targets_does_not_open_a_pool(monkeypatch) -> None:
    seen = _fake_run_command(monkeypatch, (0, _probe_stdout(), ""))

    assert buildcheck.probe_group([]) == {}
    assert seen == []
