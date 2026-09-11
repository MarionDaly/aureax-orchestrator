"""`scripts/lib/common.sh` — the shared shell helpers.

Seven operational scripts source this file, so a regression in one of these
functions is a regression in all of them at once.

The tests drive the real bash functions rather than asserting on the text of
the file, because everything worth checking here IS bash behaviour: whether an
exit status propagates out of a command substitution, which argv words ssh
actually receives, and which stream a diagnostic lands on. `ssh`, `scp` and
`uv` are shadowed with recording shell functions — a function beats a PATH
lookup, and it is inherited by the subshells these helpers use.

One test is deliberately the exception:
`test_setup_all_no_longer_prompts_for_the_mac_ip` greps scripts/setup-all.sh.
The ABSENCE of an interactive prompt cannot be observed by running the script —
a `read -p` with no terminal hangs the run forever, which is the failure itself
— so the source text is the only thing left to assert on.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import PROJECT_ROOT, write_network_yaml
from tests.test_no_plaintext_urls import PLAINTEXT_ALLOWED

COMMON_SH = PROJECT_ROOT / "scripts" / "lib" / "common.sh"

# `uv run python <args>` -> this interpreter, so the tests neither need uv on
# PATH nor pay for its startup. `shift 2` drops the "run python" words.
UV_SHIM = f'uv() {{ shift 2; "{sys.executable}" "$@"; }}'

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(script: str) -> subprocess.CompletedProcess:
    """Source common.sh under `set -uo pipefail`, then run `script`.

    `set -u` is part of the contract: every consumer runs under `set -euo
    pipefail`, and helpers that read the NODE_* arrays must tolerate their
    being unset (bench-decode.sh takes a host on argv and loads no config).
    `-e` is left off so a probe can observe a non-zero return instead of dying
    on it.
    """
    env = dict(os.environ)
    env.pop("MAC_IP", None)  # a real one in the environment would mask failures
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{COMMON_SH}"\n{textwrap.dedent(script)}'],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )


# --- the log prefixes are a protocol, not formatting ------------------------

def test_log_prefixes_map_onto_the_dashboard_classifier() -> None:
    """Both halves of one contract, pinned together.

    common.sh's helpers emit `[OK]`/`[WARN]`/`[FAIL]`/`[....]`, and
    lifecycle._classify_setup_line turns those back into dashboard log levels
    while it streams setup-all.sh. Drop a bracket in either file and every
    script still works while the whole setup log is silently reclassified.
    """
    from orchestrator.lifecycle import _classify_setup_line

    proc = _run("""
        ok "built"
        warn "slow"
        info "working"
        fail "SSH to node" "unreachable"
    """)
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 4, proc.stderr
    assert [_classify_setup_line(line) for line in lines] == [
        "success", "warn", "info", "error",
    ]


def test_section_banner_takes_its_rule_width_from_the_caller() -> None:
    """setup-all.sh and test-all.sh print different-width rules."""
    proc = _run("""
        section_banner "Step 0: Checking Mac prerequisites"
        section_banner "Narrow" 8
    """)
    lines = proc.stdout.splitlines()
    assert "=" * 64 in lines
    assert "=" * 8 in lines
    assert "  Step 0: Checking Mac prerequisites" in lines


# --- ssh_cmd / scp_cmd: the configured port ---------------------------------

def test_ssh_cmd_passes_the_targets_configured_port() -> None:
    """A target on a non-22 port was unreachable from every bash step."""
    proc = _run("""
        ssh() { echo "ARGV: $*"; }
        NODE_COUNT=2
        NODE_HOSTS=(alpha beta)
        NODE_PORTS=(22 2222)
        ssh_cmd beta root /tmp/key true
        ssh_cmd alpha root /tmp/key true
    """)
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 2, proc.stderr
    assert "-p 2222" in lines[0] and "root@beta" in lines[0]
    assert "-p 22" in lines[1] and "root@alpha" in lines[1]


def test_ssh_cmd_defaults_to_22_without_any_node_arrays() -> None:
    """bench-decode.sh sources the lib and loads no config at all."""
    proc = _run("""
        ssh() { echo "ARGV: $*"; }
        ssh_cmd somehost root /tmp/key true
    """)
    assert proc.returncode == 0, proc.stderr
    assert "-p 22" in proc.stdout


def test_rsync_rsh_carries_the_targets_configured_port() -> None:
    """The transport that actually ships the project tree had no coverage."""
    proc = _run("""
        NODE_COUNT=2
        NODE_HOSTS=(alpha beta)
        NODE_PORTS=(22 2222)
        rsync_rsh beta /tmp/key
        rsync_rsh gamma /tmp/key
    """)
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 2, proc.stderr
    assert "-p 2222" in lines[0] and "-i /tmp/key" in lines[0]
    assert "-p 22" in lines[1]


def test_every_ssh_helper_runs_in_batch_mode() -> None:
    """Key auth can fail while a target still has a password in the config.

    ssh then falls back to an interactive prompt on /dev/tty — invisible inside
    a backgrounded build whose output is redirected to a log file, and `wait`
    never returns. ssh_cmd also has to survive a node that freezes AFTER the
    handshake, which ConnectTimeout does not cover.
    """
    proc = _run("""
        ssh() { echo "SSH: $*"; }
        scp() { echo "SCP: $*"; }
        ssh_cmd host user /tmp/key true
        scp_cmd /tmp/key ./local.txt user@host:/tmp/x
        rsync_rsh host /tmp/key
    """)
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 3, proc.stderr
    assert all("BatchMode=yes" in line for line in lines)
    assert "ServerAliveInterval=15" in lines[0] and "ServerAliveCountMax=4" in lines[0]


def test_scp_cmd_reads_the_port_off_whichever_operand_is_remote() -> None:
    """scp spells the port -P and hides the host inside `user@host:path`."""
    proc = _run("""
        scp() { echo "ARGV: $*"; }
        NODE_COUNT=1
        NODE_HOSTS=(beta)
        NODE_PORTS=(2222)
        scp_cmd /tmp/key ./local.txt root@beta:/tmp/x
        scp_cmd /tmp/key root@beta:/tmp/x ./local.txt
    """)
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 2, proc.stderr
    assert all("-P 2222" in line for line in lines)
    # The key stays one argv word, so a path containing a space survives.
    assert lines[0].endswith("-i /tmp/key ./local.txt root@beta:/tmp/x")


# --- load_network_env: the child's exit status ------------------------------

def test_load_network_env_assigns_globals_on_success(tmp_path) -> None:
    """The eval must not run in a subshell, or nothing it assigns survives.

    Against a config this test owns. Called with no path, common.sh falls back
    to the developer's real configs/aureax-network.yaml — so the assertion had
    to be "count > 0", which pins no value, and emptying or disabling the fleet
    (a state the suite treats as valid) failed it with no code defect.
    """
    net = write_network_yaml(
        tmp_path,
        [
            {"node_id": "alpha", "host": "10.77.0.30"},
            {"node_id": "bravo", "host": "10.77.0.31"},
        ],
    )
    proc = _run(f"""
        {UV_SHIM}
        load_network_env all "{net}"
        echo "count=$NODE_COUNT hosts=${{NODE_HOSTS[*]}} ports=${{#NODE_PORTS[@]}}"
    """)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    assert "count=2" in out
    assert "hosts=10.77.0.30 10.77.0.31" in out
    assert "ports=2" in out


def test_load_network_env_returns_nonzero_on_a_broken_config(tmp_path) -> None:
    """`eval "$(...)"` could not see this: eval reports its own status."""
    bad = tmp_path / "network.yaml"
    bad.write_text("ssh: {targets: [\n")  # truncated flow sequence
    proc = _run(f"""
        {UV_SHIM}
        if load_network_env all "{bad}"; then echo "REPORTED-OK"; else echo "rc=$?"; fi
    """)
    assert "REPORTED-OK" not in proc.stdout
    assert proc.stdout.strip().startswith("rc=")
    assert proc.stdout.strip() != "rc=0"
    # The reason reaches stderr rather than being swallowed into the eval'd text.
    assert proc.stderr.strip()


# --- detect_mac_ip: loud failure, never loopback, never a prompt -------------

@pytest.mark.parametrize("answer", ["", "127.0.0.1", "0.0.0.0", "169.254.9.9"])
def test_detect_mac_ip_fails_rather_than_returning_an_unusable_address(answer) -> None:
    """This value is baked into every K3S_URL and every per-node config.

    update-all.sh used to degrade to 127.0.0.1 with stderr discarded, which
    deployed a fleet whose every node reported telemetry to its own loopback.
    """
    proc = _run(f"""
        uv() {{ echo "{answer}"; }}
        detect_mac_ip
    """)
    assert proc.returncode != 0
    assert proc.stdout.strip() == ""  # stdout is the answer — it must stay empty
    assert "[FAIL]" in proc.stderr


def test_detect_mac_ip_honours_and_validates_the_mac_ip_override() -> None:
    """lifecycle passes the address it already resolved via $MAC_IP."""
    good = _run("""
        uv() { echo "SHOULD NOT BE CALLED"; }
        MAC_IP=192.0.2.1 detect_mac_ip
    """)
    assert good.returncode == 0, good.stderr
    assert good.stdout.strip() == "192.0.2.1"

    bad = _run("""
        uv() { echo "SHOULD NOT BE CALLED"; }
        MAC_IP=127.0.0.1 detect_mac_ip
    """)
    assert bad.returncode != 0
    assert bad.stdout.strip() == ""


# --- k3s_token_from_cluster: one container name, one path -------------------

def test_k3s_token_from_cluster_names_the_server_container_and_token_path() -> None:
    """This was spelled out twice in the shell layer, in two styles."""
    proc = _run("""
        docker() { echo "DOCKER: $*"; }
        k3s_token_from_cluster aureax
    """)
    assert proc.returncode == 0, proc.stderr
    assert "k3d-aureax-server-0" in proc.stdout
    assert "/var/lib/rancher/k3s/server/node-token" in proc.stdout


def test_k3s_token_from_cluster_fails_when_the_container_is_not_running() -> None:
    """The caller turns an empty token into a named failure, not a silent join."""
    proc = _run("""
        docker() { return 1; }
        k3s_token_from_cluster aureax || echo "rc=$?"
    """)
    assert proc.stdout.strip() == "rc=1"


def test_setup_all_no_longer_prompts_for_the_mac_ip() -> None:
    """The dashboard runs setup-all.sh non-interactively.

    A `read -p` there has no terminal to read from and hangs the run forever,
    so the detector must fail loudly instead of asking.
    """
    text = (PROJECT_ROOT / "scripts" / "setup-all.sh").read_text()
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "read -p" in ln]


def test_the_sharded_inference_check_passes_the_head_node_id_not_its_address() -> None:
    """test-all.sh section 6 asks the harness for a node id, so a key is sent.

    The MCP port takes a bearer key. `aureax-harness` reads
    ~/.aureax/keys/<node_id>.mcp.key only when its positional resolves to an
    enabled node id in the fleet config; given a bare address it sends no
    Authorization header at all, and every keyed head answers 401, while the
    node id resolves the key and the same call succeeds. The check was passing
    $GHEAD, which is the address
    out of GROUP_HEAD_HOSTS, so section 6's one end-to-end generation failed on
    a perfectly healthy group. Asserted on the source because reproducing it
    needs a running fleet, which is exactly what this suite may not have.
    """
    text = (PROJECT_ROOT / "scripts" / "test-all.sh").read_text()
    # The invocation is line-continued, so join before matching.
    joined = text.replace("\\\n", " ")
    calls = [ln.strip() for ln in joined.splitlines()
             if "aureax-harness infer" in ln and not ln.lstrip().startswith("#")]
    assert calls, "no `aureax-harness infer` invocation found in test-all.sh"
    for call in calls:
        assert '"$GHEAD_ID"' in call, f"harness given an address, not a node id: {call}"
        assert '"$GHEAD"' not in call, f"harness given an address, not a node id: {call}"
        # The suite falls back to configs/aureax-network.example.yaml on a fresh
        # clone, and the harness resolves ids against the DEFAULT fleet file
        # unless told otherwise — so the id would resolve to nothing there.
        assert '--config "$NETWORK_CONFIG"' in call, f"fleet file not passed: {call}"


def test_no_shell_script_evals_what_a_node_answered() -> None:
    """ssh_facts output is node-controlled text, so it is parsed, never eval'd.

    The %q quoting ssh_facts asks for is performed BY THE NODE, so it
    constrains nothing a compromised node prints — and a shard worker's
    unauthenticated rpc-server is a documented way for a node to be
    compromised. An `eval` of that text runs it on the orchestrator with the
    operator's privileges, so the absence of one is the control. Asserted on
    the source because the failure only shows up against a hostile node.

    setup-all.sh's Step 0 preflight is the site this pins; the same contract is
    stated at common.sh::ssh_facts for every other caller.
    """
    text = (PROJECT_ROOT / "scripts" / "setup-all.sh").read_text()
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "eval" in ln and "FACTS" in ln]


# --- one round trip, one Dockerfile mapping, one stat idiom -----------------

def test_ssh_facts_asks_every_question_in_one_round_trip() -> None:
    proc = _run("""
        ssh() { echo "--- ssh call ---"; echo "$@"; }
        ssh_facts host user /tmp/key "ARCH=uname -m" "NV=echo no"
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("--- ssh call ---") == 1
    assert "'ARCH'" in proc.stdout and "'NV'" in proc.stdout


@pytest.mark.parametrize(("arch", "nvidia", "expected"), [
    ("aarch64", "no", "docker/Dockerfile.client"),
    ("arm64", "no", "docker/Dockerfile.client"),
    ("x86_64", "yes", "docker/Dockerfile.client.cuda"),
])
def test_pick_dockerfile_maps_supported_machines(arch, nvidia, expected) -> None:
    proc = _run(f'pick_dockerfile {arch} {nvidia} somehost')
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected


@pytest.mark.parametrize("arch", ["x86_64", "riscv64"])
def test_pick_dockerfile_refuses_a_machine_with_no_image(arch) -> None:
    """Falling through to the ARM Dockerfile builds an image that hangs at the
    RPC handshake instead of failing visibly."""
    proc = _run(f'pick_dockerfile {arch} no somehost || echo "rc=$?"')
    assert proc.stdout.strip() == "rc=1"
    assert "[FAIL]" in proc.stderr


def test_file_size_reports_zero_for_a_missing_file(tmp_path) -> None:
    present = tmp_path / "f"
    present.write_bytes(b"0123456789")
    proc = _run(f"""
        file_size "{present}"
        file_size "{tmp_path / 'absent'}"
    """)
    assert proc.stdout.split() == ["10", "0"]


def test_rsync_usable_declines_a_key_path_containing_a_space() -> None:
    """rsync splits its own `-e` string on whitespace, with no quoting."""
    proc = _run("""
        rsync() { :; }
        rsync_usable /tmp/key && echo plain-ok
        rsync_usable "/tmp/my key" || echo spaced-declined
    """)
    assert proc.stdout.split() == ["plain-ok", "spaced-declined"]


def test_ssh_facts_returns_every_key_and_quotes_the_values() -> None:
    """The remote text is EXECUTED here, not echoed.

    The docstring's promise is that a caller gets back a line per key it asked
    for, empty results included — skip those and the caller's variable is
    unset, which is an abort under the `set -u` every consumer runs. The
    quoting is the other half: both callers eval this output on the Mac, so a
    bare `%s` would run a node's answer as a command.
    """
    proc = _run("""
        ssh() { bash -c "${!#}"; }
        FACTS="$(ssh_facts host user /tmp/key "ARCH=uname -m" "EMPTY=true" "SPACED=echo one two")"
        echo "keys=$(echo "$FACTS" | grep -c .)"
        ARCH= EMPTY= SPACED=
        eval "$FACTS"
        echo "arch=[$ARCH] empty=[$EMPTY] spaced=[$SPACED]"
    """)
    assert proc.returncode == 0, proc.stderr
    assert "keys=3" in proc.stdout
    assert "empty=[]" in proc.stdout
    assert "spaced=[one two]" in proc.stdout
    assert "arch=[]" not in proc.stdout


def test_detect_node_arch_fills_both_names_from_one_round_trip() -> None:
    """One probe, two facts, and no ARCH/HAS_NVIDIA left in the caller's scope."""
    proc = _run("""
        ssh() { echo "--- ssh call ---" >&2; bash -c "${!#}"; }
        detect_node_arch host user /tmp/key
        echo "arch=[$NODE_ARCH] nvidia=[$NODE_HAS_NVIDIA]"
        echo "leaked=[${ARCH:-<none>}]"
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.count("--- ssh call ---") == 1
    assert "arch=[]" not in proc.stdout
    assert "nvidia=[no]" in proc.stdout or "nvidia=[yes]" in proc.stdout
    assert "leaked=[<none>]" in proc.stdout


def test_detect_node_arch_returns_nonzero_for_an_unreachable_node() -> None:
    """Otherwise both names come back empty and pick_dockerfile blames the arch."""
    proc = _run("""
        ssh() { return 255; }
        detect_node_arch host user /tmp/key || echo "rc=$?"
    """)
    assert proc.stdout.strip() == "rc=1"
    assert "[FAIL]" in proc.stderr


# --- sync_project_tree: what reaches a node, and what is left behind --------

def test_sync_project_tree_rsync_branch_excludes_configs_and_models() -> None:
    """configs/aureax-network.yaml carries every node's SSH password.

    --delete-excluded as well as --delete, so the two branches converge: plain
    --delete leaves a file MATCHING an --exclude untouched on the receiver, so
    a directory that once held configs/ would keep that copy through every
    later sync and still hand it to the `docker build` context.
    """
    proc = _run("""
        rsync() { echo "RSYNC: $*"; }
        sync_project_tree beta root /tmp/key /tmp/remote
    """)
    assert proc.returncode == 0, proc.stderr
    assert "--exclude configs/" in proc.stdout
    assert "--exclude models/" in proc.stdout
    assert "--delete" in proc.stdout
    assert "--delete-excluded" in proc.stdout
    assert "root@beta:/tmp/remote/" in proc.stdout


def test_sync_project_tree_tar_fallback_excludes_configs_and_leaves_no_archive() -> None:
    """A whitespace key path forces the fallback rsync cannot express.

    The archive used to be a fixed /tmp name removed only on the success path,
    so a failed build left the whole project tree — configs/ included — on the
    Mac.
    """
    proc = _run("""
        tar() { echo "TAR: $*"; TARBALL="$3"; }
        scp() { echo "SCP: $*"; }
        ssh() { echo "SSH: $*"; }
        sync_project_tree beta root "/tmp/my key" /tmp/remote
        [ -e "$TARBALL" ] && echo "LEAKED" || echo "CLEANED"
    """)
    assert proc.returncode == 0, proc.stderr
    assert "--exclude=configs" in proc.stdout
    assert "--exclude=models" in proc.stdout
    assert "root@beta:/tmp/remote.tar.gz" in proc.stdout
    # Extract-into-.new-and-swap, so both branches converge on the same tree.
    assert "/tmp/remote.new" in proc.stdout
    assert "CLEANED" in proc.stdout


def test_sync_project_tree_reuses_and_keeps_a_caller_supplied_archive(tmp_path) -> None:
    """One archive for a whole fleet: the caller builds it once and owns it."""
    archive = tmp_path / "fleet.tar.gz"
    archive.write_text("pretend archive")
    proc = _run(f"""
        tar() {{ echo "TAR-REBUILT"; }}
        scp() {{ echo "SCP: $*"; }}
        ssh() {{ echo "SSH: $*"; }}
        sync_project_tree beta root "/tmp/my key" /tmp/remote "{archive}"
    """)
    assert proc.returncode == 0, proc.stderr
    assert "TAR-REBUILT" not in proc.stdout
    assert archive.exists()


# --- curl_tls: the fleet CA, or nothing -------------------------------------
#
# TLS is mandatory fleet-wide, so these pin the two halves of that: no CA means
# no request at all (not a plaintext one, not an unverified one), and a CA
# means --cacert reaches curl ahead of whatever the caller asked for.


def test_curl_tls_refuses_to_run_without_the_fleet_ca(tmp_path) -> None:
    """No CA on disk -> curl is never invoked and the status is 77.

    The failure this prevents is the quiet one: an operator who has not run
    `aureax generate` gets told so, instead of a check that fell back to
    plaintext (or to -k) and reported a green tick for a connection nothing
    authenticated. 77 is curl's own "problem with the CA cert" status, so a
    caller that already classifies curl statuses classifies this correctly.
    """
    missing = tmp_path / "no-such-ca.pem"
    proc = _run(f"""
        curl() {{ echo "CURL-RAN: $*"; }}
        AUREAX_CA="{missing}"
        curl_tls -sf https://example.invalid/health || echo "rc=$?"
    """)
    assert "CURL-RAN" not in proc.stdout, proc.stdout
    assert "rc=77" in proc.stdout, proc.stdout
    assert str(missing) in proc.stderr
    assert "aureax generate" in proc.stderr


def test_curl_tls_passes_the_cacert_and_keeps_the_callers_flags(tmp_path) -> None:
    """--cacert <path> first, then the caller's words in their original order.

    Order matters to curl for `-K` and for the URL position used by
    test-all.sh's http_status helper, so this asserts the whole argv rather
    than merely that --cacert appears somewhere in it.
    """
    ca = tmp_path / "ca.pem"
    ca.write_text("not a real certificate")
    proc = _run(f"""
        curl() {{ printf '%s\n' "$@"; }}
        AUREAX_CA="{ca}"
        curl_tls -s -o /dev/null -w "%{{http_code}}" "https://node:30900/mcp"
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "--cacert",
        str(ca),
        "-s",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "https://node:30900/mcp",
    ]


def test_curl_tls_defaults_to_the_generated_store(tmp_path) -> None:
    """With AUREAX_CA unset the default is $HOME/.aureax/tls/ca.pem.

    That is where tlsstore.ensure_ca writes it, so the scripts find the CA a
    plain `aureax generate` produced without anyone exporting anything.
    """
    home = tmp_path / "home"
    (home / ".aureax" / "tls").mkdir(parents=True)
    (home / ".aureax" / "tls" / "ca.pem").write_text("not a real certificate")
    env = dict(os.environ)
    env["HOME"] = str(home)
    # Exported but EMPTY, not absent: `${AUREAX_CA:-...}` treats empty as unset,
    # and the `:-` spelling (rather than `:=` or `-`) is what makes that true.
    env["AUREAX_CA"] = ""
    script = (
        f'set -uo pipefail\nsource "{COMMON_SH}"\n'
        "curl() { printf '%s\\n' \"$@\"; }\n"
        "curl_tls https://node/health\n"
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == [
        "--cacert",
        str(home / ".aureax" / "tls" / "ca.pem"),
        "https://node/health",
    ]


def test_mcp_health_wait_dials_https_through_curl_tls(tmp_path) -> None:
    """The health poll is HTTPS and verified, not plaintext.

    mcp_health_wait swallows curl's stdout, so the recording shim writes to a
    file instead — the argv is the assertion.
    """
    ca = tmp_path / "ca.pem"
    ca.write_text("not a real certificate")
    log = tmp_path / "argv.txt"
    proc = _run(f"""
        curl() {{ printf '%s\n' "$@" > "{log}"; return 0; }}
        AUREAX_CA="{ca}"
        mcp_health_wait node-a 30900 0 && echo "rc=0"
    """)
    assert "rc=0" in proc.stdout, proc.stderr
    argv = log.read_text().splitlines()
    assert argv[:2] == ["--cacert", str(ca)]
    assert "https://node-a:30900/health" in argv
    assert not [a for a in argv if a.startswith("http://")]


def test_mcp_health_wait_gives_up_when_there_is_no_fleet_ca(tmp_path) -> None:
    """A missing CA is a failed health check, never a plaintext retry."""
    proc = _run(f"""
        curl() {{ echo "CURL-RAN"; return 0; }}
        AUREAX_CA="{tmp_path / 'absent.pem'}"
        mcp_health_wait node-a 30900 0 || echo "rc=$?"
    """)
    assert "CURL-RAN" not in proc.stdout
    assert "rc=1" in proc.stdout, proc.stdout


# --- bench-decode.sh: the bearer key never reaches argv ---------------------

BENCH_DECODE = PROJECT_ROOT / "scripts" / "bench-decode.sh"

# A recording stand-in for curl, installed on PATH as a real executable rather
# than shadowed with a shell function: curl_tls invokes `curl`, and what this
# test asserts is precisely what a SEPARATE process's argv looks like from the
# outside. It logs one line per argv word, and the contents of any `-K` file —
# which is a /dev/fd pipe it inherits, exactly as curl would.
_CURL_STUB = r"""#!/usr/bin/env bash
for a in "$@"; do
    printf 'ARGV %s\n' "$a" >> "$STUB_LOG"
done
prev=""
for a in "$@"; do
    if [ "$prev" = "-K" ]; then printf 'CONFIG %s\n' "$(cat "$a")" >> "$STUB_LOG"; fi
    prev="$a"
done
case "$*" in
    *"/health"*)
        exit 0 ;;
    *"/props"*)
        printf '{"model_path":"/opt/aureax/models/test.gguf"}'
        exit 0 ;;
    *"/completion"*)
        printf '{"timings":{"prompt_n":9,"prompt_ms":100.0,"prompt_per_second":90.0,'
        printf '"predicted_n":4,"predicted_ms":200.0,"predicted_per_second":20.0}}\n200'
        exit 0 ;;
esac
exit 1
"""


def _run_bench_decode(tmp_path, api_key: str | None):
    """One full bench-decode.sh run against the stub. Returns (proc, log lines)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "curl"
    stub.write_text(_CURL_STUB)
    stub.chmod(0o755)
    ca = tmp_path / "ca.pem"
    ca.write_text("not a real certificate")
    log = tmp_path / "curl-argv.txt"

    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    env["AUREAX_CA"] = str(ca)
    env["STUB_LOG"] = str(log)
    env.pop("AUREAX_API_KEY", None)
    if api_key is not None:
        env["AUREAX_API_KEY"] = api_key

    proc = subprocess.run(
        ["bash", str(BENCH_DECODE), "node:30800", "4", "1"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env,
    )
    lines = log.read_text().splitlines() if log.exists() else []
    return proc, lines


def test_bench_decode_never_puts_the_bearer_key_in_argv(tmp_path) -> None:
    """The key reaches curl through `-K` on a pipe, never as a `-H` argv word.

    argv is world-readable for the life of the request, and /completion runs
    under `-m 1800`, so a `-H "Authorization: Bearer ..."` there hands the key
    to every local process for half an hour per run. The same rule the
    supervisor follows for LLAMA_API_KEY, and the same `-K` form test-all.sh's
    http_status uses.
    """
    key = "aureax_sk_benchdecodetestkey"
    proc, lines = _run_bench_decode(tmp_path, key)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    argv = [ln[len("ARGV "):] for ln in lines if ln.startswith("ARGV ")]
    assert argv, "the stub recorded nothing — bench-decode.sh never called curl"
    assert not [a for a in argv if key in a], f"key in argv: {argv}"
    assert not [a for a in argv if a.startswith("Authorization:")]

    # Both authenticated calls carried the header, and each got it by -K.
    configs = [ln[len("CONFIG "):] for ln in lines if ln.startswith("CONFIG ")]
    assert configs == [f'header = "Authorization: Bearer {key}"'] * 2, configs
    assert argv.count("-K") == 2


def test_bench_decode_sends_no_auth_option_without_a_key(tmp_path) -> None:
    """An unset AUREAX_API_KEY omits `-K` entirely, rather than passing an empty one.

    Most targets are unauthenticated, so this is the common path: an empty
    config file would be a second thing to get right, and `curl -K ""` is an
    error.
    """
    proc, lines = _run_bench_decode(tmp_path, None)

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not [ln for ln in lines if ln.startswith("CONFIG ")]
    assert "ARGV -K" not in lines


# --- no plaintext, and no way to opt out of verification --------------------


def _code_lines(path):
    """Every line of a shell script that is not a whole-line comment.

    Comment lines are excluded because several of them legitimately quote a URL
    or an option while explaining why it is not used. Nothing here tries to
    strip trailing comments: a `curl` on a line with one is still a call.
    """
    return [
        (n, line)
        for n, line in enumerate(path.read_text().splitlines(), start=1)
        if not line.lstrip().startswith("#")
    ]


SHELL_SCRIPTS = sorted(
    [*(PROJECT_ROOT / "scripts").glob("*.sh"), *(PROJECT_ROOT / "scripts" / "lib").glob("*.sh")]
)


def _without_allowed_urls(path, line: str) -> str:
    """One line with this file's allowlisted plaintext URLs removed.

    The allowlist itself is ``tests/test_no_plaintext_urls.PLAINTEXT_ALLOWED``,
    imported rather than restated: two lists of exempted URLs drift, and the
    one that drifts wider is the one nobody reads. Only the exact string is
    removed, so any OTHER http:// on the same line still fails below.
    """
    for url in PLAINTEXT_ALLOWED.get(path.relative_to(PROJECT_ROOT).as_posix(), set()):
        line = line.replace(url, "")
    return line


def test_the_shell_layer_dials_no_plaintext_url() -> None:
    """No `http://` in any script's code.

    TLS is mandatory fleet-wide: llama-server, the MCP server and the dashboard
    all serve HTTPS with fleet-CA certificates, so a surviving http:// URL is a
    check that can only ever fail — or, worse, one that succeeds against
    something else listening on that port.

    The one exemption is the dashboard's loopback listener, which test-all.sh
    section 3 dials: `aureax dashboard` serves plain HTTP on 127.0.0.1/::1
    beside the fleet's HTTPS listener, so that URL is not on the LAN at all.
    """
    assert SHELL_SCRIPTS, "no shell scripts found — the glob is wrong"
    hits = [
        f"{path.relative_to(PROJECT_ROOT)}:{n}: {line.strip()}"
        for path in SHELL_SCRIPTS
        for n, line in _code_lines(path)
        if "http://" in _without_allowed_urls(path, line)
    ]
    assert not hits, "plaintext URL in the shell layer:\n" + "\n".join(hits)


def test_no_script_turns_certificate_verification_off() -> None:
    """`curl -k` / `--insecure` would make every TLS check vacuous.

    A check that accepts any certificate stops distinguishing "the fleet
    answered" from "something answered on that port", which is most of what
    this work bought.
    """
    hits = [
        f"{path.relative_to(PROJECT_ROOT)}:{n}: {line.strip()}"
        for path in SHELL_SCRIPTS
        for n, line in _code_lines(path)
        if "--insecure" in line or re.search(r"(?<![\w-])curl(?:_tls)?\s+(?:-\w*k\b|-k\b)", line)
    ]
    assert not hits, "certificate verification disabled:\n" + "\n".join(hits)


def test_rsync_transport_uses_the_same_trust_store_and_re_trust_policy():
    """A changed key must not refuse (nodes are re-imaged as routine), and the
    store must be the one the Python side writes, so both halves agree."""
    result = _run('rsync_rsh example /tmp/key')
    assert 'StrictHostKeyChecking=no' in result.stdout
    assert f'UserKnownHostsFile={os.path.expanduser("~")}/.aureax/known_hosts' in result.stdout
    assert 'ConnectTimeout=10' in result.stdout


# --- the build context is the node account's own directory ------------------
#
# `sudo docker build` runs against whatever the sync left behind, so the
# directory it reads decides what root's daemon bakes into the image. Under a
# world-writable /tmp another local account picks that; inside the node
# account's own 0700 directory it cannot.


def test_remote_build_dir_stages_inside_the_node_accounts_private_directory() -> None:
    """A relative path under a 0700 ~/.aureax-stage, never a /tmp name."""
    # The stub records on stderr: the helper sends the remote command's own
    # stdout to /dev/null, because only the path it prints may reach a caller's
    # `$( )`.
    proc = _run("""
        ssh() { echo "SSH: $*" >&2; }
        remote_build_dir beta root /tmp/key
    """)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ".aureax-stage/build", proc.stdout
    assert "mkdir -p ~/.aureax-stage" in proc.stderr
    assert "chmod 700 ~/.aureax-stage" in proc.stderr


def test_remote_build_dir_fails_when_the_staging_directory_is_not_ours() -> None:
    """The chmod is the ownership check: a directory someone else owns refuses it.

    Failing closed matters more here than anywhere else in the sync — the
    caller's next step hands this directory to a root `docker build`.
    """
    proc = _run("""
        ssh() { return 1; }
        remote_build_dir beta root /tmp/key || echo "rc=$?"
    """)
    assert proc.returncode == 0, proc.stderr
    assert "rc=1" in proc.stdout
    assert ".aureax-stage/build" not in proc.stdout


def test_remote_build_dir_reclaims_the_path_it_replaces() -> None:
    """/tmp/aureax-build is not touched by anything else once this exists.

    A tree left there keeps its copy of whatever the sync excludes today —
    configs/aureax-network.yaml is the one that matters — so it goes.
    """
    proc = _run("""
        ssh() { echo "SSH: $*" >&2; }
        remote_build_dir beta root /tmp/key
    """)
    assert "rm -rf /tmp/aureax-build /tmp/aureax-build.new /tmp/aureax-build.tar.gz" in proc.stderr


def test_the_build_scripts_never_name_a_fixed_tmp_build_context() -> None:
    """Source text, because the absence of a path cannot be observed by running.

    Both build scripts go through remote_build_dir; a literal /tmp path back in
    either one would quietly reopen the same door on the node.
    """
    for name in ("build-on-node.sh", "update-all.sh"):
        path = PROJECT_ROOT / "scripts" / name
        hits = [
            f"{name}:{n}: {line.strip()}"
            for n, line in _code_lines(path)
            if "/tmp/aureax-build" in line
        ]
        assert not hits, "fixed /tmp build context:\n" + "\n".join(hits)


def test_operator_set_integer_knobs_are_validated_before_they_are_used() -> None:
    """A non-numeric knob must be diagnosed, not divided by.

    `[` failing inside a `while` condition is exempt from `set -e`, so an
    unvalidated concurrency cap runs the fan-out unbounded across the whole
    fleet, and an unvalidated chunk count divides by zero mid-provision. Source
    text, because neither script can be run in a test: both read the fleet
    config and start work at file scope.
    """
    update_all = (PROJECT_ROOT / "scripts" / "update-all.sh").read_text()
    assert '"$BUILD_CONCURRENCY" =~ ^[1-9][0-9]*$' in update_all
    download = (PROJECT_ROOT / "scripts" / "download-model.sh").read_text()
    assert 'case "$DOWNLOAD_CHUNKS" in' in download
    assert "*[!0-9]*)" in download


# --- download-model.sh: a part is complete only at the exact length ----------

DOWNLOAD_MODEL = PROJECT_ROOT / "scripts" / "download-model.sh"


def _shell_function(path, name: str) -> str:
    """The source text of one shell function, for a script that cannot be sourced.

    download-model.sh does its work at file scope — it reads the fleet config
    and starts downloading — so a test can only drive its functions by lifting
    them out. The emptiness assertion is what keeps a rename from turning this
    into a test that runs nothing.
    """
    text = path.read_text()
    start = text.index(f"\n{name}() {{\n")
    end = text.index("\n}\n", start)
    body = text[start : end + 3]
    assert f"{name}()" in body, f"{name} not found in {path}"
    return body


def _run_fetch_file(dest_bytes: int, upstream_bytes: int, tmp_path):
    """fetch_file against a stubbed upstream, with `dest` holding dest_bytes."""
    dest = tmp_path / "part.gguf"
    dest.write_bytes(b"x" * dest_bytes)
    return _run(f"""
        {_shell_function(DOWNLOAD_MODEL, "fetch_file")}
        gb() {{ echo 1; }}
        remote_size() {{ echo {upstream_bytes}; }}
        _cleanup_pfetch() {{ :; }}
        single_stream_fetch() {{ echo "FETCHED have=$(file_size "$2") total=$3"; }}
        DOWNLOAD_CHUNKS=8
        DOWNLOAD_PARALLEL_MIN=$((256 * 1024 * 1024))
        PFETCH_TMPDIR=""
        fetch_file https://example.invalid/part.gguf "{dest}"
        echo "SIZE=$(file_size "{dest}")"
    """), dest


def test_fetch_file_reports_complete_only_at_the_exact_upstream_length(tmp_path) -> None:
    """The one case that needs no work: local length == upstream length."""
    proc, _ = _run_fetch_file(1000, 1000, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "already complete (1000 bytes)" in proc.stdout
    assert "FETCHED" not in proc.stdout


def test_fetch_file_resumes_a_short_part(tmp_path) -> None:
    """A short part keeps what it has and resumes from there."""
    proc, _ = _run_fetch_file(400, 1000, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "FETCHED have=400 total=1000" in proc.stdout
    assert "already complete" not in proc.stdout


def test_fetch_file_truncates_and_refetches_an_over_length_part(tmp_path) -> None:
    """Longer than upstream is corruption, and it has to repair itself.

    `-C -` against a file that changed upstream appends the new body to the old
    one, and a throttled resume appends an error body the same way. Reporting
    that complete makes every later run a no-op: the caller re-detects the size
    mismatch, is told "already complete", and SCPs the corrupt part to a node —
    a manual `rm` on a fleet that is re-provisioned as routine.
    """
    proc, _dest = _run_fetch_file(1500, 1000, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "already complete" not in proc.stdout
    assert "corrupt, re-fetching" in proc.stdout
    # Truncated BEFORE the re-fetch, so the resume starts from zero rather than
    # from the inflated offset that produced the file.
    assert "FETCHED have=0 total=1000" in proc.stdout


# --- download-model.sh: declared digests are verified before any upload -----


def _write_parts(tmp_path, bodies: dict[str, bytes]) -> None:
    """Write the model parts a verification run reads. Nothing is downloaded."""
    for name, body in bodies.items():
        (tmp_path / name).write_bytes(body)


def _digest(body: bytes) -> str:
    """The digest as `shasum -a 256` prints it: 64 lowercase hex characters."""
    return hashlib.sha256(body).hexdigest()


def _run_verify(tmp_path, parts, digests, model="Probe-Model"):
    """Drive verify_model_parts against files in `tmp_path`.

    Both list arguments are space-separated strings, which is the shape
    download-model.sh keeps them in (bash 3.2 on the Mac has no nested arrays,
    so every per-node list travels as one string per index).
    """
    return _run(f"""
        {_shell_function(DOWNLOAD_MODEL, "sha256_of")}
        {_shell_function(DOWNLOAD_MODEL, "verify_model_parts")}
        LOCAL_MODEL_DIR="{tmp_path}"
        verify_model_parts "{model}" "{" ".join(parts)}" "{" ".join(digests)}"
    """)


def test_a_matching_digest_passes_and_keeps_the_file(tmp_path) -> None:
    """The ordinary case: declared digests, files that match them."""
    bodies = {f"part-{i}.gguf": bytes([i]) * 128 for i in range(3)}
    _write_parts(tmp_path, bodies)

    proc = _run_verify(tmp_path, list(bodies), [_digest(b) for b in bodies.values()])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for name in bodies:
        assert f"{name}: sha256 verified" in proc.stdout
        assert (tmp_path / name).exists()


def test_a_mismatched_part_is_deleted_and_the_run_fails(tmp_path) -> None:
    """A wrong-bytes part must not survive the run.

    Every other gate compares LENGTHS, so a substituted part of the right size
    is reported complete on this run and on every later one — the file would be
    pushed to a head and handed to llama.cpp's GGUF deserializer forever. Both
    digests are named because the operator's next question is whether the
    registry entry or the file is the wrong one.
    """
    good, bad = b"g" * 64, b"b" * 64
    _write_parts(tmp_path, {"good.gguf": good, "bad.gguf": bad})
    declared_for_bad = _digest(b"expected-bytes")

    proc = _run_verify(tmp_path, ["good.gguf", "bad.gguf"], [_digest(good), declared_for_bad])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "sha256 mismatch" in proc.stdout
    assert declared_for_bad in proc.stdout, "the expected digest is not named"
    assert _digest(bad) in proc.stdout, "the digest actually found is not named"
    assert not (tmp_path / "bad.gguf").exists(), "a mismatched part was left on disk"
    # The parts that verified are untouched — re-running re-fetches only the
    # one that failed.
    assert (tmp_path / "good.gguf").exists()


def test_no_declared_digest_says_so_once_and_keeps_the_size_checks(tmp_path) -> None:
    """An entry with no digest stays legal, and stays audible.

    Every shipped registry entry declares none today, so this is the normal
    path; it is one explicit line per model rather than silence, because "the
    right length arrived" is not the claim a digest makes.
    """
    _write_parts(tmp_path, {"part.gguf": b"x" * 32})

    proc = _run_verify(tmp_path, ["part.gguf"], [], model="Qwen-Probe")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no digest declared for Qwen-Probe; size check only" in proc.stdout
    assert proc.stdout.count("no digest declared") == 1
    assert (tmp_path / "part.gguf").exists()


def test_a_digest_list_of_the_wrong_length_verifies_nothing(tmp_path) -> None:
    """The lists are paired by position, so an unequal one is not a partial check.

    Verifying part N against part M's digest reports a mismatch on a file that
    is fine — and would then delete it.
    """
    bodies = {"a.gguf": b"a" * 16, "b.gguf": b"b" * 16}
    _write_parts(tmp_path, bodies)

    proc = _run_verify(tmp_path, list(bodies), [_digest(b"a" * 16)])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "paired by position" in proc.stdout
    assert all((tmp_path / name).exists() for name in bodies), "a part was deleted"


def test_an_unreadable_part_fails_rather_than_verifying_silently(tmp_path) -> None:
    """No digest computed is a failure, not a pass.

    sha256_of answers with an empty line for a missing file or a host with no
    digest tool, because a non-zero status inside a command substitution aborts
    the run under `set -e`. An empty answer must therefore never compare equal
    to anything.
    """
    proc = _run_verify(tmp_path, ["absent.gguf"], ["a" * 64])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "could not compute sha256" in proc.stdout


def test_both_step_2_paths_verify_before_step_3_uploads() -> None:
    """Source text: the ORDER is the property, and it needs a fleet to observe.

    A model that is already complete locally is skipped by the download loop
    but still SCP'd in Step 3, so verification on the download path alone would
    leave the common case — a re-provision against a populated models/ — pushing
    unverified files.
    """
    text = DOWNLOAD_MODEL.read_text()
    # `|| exit 1` is part of what is pinned: a failure has to stop the run
    # rather than be reported and walked past into the upload.
    call = 'verify_model_parts "$f" "${NODE_MODEL_PARTS[$i]}" "$digests" || exit 1'
    offsets = [m.start() for m in re.finditer(re.escape(call), text)]
    assert len(offsets) == 2, f"expected both Step 2 paths to verify, found {len(offsets)}"
    step3 = text.index("--- Step 3:")
    assert all(offset < step3 for offset in offsets)


def test_the_k3s_installer_is_retried_not_run_once() -> None:
    """Source text: the installer's own download of the k3s binary has no retry,
    and one refused download on one node failed a whole nine-node provision.
    The invocation is a function called from a bounded `until` loop, so both
    setup-all.sh and the Python deployer (which run this same script) retry."""
    text = (PROJECT_ROOT / "scripts" / "join-node.sh").read_text()
    fn = text.index("install_agent() {")
    loop = text.index("until install_agent; do")
    assert fn < loop, "the retry loop must call the installer through its function"
    assert "sh -s - agent $EXTRA_AGENT_ARGS" in text[fn:loop]
    assert "INSTALL_ATTEMPTS=3" in text
    assert text.count("sh -s - agent") == 1, "the installer must run only through the retried function"


def test_both_draft_paths_verify_before_step_3_uploads() -> None:
    """Same property for the speculative-decoding draft: it is pushed by the
    same Step 3 loop, so both Step 2b paths verify its declared digest first."""
    text = DOWNLOAD_MODEL.read_text()
    call = 'verify_model_parts "$df" "$df" "${NODE_DRAFT_SHA256[$i]}" || exit 1'
    offsets = [m.start() for m in re.finditer(re.escape(call), text)]
    assert len(offsets) == 2, f"expected both Step 2b paths to verify, found {len(offsets)}"
    step2b = text.index("--- Step 2b:")
    step3 = text.index("--- Step 3:")
    assert all(step2b < offset < step3 for offset in offsets)


def test_config_derived_model_names_reach_the_node_quoted() -> None:
    """`filename` is pasted from a third-party model page, and it reaches a shell.

    ssh_facts embeds each snippet inside `$( )` on the remote command line, so
    an unquoted name carrying `$(...)`, a backtick or `;` runs on every node as
    the login user. The scp operands next to it are NOT quoted and must not be:
    scp has used the SFTP protocol since OpenSSH 9.0, which sends the remote
    path verbatim, so quotes there would become part of the filename. Source
    text: driving either would need a fleet.
    """
    text = DOWNLOAD_MODEL.read_text()
    assert "stat -c%s '${REMOTE_MODEL_DIR}/${parts[$pi]}'" in text
    assert "stat -c%s '${REMOTE_MODEL_DIR}/${df}'" in text
    assert '"${USER}@${HOST}:${REMOTE_MODEL_DIR}/${p}"' in text
    assert '"${USER}@${HOST}:${REMOTE_MODEL_DIR}/${df}"' in text


# --- join-node.sh: the cluster token never arrives in argv -------------------


def test_join_node_refuses_a_token_on_the_command_line(tmp_path) -> None:
    """The rejection happens before any step, so this runs nothing on the host.

    An argv word is visible in `ps` on both machines for the length of the
    install, and sudo copies the whole command line into the node's auth log,
    where it outlives the run. The token is the one credential that lets any
    machine join the cluster.
    """
    proc = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "join-node.sh"),
         "https://192.0.2.10:6443", "a-token-that-must-not-be-accepted"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "not on the command line" in proc.stderr
    assert "Enabling cgroups" not in proc.stdout
    # The usage text must not advertise the form either.
    assert "<k3s_token>" not in (PROJECT_ROOT / "scripts" / "join-node.sh").read_text()


# --- test-all.sh does not gamble with the operator's fleet definition -------


def test_the_invalid_yaml_check_backs_up_the_fleet_definition() -> None:
    """Source text, because the suite cannot run itself inside its own tests.

    Section 3 aims a deliberately invalid body at the route that overwrites
    configs/aureax-network.yaml. That file is gitignored, holds every node's
    address and credential, and has no other copy anywhere; the route validates
    before it writes, so the backup is what stands between a future regression
    in that order and a fleet nobody can reconstruct.
    """
    text = (PROJECT_ROOT / "scripts" / "test-all.sh").read_text()
    put_at = text.index("PUT invalid YAML returns ok:false")
    before, after = text[:put_at], text[put_at:]
    assert 'FLEET_BACKUP="$(mktemp' in before, "no backup is taken before the PUT"
    assert "restore_fleet_file" in after, "nothing restores the file after the PUT"
    # And whatever happens to the run, the traps put it back.
    assert "trap suite_cleanup EXIT" in text
    assert "trap 'suite_cleanup; exit 130' INT TERM" in text
