"""`scripts/bench-decode.sh` — the knobs the sharding review measures with.

PREFILL_TOKENS, CONCURRENCY, TIMEOUT_SECONDS and HEAD_SSH/HEAD_SSH_KEY/HEAD_NIC
are environment variables so that the positional form, and every result
recorded with it, keeps its meaning. Most of what is checked here is therefore
two-sided: the new knob does what it says, AND a run without it is the run the
script always made.

The script is driven end to end, as tests/test_common_sh.py's bench-decode
tests do. `curl` and `ssh` are recording executables on PATH rather than shell
functions, because what matters is what a SEPARATE process received — its argv,
the `-K` pipe, the body file. Each stub process logs to its own file (named by
its pid), so the concurrent requests cannot interleave their records.

One test dials for real: the unreachable-endpoint case runs the real curl
against 127.0.0.1:1, where nothing listens, to prove the banner (with the
requested fill) is printed before the health check fails.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import pytest

from tests.conftest import PROJECT_ROOT

BENCH_DECODE = PROJECT_ROOT / "scripts" / "bench-decode.sh"

# The fixed prompt. Pinned here because it is the workload: a fill-0 run must
# send exactly this, or every earlier measurement stops being comparable.
PROMPT = "List three signs that an industrial sensor is about to fail, and explain each one."

KNOBS = (
    "AUREAX_API_KEY", "PREFILL_TOKENS", "CONCURRENCY", "TIMEOUT_SECONDS",
    "HEAD_SSH", "HEAD_SSH_KEY", "HEAD_NIC",
)

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

_CURL_STUB = r"""#!/usr/bin/env bash
out="$STUB_DIR/curl.$$.log"
prev=""
for a in "$@"; do
    printf 'ARGV %s\n' "$a" >> "$out"
    if [ "$prev" = "-K" ]; then printf 'CONFIG %s\n' "$(cat "$a")" >> "$out"; fi
    if [ "$prev" = "--data-binary" ]; then printf 'BODY %s\n' "$(cat "${a#@}")" >> "$out"; fi
    prev="$a"
done
case "$*" in
    *"/health"*)
        exit 0 ;;
    *"/props"*)
        printf '%s' "$STUB_PROPS"
        exit 0 ;;
    *"/completion"*)
        sleep "${STUB_SLEEP:-0}"
        printf '{"timings":{"prompt_n":9,"prompt_ms":100.0,"prompt_per_second":90.0,'
        printf '"predicted_n":4,"predicted_ms":200.0,"predicted_per_second":20.0}}\n200'
        exit 0 ;;
esac
exit 1
"""

# A head whose counters advance by a fixed step per read: eth0 by 400 kB
# received and 4 MB transmitted, eth1 (printed flush against its colon, the
# way an old kernel prints a wide counter) by ten times that.
_SSH_STUB = r"""#!/usr/bin/env bash
last=""
for a in "$@"; do last="$a"; done
printf '%s\n' "$*" >> "$STUB_DIR/ssh.log"
if [ "${STUB_SSH_MODE:-ok}" = "fail" ]; then
    echo "ssh: connect to host 192.0.2.10 port 22: Connection refused" >&2
    exit 255
fi
case "$last" in
    *"ip route get 1.1.1.1"*)
        printf '1.1.1.1 via 192.0.2.1 dev eth0 src 192.0.2.10 uid 1000 \n    cache \n'
        exit 0 ;;
    "cat /proc/net/dev")
        n=$(cat "$STUB_DIR/ssh.count" 2>/dev/null || echo 0)
        n=$((n + 1))
        echo "$n" > "$STUB_DIR/ssh.count"
        printf 'Inter-|   Receive                            |  Transmit\n'
        printf ' face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets\n'
        printf '    lo:    5000      50    0    0    0     0          0         0     5000      50    0    0    0     0       0          0\n'
        printf '  eth0: %s 900 0 0 0 0 0 0 %s 800 0 0 0 0 0 0 0\n' $((1000000 + n * 400000)) $((2000000 + n * 4000000))
        printf 'eth1:%s 900 0 0 0 0 0 0 %s 800 0 0 0 0 0 0 0\n' $((1000000 + n * 4000000)) $((2000000 + n * 40000000))
        exit 0 ;;
esac
exit 1
"""


def _env(tmp_path, stubs: bool = True, **extra: str) -> dict[str, str]:
    """A clean environment: no inherited knob, a dummy CA, a private TMPDIR."""
    env = dict(os.environ)
    for name in KNOBS:
        env.pop(name, None)
    ca = tmp_path / "ca.pem"
    ca.write_text("not a real certificate")
    env["AUREAX_CA"] = str(ca)
    scratch = tmp_path / "tmp"
    scratch.mkdir(exist_ok=True)
    env["TMPDIR"] = str(scratch)
    if stubs:
        bindir = tmp_path / "bin"
        stub_dir = tmp_path / "stub"
        if not bindir.exists():
            bindir.mkdir()
            stub_dir.mkdir()
            for name, text in (("curl", _CURL_STUB), ("ssh", _SSH_STUB)):
                path = bindir / name
                path.write_text(text)
                path.chmod(0o755)
        env["PATH"] = f"{bindir}:{env['PATH']}"
        env["STUB_DIR"] = str(stub_dir)
        env["STUB_PROPS"] = '{"model_path":"/opt/aureax/models/test.gguf"}'
    env.update(extra)
    return env


def _bench(tmp_path, *args: str, stubs: bool = True, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(BENCH_DECODE), *args],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env=_env(tmp_path, stubs=stubs, **extra), timeout=120,
    )


def _curl_calls(tmp_path) -> list[dict]:
    """One record per curl process: its argv, -K contents and body file."""
    calls = []
    for log in sorted((tmp_path / "stub").glob("curl.*.log")):
        call = {"argv": [], "configs": [], "bodies": []}
        for line in log.read_text().splitlines():
            kind, _, value = line.partition(" ")
            call[{"ARGV": "argv", "CONFIG": "configs", "BODY": "bodies"}[kind]].append(value)
        calls.append(call)
    return calls


def _completion_calls(tmp_path) -> list[dict]:
    return [c for c in _curl_calls(tmp_path) if any(a.endswith("/completion") for a in c["argv"])]


def _ssh_lines(tmp_path) -> list[str]:
    log = tmp_path / "stub" / "ssh.log"
    return log.read_text().splitlines() if log.exists() else []


def _scratch_left(tmp_path) -> list[str]:
    return [p.name for p in (tmp_path / "tmp").glob("aureax-bench.*")]


# --- the script's surface ---------------------------------------------------

def test_the_script_parses() -> None:
    proc = subprocess.run(["bash", "-n", str(BENCH_DECODE)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_lists_every_knob_and_dials_nothing(tmp_path, flag) -> None:
    proc = _bench(tmp_path, flag)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("Usage: bench-decode.sh HOST:PORT [N_PREDICT] [RUNS]")
    for name in (*KNOBS, "AUREAX_CA"):
        assert name in proc.stdout, f"--help does not mention {name}"
    assert "http://" not in proc.stdout
    assert _curl_calls(tmp_path) == []


def test_no_argument_is_refused_with_exit_2(tmp_path) -> None:
    """The endpoint has no default — nothing is dialled that was not named."""
    proc = _bench(tmp_path)

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "[FAIL] usage" in proc.stderr
    assert _curl_calls(tmp_path) == []


@pytest.mark.parametrize("name, value", [
    ("PREFILL_TOKENS", "abc"),
    ("PREFILL_TOKENS", "-1"),
    ("PREFILL_TOKENS", "010"),
    ("CONCURRENCY", "0"),
    ("CONCURRENCY", "two"),
    ("TIMEOUT_SECONDS", "0"),
    ("TIMEOUT_SECONDS", "08"),
    ("HEAD_SSH", "192.0.2.10"),
    ("HEAD_SSH", "-oProxyCommand=evil@192.0.2.10"),
    ("HEAD_SSH", "aureax@-oProxyCommand=evil"),
    ("HEAD_NIC", "eth 0"),
])
def test_a_bad_knob_is_refused_before_anything_is_dialled(tmp_path, name, value) -> None:
    proc = _bench(tmp_path, "node:30800", **{name: value})

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert f"[FAIL] {name}=" in proc.stderr
    assert _curl_calls(tmp_path) == []
    assert _ssh_lines(tmp_path) == []


def test_an_unreachable_endpoint_fails_after_the_banner_shows_the_fill(tmp_path) -> None:
    """Real curl, against a port where nothing listens: the health-check path."""
    proc = _bench(tmp_path, "127.0.0.1:1", stubs=False, PREFILL_TOKENS="1000")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "  fill:      1000 tokens requested (4000 characters of filler" in proc.stdout
    assert "[FAIL] https://127.0.0.1:1/health" in proc.stderr
    assert _scratch_left(tmp_path) == []


# --- PREFILL_TOKENS ---------------------------------------------------------

def _prompts(tmp_path) -> list[str]:
    bodies = [b for c in _completion_calls(tmp_path) for b in c["bodies"]]
    return [json.loads(b)["prompt"] for b in bodies]


def test_the_filler_prompt_is_byte_identical_across_runs_and_invocations(tmp_path) -> None:
    for _ in range(2):
        proc = _bench(tmp_path, "node:30800", "4", "2", PREFILL_TOKENS="1000")
        assert proc.returncode == 0, proc.stdout + proc.stderr

    bodies = [b for c in _completion_calls(tmp_path) for b in c["bodies"]]
    assert len(bodies) == 4, bodies
    assert len(set(bodies)) == 1, "the request body changed between runs"

    prompt = json.loads(bodies[0])["prompt"]
    assert prompt.endswith("\n\n" + PROMPT)
    filler = prompt[: -len("\n\n" + PROMPT)]
    assert len(filler) == 4000
    assert filler.startswith("Maintenance logs")


def test_fill_zero_sends_the_bare_prompt_and_the_same_request_fields(tmp_path) -> None:
    proc = _bench(tmp_path, "node:30800", "4", "1")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    [call] = _completion_calls(tmp_path)
    assert json.loads(call["bodies"][0]) == {
        "prompt": PROMPT,
        "n_predict": 4,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
    }


# --- the default output, and TIMEOUT_SECONDS ---------------------------------

def test_a_run_with_no_knobs_prints_the_format_it_always_has(tmp_path) -> None:
    proc = _bench(tmp_path, "node:30800", "4", "2")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    lines = proc.stdout.splitlines()
    run_line = "  run {}: decode 20.0 tok/s (4 tok, 50.0 ms/tok)  |  prefill 90.0 tok/s (9 tok, 100.0 ms)"
    assert run_line.format(1) in lines
    assert run_line.format(2) in lines
    assert lines[-2:] == ["  decode  median: 20.0 tok/s", "  prefill median: 90.0 tok/s"]
    assert not [ln for ln in lines if "aggregate" in ln or "req " in ln or "head NIC" in ln]
    assert _ssh_lines(tmp_path) == []

    for call in _completion_calls(tmp_path):
        argv = call["argv"]
        assert argv[argv.index("-m") + 1] == "1800"


def test_timeout_seconds_is_the_completion_requests_ceiling(tmp_path) -> None:
    proc = _bench(tmp_path, "node:30800", "4", "1", TIMEOUT_SECONDS="7200")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    [call] = _completion_calls(tmp_path)
    assert call["argv"][call["argv"].index("-m") + 1] == "7200"
    others = [c for c in _curl_calls(tmp_path) if c not in _completion_calls(tmp_path)]
    assert others and all(c["argv"][c["argv"].index("-m") + 1] == "10" for c in others)


# --- CONCURRENCY ------------------------------------------------------------

def test_concurrent_requests_run_in_parallel_and_report_an_aggregate(tmp_path) -> None:
    """Three 1 s requests finish in about 1 s, each with its own key pipe."""
    key = "aureax_sk_benchconcurrencykey"
    proc = _bench(
        tmp_path, "node:30800", "4", "1",
        CONCURRENCY="3", STUB_SLEEP="1", AUREAX_API_KEY=key,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()

    for req in (1, 2, 3):
        assert f"  run 1 req {req}: decode 20.0 tok/s (4 tok, 50.0 ms/tok)  |  prefill 90.0 tok/s (9 tok, 100.0 ms)" in lines

    [agg] = [ln for ln in lines if ln.startswith("  run 1 aggregate: ")]
    m = re.search(
        r"decode ([\d.]+) tok/s \(12 tok over a ([\d.]+) s decode window\)"
        r"  \|  end to end ([\d.]+) tok/s \(([\d.]+) s wall\)"
        r"  \|  per-request decode median 20\.0 tok/s$",
        agg,
    )
    assert m, agg
    agg_ps, window, _, wall = (float(g) for g in m.groups())
    assert wall < 2.5, f"requests ran one after another: {agg}"
    # The window is floored at one request's own 200 ms decode.
    assert window >= 0.2 and 0 < agg_ps <= 12 / 0.2

    assert "  decode  median: 20.0 tok/s per request (3 requests)" in lines
    assert "  prefill median: 90.0 tok/s per request" in lines
    assert [ln for ln in lines if ln.startswith("  aggregate decode median: ")] == [
        f"  aggregate decode median: {m.group(1)} tok/s at concurrency 3"
    ]

    completions = _completion_calls(tmp_path)
    assert len(completions) == 3
    for call in completions:
        assert call["configs"] == [f'header = "Authorization: Bearer {key}"']
        assert not [a for a in call["argv"] if key in a]
    assert _scratch_left(tmp_path) == []


@pytest.mark.parametrize("slots, warned", [(1, True), (2, False)])
def test_concurrency_above_the_servers_slots_is_called_a_queue(tmp_path, slots, warned) -> None:
    proc = _bench(
        tmp_path, "node:30800", "4", "1", CONCURRENCY="2",
        STUB_PROPS=json.dumps({"model_path": "/m/test.gguf", "total_slots": slots}),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"  slots:     {slots}" in proc.stdout
    assert ("will queue, not batch" in proc.stdout) is warned


# --- HEAD_SSH: NIC bytes per decoded token ----------------------------------

@pytest.mark.parametrize("concurrency, tx, rx, tokens", [
    ("1", "1000.0", "100.0", 4),
    ("2", "500.0", "50.0", 8),
])
def test_head_nic_bytes_are_divided_by_the_runs_decoded_tokens(tmp_path, concurrency, tx, rx, tokens) -> None:
    proc = _bench(
        tmp_path, "node:30800", "4", "2",
        CONCURRENCY=concurrency, HEAD_SSH="aureax@192.0.2.10", HEAD_SSH_KEY="/keys/head",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()

    assert "  head NIC:  eth0 on aureax@192.0.2.10 (the default-route interface)" in lines
    for run in (1, 2):
        assert (
            f"  run {run} head NIC: tx {tx} KB/token, rx {rx} KB/token "
            f"(4.0 MB tx, 0.4 MB rx over {tokens} tok)"
        ) in lines
    assert lines[-1] == f"  head NIC: tx {tx} KB/token, rx {rx} KB/token (interface eth0)"

    # One detection, then a read before and after each run — all through
    # ssh_cmd's options, with the given key and login.
    ssh = _ssh_lines(tmp_path)
    assert len(ssh) == 5 and ssh[0].endswith("ip route get 1.1.1.1 2>/dev/null || "
                                             "/usr/sbin/ip route get 1.1.1.1 2>/dev/null || "
                                             "/sbin/ip route get 1.1.1.1")
    assert all(s.endswith("cat /proc/net/dev") for s in ssh[1:])
    for s in ssh:
        assert "-o BatchMode=yes" in s and "-i /keys/head aureax@192.0.2.10" in s
        assert "UserKnownHostsFile=" in s and "/.aureax/known_hosts" in s


def test_head_nic_given_skips_detection(tmp_path) -> None:
    """eth1's line is printed flush against its colon, as an old kernel does."""
    proc = _bench(tmp_path, "node:30800", "4", "1", HEAD_SSH="aureax@192.0.2.10", HEAD_NIC="eth1")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    assert "  head NIC:  eth1 on aureax@192.0.2.10 (HEAD_NIC)" in proc.stdout
    assert proc.stdout.splitlines()[-1] == "  head NIC: tx 10000.0 KB/token, rx 1000.0 KB/token (interface eth1)"
    assert not [s for s in _ssh_lines(tmp_path) if "ip route" in s]


def test_an_ssh_failure_warns_once_and_the_benchmark_carries_on(tmp_path) -> None:
    proc = _bench(
        tmp_path, "node:30800", "4", "2",
        HEAD_SSH="aureax@192.0.2.10", HEAD_NIC="eth0", STUB_SSH_MODE="fail",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = proc.stdout.splitlines()

    warns = [ln for ln in lines if ln.startswith("  [WARN] head NIC:")]
    assert warns == ["  [WARN] head NIC: ssh aureax@192.0.2.10 failed — continuing without NIC numbers"]
    assert any("Connection refused" in ln for ln in lines)
    assert len(_ssh_lines(tmp_path)) == 1, "a failed head kept being dialled"
    assert lines[-2:] == ["  decode  median: 20.0 tok/s", "  prefill median: 90.0 tok/s"]
    assert not [ln for ln in lines if "KB/token" in ln]


def test_an_unknown_interface_warns_once_and_the_benchmark_carries_on(tmp_path) -> None:
    proc = _bench(tmp_path, "node:30800", "4", "2", HEAD_SSH="aureax@192.0.2.10", HEAD_NIC="wlan9")
    assert proc.returncode == 0, proc.stdout + proc.stderr

    warns = [ln for ln in proc.stdout.splitlines() if "[WARN]" in ln]
    assert warns == [
        "  [WARN] head NIC: no 'wlan9' counters in /proc/net/dev on 192.0.2.10 — set HEAD_NIC"
        " — continuing without NIC numbers"
    ]
    assert len(_ssh_lines(tmp_path)) == 1
    assert "KB/token" not in proc.stdout
