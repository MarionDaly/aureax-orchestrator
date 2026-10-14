#!/usr/bin/env bash
# ===========================================================================
# Measure decode and prefill throughput on a running llama-server.
#
# Exists because nothing else in this repo separates the two. The MCP tool
# path (client/mcp/server.py) reports only timings.predicted_ms, and the
# harness spinner reports wall-clock including prefill and HTTP — two
# different numbers on screen during one run, neither of them attributable.
# Every performance change wants an A/B against a FIXED workload, which is
# what this gives.
#
# Usage (HOST:PORT is required — there is no default endpoint, so the script
# can never dial a machine the operator did not name):
#   ./scripts/bench-decode.sh HOST:PORT [N_PREDICT] [RUNS]
#   ./scripts/bench-decode.sh 192.0.2.10:30800 128 3
#   ./scripts/bench-decode.sh --help
#
# Against a node with api_key auth on, export the key first — /props and
# /completion answer 401 without it (/health stays public either way):
#   AUREAX_API_KEY=$(cat ~/.aureax/keys/<node-id>.key) \
#       ./scripts/bench-decode.sh 192.0.2.10:30800
#
# Every call is HTTPS, verified against the fleet CA (curl_tls in
# scripts/lib/common.sh). Point AUREAX_CA elsewhere to bench a second fleet;
# there is no plaintext mode, so a machine answering http:// on that port is a
# machine this script will not measure.
#
# ENVIRONMENT KNOBS. Everything beyond the fixed short-prompt run is an
# environment variable, so argv keeps its three positionals and a run with
# none of these set is the run this script has always made: the same prompt
# text, one request at a time, a 1800 s ceiling, no SSH.
#
#   PREFILL_TOKENS=N    prepend roughly N tokens of filler to the prompt
#                       (default 0, the bare prompt). One fixed English
#                       paragraph, repeated and cut to N x 4 characters. Four
#                       characters per token is an estimate, so every run line
#                       reports the server's own prompt_n. Same N, same bytes,
#                       in every run of every invocation.
#   CONCURRENCY=C       fire C identical /completion requests at once in every
#                       run (default 1). Above 1, each request gets its own
#                       line and the run an AGGREGATE decode tok/s: the summed
#                       predicted_n over the decode window, which runs from the
#                       earliest decode start to the slowest request's finish.
#                       Each finish is clocked here; a request's decode start is
#                       its finish minus its own predicted_ms. The end-to-end
#                       figure over the whole wall clock, prefill included, is
#                       printed beside it. Batching needs C server slots
#                       (model.server.parallel_slots), and the script warns when
#                       /props reports fewer: the extra requests then queue
#                       instead of batching.
#   TIMEOUT_SECONDS=S   curl -m for each /completion request (default 1800). A
#                       32768-token prefill on CPU workers can take an hour, so
#                       give a 32768 fill e.g. 7200. A request queued behind a
#                       busy slot spends its wait on this clock too.
#   HEAD_SSH=user@host  the group head. When set, the head's /proc/net/dev is
#                       read before and after every run, and each run prints the
#                       bytes its NIC transmitted and received per decoded
#                       token, with a median line at the end. Reached through
#                       ssh_cmd (scripts/lib/common.sh: BatchMode, the shared
#                       ~/.aureax/known_hosts) on port 22 — this script loads no
#                       fleet config. An SSH failure warns once and the
#                       benchmark carries on without NIC numbers; it never
#                       aborts a run.
#   HEAD_SSH_KEY=path   private key for HEAD_SSH (default ~/.ssh/id_ed25519,
#                       the private half of ssh.bootstrap_public_key's default —
#                       the same default build-on-node.sh takes).
#   HEAD_NIC=ifname     the interface to count (default: whichever one
#                       `ip route get 1.1.1.1` names on the head, i.e. the
#                       default-route NIC; set it when the workers are reached
#                       over another one).
#
# What a NIC number covers: every byte on that interface between the two reads,
# divided by the run's total predicted_n. That is the whole run, prefill
# included, plus everything else on the link — telemetry, this script's own
# HTTPS, the SSH read itself, tens of KB a run. At fill 0 the prefill is a few
# dozen tokens and the figure is the decode wire cost; at a 32768 fill it also
# carries the prefill's traffic, which the MB totals on each run line let you
# subtract. KB and MB are decimal (10^3 and 10^6 bytes).
#
# THE PHASE 0 SWEEP of the sharding review is fill 0 and 32768, each at
# concurrency 1 and 2, with the head's NIC counted:
#   export AUREAX_API_KEY=$(cat ~/.aureax/keys/<head-id>.key)
#   export HEAD_SSH=aureax@192.0.2.10
#   ./scripts/bench-decode.sh 192.0.2.10:30800
#   CONCURRENCY=2 ./scripts/bench-decode.sh 192.0.2.10:30800
#   PREFILL_TOKENS=32768 TIMEOUT_SECONDS=7200 ./scripts/bench-decode.sh 192.0.2.10:30800
#   PREFILL_TOKENS=32768 TIMEOUT_SECONDS=7200 CONCURRENCY=2 \
#       ./scripts/bench-decode.sh 192.0.2.10:30800
# The expected decode wire traffic on a 3-worker group is about 1 MB per token,
# most of it the logits readback. The two concurrency-2 rows measure batching
# only on a head running parallel_slots: 2 — llama-server divides context_size
# across its slots, so a 131072 head leaves 65536 per slot, still room for the
# 32768 fill.
#
# Sampling clocks on a worker during the run (the DVFS question) is a separate
# terminal. The sysfs paths below are per-SoC — read the node's own
# /sys/class/devfreq and /sys/devices/system/cpu/*/cpufreq to find its names:
#   ssh <user>@<worker> 'while :; do \
#       printf "%s %s\n" "$(cat /sys/class/devfreq/<dev>/cur_freq)" \
#                        "$(cat /sys/devices/system/cpu/<cpu>/cpufreq/scaling_cur_freq)"; \
#       sleep 0.2; done'
# ===========================================================================
set -uo pipefail

# The shared `fail` and `warn` helpers, so a benchmark failure reads the same
# as one from setup-all.sh (the measurement lines below use this script's own
# format) — curl_tls, which is how every request here reaches the node over
# TLS verified against the fleet CA — and ssh_cmd, which is how HEAD_SSH reaches
# the head with the options every other script uses. Note this script keeps
# `set -uo pipefail` WITHOUT -e: a run that fails is reported and exited
# explicitly below.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

usage() {
    cat <<'EOF'
Usage: bench-decode.sh HOST:PORT [N_PREDICT] [RUNS]
       bench-decode.sh -h | --help

Measure llama-server decode and prefill throughput over HTTPS, verified
against the fleet CA. HOST:PORT has no default; N_PREDICT (tokens generated
per request) defaults to 128 and RUNS to 3.

Environment (all optional; unset, the run is the fixed short prompt, alone):
  AUREAX_API_KEY   bearer key for an auth-enabled node (handed to curl by -K,
                   never on argv)
  AUREAX_CA        fleet CA bundle (default ~/.aureax/tls/ca.pem)
  PREFILL_TOKENS   prepend ~N tokens of fixed filler to the prompt (default 0;
                   ~4 characters per token, byte-identical for the same N)
  CONCURRENCY      identical /completion requests fired in parallel per run
                   (default 1); above 1, prints each request and the run's
                   aggregate decode tok/s. Needs that many server slots
                   (model.server.parallel_slots), or the requests queue
  TIMEOUT_SECONDS  curl -m for each /completion request (default 1800); a
                   32768 fill on CPU workers can take an hour, so use e.g. 7200
  HEAD_SSH         user@host of the group head: reads /proc/net/dev there
                   before and after each run and prints the NIC's tx and rx KB
                   per decoded token (SSH port 22; a failure warns once and
                   the benchmark carries on)
  HEAD_SSH_KEY     private key for HEAD_SSH (default ~/.ssh/id_ed25519)
  HEAD_NIC         head interface to count (default: the one
                   `ip route get 1.1.1.1` names on the head)

Phase 0 sweep: PREFILL_TOKENS 0 and 32768, each at CONCURRENCY 1 and 2, with
HEAD_SSH set (and TIMEOUT_SECONDS=7200 for the 32768 rows), e.g.
  HEAD_SSH=aureax@192.0.2.10 PREFILL_TOKENS=32768 TIMEOUT_SECONDS=7200 \
      CONCURRENCY=2 ./scripts/bench-decode.sh 192.0.2.10:30800
EOF
}

# Help before the endpoint check, so `--help` is answered rather than taken
# for an endpoint and dialled.
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

# The endpoint has no default: a benchmark that dials some baked-in address
# when run with no arguments measures whatever happens to answer there.
if [ "$#" -lt 1 ]; then
    fail "usage" "bench-decode.sh HOST:PORT [N_PREDICT] [RUNS]  (--help lists the environment knobs)" >&2
    exit 2
fi
ENDPOINT="$1"
N_PREDICT="${2:-128}"
RUNS="${3:-3}"

# Both numbers are used unvalidated further down — N_PREDICT becomes the
# body's n_predict, RUNS is the `seq` bound — and neither has a sane
# reading when it is not a positive integer. A non-numeric RUNS in particular
# leaves TG_LIST/PP_LIST empty, and with `set -e` off the median lines print
# "  decode  median:  tok/s" and the script exits 0: a benchmark reporting no
# number as success.
case "$N_PREDICT" in
    ''|*[!0-9]*|0) fail "N_PREDICT='${N_PREDICT}'" "must be a positive integer" >&2; exit 2 ;;
esac
case "$RUNS" in
    ''|*[!0-9]*|0) fail "RUNS='${RUNS}'" "must be a positive integer" >&2; exit 2 ;;
esac

# The environment knobs (see the header). Validated here, before anything is
# dialled, for the same reason as the two above: PREFILL_TOKENS sizes the
# prompt, CONCURRENCY is a `seq` bound and TIMEOUT_SECONDS is curl's -m, and
# none of them has a sane reading when it is not an integer. A leading zero is
# refused as well, so `08` cannot read as eight to one consumer and as a bad
# octal number to another.
PREFILL_TOKENS="${PREFILL_TOKENS:-0}"
CONCURRENCY="${CONCURRENCY:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
HEAD_SSH="${HEAD_SSH:-}"
# $HOME, not `~`: tilde does not expand inside double quotes, so a `~` default
# would be a literal path that resolves nowhere (build-on-node.sh, same note).
HEAD_SSH_KEY="${HEAD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
HEAD_NIC="${HEAD_NIC:-}"

case "$PREFILL_TOKENS" in
    0) ;;
    *[!0-9]*|0*) fail "PREFILL_TOKENS='${PREFILL_TOKENS}'" "must be a non-negative integer" >&2; exit 2 ;;
esac
case "$CONCURRENCY" in
    *[!0-9]*|0*) fail "CONCURRENCY='${CONCURRENCY}'" "must be a positive integer" >&2; exit 2 ;;
esac
case "$TIMEOUT_SECONDS" in
    *[!0-9]*|0*) fail "TIMEOUT_SECONDS='${TIMEOUT_SECONDS}'" "must be a positive integer" >&2; exit 2 ;;
esac

# HEAD_SSH lands in ssh's argv as one "user@host" word, so a part ssh would
# read as an OPTION (a leading `-`) or one carrying anything a login name or an
# address never does is refused here rather than dialled. The interface name
# is checked the same way: it only ever reaches awk and the output, but a
# name with a space in it would match nothing and read as a missing NIC.
HEAD_USER=""
HEAD_HOST=""
if [ -n "$HEAD_SSH" ]; then
    case "$HEAD_SSH" in
        *@*) HEAD_USER="${HEAD_SSH%%@*}"; HEAD_HOST="${HEAD_SSH#*@}" ;;
    esac
    case "$HEAD_USER" in ''|-*|*[![:alnum:]._-]*) HEAD_USER="" ;; esac
    case "$HEAD_HOST" in ''|-*|*[![:alnum:].:_-]*) HEAD_HOST="" ;; esac
    if [ -z "$HEAD_USER" ] || [ -z "$HEAD_HOST" ]; then
        fail "HEAD_SSH='${HEAD_SSH}'" "must be user@host — the group head's SSH login" >&2
        exit 2
    fi
fi
case "$HEAD_NIC" in
    -*|*[![:alnum:]._@-]*) fail "HEAD_NIC='${HEAD_NIC}'" "is not an interface name" >&2; exit 2 ;;
esac

# Fixed prompt: short enough that prefill is not the story, long enough to be
# a real forward pass. Kept verbatim across runs — changing it invalidates
# comparison with every earlier measurement. PREFILL_TOKENS prepends filler to
# it and never edits it, so a fill-0 run sends exactly this text.
PROMPT="List three signs that an industrial sensor is about to fail, and explain each one."

# The two calls that need the key go through this wrapper rather than calling
# curl_tls directly. The key reaches curl through `-K` on a process
# substitution — a pipe under /dev/fd, so it lands in no file — rather than as
# a `-H` argv word: argv is readable by any local process for the life of the
# request, which on the /completion call below is up to TIMEOUT_SECONDS (1800
# by default, hours for a long fill), and it is the disclosure channel this
# codebase closes everywhere else (the supervisor hands llama-server its key
# in the environment for exactly this reason — see ARCHITECTURE.md,
# "llama-server API-key authentication"). scripts/test-all.sh's http_status
# uses the same form deliberately: two spellings of "how a bearer token
# reaches curl" would be two things to keep right. The concurrent requests
# each run this in their own subshell, so each gets its own pipe.
#
# It does NOT survive xtrace: `bash -x scripts/bench-decode.sh`, the natural
# move when a run fails, traces the printf and prints the key. Don't debug that
# way against a live fleet.
#
# The `:-` guard is required: this script runs under `set -u`, where a bare
# ${AUREAX_API_KEY} would abort rather than default. An unset key takes the
# second branch, which passes no auth option at all, so an unauthenticated
# target works exactly as before.
curl_auth() {
    if [ -n "${AUREAX_API_KEY:-}" ]; then
        curl_tls -K <(printf 'header = "Authorization: Bearer %s"\n' "${AUREAX_API_KEY}") "$@"
    else
        curl_tls "$@"
    fi
}

# Median of the arguments, lower middle for an even count. Callers guarantee
# at least one argument: `"${arr[@]}"` on an EMPTY array aborts under bash
# 3.2's `set -u`, inside a command substitution, where without `set -e` the
# script carries on and prints a median with no number in it.
median() {
    printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

# Seconds since the epoch, to the microsecond. bash 3.2 has no EPOCHREALTIME
# and BSD date no %N, so the python3 this script already requires is the
# clock. Its start-up (tens of ms) lands on every reading alike, which the
# decode window below does not notice and a decode of many seconds would not
# either.
now() {
    python3 -c 'import time; print("%.6f" % time.time())'
}

echo "=== Aureax decode benchmark ==="
echo "  endpoint:  https://${ENDPOINT}"
echo "  n_predict: ${N_PREDICT}"
echo "  runs:      ${RUNS}"
if [ "$PREFILL_TOKENS" -gt 0 ]; then
    echo "  fill:      ${PREFILL_TOKENS} tokens requested ($((PREFILL_TOKENS * 4)) characters of filler; prompt_n is the real count)"
else
    echo "  fill:      none (the bare prompt)"
fi
echo "  parallel:  ${CONCURRENCY} request(s) per run"
echo "  timeout:   ${TIMEOUT_SECONDS} s per request"
if [ -n "$HEAD_SSH" ]; then
    echo "  head ssh:  ${HEAD_SSH} (key ${HEAD_SSH_KEY})"
fi
echo

# python3 builds the request body and parses the timings. Checked here, before
# anything is dialled, because without it there is no body to send, and the
# run below would report "no response" — sending the operator to inspect a
# perfectly healthy node.
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3" "not on PATH — needed to build the request body and parse timings" >&2
    exit 1
fi

# stderr is NOT redirected here, unlike everywhere else in this script:
# `curl -sf` is silent on failure, so the only thing that can reach it is
# curl_tls's "no fleet CA at ..." line — which is the one diagnosis the
# generic "not answering" below cannot give.
if ! curl_tls -sf -m 10 "https://${ENDPOINT}/health" >/dev/null; then
    fail "https://${ENDPOINT}/health" "not answering (or the fleet CA does not trust it)" >&2
    exit 1
fi

# Report the server's own view of what it is running, so a result can never be
# attributed to the wrong model or the wrong flags after the fact.
#
# Parsed with the python3 this script already requires, not with a regex: a
# pattern like `"model_path":"..."` matches only compact JSON, so a
# pretty-printed or reshaped answer would read as no answer at all. The two
# failures are also kept apart, because they have different fixes — an empty
# HTTP response means nothing was returned to parse (unreachable, or the 401 an
# unset key earns on an auth-protected node), while a parsed body with no
# model_path means this llama.cpp build simply does not report one.
PROPS=$(curl_auth -sf -m 10 "https://${ENDPOINT}/props" 2>/dev/null || true)
MODEL=""
if [ -n "$PROPS" ]; then
    MODEL=$(printf '%s' "$PROPS" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("model_path") or "")' \
        2>/dev/null) || MODEL=""
fi
if [ -n "$MODEL" ]; then
    echo "  model:     $(basename "$MODEL")"
elif [ -z "$PROPS" ]; then
    # Said out loud: the numbers below are still real, they just carry no record
    # of which model produced them.
    warn "/props gave no answer — unreachable, or a 401 (set AUREAX_API_KEY) — result is unattributed"
else
    warn "/props carries no model_path on this build — result is unattributed"
fi

# Concurrent requests batch only when the server has a slot for each. With
# fewer, the extra requests queue and the aggregate measures a queue, so say
# so before the numbers rather than leave a concurrency row that cannot be told
# from a batching one. Silent when /props does not carry total_slots: absence
# of the field is not evidence either way.
if [ "$CONCURRENCY" -gt 1 ] && [ -n "$PROPS" ]; then
    SLOTS=$(printf '%s' "$PROPS" \
        | python3 -c 'import json,sys; v = json.load(sys.stdin).get("total_slots"); print(v if type(v) is int else "")' \
        2>/dev/null) || SLOTS=""
    if [ -n "$SLOTS" ]; then
        echo "  slots:     ${SLOTS}"
        if [ "$SLOTS" -lt "$CONCURRENCY" ]; then
            warn "the server has ${SLOTS} slot(s) (model.server.parallel_slots) — ${CONCURRENCY} requests will queue, not batch"
        fi
    fi
fi

# Scratch space for the request body, the concurrent requests' answers and the
# head's SSH stderr. mktemp -d makes it 0700, and it is removed on every exit
# path, failures included.
if ! WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/aureax-bench.XXXXXX"); then
    fail "mktemp" "could not create a scratch directory" >&2
    exit 1
fi
trap 'rm -rf "$WORK_DIR"' EXIT

# The request body, built ONCE and sent by every request of every run, so the
# workload is byte-identical across them by construction. temperature 0 and a
# fixed seed keep the token count and the work per token stable between runs.
# cache_prompt false forces a real prefill each time rather than reusing the
# previous run's KV.
#
# The filler is one fixed paragraph repeated and cut at PREFILL_TOKENS x 4
# characters — a pure function of the number, so the same fill sends the same
# bytes in every invocation, which is what lets two fill-32768 results be
# compared at all. NEVER edit the paragraph: like PROMPT, it is part of the
# workload. A fill of 0 adds nothing, so the prompt is PROMPT exactly.
#
# A file, sent with --data-binary @: a 32768 fill is ~131 KB of JSON, and argv
# is capped (1 MiB in total on macOS), while a file has no such ceiling and
# --data-binary sends it verbatim.
BODY_FILE="$WORK_DIR/body.json"
if ! python3 - "$PREFILL_TOKENS" "$N_PREDICT" "$PROMPT" > "$BODY_FILE" <<'PY'
import json
import sys

fill, n_predict, prompt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
PARAGRAPH = (
    "Maintenance logs from a water treatment plant record routine readings "
    "from pumps, valves, flow meters and dosing equipment. Each shift an "
    "operator walks the same route, notes pressures and temperatures, listens "
    "for bearings that have started to whine, and checks that alarms still "
    "reach the control room. Most entries are unremarkable, which is the "
    "point: a long record of normal behaviour is what makes an abnormal "
    "reading stand out when it finally arrives."
)
if fill > 0:
    chars = fill * 4
    unit = PARAGRAPH + " "
    filler = unit * (chars // len(unit) + 1)
    prompt = filler[:chars] + "\n\n" + prompt
json.dump(
    {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": False,
    },
    sys.stdout,
)
PY
then
    fail "request body" "python3 could not build it (see the error above)" >&2
    exit 1
fi

# One /completion request: the body, then a last line carrying the HTTP status.
#
# The status is captured alongside the body instead of using `curl -f`, which
# collapses every HTTP error into an empty body and exit 22. That made a 401
# from an auth-protected node — the one failure this script can tell the
# operator how to fix — read as a server timeout. `%{http_code}` is 000 when
# curl never got a response at all.
completion_request() {
    curl_auth -s -m "$TIMEOUT_SECONDS" -w '\n%{http_code}' "https://${ENDPOINT}/completion" \
        -H 'Content-Type: application/json' \
        --data-binary "@${BODY_FILE}" 2>/dev/null
}

# Check one answer as completion_request prints it, and leave its six timing
# fields in TIMINGS; or report against <label> ("run 2", or "run 2 req 1") and
# return 1, which every caller turns into exit 1. Not called inside a command
# substitution, so TIMINGS reaches the caller.
parse_completion() {
    local label="$1" resp="$2" code hint
    code="${resp##*$'\n'}"
    resp="${resp%$'\n'*}"

    if [ -z "$code" ] || [ "$code" = "000" ]; then
        fail "$label" "no response (timeout or connection error; TIMEOUT_SECONDS=${TIMEOUT_SECONDS})" >&2
        return 1
    fi
    if [ "$code" != "200" ]; then
        hint=""
        if [ "$code" = "401" ]; then
            hint=" — set AUREAX_API_KEY to the node's key (~/.aureax/keys/<node-id>.key)"
        fi
        fail "$label" "/completion answered HTTP ${code}${hint}" >&2
        return 1
    fi
    if [ -z "$resp" ]; then
        fail "$label" "empty body from /completion" >&2
        return 1
    fi

    # Parsed into a variable and CHECKED before the read. A non-empty body that
    # does not parse — curl -m truncating mid-stream, an HTML error page — left
    # the substitution empty; `read` then assigned every field the empty
    # string without tripping `set -u`, the run printed
    # "decode  tok/s ( tok, n/a ms/tok)", and the empty element went into
    # TG_LIST/PP_LIST. `sort -g` orders empties first, so one could land on the
    # median index and be reported as the measurement, with the script exiting
    # 0. A benchmark that cannot parse its own result has no number to report.
    if ! TIMINGS=$(printf '%s' "$resp" | python3 -c '
import json, sys
t = json.load(sys.stdin).get("timings")
if not t:
    sys.exit("no timings in /completion response")
print(t.get("prompt_n", 0), round(t.get("prompt_ms", 0), 1), round(t.get("prompt_per_second", 0) or 0, 3),
      t.get("predicted_n", 0), round(t.get("predicted_ms", 0), 1), round(t.get("predicted_per_second", 0) or 0, 3))
'); then
        fail "$label" "could not parse the /completion response (see the error above)" >&2
        return 1
    fi
}

# Print one request's line from TIMINGS and record it: TG_LIST/PP_LIST for the
# final medians, RUN_N for the run's NIC division. The fields stay global so
# the concurrent branch can read TG_N/TG_MS/TG_PS straight after.
report_request() {
    local label="$1"
    read -r PP_N PP_MS PP_PS TG_N TG_MS TG_PS <<< "$TIMINGS"
    MS_PER_TOK=$(awk -v ms="$TG_MS" -v n="$TG_N" 'BEGIN{ if (n>0) printf "%.1f", ms/n; else print "n/a" }')
    echo "  ${label}: decode ${TG_PS} tok/s (${TG_N} tok, ${MS_PER_TOK} ms/tok)  |  prefill ${PP_PS} tok/s (${PP_N} tok, ${PP_MS} ms)"
    TG_LIST+=("$TG_PS")
    PP_LIST+=("$PP_PS")
    RUN_N=$((RUN_N + TG_N))
}

# ---- Head NIC counters (HEAD_SSH) -----------------------------------------
#
# Best effort by design: the benchmark is the deliverable, so nothing here ever
# exits. The first failure of any kind warns once, switches NIC numbers off for
# the rest of the benchmark, and the runs carry on.
#
# Both remote commands are POSIX on purpose — the node account's login shell
# is not pinned (scripts/lib/common.sh, at ssh_cmd) — and both answers are
# PARSED here, never eval'd: they are whatever the node printed (the reasoning
# is spelled out at detect_node_arch in scripts/lib/common.sh). stdin is
# /dev/null so ssh cannot read the terminal, and stderr is kept in a file so a
# healthy run is not interleaved with ssh's notices while a failure can still
# show why.
NIC_ON=0
NIC_IF=""
NIC_SAMPLE=""
NIC_TX_LIST=()
NIC_RX_LIST=()

head_ssh() {
    ssh_cmd "$HEAD_HOST" "$HEAD_USER" "$HEAD_SSH_KEY" "$1" </dev/null 2>"$WORK_DIR/ssh.err"
}

# Switch NIC numbers off with one warning. A second argument of `ssh` means the
# failure was ssh's own, so its stderr is the explanation and is shown.
nic_off() {
    warn "head NIC: $1 — continuing without NIC numbers"
    if [ "${2:-}" = "ssh" ] && [ -s "$WORK_DIR/ssh.err" ]; then
        sed 's/^/         /' "$WORK_DIR/ssh.err"
    fi
    NIC_ON=0
}

# "<rx bytes> <tx bytes>" for NIC_IF, from the head's /proc/net/dev. Returns
# ssh's 255 when the connection itself failed, 1 when the interface is not in
# the answer. Lines read `  eth0: <8 receive fields> <8 transmit fields>`, and
# an old kernel prints a wide counter flush against the colon, hence the split
# on `:` rather than on whitespace.
head_counters() {
    local dev rc=0
    dev=$(head_ssh 'cat /proc/net/dev') || rc=$?
    [ "$rc" -eq 0 ] || return "$rc"
    printf '%s\n' "$dev" | awk -v ifc="$NIC_IF" '
        { line = $0; sub(/^[ \t]+/, "", line); c = index(line, ":") }
        c > 0 && substr(line, 1, c - 1) == ifc {
            n = split(substr(line, c + 1), f, " ")
            if (n >= 9 && f[1] ~ /^[0-9]+$/ && f[9] ~ /^[0-9]+$/) { print f[1], f[9]; found = 1; exit }
        }
        END { if (!found) exit 1 }'
}

# Take a sample into NIC_SAMPLE, or switch NIC numbers off. Called directly,
# never in a command substitution, so nic_off's NIC_ON=0 reaches this shell.
nic_sample() {
    local rc=0
    NIC_SAMPLE=$(head_counters) || rc=$?
    case "$rc" in
        0) return 0 ;;
        255) nic_off "ssh ${HEAD_SSH} failed" ssh ;;
        *) nic_off "no '${NIC_IF}' counters in /proc/net/dev on ${HEAD_HOST} — set HEAD_NIC" ;;
    esac
    return 1
}

# One run's line: the counter deltas over the run's total predicted_n. awk,
# not $(( )), for the arithmetic: the counters are 64-bit and the division is
# fractional; awk's doubles are exact to 2^53 bytes.
nic_report() {
    local run="$1" before="$2" after="$3" n="$4" rx0 tx0 rx1 tx1 out tx rx txmb rxmb
    read -r rx0 tx0 <<< "$before"
    read -r rx1 tx1 <<< "$after"
    out=$(awk -v rx0="$rx0" -v tx0="$tx0" -v rx1="$rx1" -v tx1="$tx1" -v n="$n" 'BEGIN {
        drx = rx1 - rx0; dtx = tx1 - tx0
        if (drx < 0 || dtx < 0) { print "backwards"; exit }
        if (n <= 0) { print "none"; exit }
        printf "%.1f %.1f %.1f %.1f\n", dtx / 1000 / n, drx / 1000 / n, dtx / 1e6, drx / 1e6
    }')
    case "$out" in
        backwards)
            nic_off "the ${NIC_IF} counters on ${HEAD_HOST} went backwards (interface reset?)" ;;
        none)
            echo "  run ${run} head NIC: no tokens decoded — nothing to divide by" ;;
        *)
            read -r tx rx txmb rxmb <<< "$out"
            echo "  run ${run} head NIC: tx ${tx} KB/token, rx ${rx} KB/token (${txmb} MB tx, ${rxmb} MB rx over ${n} tok)"
            NIC_TX_LIST+=("$tx")
            NIC_RX_LIST+=("$rx")
            ;;
    esac
}

# Which interface. `ip route get` asks the routing table without sending a
# packet; the full paths cover an ssh PATH without the sbin directories. Only
# ssh's own 255 counts as an SSH failure — any other status means the head
# answered and simply named no interface.
if [ -n "$HEAD_SSH" ]; then
    NIC_ON=1
    if [ -n "$HEAD_NIC" ]; then
        NIC_IF="$HEAD_NIC"
        echo "  head NIC:  ${NIC_IF} on ${HEAD_SSH} (HEAD_NIC)"
    else
        ROUTE_RC=0
        ROUTE=$(head_ssh 'ip route get 1.1.1.1 2>/dev/null || /usr/sbin/ip route get 1.1.1.1 2>/dev/null || /sbin/ip route get 1.1.1.1') \
            || ROUTE_RC=$?
        if [ "$ROUTE_RC" -eq 255 ]; then
            nic_off "ssh ${HEAD_SSH} failed" ssh
        else
            NIC_IF=$(printf '%s\n' "$ROUTE" | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -n 1)
            case "$NIC_IF" in
                ''|-*|*[![:alnum:]._@-]*)
                    nic_off "could not tell which interface carries the default route on ${HEAD_HOST} — set HEAD_NIC" ;;
                *)
                    echo "  head NIC:  ${NIC_IF} on ${HEAD_SSH} (the default-route interface)" ;;
            esac
        fi
    fi
fi
echo

TG_LIST=()
PP_LIST=()
AGG_LIST=()

for run in $(seq 1 "$RUNS"); do
    # Last run's answers must not stand in for this run's: a request that
    # never wrote its file would otherwise be read from the run before.
    rm -f "$WORK_DIR"/resp.* "$WORK_DIR"/end.*
    RUN_N=0

    # Sampled immediately around the requests, so the SSH round trips fall
    # outside the timed window and only their own few KB inside the counted one.
    NIC_BEFORE=""
    if [ "$NIC_ON" -eq 1 ] && nic_sample; then
        NIC_BEFORE="$NIC_SAMPLE"
    fi

    if [ "$CONCURRENCY" -eq 1 ]; then
        RESP=$(completion_request)
        parse_completion "run ${run}" "$RESP" || exit 1
        report_request "run ${run}"
    else
        # Every request in its own background subshell, answer and finish time
        # to files, then wait for all of them: the aggregate needs each one's
        # finish, and a request that fails is reported after the others land
        # rather than leaving them running behind an exit.
        T0=$(now)
        PIDS=()
        for req in $(seq 1 "$CONCURRENCY"); do
            (
                completion_request > "$WORK_DIR/resp.${req}"
                now > "$WORK_DIR/end.${req}"
            ) &
            PIDS+=("$!")
        done
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done

        RUN_TG_LIST=()
        AGG_ARGS=()
        for req in $(seq 1 "$CONCURRENCY"); do
            RESP=$(cat "$WORK_DIR/resp.${req}" 2>/dev/null)
            parse_completion "run ${run} req ${req}" "$RESP" || exit 1
            report_request "run ${run} req ${req}"
            RUN_TG_LIST+=("$TG_PS")
            AGG_ARGS+=("$TG_N" "$TG_MS" "$(cat "$WORK_DIR/end.${req}" 2>/dev/null)")
        done

        # The decode window runs from the earliest decode start (a finish minus
        # its own predicted_ms) to the latest finish, so the prefill before it
        # is not in the denominator — at a 32768 fill that prefill is most of
        # the wall clock. It is floored at the longest single decode, so clock
        # jitter can never make the window shorter than a decode inside it.
        # The end-to-end figure, prefill included, divides by the wall clock
        # from the first request fired to the slowest one finished.
        if ! AGG=$(python3 -c '
import sys
t0, v = float(sys.argv[1]), sys.argv[2:]
reqs = [(int(v[i]), float(v[i + 1]) / 1000.0, float(v[i + 2])) for i in range(0, len(v), 3)]
total = sum(n for n, _, _ in reqs)
last = max(end for _, _, end in reqs)
window = max(last - min(end - secs for _, secs, end in reqs), max(secs for _, secs, _ in reqs))
wall = last - t0
print(total, round(total / window, 3) if window > 0 else 0, round(window, 1),
      round(total / wall, 3) if wall > 0 else 0, round(wall, 1))
' "$T0" "${AGG_ARGS[@]}"); then
            fail "run ${run}" "could not compute the aggregate (see the error above)" >&2
            exit 1
        fi
        read -r AGG_N AGG_PS AGG_WINDOW AGG_E2E_PS AGG_WALL <<< "$AGG"
        echo "  run ${run} aggregate: decode ${AGG_PS} tok/s (${AGG_N} tok over a ${AGG_WINDOW} s decode window)  |  end to end ${AGG_E2E_PS} tok/s (${AGG_WALL} s wall)  |  per-request decode median $(median "${RUN_TG_LIST[@]}") tok/s"
        AGG_LIST+=("$AGG_PS")
    fi

    if [ -n "$NIC_BEFORE" ] && [ "$NIC_ON" -eq 1 ] && nic_sample; then
        nic_report "$run" "$NIC_BEFORE" "$NIC_SAMPLE" "$RUN_N"
    fi
done

echo
if [ "${#TG_LIST[@]}" -eq 0 ]; then
    fail "benchmark" "no runs completed (RUNS=${RUNS})" >&2
    exit 1
fi
# At CONCURRENCY=1 these two lines are byte-for-byte the format they have
# always had; above it they say what the median is over, and the aggregate
# gets its own line.
if [ "$CONCURRENCY" -eq 1 ]; then
    printf '  decode  median: %s tok/s\n' "$(median "${TG_LIST[@]}")"
    printf '  prefill median: %s tok/s\n' "$(median "${PP_LIST[@]}")"
else
    printf '  decode  median: %s tok/s per request (%s requests)\n' "$(median "${TG_LIST[@]}")" "${#TG_LIST[@]}"
    printf '  prefill median: %s tok/s per request\n' "$(median "${PP_LIST[@]}")"
    printf '  aggregate decode median: %s tok/s at concurrency %s\n' "$(median "${AGG_LIST[@]}")" "$CONCURRENCY"
fi
# Only when every run was counted: a median over the runs that happened to
# precede an SSH failure would read as covering them all.
if [ "$NIC_ON" -eq 1 ] && [ "${#NIC_TX_LIST[@]}" -gt 0 ]; then
    printf '  head NIC: tx %s KB/token, rx %s KB/token (interface %s)\n' \
        "$(median "${NIC_TX_LIST[@]}")" "$(median "${NIC_RX_LIST[@]}")" "$NIC_IF"
fi
