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
# format) — and curl_tls, which is how every request here reaches the node over
# TLS verified against the fleet CA. Note this script keeps `set -uo pipefail`
# WITHOUT -e: a run that fails is reported and exited explicitly below.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

# The endpoint has no default: a benchmark that dials some baked-in address
# when run with no arguments measures whatever happens to answer there.
if [ "$#" -lt 1 ]; then
    fail "usage" "bench-decode.sh HOST:PORT [N_PREDICT] [RUNS]" >&2
    exit 2
fi
ENDPOINT="$1"
N_PREDICT="${2:-128}"
RUNS="${3:-3}"

# Both numbers are used unvalidated further down — N_PREDICT is interpolated
# raw into the JSON body, RUNS is the `seq` bound — and neither has a sane
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

# Fixed prompt: short enough that prefill is not the story, long enough to be
# a real forward pass. Kept verbatim across runs — changing it invalidates
# comparison with every earlier measurement.
PROMPT="List three signs that an industrial sensor is about to fail, and explain each one."

# The two calls that need the key go through this wrapper rather than calling
# curl_tls directly. The key reaches curl through `-K` on a process
# substitution — a pipe under /dev/fd, so it lands in no file — rather than as
# a `-H` argv word: argv is readable by any local process for the life of the
# request, which on the /completion call below is up to the 1800s `-m`, and it
# is the disclosure channel this codebase closes everywhere else (the
# supervisor hands llama-server its key in the environment for exactly this
# reason — see ARCHITECTURE.md, "llama-server API-key authentication").
# scripts/test-all.sh's http_status uses the same form deliberately: two
# spellings of "how a bearer token reaches curl" would be two things to keep
# right.
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

echo "=== Aureax decode benchmark ==="
echo "  endpoint:  https://${ENDPOINT}"
echo "  n_predict: ${N_PREDICT}"
echo "  runs:      ${RUNS}"
echo

# python3 builds the request body and parses the timings. Checked here, before
# anything is dialled, because without it the body becomes `"prompt": ,`,
# llama-server answers 400, and the run below reports "no response (timeout or
# server error)" — sending the operator to inspect a perfectly healthy node.
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
echo

# Hoisted out of the run loop: PROMPT is a constant, so re-encoding it per run
# spawned a second python3 per iteration to rebuild a byte-identical string.
PROMPT_JSON=$(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')

TG_LIST=()
PP_LIST=()

for run in $(seq 1 "$RUNS"); do
    # temperature 0 and a fixed seed keep the token count and the work per
    # token stable between runs. cache_prompt false forces a real prefill each
    # time rather than reusing the previous run's KV.
    #
    # The status is captured alongside the body instead of using `curl -f`,
    # which collapses every HTTP error into an empty body and exit 22. That
    # made a 401 from an auth-protected node — the one failure this script can
    # tell the operator how to fix — read as a server timeout. `%{http_code}`
    # is 000 when curl never got a response at all.
    RESP=$(curl_auth -s -m 1800 -w '\n%{http_code}' "https://${ENDPOINT}/completion" \
        -H 'Content-Type: application/json' \
        -d "{
              \"prompt\": ${PROMPT_JSON},
              \"n_predict\": ${N_PREDICT},
              \"temperature\": 0,
              \"seed\": 42,
              \"cache_prompt\": false
            }" 2>/dev/null)
    HTTP_CODE="${RESP##*$'\n'}"
    RESP="${RESP%$'\n'*}"

    if [ "$HTTP_CODE" = "000" ]; then
        fail "run ${run}" "no response (timeout or connection error)" >&2
        exit 1
    fi
    if [ "$HTTP_CODE" != "200" ]; then
        HINT=""
        [ "$HTTP_CODE" = "401" ] && HINT=" — set AUREAX_API_KEY to the node's key (~/.aureax/keys/<node-id>.key)"
        fail "run ${run}" "/completion answered HTTP ${HTTP_CODE}${HINT}" >&2
        exit 1
    fi
    if [ -z "$RESP" ]; then
        fail "run ${run}" "empty body from /completion" >&2
        exit 1
    fi

    # Parsed into a variable and CHECKED before the read. A non-empty body that
    # does not parse — curl -m 1800 truncating mid-stream, an HTML error page —
    # left the substitution empty; `read` then assigned every field the empty
    # string without tripping `set -u`, the run printed
    # "decode  tok/s ( tok, n/a ms/tok)", and the empty element went into
    # TG_LIST/PP_LIST. `sort -g` orders empties first, so one could land on the
    # median index and be reported as the measurement, with the script exiting
    # 0. A benchmark that cannot parse its own result has no number to report.
    if ! TIMINGS=$(printf '%s' "$RESP" | python3 -c '
import json, sys
t = json.load(sys.stdin).get("timings")
if not t:
    sys.exit("no timings in /completion response")
print(t.get("prompt_n", 0), round(t.get("prompt_ms", 0), 1), round(t.get("prompt_per_second", 0) or 0, 3),
      t.get("predicted_n", 0), round(t.get("predicted_ms", 0), 1), round(t.get("predicted_per_second", 0) or 0, 3))
'); then
        fail "run ${run}" "could not parse the /completion response (see the error above)" >&2
        exit 1
    fi
    read -r PP_N PP_MS PP_PS TG_N TG_MS TG_PS <<< "$TIMINGS"

    MS_PER_TOK=$(awk -v ms="$TG_MS" -v n="$TG_N" 'BEGIN{ if (n>0) printf "%.1f", ms/n; else print "n/a" }')
    echo "  run ${run}: decode ${TG_PS} tok/s (${TG_N} tok, ${MS_PER_TOK} ms/tok)  |  prefill ${PP_PS} tok/s (${PP_N} tok, ${PP_MS} ms)"

    TG_LIST+=("$TG_PS")
    PP_LIST+=("$PP_PS")
done

echo
# `printf '%s\n' "${TG_LIST[@]}"` on an EMPTY array aborts under bash 3.2's
# `set -u` — inside a command substitution, where without `set -e` the script
# carries on and prints a median with no number in it.
if [ "${#TG_LIST[@]}" -eq 0 ]; then
    fail "benchmark" "no runs completed (RUNS=${RUNS})" >&2
    exit 1
fi
printf '  decode  median: %s tok/s\n' "$(printf '%s\n' "${TG_LIST[@]}" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
printf '  prefill median: %s tok/s\n' "$(printf '%s\n' "${PP_LIST[@]}" | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')"
