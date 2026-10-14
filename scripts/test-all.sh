#!/usr/bin/env bash
# ==========================================================================
# Aureax Prototype — Comprehensive Test Suite
#
# Tests every layer of the system. Sections and what gates them:
#   0. Unit tests (pytest)        — always runs, needs the dev group installed
#   1. Orchestrator (config, CLI) — always runs, no infra needed
#   2. Infrastructure (SSH, K3s)  — gated on kubectl reachability
#   3. Dashboard API              — gated on the dashboard being reachable
#   4. MCP endpoints + auth       — gated on an MCP node answering /health
#   5. Cognitive planner          — gated on a planner target existing
#   6. Sharded groups             — gated on kubectl AND a non-empty node_groups
#   7. llama-server API auth      — gated on kubectl AND an auth-enabled node
#                                   whose key file exists on the orchestrator
#
# Three checks for key material at rest run after section 7 at file scope,
# because none of them needs a cluster: two for the shipped logs and one for
# the permissions on the generated configs.
#
# A gated-out section emits `skip` lines rather than passing silently. Those
# labels are indicative of what the section covers, not one per check: sections
# 6 and 7 run a set of checks per group and per node.
#
# Node/group config is read from configs/aureax-network.yaml, falling back to
# the shipped configs/aureax-network.example.yaml when an operator copy does
# not exist; the dashboard port is assumed to be the schema default
# (server.dashboard.port, 7001).
#
# Every endpoint this suite dials over the network is HTTPS, verified against
# the fleet CA at $AUREAX_CA (~/.aureax/tls/ca.pem by default) through curl_tls
# in scripts/lib/common.sh. There is no plaintext fallback and no -k anywhere:
# a check that accepted any certificate would stop distinguishing "the fleet
# answered" from "something answered", which is most of what TLS buys here. A
# missing CA makes every such check fail with curl's status 77 and a line
# naming `aureax generate`.
#
# The single exception is section 3, which talks to the dashboard's loopback
# listener over plain HTTP — see DASHBOARD_URL there. That listener is bound to
# 127.0.0.1/::1 only, so it is not on the LAN at all.
#
# THIS SUITE WRITES TO ~/.aureax. Section 1 runs `aureax init` and `aureax
# generate` for real, against the operator's own HOME whenever
# configs/aureax-network.yaml exists: that rewrites ~/.aureax/configs for the
# whole fleet and re-stamps every node's telemetry endpoint with the LAN
# address detected now. A run on a Mac with no detectable LAN address stamps
# 127.0.0.1, which leaves every node pointing telemetry at its own loopback
# until `uv run aureax generate --orchestrator-ip <MAC_LAN_IP>` is run again.
# A checkout with no operator config generates into a throwaway HOME instead
# and touches nothing.
#
# Prerequisites:
#   - Orchestrator tests: just uv + project deps
#   - Infra tests: setup-all.sh completed, K3s running
#   - Dashboard tests: `uv run aureax dashboard` running in another terminal
#   - MCP tests: pods Running+Ready on edge nodes
#   - Anything past section 1: `uv run aureax generate` has minted the fleet CA
#
# Usage: ./scripts/test-all.sh
# ==========================================================================
set -euo pipefail

# ssh_cmd, ssh_facts, node_index_of, detect_mac_ip, load_network_env,
# section_banner, mcp_health_wait, curl_tls (with $AUREAX_CA) and the shared
# `warn` helper. Sourced FIRST on purpose — see the counting overrides below,
# which deliberately replace `fail` with a one-argument counting variant.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

PROJECT_DIR="$AUREAX_ROOT"
cd "$PROJECT_DIR"

# Where `aureax generate` writes the per-node keys
# (orchestrator/config/keystore.py): <node_id>.key for llama-server,
# <node_id>.mcp.key for the MCP server. Defined here rather than in section 7,
# which used to own it, because section 4 now reads the MCP keys too and every
# section runs under `set -u`. No key material is ever read outside the two
# `tr -d` lines at the point of use.
KEY_DIR="$HOME/.aureax/keys"

export PATH="$HOME/.local/bin:$PATH"

# --- Counters ---
PASS=0
FAIL=0
SKIP=0

# --- Read config from YAML ---
#
# Which fleet definition to read. The operator's own
# configs/aureax-network.yaml carries SSH credentials and real addresses and is
# therefore not part of the repository, so a fresh clone has only the shipped
# example. Resolving one here — and passing it to every parse-config.py call,
# every `aureax` invocation and the section-5 probe — is what lets sections 0
# and 1, which need no fleet at all, run on a clone.
NETWORK_CONFIG="configs/aureax-network.yaml"
# Where `aureax generate` writes. The example shares node ids with a real fleet
# (shard-head, planner-01, ...), so when the suite falls back to it the resolved
# configs go into a throwaway HOME — otherwise a run on a machine whose real
# fleet file is merely absent for the moment would overwrite ~/.aureax/configs
# with documentation-range addresses that every later start would then push.
GEN_HOME="$HOME"
GEN_TMP_HOME=""
if [ ! -f "$NETWORK_CONFIG" ]; then
    if [ -f "configs/aureax-network.example.yaml" ]; then
        NETWORK_CONFIG="configs/aureax-network.example.yaml"
        GEN_TMP_HOME="$(mktemp -d)"
        GEN_HOME="$GEN_TMP_HOME"
        echo "  (no configs/aureax-network.yaml — reading ${NETWORK_CONFIG};"
        echo "   copy it to configs/aureax-network.yaml and edit it to test a real fleet;"
        echo "   generated configs go to a temporary HOME, not ~/.aureax)"
    else
        echo "  no fleet definition found (configs/aureax-network.yaml or"
        echo "  configs/aureax-network.example.yaml) — cannot run any section"
        exit 1
    fi
fi

# --- What this run must put back, however it ends ---------------------------
#
# ONE cleanup function and ONE pair of traps for the whole suite, armed here
# because the first thing they protect (the fleet definition, in section 3)
# runs long before the last (the groups error file, in section 6). Each entry
# is written to tolerate never having been set: a section that was gated out
# leaves its variable empty and the cleanup skips it.
#
# The signal handlers exit explicitly. A trap that only cleans up would swallow
# the interrupt and let the suite carry on.
FLEET_FILE="configs/aureax-network.yaml"
FLEET_BACKUP=""      # a copy of $FLEET_FILE while a write check is in flight
FLEET_EXISTED=0

restore_fleet_file() {
    # Put the operator's fleet definition back if a check changed it, then drop
    # the backup. Idempotent and quiet when nothing moved, so the traps and the
    # inline call after the check can both run it.
    [ -n "$FLEET_BACKUP" ] || return 0
    if [ "$FLEET_EXISTED" -eq 1 ]; then
        if [ -f "$FLEET_BACKUP" ] && ! cmp -s "$FLEET_BACKUP" "$FLEET_FILE"; then
            cp "$FLEET_BACKUP" "$FLEET_FILE"
            warn "restored ${FLEET_FILE} from the pre-check backup"
        fi
    elif [ -f "$FLEET_FILE" ]; then
        # There was no operator config before this run, so a file here is one a
        # check created — and it would silently become the fleet definition
        # every later run reads.
        rm -f "$FLEET_FILE"
        warn "removed ${FLEET_FILE}, which a check created"
    fi
    rm -f "$FLEET_BACKUP"
    FLEET_BACKUP=""
}

suite_cleanup() {
    restore_fleet_file
    rm -f "${GROUPS_ERRFILE:-}"
    rm -rf "${GEN_TMP_HOME:-}"
}
trap suite_cleanup EXIT
trap 'suite_cleanup; exit 130' INT TERM

# load_network_env command-substitutes first and then evals, so it reports the
# CHILD's status. `eval "$(...)"` reports the status of *eval*, which is 0 for
# the empty string a failed parse-config.py leaves behind — so under `set -e` a
# config error would sail past this line and surface much later as an unbound
# NODE_COUNT, after a full pytest run and with no Results summary at all.
# parse-config.py's stderr is not captured, so its traceback (and its non-fatal
# `#`-prefixed notes) reach the terminal directly.
if ! load_network_env all "$NETWORK_CONFIG"; then
    echo "  parse-config.py all FAILED — cannot run any section (see above)"
    exit 1
fi
# `ssh.targets` has no minimum length and every target may be `enabled: false`,
# so NODE_COUNT can be 0. Caught here because every loop below expands a NODE_*
# element — an unbound-variable abort under `set -u` — and BSD `seq 0 -1` prints
# "0" and "-1" rather than iterating zero times, so the run would die mid-suite
# with no Results summary. Same guard update-all.sh and download-model.sh carry.
if [ "${NODE_COUNT:-0}" -eq 0 ]; then
    fail "config" "no enabled SSH targets in ${NETWORK_CONFIG} — nothing to test"
    exit 1
fi

# --- Helpers ---
#
# The four below (check/check_output/skip/fail) deliberately shadow or extend
# what scripts/lib/common.sh provides: this script counts results, so its
# `fail` takes ONE argument (a description) and bumps the FAIL counter, where
# the shared `fail` takes two (what, why) and only prints. Last definition
# wins, which is why the source line above comes first. Everything else —
# ssh_cmd, node_index_of, load_network_env, and section_banner (called with an
# explicit rule width of 40, this file's narrower heading) — is used as
# shipped.

# Run a command silently. Pass = exit 0, Fail = nonzero.
check() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name"
        FAIL=$((FAIL + 1))
    fi
}

# Run a command, check that stdout contains a substring.
check_output() {
    local name="$1" expected="$2"; shift 2
    local output
    output=$("$@" 2>&1) || true
    if echo "$output" | grep -q "$expected"; then
        echo "  [PASS] $name"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $name — expected '$expected' in output"
        FAIL=$((FAIL + 1))
    fi
}

# One request, authenticated or not, compared against an expected HTTP status.
#
#   http_status <name> <want> <url> [key] [-- extra curl args...]
#
# Two sections need it: section 4's POST /mcp probes and section 7's llama-server
# GETs. It lives up here with check_output rather than inside either of them
# because a second copy is how the two would drift into disagreeing about how a
# bearer token reaches curl — and that is the part that has to be right.
#
# Status code rather than body: both servers answer a 401 with a JSON error
# body, so a body-pattern check cannot tell "auth is on" from "the node is
# down" — and telling those apart is the whole job.
#
# The key reaches curl through `-K` on a process substitution — a pipe under
# /dev/fd, so it lands in no file — rather than as a `-H` argv word. argv is
# readable by any local process for the life of the request, which is the
# disclosure channel this codebase closes everywhere else — the supervisor
# passes LLAMA_API_KEY to llama-server in its environment for the same reason
# (see ARCHITECTURE.md, "llama-server API-key authentication"). It does NOT
# survive xtrace: `bash -x scripts/test-all.sh`, the natural move when either
# section fails, traces the printf below and prints the key. Don't run the
# suite that way against a live fleet. The `--cacert` curl_tls adds is a PATH,
# not a secret, so TLS changes nothing about that exposure either way.
#
# The unauthenticated call passes no config at all, so its branch omits -K
# rather than passing an empty one.
#
# Anything after the key (with an optional `--` separator, which is dropped) is
# handed to curl verbatim, before the URL: that is how the POST bodies and
# Accept headers of the MCP checks get in without a second helper.
http_status() {
    local name="$1" want="$2" url="$3" key="${4:-}"
    if [ "$#" -ge 4 ]; then shift 4; else shift "$#"; fi
    if [ "${1:-}" = "--" ]; then shift; fi
    local got
    if [ -n "$key" ]; then
        got=$(curl_tls -s -o /dev/null -w "%{http_code}" --max-time 15 \
                  -K <(printf 'header = "Authorization: Bearer %s"\n' "$key") \
                  "$@" "$url" 2>/dev/null) || got="000"
    else
        got=$(curl_tls -s -o /dev/null -w "%{http_code}" --max-time 15 \
                  "$@" "$url" 2>/dev/null) || got="000"
    fi
    if [ "$got" = "$want" ]; then
        echo "  [PASS] ${name}"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] ${name} — expected HTTP ${want}, got ${got}"
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    echo "  [SKIP] $1"
    SKIP=$((SKIP + 1))
}

# Fold `PASS|name|detail` / `FAIL|name|detail` lines from a helper that tests
# several things in one process into the counters.
#
# `status` is the helper's own exit status and is not optional. A helper that
# dies before printing anything — a renamed import, a bad argv — emits only a
# traceback, which parses as zero results: every one of its tests disappears
# without moving a single counter and the suite can still print ALL TESTS
# PASSED. Same hole section 0 guards against for pytest's summary line.
count_results() {
    local label="$1" status="$2" output="$3"
    local seen=0 failed=0 st name detail
    while IFS='|' read -r st name detail; do
        if [ "$st" = "PASS" ]; then
            echo "  [PASS] ${name}${detail:+ — ${detail}}"
            PASS=$((PASS + 1)); seen=$((seen + 1))
        elif [ "$st" = "FAIL" ]; then
            echo "  [FAIL] ${name}${detail:+ — ${detail}}"
            FAIL=$((FAIL + 1)); seen=$((seen + 1)); failed=$((failed + 1))
        fi
    done <<< "$output"
    if [ "$seen" -eq 0 ]; then
        fail "${label} (helper produced no results, exit ${status})"
        echo "$output" | sed 's/^/    /'
    elif [ "$status" -ne 0 ] && [ "$failed" -eq 0 ]; then
        fail "${label} (helper exited ${status} after reporting no failures)"
        echo "$output" | sed 's/^/    /'
    fi
}

# Record a failure without running anything — for when the harness itself
# could not get far enough to test. A section that cannot run must not be
# reported as a section with nothing to test.
#
# DELIBERATE OVERRIDE of scripts/lib/common.sh's `fail`, which takes (what,
# why) and only prints. Here a failure has to move the counter, and every call
# site in this file passes one argument. Defined after the `source` line at the
# top of the file so this definition wins.
fail() {
    echo "  [FAIL] $1"
    FAIL=$((FAIL + 1))
}

# One skip list per gated section, defined once and called from each of that
# section's two or three gate branches, so a gated-out section cannot pass
# silently and the branches cannot drift apart from each other.
#
# The labels are INDICATIVE, not per-check. Sections 6 and 7 emit a set of
# checks per group and per node, so no fixed list can match their counts; what
# these lists guarantee is that a section which did not run says so.
skip_dashboard_checks() {
    skip "GET /api/cluster"
    skip "GET /api/pods"
    skip "GET /api/nodes"
    skip "POST /api/v1/telemetry"
    skip "POST /api/v1/logs"
    skip "GET /api/lifecycle/state"
    skip "GET /api/lifecycle/nodes"
    skip "GET /api/access/nodes"
    skip "/api/access/nodes carries no plaintext key"
    skip "GET /api/config"
    skip "GET /api/config/<file>"
    skip "GET /api/config/<unknown> 404"
    skip "PUT invalid YAML rejected"
}

skip_shard_checks() {
    skip "Shard DaemonSet exists"
    skip "Shard worker pods Ready"
    skip "Worker RPC listeners"
    skip "Head llama endpoint"
    skip "Workers holding weights (rpc-server RSS)"
    skip "llama.cpp build consistency"
    skip "Sharded inference"
}

skip_auth_checks() {
    skip "/health public"
    skip "/v1/models unauthenticated rejected"
    skip "/props unauthenticated rejected"
    skip "/props with key accepted"
    skip "/metrics with key accepted"
    skip "curl without --cacert is refused"
    # The three key-material-at-rest checks are NOT listed here: they run at
    # file scope, below the section, whatever this gate decides. Listing them
    # would report each of them twice on a gated-out run.
}

# ssh_cmd and node_index_of come from scripts/lib/common.sh. The group sections
# below know hosts, not indices, and `user`, `key_file` and `port` are all PER
# TARGET in the schema — reaching every group member with node 0's credentials
# fails auth the moment a group spans nodes with different logins, and
# `2>/dev/null` hides the reason.

# ssh_cmd against a node identified by its index, using that node's own
# credentials — never another node's, which is why this takes an index rather
# than a host.
#
# An unknown host returns 1 with a message on stderr. Note that the section-6
# callers all redirect stderr to /dev/null to suppress ssh's own noise, so what
# they actually observe is the empty output and nonzero status, not the text.
ssh_at() {
    local i="$1"; shift
    if [ "$i" -lt 0 ]; then
        echo "node_index_of: host not found in NODE_HOSTS" >&2
        return 1
    fi
    ssh_cmd "${NODE_HOSTS[$i]}" "${NODE_USERS[$i]}" "${NODE_KEY_FILES[$i]}" "$@"
}

# scripts/lib/common.sh::ssh_facts against a node index, same reasoning as
# ssh_at. Section 6 asks each shard worker three one-line questions; without
# this that is three full handshakes per worker, eighteen for the current two
# groups, because there is no connection multiplexing here.
ssh_facts_at() {
    local i="$1"; shift
    if [ "$i" -lt 0 ]; then
        echo "node_index_of: host not found in NODE_HOSTS" >&2
        return 1
    fi
    ssh_facts "${NODE_HOSTS[$i]}" "${NODE_USERS[$i]}" "${NODE_KEY_FILES[$i]}" "$@"
}

# The image architecture and the recorded llama.cpp commit out of a SUM_CMD
# blob (section 6). Each fact sits on its own tagged line, so neither depends
# on the order the three facts arrive in; empty when the fact is absent.
image_arch_of() {
    printf '%s\n' "$1" | sed -n 's/^.*image_arch=\([A-Za-z0-9_]*\).*$/\1/p' | head -1
}
llama_commit_of() {
    printf '%s\n' "$1" | sed -n 's/.*"commit" *: *"\([0-9a-f]\{7,64\}\)".*/\1/p' | head -1
}

# The 64-hex digest from a `sha256sum` line, or empty when the line is
# something else. The remote checksum command merges stderr into its output so
# the reason for a failure survives the trip, which means the answer has to be
# recognised rather than assumed: a sudo refusal must read as unreadable, never
# as a build mismatch that sends the operator rebuilding images.
sha_of() {
    local first="${1%% *}"
    case "$first" in
        ""|*[!0-9a-f]*) return 0 ;;
    esac
    if [ "${#first}" -eq 64 ]; then printf '%s' "$first"; fi
}

# Memoized `hostname` for a node index. Sets NODE_HOSTNAME to the answer, or to
# the empty string when the node did not respond.
#
# There is no connection multiplexing here, so every ask is a full SSH
# handshake with ConnectTimeout=10, and three call sites want the same answer:
# section 2's reachability check, section 2's label lookup, and section 6 for
# every group member. A node's hostname cannot change mid-run.
#
# It reports through a variable rather than stdout because `$(node_hostname i)`
# would run the whole function in a subshell — the cache it fills would die
# with that subshell and every call would still pay for a handshake, which is
# the entire point of the helper. Misses are cached too, so an unreachable node
# is dialled once rather than once per call site.
NODE_HOSTNAMES=()
NODE_HOSTNAME=""
node_hostname() {
    local i="$1" h
    if [ -z "${NODE_HOSTNAMES[$i]:-}" ]; then
        h="$(ssh_at "$i" "hostname" 2>/dev/null | tr -d '[:space:]')" || h=""
        NODE_HOSTNAMES[$i]="${h:-?}"
    fi
    if [ "${NODE_HOSTNAMES[$i]}" = "?" ]; then
        NODE_HOSTNAME=""
    else
        NODE_HOSTNAME="${NODE_HOSTNAMES[$i]}"
    fi
}

# Validate a batch of edge configs in ONE interpreter, from alternating
# name/path arguments. A `uv run` per config file costs an environment
# resolution and a full orchestrator import each, in a section that needs no
# infrastructure at all. Per-file attribution survives because this emits the
# same PASS|/FAIL| protocol section 4's MCP helper uses.
validate_edge_configs() {
    local label="$1"; shift
    local out status=0
    out=$(uv run python - "$@" 2>&1 <<'PY'
import sys

from orchestrator.config.loader import load_edge_config

args = sys.argv[1:]
for name, path in zip(args[::2], args[1::2]):
    try:
        load_edge_config(path)
    except Exception as e:  # one bad config must not hide the rest of the batch
        print(f"FAIL|{name}|" + " ".join(f"{type(e).__name__}: {e}".split()))
    else:
        print(f"PASS|{name}")
PY
    ) || status=$?
    count_results "$label" "$status" "$out"
}

# ===================== SECTION 0: PYTEST UNIT TESTS =====================
# Always runs — no infrastructure needed. Gated on the dev group being installed
# and on tests/ existing: a workspace created by `aureax init --workspace`
# carries scripts and templates but not the test tree (the wheel does not
# bundle it), so there the section is a skip, not a pytest collection error.

section_banner "0. Unit Tests (pytest)" 40

if [ ! -d "${AUREAX_ROOT}/tests" ]; then
    skip "pytest unit tests (no tests/ here — run them from a Git checkout)"
elif uv run --group dev python -c "import pytest" >/dev/null 2>&1; then
    PYTEST_FAIL=0
    PYTEST_OUT=$(uv run --group dev pytest tests/ -q 2>&1) || PYTEST_FAIL=1
    echo "$PYTEST_OUT" | tail -5
    PYTEST_PASSED=$(echo "$PYTEST_OUT" | grep -oE '[0-9]+ passed' | head -1 | grep -oE '[0-9]+' || echo "0")
    PYTEST_FAILED=$(echo "$PYTEST_OUT" | grep -oE '[0-9]+ failed' | head -1 | grep -oE '[0-9]+' || echo "0")
    PASS=$((PASS + PYTEST_PASSED))
    FAIL=$((FAIL + PYTEST_FAILED))
    # A collection or config error prints no "N failed" line at all, so the two
    # greps above both yield 0 and the suite would go on to report ALL TESTS
    # PASSED having run nothing. Trust pytest's exit status over its summary.
    if [ "$PYTEST_FAIL" -eq 1 ] && [ "$PYTEST_FAILED" -eq 0 ]; then
        echo "  [FAIL] pytest exited non-zero with no failure count (collection or config error)"
        FAIL=$((FAIL + 1))
    fi
else
    skip "pytest unit tests (run 'uv sync --group dev' first)"
fi

# ===================== SECTION 1: ORCHESTRATOR =====================
# Always runs — no infrastructure needed.

section_banner "1. Orchestrator Layer" 40

check "Load network config" \
    uv run python -m orchestrator.config.loader "$NETWORK_CONFIG"

# Raw per-node configs DO validate — loader.load_edge_config substitutes
# PLACEHOLDER_DUMMIES for the ${...} names first — so validate them properly
# rather than only checking that the YAML parses. Deduped because the shard
# workers share one template.
RAW_CFG_ARGS=()
SEEN_CFGS=" "
for i in $(seq 0 $((NODE_COUNT - 1))); do
    CFG_FILE="${NODE_CONFIG_FILES[$i]}"
    # `config_file` is optional in the schema and parse-config.py preserves the
    # gap as an empty element. Without this guard the empty slot is validated as
    # the literal path `configs/` — a directory — and reports a spurious FAIL.
    if [ -z "$CFG_FILE" ]; then
        skip "per-node config for ${NODE_IDS[$i]} (no config_file)"
        continue
    fi
    case "$SEEN_CFGS" in *" $CFG_FILE "*) continue ;; esac
    SEEN_CFGS="${SEEN_CFGS}${CFG_FILE} "
    RAW_CFG_ARGS+=("Validate per-node config: ${CFG_FILE}" "configs/${CFG_FILE}")
done
if [ "${#RAW_CFG_ARGS[@]}" -gt 0 ]; then
    validate_edge_configs "Per-node config validation" "${RAW_CFG_ARGS[@]}"
fi

# Both Dockerfiles must pin the SAME llama.cpp build: the RPC wire format is
# version-locked and a mismatch hangs at the handshake with no error.
#
# Compared as extracted values, per file. Counting distinct lines across a
# `grep -h` of both files asks only whether the pins seen ANYWHERE form a
# singleton, so a Dockerfile that lost its pin — or is missing entirely —
# contributes nothing and the surviving file's one line still counts as one.
# Each file's own pins are deduped too: both are multi-stage and every stage
# has to agree.
llama_pins() {
    grep -h '^ARG LLAMA_CPP_VERSION=' "$1" 2>/dev/null | sed 's/^ARG LLAMA_CPP_VERSION=//' | sort -u
}
LLAMA_PIN_CLIENT="$(llama_pins docker/Dockerfile.client || true)"
LLAMA_PIN_CUDA="$(llama_pins docker/Dockerfile.client.cuda || true)"
if [ -z "$LLAMA_PIN_CLIENT" ] || [ -z "$LLAMA_PIN_CUDA" ]; then
    echo "  [FAIL] LLAMA_CPP_VERSION pin missing from a Dockerfile" \
         "(client='${LLAMA_PIN_CLIENT:-<none>}', cuda='${LLAMA_PIN_CUDA:-<none>}')"
    FAIL=$((FAIL + 1))
elif [ "$LLAMA_PIN_CLIENT" = "$LLAMA_PIN_CUDA" ]; then
    echo "  [PASS] LLAMA_CPP_VERSION identical across both Dockerfiles (${LLAMA_PIN_CLIENT})"
    PASS=$((PASS + 1))
else
    echo "  [FAIL] LLAMA_CPP_VERSION differs between Dockerfiles — RPC will hang at handshake"
    grep -Hn '^ARG LLAMA_CPP_VERSION=' docker/Dockerfile.client docker/Dockerfile.client.cuda
    FAIL=$((FAIL + 1))
fi

check "CLI: aureax init" \
    uv run aureax --config "$NETWORK_CONFIG" init

# Assert against a node_id read from the config rather than a literal: the
# fleet's names change, and a hardcoded one fails for the wrong reason.
check_output "CLI: aureax nodes" "${NODE_IDS[0]}" \
    uv run aureax --config "$NETWORK_CONFIG" nodes

# `aureax generate` REWRITES ~/.aureax/configs for the whole fleet, so the
# address it is given is the one every node's telemetry endpoint then names
# until the next generate. Asking the same detector setup-all.sh uses keeps a
# test run from installing a fleet that ships telemetry to its own loopback.
# detect_mac_ip's own [FAIL] line goes to stderr and would be uncounted noise
# in this suite, so it is discarded and reported through `warn` instead.
GEN_IP="$(detect_mac_ip 2>/dev/null || echo 127.0.0.1)"
if [ "$GEN_IP" = "127.0.0.1" ]; then
    warn "no Mac LAN IP resolved — generating with 127.0.0.1"
    echo "         ~/.aureax/configs now points every node at its own loopback;"
    echo "         restore with 'uv run aureax generate --orchestrator-ip <MAC_LAN_IP>'"
fi

# HOME is swapped for the generate only; uv's cache stays where it is, or a
# fresh HOME would make uv re-resolve into an empty cache.
check "CLI: aureax generate" \
    env HOME="$GEN_HOME" UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}" \
    uv run aureax --config "$NETWORK_CONFIG" generate --orchestrator-ip "$GEN_IP"

# Quoted: an unquoted array expansion word-splits an id containing whitespace
# into several iterations and drops an empty one entirely, which is exactly the
# quoting contract parse-config.py's _print_shell_array exists to provide.
GEN_CFG_ARGS=()
for node_id in "${NODE_IDS[@]}"; do
    GEN_CFG_ARGS+=(
        "Validate generated edge config (${node_id})"
        "$GEN_HOME/.aureax/configs/aureax-edge-${node_id}.yaml"
    )
done
if [ "${#GEN_CFG_ARGS[@]}" -gt 0 ]; then
    validate_edge_configs "Generated edge config validation" "${GEN_CFG_ARGS[@]}"
fi

check "CLI: aureax stop --help" \
    uv run aureax --config "$NETWORK_CONFIG" stop --help

check "CLI: aureax start --help" \
    uv run aureax --config "$NETWORK_CONFIG" start --help

# ===================== SECTION 2: INFRASTRUCTURE =====================
# Gated on kubectl being available and cluster reachable.

section_banner "2. Infrastructure Layer" 40

# ONE probe, shared by the three sections gated on it (2, 6, 7). Against a
# cluster whose host is unreachable each `kubectl cluster-info` blocks on a
# full TCP connect timeout, and the answer cannot change mid-run. Taken here
# rather than at the top of the file: a probe from before the pytest run above
# would be minutes stale by sections 6 and 7.
KUBECTL_OK=0
if kubectl cluster-info >/dev/null 2>&1; then
    KUBECTL_OK=1
fi

if [ "$KUBECTL_OK" -eq 1 ]; then
    echo "  (kubectl reachable — running infra tests)"
    echo

    # SSH connectivity. Through node_hostname so the label pass below reuses
    # this probe instead of dialling every node a second time.
    for i in $(seq 0 $((NODE_COUNT - 1))); do
        HOST="${NODE_HOSTS[$i]}"
        node_hostname "$i"
        if [ -n "$NODE_HOSTNAME" ]; then
            echo "  [PASS] SSH to ${HOST}"
            PASS=$((PASS + 1))
        else
            echo "  [FAIL] SSH to ${HOST}"
            FAIL=$((FAIL + 1))
        fi
    done

    # K3s nodes. Counted rather than grepped: `grep -q Ready` also matches
    # NotReady, so a grep passes a fleet whose every kubelet is down. Same
    # shape as the pod-Ready count below, empty list included — no nodes at all
    # is a failure, not a vacuous pass.
    NODES_OUT=$(kubectl get nodes --no-headers --show-labels 2>/dev/null || true)
    NODES_TOTAL=$(echo "$NODES_OUT" | grep -c . || true)
    NODES_NOT_READY=$(echo "$NODES_OUT" | awk 'NF && $2 != "Ready" {n++} END {print n+0}')
    if [ "$NODES_TOTAL" -gt 0 ] && [ "$NODES_NOT_READY" -eq 0 ]; then
        echo "  [PASS] All K3s nodes Ready (${NODES_TOTAL}/${NODES_TOTAL})"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] K3s nodes not all Ready (${NODES_NOT_READY} of ${NODES_TOTAL} not Ready)"
        echo "$NODES_OUT" | awk 'NF && $2 != "Ready" {print "         " $1 " " $2}'
        FAIL=$((FAIL + 1))
    fi

    check "Namespace 'aureax' exists" \
        kubectl get namespace aureax

    # One listing serves the existence check and both counts below, rather than
    # three round trips for one answer. Guarded, because under `set -euo
    # pipefail` an absent namespace — the state after reset-all.sh, or before
    # the first deploy — would otherwise end the whole run right here, with
    # sections 3-7 unrun and no Results summary at all.
    PODS_OUT=$(kubectl -n aureax get pods --no-headers 2>/dev/null || true)

    # Existence only — a CrashLoopBackOff pod is listed too. Readiness is the
    # next check.
    if grep -q "aureax-edge" <<<"$PODS_OUT"; then
        echo "  [PASS] Edge pods exist"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] Edge pods exist"
        FAIL=$((FAIL + 1))
    fi

    read -r POD_READY POD_TOTAL <<< "$(echo "$PODS_OUT" \
        | awk 'NF {t++; if ($2 == "1/1") r++} END {print r+0, t+0}')"
    if [ "$POD_READY" = "$POD_TOTAL" ] && [ "$POD_TOTAL" -gt 0 ]; then
        echo "  [PASS] All pods Ready ($POD_READY/$POD_TOTAL)"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] Pods not all Ready ($POD_READY/$POD_TOTAL)"
        FAIL=$((FAIL + 1))
    fi

    # Node labels, read out of the NODES_OUT listing above (--show-labels puts
    # them in the last column) rather than one API round trip per node.
    #
    # BOTH directions are asserted. A role flip that only ADDS a label leaves
    # the node matching both DaemonSet selectors — running rpc-server and
    # llama-server at once, the latter with no model file — and a
    # presence-only check passes that node twice over.
    for i in $(seq 0 $((NODE_COUNT - 1))); do
        node_hostname "$i"
        K8S_NODE="${NODE_HOSTNAME:-unknown}"
        NODE_LABELS=$(echo "$NODES_OUT" | awk -v n="$K8S_NODE" '$1 == n {print $NF}')
        if [ "${NODE_ROLES[$i]}" = "$SHARD_WORKER_ROLE" ]; then
            WANT_LABEL="aureax.io/shard=worker"
            DENY_LABEL="aureax.io/role=edge"
        else
            WANT_LABEL="aureax.io/role=edge"
            DENY_LABEL="aureax.io/shard=worker"
        fi
        if ! grep -q "$WANT_LABEL" <<<"$NODE_LABELS"; then
            echo "  [FAIL] Node ${K8S_NODE} is missing ${WANT_LABEL}"
            FAIL=$((FAIL + 1))
        elif grep -q "$DENY_LABEL" <<<"$NODE_LABELS"; then
            echo "  [FAIL] Node ${K8S_NODE} carries ${DENY_LABEL} as well as ${WANT_LABEL}"
            echo "         it matches both DaemonSet selectors — a role flip that only added a label"
            FAIL=$((FAIL + 1))
        else
            echo "  [PASS] Node ${K8S_NODE} carries ${WANT_LABEL} and not ${DENY_LABEL}"
            PASS=$((PASS + 1))
        fi
    done
else
    echo "  (kubectl not reachable — skipping infra tests)"
    echo
    skip "SSH connectivity"
    skip "K3s nodes Ready"
    skip "Namespace 'aureax'"
    skip "Edge pods exist"
    skip "Pods Ready"
    skip "Node role labels"
fi

# ===================== SECTION 3: DASHBOARD API =====================
# Gated on the dashboard being reachable at localhost:7001. Nothing here is
# authenticated — the dashboard has no token and no login (schema.py:
# server.dashboard.host) — so reachability is the whole gate.

section_banner "3. Dashboard API" 40

# Port assumed default (server.dashboard.port) — see the header note.
#
# PLAINTEXT, and the one exception to this suite's TLS rule: `aureax dashboard`
# serves the same app on two listeners in one process — HTTPS on the
# orchestrator's LAN address for the fleet, and plain HTTP on 127.0.0.1/::1 for
# whoever is sitting at the Mac. This suite runs on the Mac, so it takes the
# loopback door: no --cacert, no trust store, and no dependency on the leaf
# still carrying today's address. Loopback is not routable, so nothing here
# crosses the LAN. Every other endpoint in this file is HTTPS.
#
# `localhost`, not 127.0.0.1, so the check follows whichever family the
# operator's browser would resolve first — both are bound.
DASHBOARD_URL="http://localhost:7001"

# Every dashboard call below is a plain `curl` (not curl_tls, which would add a
# --cacert that means nothing to an http:// URL) with two things that matter:
#   -f          so a 404/500 is a failed check, not a body to grep
#   --max-time  because curl has no default overall timeout: a wedged route
#               (a request queued behind a lifecycle run) would otherwise block
#               the suite forever — no summary, no failure, no exit code

if curl -sf --max-time 15 "${DASHBOARD_URL}/api/cluster" >/dev/null 2>&1; then
    echo "  (dashboard reachable — running API tests)"
    echo

    # Asserts the endpoint ANSWERS with a status, not that the status is "ok":
    # /api/cluster correctly reports {"status":"error"} when the dashboard is up
    # but kubectl is not, and this section is gated only on the dashboard. The
    # strict ok-check belongs to section 2, which is gated on kubectl.
    check_output "GET /api/cluster" '"status":"' \
        curl -sf --max-time 15 "${DASHBOARD_URL}/api/cluster"

    check_output "GET /api/pods" '"status"' \
        curl -sf --max-time 15 "${DASHBOARD_URL}/api/pods"

    check "GET /api/nodes" \
        curl -sf --max-time 15 "${DASHBOARD_URL}/api/nodes"

    # The two routes edge nodes push to. They are the reason the dashboard
    # binds 0.0.0.0, so a regression here silently stops every node's telemetry
    # and log shipping.
    TEST_JSONL='{"node_id":"test","metric_name":"test_metric","metric_value":42.0,"timestamp":"2026-01-01T00:00:00Z"}'
    check_output "POST /api/v1/telemetry" '"ingested"' \
        curl -sf --max-time 15 -X POST "${DASHBOARD_URL}/api/v1/telemetry" \
            -H "Content-Type: application/x-ndjson" \
            -d "$TEST_JSONL"

    # Fields per orchestrator/logs/receiver.py::_LOG_FIELDS — node_id, level,
    # text. A record missing one is dropped, which the route reports as a 500.
    TEST_LOGJSONL='{"node_id":"test","level":"info","text":"test log from test-all.sh","timestamp":"2026-01-01T00:00:00Z"}'
    check_output "POST /api/v1/logs" '"ingested"' \
        curl -sf --max-time 15 -X POST "${DASHBOARD_URL}/api/v1/logs" \
            -H "Content-Type: application/x-ndjson" \
            -d "$TEST_LOGJSONL"

    # Lifecycle control endpoints (read-only here — stop/start runs last)
    check_output "GET /api/lifecycle/state" '"state"' \
        curl -sf --max-time 15 "${DASHBOARD_URL}/api/lifecycle/state"

    check_output "GET /api/lifecycle/nodes" "${NODE_IDS[0]}" \
        curl -sf --max-time 15 "${DASHBOARD_URL}/api/lifecycle/nodes"

    # Access tab: per-node endpoints + staged auth state. Also the payload-level
    # security invariant — no plaintext key may ever appear in this listing.
    #
    # ONE fetch, and the key-leak assertion is gated on that fetch having
    # returned something recognisable. Falling back to an empty body would make
    # a 500, a hang or a 4xx print "[PASS] carries no plaintext key" having
    # inspected nothing — and that PASS line is the only evidence anyone reads
    # for this invariant.
    ACCESS_BODY=$(curl -sf --max-time 15 "${DASHBOARD_URL}/api/access/nodes" 2>/dev/null) || ACCESS_BODY=""
    if ! grep -q "${NODE_IDS[0]}" <<<"$ACCESS_BODY"; then
        fail "GET /api/access/nodes"
        fail "/api/access/nodes unreadable — key-leak check not performed"
    else
        echo "  [PASS] GET /api/access/nodes"
        PASS=$((PASS + 1))
        if grep -q 'aureax_sk_[A-Za-z0-9_-]\{20\}' <<<"$ACCESS_BODY"; then
            echo "  [FAIL] /api/access/nodes leaked a plaintext API key"
            FAIL=$((FAIL + 1))
        else
            echo "  [PASS] /api/access/nodes carries no plaintext key"
            PASS=$((PASS + 1))
        fi
    fi

    # Config CRUD endpoints
    check_output "GET /api/config" '"aureax-network.yaml"' \
        curl -sf --max-time 15 "${DASHBOARD_URL}/api/config"

    check_output "GET /api/config/aureax-network.yaml" '"content"' \
        curl -sf --max-time 15 "${DASHBOARD_URL}/api/config/aureax-network.yaml"

    # Path whitelist: unknown name returns 404.
    #
    # `|| STATUS="000"` is load-bearing: the section gate ran several calls
    # ago, so the dashboard can be gone by now (an operator Ctrl-C, an `aureax
    # stop` in another terminal). curl then exits 7 and an unguarded assignment
    # ends the run under `set -e` before the comparison below can report it.
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 \
                 "${DASHBOARD_URL}/api/config/does-not-exist.yaml") || STATUS="000"
    if [ "$STATUS" = "404" ]; then
        echo "  [PASS] GET /api/config/<unknown> returns 404"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] GET /api/config/<unknown> expected 404 got $STATUS"
        FAIL=$((FAIL + 1))
    fi

    # Malformed YAML body is rejected with ok:false. `-s`, not `-sf`: the route
    # answers 200 with ok:false, and the body is the assertion.
    #
    # Backed up first, because this check aims a bad body at the operator's own
    # fleet definition: gitignored, holding every node's address and credential,
    # and with no other copy anywhere. The route validates before it writes
    # (orchestrator/dashboard/config_api.py::save_config), so the body cannot
    # reach the file today — this is what stands between a future regression in
    # that order and a fleet nobody can reconstruct. restore_fleet_file runs
    # from the suite's traps too, so a Ctrl-C mid-check restores as well.
    FLEET_EXISTED=0
    FLEET_BACKUP="$(mktemp "${TMPDIR:-/tmp}/aureax-network-backup.XXXXXX")"
    if [ -f "$FLEET_FILE" ]; then
        FLEET_EXISTED=1
        cp "$FLEET_FILE" "$FLEET_BACKUP"
    fi

    BAD_BODY='{"content":"not: [valid yaml"}'
    check_output "PUT invalid YAML returns ok:false" '"ok":false' \
        curl -s --max-time 15 -X PUT \
             -H "Content-Type: application/json" \
             -d "$BAD_BODY" \
             "${DASHBOARD_URL}/api/config/aureax-network.yaml"
    restore_fleet_file
else
    echo "  (dashboard not reachable at ${DASHBOARD_URL} — skipping)"
    echo "  Hint: run 'uv run aureax dashboard' in another terminal"
    echo "        (the probe verifies against ${AUREAX_CA}; a missing or stale"
    echo "         fleet CA looks exactly like a dashboard that is not running)"
    echo
    skip_dashboard_checks
fi

# ===================== SECTION 4: MCP ENDPOINTS =====================
# Gated on the first MCP-server node answering /health on the MCP NodePort
# (pod readiness is implied by that answer, not checked).
#
# Also the section that proves MCP bearer auth is on where a template asked for
# it. That has the same shape as section 7's llama checks and for the same
# reason: an MCP key that was minted but never applied leaves /health green,
# the pod Ready and the port wide open, and nothing else in this suite looks.

section_banner "4. MCP Endpoints" 40

# Check if at least one MCP-server node is reachable. Planner nodes
# (mcp.server.enabled=false) don't expose the MCP NodePort — they're MCP
# clients driven by the dashboard, so skip them here.
#
# $MCP_NODE_PORT and $LLAMA_NODE_PORT come from `parse-config.py all`, which
# reads them from orchestrator/config/schema.py. Never type the numbers in
# here: a Service NodePort change would then update every Python caller and
# leave these checks probing the port nothing serves.
MCP_PORT="$MCP_NODE_PORT"
FIRST_MCP_HOST=""
for i in $(seq 0 $((NODE_COUNT - 1))); do
    if [ "${NODE_HAS_MCP_SERVER[$i]}" = "true" ]; then
        FIRST_MCP_HOST="${NODE_HOSTS[$i]}"
        break
    fi
done

MCP_REACHABLE=false
if [ -n "$FIRST_MCP_HOST" ] && mcp_health_wait "$FIRST_MCP_HOST" "$MCP_PORT" 0; then
    MCP_REACHABLE=true
fi

if $MCP_REACHABLE; then
    echo "  (MCP reachable — running endpoint tests)"
    echo

    # /health on each MCP-server node, UNAUTHENTICATED on purpose.
    #
    # This is an assertion about the exemption, not just a liveness poll:
    # client/mcp/auth.py exempts exactly {"/health"}, and mcp_health_wait
    # (scripts/lib/common.sh), setup-all.sh's verification loop and
    # update-all.sh's post-apply wait all poll it with no key. Narrow that
    # exemption and all three start reporting every node as dead, having
    # changed nothing about the node.
    #
    # `-f` is what makes it an assertion: a 401 would be a curl failure here,
    # not a body that happens not to match.
    for i in $(seq 0 $((NODE_COUNT - 1))); do
        if [ "${NODE_HAS_MCP_SERVER[$i]}" != "true" ]; then
            continue
        fi
        HOST="${NODE_HOSTS[$i]}"
        check_output "MCP /health on ${HOST}:${MCP_PORT} is 200 unauthenticated" '"status":"ok"' \
            curl_tls -sf --max-time 15 "https://${HOST}:${MCP_PORT}/health"
    done

    # Bearer auth on /mcp, for every node whose template carries
    # `api_key: ${mcp_api_key}` under mcp.server.
    #
    # POST, not GET: the streamable-http transport answers a GET with an SSE
    # stream that never closes, so a GET probe measures --max-time rather than
    # the status. The Accept header carrying BOTH types is required by the
    # transport itself (mcp.server.streamable_http) — omit it and an
    # authenticated call answers 406, which reads as an auth failure.
    #
    # A real `initialize` body rather than `{}`: the 200 arm has to be a
    # request the server would actually accept, or "200 with the key" would
    # only prove that a malformed request is rejected after the auth check
    # instead of before it. The session this registers is left to expire; the
    # server prunes it, and DELETEing it would need the session id from a
    # response body this helper deliberately discards.
    if [ "$NODE_COUNT" -gt 0 ] && [ -z "${NODE_MCP_AUTH+x}" ]; then
        # Same guard, same reason as section 7's NODE_AUTH one: NODE_MCP_AUTH
        # is newer than the other NODE_* arrays, so a stale
        # scripts/parse-config.py leaves it unset and indexing it under `set -u`
        # would abort the run here rather than report anything.
        fail "NODE_MCP_AUTH not exported by parse-config.py — cannot tell which nodes enforce MCP auth"
    else
        MCP_INIT_BODY='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test-all","version":"0"}}}'
        for i in $(seq 0 $((NODE_COUNT - 1))); do
            [ "${NODE_HAS_MCP_SERVER[$i]}" = "true" ] || continue
            [ "${NODE_MCP_AUTH[$i]}" = "true" ] || continue
            HOST="${NODE_HOSTS[$i]}"
            NID="${NODE_IDS[$i]}"
            # Reported, never a bare `continue`: on a mixed fleet a silent skip
            # leaves this node with no PASS, no FAIL and no SKIP line, and its
            # checks vanish from the counters with nothing to notice.
            if [ ! -f "${KEY_DIR}/${NID}.mcp.key" ]; then
                skip "${NID}: MCP auth checks (no key at ${KEY_DIR}/${NID}.mcp.key — run aureax generate)"
                continue
            fi
            # Read at the point of use, into a variable that goes no further
            # than the http_status call below.
            MCP_KEY=$(tr -d '[:space:]' < "${KEY_DIR}/${NID}.mcp.key" 2>/dev/null) || MCP_KEY=""
            if [ -z "$MCP_KEY" ]; then
                fail "${NID}: MCP key file ${KEY_DIR}/${NID}.mcp.key is empty or unreadable"
                continue
            fi

            http_status "${NID}: POST /mcp unauthenticated is 401 (MCP auth is enforced)" \
                401 "https://${HOST}:${MCP_PORT}/mcp" "" \
                -- -X POST -H 'Content-Type: application/json' \
                   -H 'Accept: application/json, text/event-stream' \
                   --data-binary "$MCP_INIT_BODY"

            http_status "${NID}: POST /mcp with key is 200 (our key is the one enforced)" \
                200 "https://${HOST}:${MCP_PORT}/mcp" "$MCP_KEY" \
                -- -X POST -H 'Content-Type: application/json' \
                   -H 'Accept: application/json, text/event-stream' \
                   --data-binary "$MCP_INIT_BODY"
        done
    fi

    # Run Python MCP tool tests on each MCP-server node
    for i in $(seq 0 $((NODE_COUNT - 1))); do
        if [ "${NODE_HAS_MCP_SERVER[$i]}" != "true" ]; then
            echo
            echo "  --- ${NODE_IDS[$i]} (${NODE_ROLES[$i]}) — no MCP server, skipping tools ---"
            continue
        fi
        HOST="${NODE_HOSTS[$i]}"
        NODE_ID="${NODE_IDS[$i]}"
        echo
        echo "  --- MCP tools on ${HOST} (${NODE_ID}) ---"

        # Run the Python MCP test helper and parse its output. Its exit status
        # is kept: the helper imports httpx/mcp outside its own try block, so a
        # renamed symbol there emits a traceback and nothing else — every tool
        # test vanishing without a single FAIL unless count_results notices
        # that no results parsed.
        #
        # Both credentials travel in the ENVIRONMENT, never on argv: the helper
        # reads $AUREAX_MCP_KEY_FILE (a path to the key, so the key itself is in
        # no process's command line) and $AUREAX_CA (the fleet CA it verifies
        # the node's certificate against — same file curl_tls uses, so a run
        # against a second fleet moves both together). AUREAX_MCP_KEY_FILE is
        # set only when the file exists; an auth-off node needs no key and a
        # missing one is already reported by the checks above.
        MCP_STATUS=0
        MCP_KEY_FILE=""
        if [ -f "${KEY_DIR}/${NODE_ID}.mcp.key" ]; then
            MCP_KEY_FILE="${KEY_DIR}/${NODE_ID}.mcp.key"
        fi
        MCP_OUTPUT=$(AUREAX_CA="$AUREAX_CA" AUREAX_MCP_KEY_FILE="$MCP_KEY_FILE" \
            uv run python scripts/test-mcp-tools.py "$HOST" "$MCP_PORT" 2>&1) || MCP_STATUS=$?
        count_results "MCP tool tests on ${HOST}" "$MCP_STATUS" "$MCP_OUTPUT"
    done
else
    echo "  (MCP not reachable at ${FIRST_MCP_HOST:-<no MCP node>}:${MCP_PORT} — skipping)"
    echo "  Hint: check 'uv run aureax status --verbose' for pod status"
    echo
    # ONE line for the whole helper, deliberately: scripts/test-mcp-tools.py
    # derives the sensor set it checks from what get_sensor_status reports at
    # runtime, so its result count is a property of the node — any per-tool
    # list written here would go stale the moment a node declares one more.
    skip "MCP /health is 200 unauthenticated"
    skip "MCP bearer auth on /mcp (401 without a key, 200 with one)"
    skip "MCP tool tests (every tool, on every MCP-server node)"
fi

# ===================== SECTION 5: COGNITIVE PLANNER =====================
# The planner runs as a node pod (cognitive_planner role). This section asks
# one question — is such a target enabled — and that answer is its own gate:
# YES passes, NO skips, and a probe that could not run fails.
#
# It does NOT re-validate the planner's own config file. Section 1 already
# loads every enabled target's config_file through load_edge_config, which is
# the same call `python -m orchestrator.config.loader` makes — and naming the
# file here would record a [FAIL] on a planner-free fleet that also drops it,
# for a configuration the branch below calls valid.

section_banner "5. Cognitive Planner" 40

# `|| PLANNER_TARGET_PRESENT="ERROR"` is load-bearing: under `set -e` a bare
# VAR=$(...) aborts the whole suite when the command fails, so a stale import
# name in the snippet below would silently kill every section after it. stderr
# is captured with 2>&1 so the reason is printed by the FAIL branch rather than
# swallowed.
#
# The fleet definition arrives as argv rather than a literal, so this probe
# reads the same file the rest of the suite resolved at the top. Per-node
# config_file paths are relative to that file's own directory.
PLANNER_TARGET_PRESENT=$(uv run python -c "
import sys
from pathlib import Path

from client.config.parser import load_edge_config_raw
from orchestrator.config.loader import load_network_config
from orchestrator.config.schema import PLANNER_ROLE

network_path = Path(sys.argv[1])
cfg = load_network_config(network_path)
for t in cfg.ssh.enabled_targets:
    if not t.config_file:
        continue
    try:
        node = load_edge_config_raw(network_path.parent / t.config_file).get('node', {})
    except Exception:
        continue
    if node.get('role') == PLANNER_ROLE:
        print('YES')
        break
else:
    print('NO')
" "$NETWORK_CONFIG" 2>&1) || PLANNER_TARGET_PRESENT="ERROR"

# Matched per line rather than as a whole string: 2>&1 above merges stderr in,
# so one DeprecationWarning from a dependency or one uv notice makes the
# capture "warning...\nYES", which equals neither literal — a correctly
# configured fleet reported as a probe error. check_output survives the same
# noise for the same reason: it greps.
if printf '%s\n' "$PLANNER_TARGET_PRESENT" | grep -qx YES; then
    echo "  [PASS] cognitive_planner SSH target — found"
    PASS=$((PASS + 1))
elif printf '%s\n' "$PLANNER_TARGET_PRESENT" | grep -qx NO; then
    # Not a failure: a fleet with no planner is a valid configuration. (The
    # shipped config DOES have one enabled, so reaching this branch means the
    # config was edited — which is fine, but is not the default.)
    skip "cognitive_planner SSH target (none enabled in aureax-network.yaml)"
else
    echo "  [FAIL] cognitive_planner target probe errored: ${PLANNER_TARGET_PRESENT}"
    FAIL=$((FAIL + 1))
fi

# ===================== SECTION 6: SHARDED GROUPS =====================
# Gated on kubectl reachability AND a non-empty node_groups block.
#
# The last three checks are the point of this section: a head can answer
# /health perfectly while serving on fewer devices than intended, or while one
# peer runs a mismatched llama.cpp build. Neither shows up as an error.

section_banner "6. Sharded Groups" 40

# Distinguish "no groups configured" from "we could not ask". Discarding
# stderr and rewriting any failure as GROUP_COUNT=0 makes an import error, a
# schema change or an unreadable config look identical to a groupless fleet —
# every check below skips silently and the run still prints ALL TESTS PASSED.
#
# The exit status is what decides, not the captured text: a non-zero exit that
# leaves stderr empty (signal death, an unwritable error file) gave an empty
# GROUPS_ERR, an unset GROUP_COUNT, and the "no groups configured" skip — the
# very outcome this block exists to prevent. The error file is mktemp'd because
# a fixed /tmp path owned by another user makes the redirect itself fail.
#
# The suite's traps (armed near the top of this file) are what remove it:
# `eval` on a syntax error in parse-config.py's output, and Ctrl-C during the
# `uv run`, both leave before the explicit rm — leaving a file of tracebacks
# and local config paths in TMPDIR, one per aborted run.
GROUPS_ENV=""
GROUPS_ERR=""
GROUPS_OK=1
GROUPS_ERRFILE="$(mktemp "${TMPDIR:-/tmp}/aureax-groups-err.XXXXXX")"
if GROUPS_ENV="$(uv run python scripts/parse-config.py groups "$NETWORK_CONFIG" 2>"$GROUPS_ERRFILE")"; then
    eval "$GROUPS_ENV"
else
    GROUPS_OK=0
    GROUPS_ERR="$(tail -3 "$GROUPS_ERRFILE" 2>/dev/null || true)"
fi
rm -f "$GROUPS_ERRFILE"

if [ "$GROUPS_OK" -eq 0 ]; then
    echo "  parse-config.py groups FAILED:"
    echo "${GROUPS_ERR:-<no output on stderr>}" | sed 's/^/    /'
    echo
    fail "Read node_groups from config"
    skip_shard_checks
elif [ "${GROUP_COUNT:-0}" -eq 0 ]; then
    echo "  (no node_groups configured — skipping)"
    echo
    skip_shard_checks
elif [ "$KUBECTL_OK" -ne 1 ]; then
    echo "  (kubectl not reachable — skipping)"
    echo
    skip_shard_checks
else
    check "Shard DaemonSet exists" kubectl -n aureax get daemonset aureax-shard

    # ONE namespace-wide shard-pod listing for every group's Ready count,
    # filtered per worker below, rather than one API round trip per worker.
    # `|| true` for the same reason the other captures carry it: under `set
    # -euo pipefail` an absent namespace would end the run mid-group, before
    # the remaining groups and before the Results summary, instead of
    # reporting FAIL.
    # NODE + Ready condition per shard pod, one stable pair per line. Not
    # `-o wide`: after a container restart its RESTARTS column reads
    # "4 (110m ago)", three fields instead of one, which shifts NODE from $7
    # to $9 and silently drops that worker from the per-group Ready count.
    SHARD_PODS=$(kubectl -n aureax get pods -l app=aureax-shard --no-headers \
        -o custom-columns='NODE:.spec.nodeName,READY:.status.conditions[?(@.type=="Ready")].status' 2>/dev/null || true)

    # Every peer must run the byte-identical llama.cpp build: the RPC wire
    # format is version-locked, and a mismatch HANGS at the handshake.
    #
    # `sudo -n`: this suite is meant to run unattended and ssh_cmd allocates no
    # remote tty, so a plain `sudo` on a node whose sudoers wants a password
    # prompted into a discarded stderr and then read the operator's stdin. -n
    # turns that hang into an immediate, attributable failure.
    #
    # Remote stderr is MERGED into the output (2>&1) so the reason survives the
    # trip and can be printed with the skip. That is why every answer goes
    # through sha_of: a value that is not 64 hex digits is an error message,
    # and must read as unreadable rather than as a build mismatch.
    #
    # `--pull=never` and `timeout`, because this asks the same question as
    # orchestrator/cluster/buildcheck.py::BUILD_PROBE and must not answer it
    # differently: `docker run` defaults to `--pull=missing`, so on a node whose
    # image is genuinely absent it reaches for docker.io instead of failing
    # fast, and an unbounded probe against a wedged daemon sits far past the
    # point where a build-consistency check has any answer to give.
    # One container start per member, three facts: the rpc-server checksum,
    # the image's machine architecture and the llama.cpp commit the image was
    # built from (docker/Dockerfile.client writes it into llama-build.json).
    # The commit is what the RPC wire format is locked to and it is
    # architecture-neutral, so it is compared across EVERY member; the
    # checksum can only match within one architecture (a CUDA x86_64 member
    # built from the head's commit is a different binary by construction), so
    # it is compared only among members that share the head's architecture.
    # orchestrator/cluster/buildcheck.py applies the same two rules for
    # `aureax start` and `group doctor`; this section is the shell-side echo.
    SUM_CMD="timeout 20 sudo -n docker run --rm --pull=never --entrypoint sh aureax-client:dev -c 'sha256sum /usr/local/bin/rpc-server; echo image_arch=\$(uname -m); cat /usr/local/share/aureax-llama-build.json 2>/dev/null' 2>&1"

    for gi in $(seq 0 $((GROUP_COUNT - 1))); do
        GNAME="${GROUP_NAMES[$gi]}"
        GHEAD="${GROUP_HEAD_HOSTS[$gi]}"
        # The head's node_id as well as its address: the checks below that dial
        # a port need the host, and the harness needs the id to find the key
        # (see the end-to-end check at the bottom of this loop).
        GHEAD_ID="${GROUP_HEAD_IDS[$gi]}"
        GPORT="${GROUP_RPC_PORTS[$gi]}"
        read -r -a GWORKERS <<< "${GROUP_WORKER_HOSTS[$gi]}"

        # ONE SSH round trip per worker for all three per-worker facts: the RPC
        # listener, the rpc-server RSS and the rpc-server checksum. There is no
        # connection multiplexing (scripts/lib/common.sh::ssh_facts), so asking
        # them separately costs three full handshakes at ConnectTimeout=10
        # against every worker, per group. The reporting loops below read the
        # arrays this fills.
        #
        # `|| FACTS=""` is load-bearing under `set -euo pipefail`: an
        # unreachable worker exits 255 and would end the whole run here. Every
        # value then comes back empty and lands in the FAIL / unreadable
        # branches below, which is exactly what those branches are for.
        #
        # The listener snippet starts with `{`, not `(`: ssh_facts wraps each
        # snippet in "$( ... )", and a leading paren would make that "$((" —
        # arithmetic expansion, on the remote shell.
        W_LISTENING=()
        W_RSS=()
        W_SUM=()
        W_ARCH=()
        W_COMMIT=()
        for w in "${GWORKERS[@]}"; do
            FACTS=$(ssh_facts_at "$(node_index_of "$w")" \
                "LISTENING={ ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null; } | grep -c ':${GPORT}'" \
                "RSS=ps -eo rss,comm | awk '/rpc-server/{s+=\$1} END{printf \"%d\", s/1024}'" \
                "SUM=${SUM_CMD}" 2>/dev/null) || FACTS=""
            # PARSED, never eval'd — the same contract
            # scripts/lib/common.sh::detect_node_arch and download-model.sh's
            # remote-size read follow. This text is whatever the WORKER's shell
            # printed, so eval would run it here with the operator's privileges,
            # and every shard worker exposes an unauthenticated rpc-server on
            # the LAN (see ARCHITECTURE.md, "Security"). ssh_facts %q-quotes
            # each value, but that quoting happens ON THE NODE and so constrains
            # nothing a compromised node chooses to print.
            #
            # Each read is narrowed to what the check can use, which is also all
            # the validation these need. The two counters are digits only. The
            # checksum arrives %q-escaped as `<64hex>\ \ /path/to/rpc-server`,
            # so the digest is taken as the anchored leading 64-hex token —
            # `tr`-ing the whole value would keep the path's hex-ish characters,
            # and passing the escaped form on would make sha_of reject a
            # perfectly good checksum and turn build consistency into a silent
            # skip. When no digest is there the raw value is kept instead, so
            # the remote's own error text still reaches the skip line below.
            LISTENING=$(sed -n 's/^LISTENING=//p' <<<"$FACTS" | tr -cd '0-9')
            RSS=$(sed -n 's/^RSS=//p' <<<"$FACTS" | tr -cd '0-9')
            # SUM_CMD prints three lines; ssh_facts prefixes only the first
            # with "SUM=", so the checksum is the SUM line and the other two
            # facts are found by their own tags anywhere in the blob.
            SUM=$(sed -n 's/^SUM=\([0-9a-f]\{64\}\).*/\1/p' <<<"$FACTS")
            [ -n "$SUM" ] || SUM=$(sed -n 's/^SUM=//p' <<<"$FACTS")
            W_LISTENING+=("$LISTENING")
            W_RSS+=("$RSS")
            W_SUM+=("$SUM")
            W_ARCH+=("$(image_arch_of "$FACTS")")
            W_COMMIT+=("$(llama_commit_of "$FACTS")")
        done

        # Worker pods Ready — counted on THIS GROUP's worker nodes only.
        # `-l app=aureax-shard` alone spans every group in the namespace, so
        # with two groups one group's Ready workers satisfy the other's count.
        # (Same reasoning as the `app=` filter in lifecycle._wait_pods_on_hosts.)
        WREADY=0
        for w in "${GWORKERS[@]}"; do
            wi=$(node_index_of "$w")
            [ "$wi" -lt 0 ] && continue
            node_hostname "$wi"
            WHOST="$NODE_HOSTNAME"
            [ -z "$WHOST" ] && continue
            N=$(echo "$SHARD_PODS" | awk -v n="$WHOST" '$1 == n && $2 == "True" {c++} END {print c+0}')
            WREADY=$((WREADY + N))
        done
        if [ "$WREADY" -ge "${#GWORKERS[@]}" ]; then
            echo "  [PASS] ${GNAME}: shard worker pods Ready (${WREADY}/${#GWORKERS[@]})"
            PASS=$((PASS + 1))
        else
            echo "  [FAIL] ${GNAME}: shard worker pods Ready (${WREADY}/${#GWORKERS[@]})"
            FAIL=$((FAIL + 1))
        fi

        # Each worker must be LISTENING on its RPC port.
        #
        # Checked on the node, not by dialing from here: rpc-server serves one
        # client at a time with a listen backlog of 1, so once the head is
        # connected an external `nc -z` fails — for a worker that is busy doing
        # exactly its job. The listener check is true whether or not a client
        # is currently attached.
        #
        # An empty value means the worker was unreachable, and a `0` means it
        # answered with no listener: both are the failure this check is for.
        wn=0
        for w in "${GWORKERS[@]}"; do
            if [ "${W_LISTENING[$wn]:-0}" -ge 1 ] 2>/dev/null; then
                echo "  [PASS] ${GNAME}: worker ${w} listening on :${GPORT}"
                PASS=$((PASS + 1))
            else
                echo "  [FAIL] ${GNAME}: worker ${w} is not listening on :${GPORT}"
                FAIL=$((FAIL + 1))
            fi
            wn=$((wn + 1))
        done

        # Bare `check`, NOT check_output with an empty pattern: check_output
        # swallows the exit status and `grep -q ""` matches the empty line, so
        # an empty expectation makes the one check that proves the group is
        # serving incapable of ever failing.
        check "${GNAME}: head llama endpoint" \
            curl_tls -sf --max-time 10 "https://${GHEAD}:${LLAMA_NODE_PORT}/health"

        # Every worker must actually be holding weights. This is the check that
        # catches the nastiest failure mode: the model loads, the endpoint is
        # healthy, every probe is green — and one machine is quietly doing all
        # the work because a peer's layers were never assigned.
        #
        # Resident memory of the rpc-server PROCESS, not the node's total used
        # RAM: an idle 16 GB node sits near a gigabyte all by itself, so a
        # whole-node threshold passes for a worker holding nothing — which is
        # precisely the case this check exists to catch. The threshold is
        # derived from the model: each worker should hold roughly
        # size_gb / n_workers, and half of that is a generous floor.
        # Indexed from THIS GROUP's head, not the literal 0. `model_for_node`
        # indexes ssh.targets, so 0 resolved whatever node happens to be listed
        # first — which with several groups, or a planner at the top of the
        # file, is a different machine running a different model. The head is
        # the right node to ask: it is the one that holds the GGUF.
        GHEAD_IDX=$(node_index_of "$GHEAD")   # -1 when the host is not a target
        MODEL_SIZE_GB=""
        if [ "$GHEAD_IDX" -ge 0 ] 2>/dev/null; then
            # Soft-fail on purpose: MODEL_SIZE_GB stays empty and the warning
            # below explains the weaker threshold that results.
            load_network_env model_for_node "$GHEAD_IDX" "$NETWORK_CONFIG" 2>/dev/null || true
        fi
        if [ -z "${MODEL_SIZE_GB:-}" ] || [ "${MODEL_SIZE_GB:-0}" = "0" ]; then
            warn "${GNAME}: could not read model size for head ${GHEAD};"
            echo "         falling back to a 500 MB floor, which only catches a worker holding nothing"
        fi
        MIN_MB=$(awk -v g="${MODEL_SIZE_GB:-0}" -v n="${#GWORKERS[@]}" \
            'BEGIN{ m = (n > 0 ? g * 1024 / n : 0) / 2; printf "%d", (m < 500 ? 500 : m) }')
        wn=0
        for w in "${GWORKERS[@]}"; do
            W_USED="${W_RSS[$wn]}"
            if [ "${W_USED:-0}" -ge "$MIN_MB" ] 2>/dev/null; then
                echo "  [PASS] ${GNAME}: worker ${w} holding weights (rpc-server RSS ${W_USED} MB >= ${MIN_MB} MB)"
                PASS=$((PASS + 1))
            else
                echo "  [FAIL] ${GNAME}: worker ${w} rpc-server holds ${W_USED:-?} MB, expected >= ${MIN_MB} MB"
                echo "         its layers were never assigned — the group looks healthy but is not sharded"
                FAIL=$((FAIL + 1))
            fi
            wn=$((wn + 1))
        done

        # Build consistency, from the facts collected above plus one call for
        # the head. Two rules, the same two orchestrator/cluster/buildcheck.py
        # applies for `aureax start`: the llama.cpp COMMIT must be the head's
        # on every member whose build record is readable (the RPC wire format
        # is locked to it and it is architecture-neutral), and the rpc-server
        # CHECKSUM must be the head's only on members of the head's
        # architecture — a CUDA x86_64 member is a different binary by
        # construction and is compared by commit alone.
        HEAD_SUM_RAW=$(ssh_at "$(node_index_of "$GHEAD")" "$SUM_CMD" 2>/dev/null) || HEAD_SUM_RAW=""
        HEAD_SUM="$(sha_of "$HEAD_SUM_RAW")"
        HEAD_ARCH="$(image_arch_of "$HEAD_SUM_RAW")"
        HEAD_COMMIT="$(llama_commit_of "$HEAD_SUM_RAW")"
        BUILD_MISMATCH=0
        UNREADABLE=""
        SUM_REASON=""
        CROSS_ARCH=""
        [ -n "$HEAD_SUM" ] || SUM_REASON="${HEAD_SUM_RAW}"
        wn=0
        for w in "${GWORKERS[@]}"; do
            W_SUM_RAW="${W_SUM[$wn]}"
            W_SUM_HEX="$(sha_of "$W_SUM_RAW")"
            W_ARCH_1="${W_ARCH[$wn]}"
            W_COMMIT_1="${W_COMMIT[$wn]}"
            if [ -n "$HEAD_COMMIT" ] && [ -n "$W_COMMIT_1" ] && [ "$W_COMMIT_1" != "$HEAD_COMMIT" ]; then
                echo "         ${w}: llama.cpp commit ${W_COMMIT_1:0:12} differs from head ${HEAD_COMMIT:0:12}"
                BUILD_MISMATCH=1
            fi
            if [ -n "$HEAD_ARCH" ] && [ -n "$W_ARCH_1" ] && [ "$W_ARCH_1" != "$HEAD_ARCH" ]; then
                # Another architecture: the commit above is the whole check.
                # No readable commit there means nothing was compared at all.
                CROSS_ARCH="${CROSS_ARCH} ${w}(${W_ARCH_1})"
                [ -n "$W_COMMIT_1" ] || UNREADABLE="${UNREADABLE} ${w}"
            elif [ -z "$W_SUM_HEX" ]; then
                # An unreadable checksum is NOT a mismatch — reporting it as one
                # sends the operator rebuilding images to fix an SSH or sudo
                # problem. The remote's own words are kept for the skip line.
                UNREADABLE="${UNREADABLE} ${w}"
                [ -n "$SUM_REASON" ] || SUM_REASON="${W_SUM_RAW}"
            elif [ -n "$HEAD_SUM" ] && [ "$W_SUM_HEX" != "$HEAD_SUM" ]; then
                echo "         ${w}: rpc-server sha256 differs from head (same architecture${HEAD_ARCH:+ ${HEAD_ARCH}})"
                BUILD_MISMATCH=1
            fi
            wn=$((wn + 1))
        done
        if [ "$BUILD_MISMATCH" -eq 1 ]; then
            echo "  [FAIL] ${GNAME}: llama.cpp builds differ across members — RPC will hang at handshake"
            FAIL=$((FAIL + 1))
        elif [ -z "$HEAD_SUM" ] || [ -n "$UNREADABLE" ]; then
            skip "${GNAME}: llama.cpp build consistency (could not read checksum on:${UNREADABLE:- head}${SUM_REASON:+ — ${SUM_REASON}})"
        elif [ -n "$CROSS_ARCH" ]; then
            echo "  [PASS] ${GNAME}: llama.cpp build identical within ${HEAD_ARCH:-the head architecture}; compared by commit${HEAD_COMMIT:+ ${HEAD_COMMIT:0:12}} on:${CROSS_ARCH}"
            PASS=$((PASS + 1))
        else
            echo "  [PASS] ${GNAME}: llama.cpp build identical across all members"
            PASS=$((PASS + 1))
        fi

        # End-to-end: real generation through the group's single MCP endpoint.
        #
        # The harness is handed the head's node_id, NOT its address, and the
        # fleet file it was resolved from. The MCP port takes a bearer key, and
        # the harness only reads ~/.aureax/keys/<node_id>.mcp.key when the
        # positional resolves to an enabled node id — given a bare address it
        # sends no key and every keyed head answers 401, failing this check on a
        # perfectly healthy group. --config because this suite may be reading
        # the shipped example rather than the default fleet file.
        if mcp_health_wait "$GHEAD" "$MCP_NODE_PORT" 0; then
            check "${GNAME}: sharded inference via MCP" \
                uv run aureax-harness infer --config "$NETWORK_CONFIG" \
                "$GHEAD_ID" "Reply with the single word: ok"
        else
            skip "${GNAME}: sharded inference (MCP endpoint not reachable)"
        fi
    done
fi

# ===================== SECTION 7: LLAMA-SERVER API AUTH =====================
# Gated on kubectl reachability AND at least one node whose template enables
# api_key auth AND that node's key file existing on this Mac.
#
# What this section is for: auth failures are quiet in both directions. A key
# that does not match locks every client out while /health stays green and the
# pod stays Ready; an exemption lost upstream locks the KUBELET out and
# crash-loops the node. Neither shows up anywhere else in this suite.

section_banner "7. llama-server API Auth" 40

LLAMA_PORT="$LLAMA_NODE_PORT"

# Auth-enabled nodes that actually serve llama HTTP. Shard workers are excluded
# by role, on the same literal section 2 gates its label check on: they run
# rpc-server, which speaks no HTTP at all, so every check below would report a
# connection failure as an auth failure.
AUTH_IDXS=()
if [ "$NODE_COUNT" -gt 0 ] && [ -z "${NODE_AUTH+x}" ]; then
    # NODE_AUTH is newer than the other NODE_* arrays, so a stale
    # scripts/parse-config.py leaves it unset — and indexing an unset array
    # under `set -u` would abort the run HERE, six sections in and before the
    # summary ever prints. Same failure the `load_network_env all` guard at the
    # top of this file exists to prevent, so it is reported the same way: loudly
    # and counted, leaving AUTH_IDXS empty for the skip branch below.
    fail "NODE_AUTH not exported by parse-config.py — cannot tell which nodes use auth"
else
    for i in $(seq 0 $((NODE_COUNT - 1))); do
        [ "${NODE_AUTH[$i]}" = "true" ] || continue
        [ "${NODE_ROLES[$i]}" = "$SHARD_WORKER_ROLE" ] && continue
        # No key file means `aureax generate` has not run since auth was turned
        # on. Skipping is right — every check below would fail for that one
        # reason — but it is reported, not silent: on a mixed fleet a bare
        # `continue` leaves this node with no PASS, no FAIL and no SKIP line,
        # so its checks vanish from the counters with nothing to notice.
        if [ ! -f "${KEY_DIR}/${NODE_IDS[$i]}.key" ]; then
            skip "${NODE_IDS[$i]}: llama-server auth checks (no key at ${KEY_DIR}/${NODE_IDS[$i]}.key — run aureax generate)"
            continue
        fi
        AUTH_IDXS+=("$i")
    done
fi

if [ "$KUBECTL_OK" -ne 1 ]; then
    echo "  (kubectl not reachable — skipping)"
    echo
    skip_auth_checks
elif [ "${#AUTH_IDXS[@]}" -eq 0 ]; then
    echo "  (no auth-enabled node with a key on this Mac — skipping)"
    echo "  Hint: add 'api_key: \${api_key}' to a node's template, then"
    echo "        'uv run aureax generate --orchestrator-ip <MAC_LAN_IP>'"
    echo
    skip_auth_checks
else
    for i in "${AUTH_IDXS[@]}"; do
        HOST="${NODE_HOSTS[$i]}"
        NID="${NODE_IDS[$i]}"
        BASE="https://${HOST}:${LLAMA_PORT}"
        echo
        echo "  --- ${NID} (${HOST}:${LLAMA_PORT}) ---"

        # Read at the point of use, into a variable that goes no further than
        # the http_status calls below.
        KEY=$(tr -d '[:space:]' < "${KEY_DIR}/${NID}.key" 2>/dev/null) || KEY=""
        if [ -z "$KEY" ]; then
            fail "${NID}: key file ${KEY_DIR}/${NID}.key is empty or unreadable"
            continue
        fi

        # MUST stay public: the DaemonSet's startupProbe and livenessProbe send
        # no bearer token, so an exemption lost here does not lock out clients,
        # it crash-loops the node.
        http_status "${NID}: /health unauthenticated is 200 (k8s probes need it)" \
            200 "${BASE}/health"

        # BUILD-SPECIFIC — /v1/models needs the key on the pinned build
        # (b11429). It was auth-exempt through b10375; upstream removed the
        # /models and /v1/models exemptions in b10518 (#26347), which is why
        # README.md's `curl .../v1/models` examples now send the bearer key.
        # A 200 here on a node whose /props check below still says 401 means
        # that node is running a pre-b10518 image: the pin moved and this node
        # was not rebuilt.
        http_status "${NID}: /v1/models unauthenticated is 401 (exemption gone since b10518)" \
            401 "${BASE}/v1/models"

        # Proves auth is actually ON. Without this, a node whose key was never
        # applied passes every other check in this section.
        http_status "${NID}: /props unauthenticated is 401 (auth is enforced)" \
            401 "${BASE}/props"

        # Proves OUR key is the one the running llama-server enforces — the
        # failure a staged-but-unapplied rotation leaves behind.
        http_status "${NID}: /props with key is 200 (key matches)" \
            200 "${BASE}/props" "$KEY"

        # The telemetry path: client/telemetry/collector.py polls /metrics with
        # this same bearer header, so a 401 here is the fleet losing its
        # throughput numbers rather than an outage anyone would notice.
        http_status "${NID}: /metrics with key is 200 (telemetry path)" \
            200 "${BASE}/metrics" "$KEY"

        # TLS, asserted from the other side: a plain `curl` with NO --cacert
        # must FAIL, and fail with 60 specifically (curl's "peer certificate
        # cannot be authenticated with known CA certificates").
        #
        # Deliberately not curl_tls — this is the one HTTPS call in the suite
        # that must not carry the fleet CA (section 3's dashboard checks are
        # plain curl too, but against a loopback listener where a CA means
        # nothing). Every other check would pass just as happily against a node
        # that had quietly fallen back to plaintext or picked up a
        # publicly-trusted certificate for some other name; this one is what
        # says the door only opens for the fleet's own CA.
        #
        # 0 would mean the node answered a stranger; 7 would mean nothing was
        # listening, which is a different defect and is reported as one.
        CURL_RC=0
        curl -s -o /dev/null --max-time 15 "${BASE}/health" >/dev/null 2>&1 || CURL_RC=$?
        if [ "$CURL_RC" -eq 60 ]; then
            echo "  [PASS] ${NID}: curl without --cacert is refused (exit 60, cert not publicly trusted)"
            PASS=$((PASS + 1))
        elif [ "$CURL_RC" -eq 0 ]; then
            echo "  [FAIL] ${NID}: curl without --cacert SUCCEEDED — ${BASE} is not fleet-CA-only"
            FAIL=$((FAIL + 1))
        else
            echo "  [FAIL] ${NID}: curl without --cacert exited ${CURL_RC}, expected 60 (cert verification)"
            FAIL=$((FAIL + 1))
        fi
    done
fi

# ===================== KEY MATERIAL AT REST =====================
# Section 7's subject, at FILE SCOPE. None of these checks depends on a cluster
# or on an auth-enabled node — two grep local files and one carries its own
# dashboard probe — so gating them on section 7 would report SKIP for all of
# them on every run with the cluster stopped, which is the most common local
# run and the one where they are cheapest to make.
#
# Both greps look for the `aureax_sk_` PREFIX, which is deliberately shared by
# the llama keys and the MCP keys (orchestrator/config/keystore.py). That is
# the concrete payoff of the shared prefix: one pattern covers both kinds, so
# adding a third key kind cannot leave a leak channel behind a grep nobody
# remembered to widen.
#
# Key material must never reach the orchestrator's log store. Read twice,
# because neither read alone covers it:
#
#   - The STORE itself, ~/.aureax/logs/system.db (the log_db_path default
#     in orchestrator/dashboard/app.py::create_app), holds every line ever
#     shipped. Grepped read-only, and needs no dashboard running.
#   - The STREAM, lifecycle_api.py::stream_logs (GET /api/lifecycle/logs),
#     replays SystemLogger's in-memory ring buffer — the only HTTP view of
#     shipped logs, since api.py's POST /api/v1/logs only ingests them. An
#     older key has already fallen out of that buffer, which is why the
#     store is read as well.

# Its own sub-header: these lines print after section 7's banner and are not
# part of its per-node auth checks.
echo
echo "  --- key material at rest ---"

LOG_DB="$HOME/.aureax/logs/system.db"
if [ ! -f "$LOG_DB" ]; then
    skip "log store carries no key material (${LOG_DB} not present)"
else
    LOG_DB_HIT=0
    grep -aq 'aureax_sk_' "$LOG_DB" || LOG_DB_HIT=$?
    if [ "$LOG_DB_HIT" -eq 0 ]; then
        echo "  [FAIL] ${LOG_DB} contains key material (found the 'aureax_sk_' prefix)"
        echo "         a key reached the log store — rotate every node's key with"
        echo "         'uv run aureax apply --all --orchestrator-ip <MAC_LAN_IP> --rotate-key' after fixing the leak"
        FAIL=$((FAIL + 1))
    elif [ "$LOG_DB_HIT" -eq 1 ]; then
        echo "  [PASS] log store carries no key material (no 'aureax_sk_' prefix)"
        PASS=$((PASS + 1))
    else
        # grep exits >1 on a read error; a file it could not search is not
        # evidence that the file is clean.
        fail "log store unreadable — key check not performed (grep exited ${LOG_DB_HIT})"
    fi
fi

# The stream is an SSE response that never closes on its own: --max-time
# bounds it and the resulting non-zero exit is expected, not a failure.
# Which is exactly why it needs a POSITIVE CONTROL. Without one, a renamed
# route, a 404, or the cutoff firing before the first frame all yield a
# body that matches nothing — and the security invariant is reported as
# verified having inspected nothing at all. `^data:` is the SSE frame
# prefix that _http_utils.sse emits.
#
# Here-strings, not `printf | grep -q`: under `set -o pipefail` that
# pipeline FAILS whenever grep exits at its first match while printf is
# still writing, because printf then dies of SIGPIPE — which only happens
# once the snapshot outgrows the pipe buffer (a few thousand lines), so
# such a check passes on a fresh dashboard and fails against a real fleet.
if curl -sf --max-time 15 "${DASHBOARD_URL}/api/cluster" >/dev/null 2>&1; then
    LOG_SNAPSHOT=$(curl -s --max-time 5 \
                        "${DASHBOARD_URL}/api/lifecycle/logs?since=0" 2>/dev/null || true)
    if ! grep -q '^data:' <<<"$LOG_SNAPSHOT"; then
        fail "GET /api/lifecycle/logs returned no SSE frame — streamed-log key check not performed"
    elif grep -q 'aureax_sk_' <<<"$LOG_SNAPSHOT"; then
        echo "  [FAIL] streamed logs contain key material (found the 'aureax_sk_' prefix)"
        echo "         a key reached the log store — rotate every node's key with"
        echo "         'uv run aureax apply --all --orchestrator-ip <MAC_LAN_IP> --rotate-key' after fixing the leak"
        FAIL=$((FAIL + 1))
    else
        echo "  [PASS] streamed logs carry no key material (no 'aureax_sk_' prefix)"
        PASS=$((PASS + 1))
    fi
else
    skip "streamed logs carry no key material (dashboard not reachable)"
fi

# The generated configs on THIS Mac carry key material in plaintext — the
# node's own llama and MCP keys, and on a planner every peer's MCP key as well,
# which makes that one file the single place a fleet's credentials are
# concentrated. generate_all_edge_configs writes them 0600 inside a 0700
# directory (orchestrator/config/templating.py); this makes that enforcement
# observable instead of merely intended, because a wrong mode here is invisible
# until somebody else on the machine reads one.
#
# $GEN_HOME, not $HOME: on a real fleet they are the same directory, and on a
# fresh clone reading the shipped example they are not — GEN_HOME is then the
# throwaway HOME section 1 generated into, so this still inspects files this
# run just wrote instead of skipping for want of an operator store.
#
# `stat -f %Lp` is the BSD spelling; this suite runs on the Mac orchestrator.
CFG_DIR="$GEN_HOME/.aureax/configs"
CFG_BAD=""
CFG_SEEN=0
for f in "$CFG_DIR"/aureax-edge-*.yaml; do
    # An unmatched glob expands to itself, which is not a file.
    [ -f "$f" ] || continue
    CFG_SEEN=$((CFG_SEEN + 1))
    MODE=$(stat -f '%Lp' "$f" 2>/dev/null) || MODE="?"
    if [ "$MODE" != "600" ]; then
        CFG_BAD="${CFG_BAD} $(basename "$f"):${MODE}"
    fi
done
if [ "$CFG_SEEN" -eq 0 ]; then
    skip "generated configs are mode 600 (none in ${CFG_DIR} — run aureax generate)"
elif [ -n "$CFG_BAD" ]; then
    echo "  [FAIL] generated configs are not all mode 600 —${CFG_BAD}"
    echo "         each one holds this node's keys in plaintext; re-run"
    echo "         'uv run aureax generate --orchestrator-ip <MAC_LAN_IP>'"
    FAIL=$((FAIL + 1))
else
    echo "  [PASS] all ${CFG_SEEN} generated configs in ${CFG_DIR} are mode 600"
    PASS=$((PASS + 1))
fi


# ===================== OPT-IN LIFECYCLE SMOKE TEST =====================
# Not a section: it belongs to section 3's dashboard API, but it must run
# HERE, after every section that talks to a node, because it stops the whole
# fleet. Run from inside section 3 it tears the fleet down under sections 4, 6
# and 7 — their MCP checks fail and their kubectl gates flip to skips.
#
# It issues a stop AND a start, and the start is not optional:
# /api/lifecycle/stop patches both DaemonSet selectors and then stops the k3d
# cluster (orchestrator/lifecycle.py::do_stop), so a stop alone leaves the
# fleet down after every run of the suite.

# Poll /api/lifecycle/state until the manager is IDLE again. Both routes return
# 202 the moment they have queued the work, so the second call would be
# refused with a 409 while the first is still in flight — and the start is the
# call that must not be skipped.
lifecycle_wait_idle() {
    local what="$1" state deadline=$((SECONDS + 1800))
    while [ "$SECONDS" -lt "$deadline" ]; do
        state=$(curl -sf --max-time 15 "${DASHBOARD_URL}/api/lifecycle/state" 2>/dev/null) || state=""
        case "$state" in
            # ERROR is terminal too (LifecycleManager._wrap), so it is matched
            # here rather than waited out for the full half hour.
            *'"state":"idle"'*)  return 0 ;;
            *'"state":"error"'*) fail "lifecycle ${what} ended in state=error"; return 1 ;;
        esac
        sleep 5
    done
    fail "lifecycle ${what} did not return to idle within 1800s"
    return 1
}

# Its own sub-header either way: these lines print under section 7's banner,
# and they are not auth checks.
echo
echo "  --- lifecycle stop/start (section 3, run last) ---"

if [ "${AUREAX_TEST_LIFECYCLE:-0}" != "1" ]; then
    skip "POST /api/lifecycle/stop (set AUREAX_TEST_LIFECYCLE=1 to enable)"
    skip "POST /api/lifecycle/start (set AUREAX_TEST_LIFECYCLE=1 to enable)"
elif ! curl -sf --max-time 15 "${DASHBOARD_URL}/api/cluster" >/dev/null 2>&1; then
    skip "POST /api/lifecycle/stop (dashboard not reachable)"
    skip "POST /api/lifecycle/start (dashboard not reachable)"
else
    check_output "POST /api/lifecycle/stop (node=null)" '"run_id"' \
        curl -sf --max-time 15 -X POST -H "Content-Type: application/json" \
             -d '{"node":null}' "${DASHBOARD_URL}/api/lifecycle/stop"
    if lifecycle_wait_idle "stop"; then
        check_output "POST /api/lifecycle/start (node=null)" '"run_id"' \
            curl -sf --max-time 15 -X POST -H "Content-Type: application/json" \
                 -d '{"node":null}' "${DASHBOARD_URL}/api/lifecycle/start"
        # Waited on, not fired and forgotten: the run has to finish before the
        # summary claims anything, and a start that never completes leaves the
        # fleet exactly as stopped as no start at all.
        lifecycle_wait_idle "start" || true
    else
        skip "POST /api/lifecycle/start (stop never finished — fleet is left STOPPED)"
    fi
fi

# ===================== SUMMARY =====================

section_banner "Results" 40

TOTAL=$((PASS + FAIL + SKIP))
echo "  Passed:  ${PASS}"
echo "  Failed:  ${FAIL}"
echo "  Skipped: ${SKIP}"
echo "  Total:   ${TOTAL}"
echo

if [ "$FAIL" -gt 0 ]; then
    echo "  SOME TESTS FAILED"
    exit 1
else
    echo "  ALL TESTS PASSED"
    exit 0
fi
