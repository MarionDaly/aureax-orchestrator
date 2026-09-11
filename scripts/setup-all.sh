#!/usr/bin/env bash
# ==========================================================================
# Aureax Prototype — Full Setup Script
#
# Runs the complete pipeline from zero to a working edge AI cluster.
# Handles completely fresh/wiped edge nodes — no manual prep
# required beyond having SSH access.
#
# All node configuration is read from configs/aureax-network.yaml, which is
# operator-owned and not in the repository: copy it from the shipped example
# first — `cp configs/aureax-network.example.yaml configs/aureax-network.yaml`
# — and edit in your own hosts, users and credentials.
#
# Runnable from anywhere: this script cds to the project root (below).
#
# Prerequisites (Mac only):
#   - macOS with Homebrew
#   - uv — the one tool this script never installs for you
#   - Docker Desktop installed and running
#   - SSH access to all edge nodes (key or password in
#     configs/aureax-network.yaml — the key is installed automatically on a
#     password-auth node, so a fresh box needs no manual ssh-copy-id)
#   - passwordless sudo on each node (checked in Step 0; every step from Step 2
#     on runs `sudo` over ssh, where a password prompt has no tty)
# k3d and kubectl are installed via Homebrew if missing.
#
# Usage: ./scripts/setup-all.sh
#   MAC_IP=<addr>          use this address instead of detecting the LAN IP
#   BUILD_CONCURRENCY=<n>  nodes worked on at once in Steps 4 and 6 (default 4)
# ==========================================================================
set -euo pipefail

# ok/fail/info/warn, section_banner, ssh_cmd/scp_cmd/ssh_facts,
# load_network_env, detect_mac_ip, k3s_token_from_cluster, mcp_health_wait.
# The [OK]/[FAIL]/[WARN]/[....] prefixes those emit are a parsed protocol:
# lifecycle._classify_setup_line maps them to dashboard log levels while it
# streams this script. See scripts/lib/common.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

PROJECT_DIR="$AUREAX_ROOT"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

MAC_IP="${MAC_IP:-}"  # Inherits env var if set; auto-detected below otherwise

# ===================== HELPERS =====================

wait_for_ssh() {
    # A wall-clock deadline, not a count of sleeps. Counting `sleep 5` ignored
    # the failed probe itself, and each one costs up to ConnectTimeout=10
    # against a rebooting node that drops SYNs — so the advertised budget was
    # really about 3x that per node, serially, for every node that failed to
    # come back. Same idiom as the readiness waits in Step 7.
    local host="$1" user="$2" key="$3" max_wait="${4:-90}"
    local deadline=$(( $(date +%s) + max_wait ))
    info "Waiting for ${host} to come back online (up to ${max_wait}s)..."
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if ssh_cmd "$host" "$user" "$key" "true" &>/dev/null; then
            return 0
        fi
        sleep 5
    done
    return 1
}

# --- Bounded concurrent fan-out: Step 4's joins and Step 6's builds ----------
#
# Both steps do per-node work that is fully independent (the Mac only holds SSH
# sessions), and both need the same three things: a cap on how many run at
# once, FIFO draining, and a failure COUNT rather than an abort, so one bad node
# does not cost the rest of the fleet its turn. One implementation, told the
# step's own wording.
#
# Output goes to one log per node because N interleaved remote builds on a
# shared terminal are unreadable; that is as much the point as the wall clock.
FANOUT_CONCURRENCY="${BUILD_CONCURRENCY:-4}"
# Validated here rather than at first use. fanout_room's test is true against an
# EMPTY pid array when the cap is 0, so fanout_wait_one indexes FANOUT_PIDS[0]
# under `set -u` and Step 4 dies with "FANOUT_PIDS[0]: unbound variable" — a
# message about an internal array, for an operator-set knob. A non-numeric value
# is worse: `[` failing inside a `while` condition is exempt from `set -e`, so
# the fan-out runs UNBOUNDED, one "integer expression expected" per node.
if ! [[ "$FANOUT_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    fail "BUILD_CONCURRENCY" "must be a positive integer, got '${FANOUT_CONCURRENCY}'"
    exit 1
fi
FANOUT_LOG_DIR="build/logs"
declare -a FANOUT_PIDS FANOUT_HOSTS
FANOUT_FAILURES=0
FANOUT_NOUN="job"
FANOUT_SUFFIX="log"
FANOUT_OK="done"

fanout_start() {
    # fanout_start <noun> <log-suffix> <success message>
    FANOUT_PIDS=()
    FANOUT_HOSTS=()
    FANOUT_FAILURES=0
    FANOUT_NOUN="$1"
    FANOUT_SUFFIX="$2"
    FANOUT_OK="$3"
    mkdir -p "$FANOUT_LOG_DIR"
}

fanout_log() { echo "${FANOUT_LOG_DIR}/${1}.${FANOUT_SUFFIX}"; }

fanout_track() { FANOUT_PIDS+=("$1"); FANOUT_HOSTS+=("$2"); }

fanout_wait_one() {
    # Block on the oldest STARTED job (FIFO — a later job that finishes first
    # still waits its turn); record its failure without aborting the rest.
    local pid="${FANOUT_PIDS[0]}" host="${FANOUT_HOSTS[0]}"
    FANOUT_PIDS=("${FANOUT_PIDS[@]:1}")
    FANOUT_HOSTS=("${FANOUT_HOSTS[@]:1}")
    if wait "$pid"; then
        ok "${host}: ${FANOUT_OK}"
    else
        fail "${host}" "${FANOUT_NOUN} failed — see $(fanout_log "$host")"
        FANOUT_FAILURES=$((FANOUT_FAILURES + 1))
    fi
}

fanout_room()  { while [ "${#FANOUT_PIDS[@]}" -ge "$FANOUT_CONCURRENCY" ]; do fanout_wait_one; done; }
fanout_drain() { while [ "${#FANOUT_PIDS[@]}" -gt 0 ]; do fanout_wait_one; done; }

_fanout_cleanup() {
    # Ctrl-C reaches this shell but NOT its asynchronous children: bash sets
    # SIGINT to ignored for background jobs of a non-interactive shell, so
    # without this the script died while up to FANOUT_CONCURRENCY remote joins
    # or builds kept running unseen — and the next run's sync into the node's
    # staging directory (build-on-node.sh) then raced a tree an orphan was
    # still building from. Killing the local pid does not stop a docker build
    # the node's daemon has already started; it does stop this Mac from
    # starting more of them and from reporting on jobs it no longer owns.
    if [ "${#FANOUT_PIDS[@]}" -gt 0 ]; then
        kill "${FANOUT_PIDS[@]}" 2>/dev/null || true
    fi
}
trap '_fanout_cleanup; exit 130' INT TERM
trap _fanout_cleanup EXIT

# --- Staging join-node.sh on a node -----------------------------------------

push_join_script() {
    # Copy scripts/join-node.sh to a private, unpredictable path on a node and
    # print that path, relative to the node user's home (where scp lands a
    # relative destination).
    #
    # NOT a fixed name under /tmp: both callers below run the staged copy with
    # `sudo`, so any other local account on the node that can pre-create the
    # file — or swap it between the `chmod +x` and the exec — chooses what root
    # runs, and a fixed name reopens that window on every run. Same convention
    # as the Python join path (orchestrator/deploy/ssh.py::stage_path): a 0700
    # $HOME/.aureax-stage directory, a random name, and the copy removed by the
    # same remote command that executes it.
    local host="$1" user="$2" key="$3" rel
    rel=".aureax-stage/join-node-$(stage_token).sh"
    ssh_cmd "$host" "$user" "$key" \
        "mkdir -p ~/.aureax-stage && chmod 700 ~/.aureax-stage" >/dev/null || return 1
    scp_cmd "$key" scripts/join-node.sh "${user}@${host}:${rel}" >/dev/null || return 1
    printf '%s\n' "$rel"
}

# ===================== STEP 0: PREREQUISITES =====================

section_banner "Step 0: Checking Mac prerequisites"

if ! command -v uv &>/dev/null; then
    fail "uv" "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
ok "uv installed"

if ! (command -v docker &>/dev/null && docker info &>/dev/null 2>&1); then
    fail "Docker" "Install and start Docker Desktop first"
    exit 1
fi
ok "Docker Desktop running"

# Homebrew is a prerequisite this script cannot install for the operator, and
# `brew install` on a machine without it dies with a bare
# "brew: command not found" that names neither the missing tool nor the fix.
brew_install() {
    local formula="$1" manual="$2"
    if ! command -v brew &>/dev/null; then
        fail "$formula" "not installed, and Homebrew is not available to install it — see ${manual}"
        exit 1
    fi
    info "Installing ${formula} via Homebrew..."
    brew install "$formula"
}

if ! command -v k3d &>/dev/null; then
    brew_install k3d "https://k3d.io/#installation"
fi
ok "k3d available"

if ! command -v kubectl &>/dev/null; then
    brew_install kubectl "https://kubernetes.io/docs/tasks/tools/"
fi
ok "kubectl available"

# --- Read config from YAML ---
# Via load_network_env, which checks the child's exit status. Written as
# `eval "$(...)"` this could not fail: eval reports its own status, so a
# traceback left every NODE_* name undefined and the run died four steps later
# on an unbound NODE_COUNT, with the real cause scrolled off the top.
#
# It also exports $SHARD_WORKER_ROLE and $PLANNER_ROLE, the two role literals
# orchestrator/config/schema.py owns. Compared against those names rather than
# spelled out here, a rename in schema.py cannot leave the role tests below
# silently matching nothing — which would label the planner as an edge in Step 4
# and count it into EDGE_COUNT in Step 7.
info "Reading node config from configs/aureax-network.yaml..."
if ! load_network_env all; then
    fail "config" "parse-config.py all failed — see the error above"
    exit 1
fi

# Named before it is indexed. With every target `enabled: false`, parse-config
# emits `NODE_HOSTS=()` and NODE_COUNT=0, and on bash 3.2 under `set -u` the
# `${NODE_HOSTS[*]}` below dies with "NODE_HOSTS[*]: unbound variable" — a
# message about an array, for a config problem. It also keeps every
# `seq 0 $((NODE_COUNT - 1))` loop below off BSD seq's descending path, where
# `seq 0 -1` counts DOWN and yields i=0 and i=-1.
if [ "${NODE_COUNT:-0}" -le 0 ]; then
    fail "config" "no enabled ssh.targets in configs/aureax-network.yaml"
    exit 1
fi

ok "Found ${NODE_COUNT} nodes: ${NODE_HOSTS[*]}"

# Install our public key on any node that only has a factory password, BEFORE
# the preflight below — which, like every other bash path here, only speaks
# `ssh -i <key>`. Without this a brand-new node is unreachable to this script
# even though the orchestrator can log into it perfectly well, and provisioning
# a fresh box would need a manual ssh-copy-id first. Every manual step is a bug.
#
# Also clears the group-write bit on $HOME: sshd's StrictModes silently refuses
# keys when the home directory is group-writable, which presents as "key
# installed, still denied".
info "Bootstrapping SSH keys on password-auth nodes..."
uv run aureax bootstrap-keys

# Verify SSH to each node, and keep the hostname each one reports.
#
# ONE handshake per node. This used to dial twice — once to test reachability,
# then again to read the answer it had just discarded — and Step 4 dialled every
# node a third time to build NODE_HOSTNAMES for the label pass. There is no
# connection multiplexing here, so each of those is a full SSH handshake with
# ConnectTimeout=10 against a node that may be on the far side of a slow link,
# and a node's hostname does not change mid-run (nor across the Step 2 reboot).
# The label pass below reads this array instead of asking again.
declare -a NODE_HOSTNAMES
for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"
    # Passwordless sudo is asked for in the SAME round trip, because it is a
    # hard requirement nothing checked: the cgroup edit (Step 2), the governor
    # unit (2b) and the K3s join (4) all run `sudo` over ssh, which allocates no
    # tty, so a node whose sudo wants a password fails there — several steps
    # later, with a message naming the wrong cause.
    #
    # An assignment takes the status of its command substitution, and `set -e`
    # is suspended inside an `if` condition — so this both captures the answer
    # and branches on the failure, in one connection.
    if ! NODE_FACTS=$(ssh_facts "$HOST" "$USER" "$KEY" \
            "REPORTED_HOSTNAME=hostname" \
            "SUDO_OK=sudo -n true >/dev/null 2>&1 && echo yes || echo no" 2>/dev/null); then
        fail "SSH to ${HOST}" "Cannot reach ${USER}@${HOST} with key ${KEY}"
        exit 1
    fi
    # PARSED, never eval'd. This text is whatever the NODE's shell printed, so
    # `eval` would run it here with the operator's privileges. ssh_facts asks
    # for %q-quoted values, but that quoting happens ON THE NODE and so
    # constrains nothing a compromised node chooses to print — the sed parse is
    # what makes the answer inert. Same reasoning, and the same shape, as
    # scripts/lib/common.sh::detect_node_arch.
    REPORTED_HOSTNAME="$(printf '%s\n' "$NODE_FACTS" | sed -n 's/^REPORTED_HOSTNAME=//p')"
    SUDO_OK="$(printf '%s\n' "$NODE_FACTS" | sed -n 's/^SUDO_OK=//p')"
    if [ "$SUDO_OK" != "yes" ]; then
        fail "sudo on ${HOST}" "passwordless sudo is required — 'sudo -n true' failed for ${USER}@${HOST}"
        exit 1
    fi
    NODE_HOSTNAMES[$i]="$REPORTED_HOSTNAME"
    ok "SSH to ${HOST} (${REPORTED_HOSTNAME})"
done

# ===================== STEP 1: INIT =====================

section_banner "Step 1: Initialize Aureax local state"

uv run aureax init
ok "~/.aureax/ created"

# ===================== STEP 2: PREPARE EDGE NODES (cgroups) =====================

section_banner "Step 2: Prepare nodes (cgroups for K3s — firmware-cmdline boots only)"

REBOOT_LIST=()

for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"
    info "Checking cgroups on ${HOST}..."

    # ONE implementation of the cgroup logic, and it lives in join-node.sh —
    # which cannot source scripts/lib/common.sh, because it runs on the node
    # from a staging directory where the repo does not exist. So the sharing
    # goes the other way: push the script (the transport Step 4 already uses)
    # and call its --cgroups-only mode. A second inline copy of the same sed
    # here would drift — deciding from /proc/cmdline where join-node.sh decides
    # from the file makes the two answer differently on the same node.
    #
    # --cgroups-only prints one state token on stdout and exits 0 in every
    # case; it deliberately does NOT use join-node.sh's `exit 100`
    # reboot-needed protocol, which belongs to the full join path.
    #
    # Staged under a random name in a 0700 directory (push_join_script) and
    # removed by the same remote command that runs it — this copy is executed
    # with the node user's own privileges, but the Step 4 copy is sudo'd, and
    # one convention for both is what keeps the safe one from being copied from
    # the unsafe one later.
    JOIN_SCRIPT="$(push_join_script "$HOST" "$USER" "$KEY")"
    CGROUP_STATE=$(ssh_cmd "$HOST" "$USER" "$KEY" \
        "chmod +x ~/${JOIN_SCRIPT} && ~/${JOIN_SCRIPT} --cgroups-only; rc=\$?; rm -f ~/${JOIN_SCRIPT}; exit \$rc")

    case "$CGROUP_STATE" in
        OK)
            ok "${HOST}: cgroups memory already enabled (firmware cmdline)"
            ;;
        NO_FIRMWARE_CMDLINE)
            ok "${HOST}: cgroups OK (no firmware cmdline — cgroups v2)"
            ;;
        FIXED_NEEDS_REBOOT)
            REBOOT_LIST+=("$i")
            ok "${HOST}: cgroup parameters added (reboot needed)"
            ;;
        STALE_NEEDS_REBOOT)
            # Added by an earlier run that never rebooted. Same handling as a
            # fresh fix — the reboot below is what makes it live.
            REBOOT_LIST+=("$i")
            ok "${HOST}: cgroup parameters already in boot config but not live (reboot needed)"
            ;;
        *)
            # The token is a protocol between two files now, so an
            # unrecognised answer has to be loud rather than a silent skip
            # that leaves K3s unable to start on this node.
            fail "${HOST}" "unexpected cgroup state '${CGROUP_STATE}' from join-node.sh --cgroups-only"
            exit 1
            ;;
    esac
done

if [ ${#REBOOT_LIST[@]} -gt 0 ]; then
    info "Rebooting nodes that need cgroup changes..."
    for i in "${REBOOT_LIST[@]}"; do
        # A reboot kills the connection before ssh can exit cleanly, so a
        # non-zero status here is the expected outcome, not a swallowed failure
        # — the wait_for_ssh below is the real check.
        ssh_cmd "${NODE_HOSTS[$i]}" "${NODE_USERS[$i]}" "${NODE_KEY_FILES[$i]}" "sudo reboot" 2>/dev/null || true
        info "Rebooted ${NODE_HOSTS[$i]}"
    done

    for i in "${REBOOT_LIST[@]}"; do
        HOST="${NODE_HOSTS[$i]}"
        USER="${NODE_USERS[$i]}"
        KEY="${NODE_KEY_FILES[$i]}"
        if wait_for_ssh "$HOST" "$USER" "$KEY" 90; then
            VERIFY=$(ssh_cmd "$HOST" "$USER" "$KEY" "grep -q 'cgroup_memory=1' /proc/cmdline && echo 'OK' || echo 'FAIL'")
            if [ "$VERIFY" = "OK" ]; then
                ok "${HOST}: Back online, cgroups enabled"
            else
                fail "${HOST}" "Rebooted but cgroups still not active"
                exit 1
            fi
        else
            fail "${HOST}" "Did not come back online after reboot"
            exit 1
        fi
    done
fi

# ===================== STEP 2b: DVFS GOVERNORS =====================
#
# Pin CPU and DRAM to their maximum operating points on every node.
#
# WHY THIS IS NOT A MICRO-OPTIMISATION. Decode on a sharded group is bounded by
# memory bandwidth, and on RK3588 the DRAM controller is itself a DVFS domain
# ("dmc") that scales 528 <-> 2112 MHz — a 4x swing on the exact resource that
# sets the token rate. Measured on this fleet: upthreshold 40, polling_interval
# 50ms, and system_status 0x1 (SYS_STATUS_NORMAL) while target_freq sat at the
# 528 MHz minimum. A reference RK3588 device tree imposes a 1560 MHz floor at
# NORMAL; this board's DT does not appear to, so nothing holds the clock up.
#
# The workload makes it worse. llama.cpp's RPC backend runs graph splits
# strictly sequentially (the RPC device reports async=false/events=false, which
# force-disables pipeline parallelism), so each worker computes in short bursts
# at roughly a 1/n duty cycle for n workers — 1/3 today; measured at ~150 ms and
# 1/7 on the seven-worker 122B fleet. A governor that samples every 50ms and
# needs 40% sustained load cannot track that at either width: it spends the
# burst ramping and drops back before the next one. `ondemand` on the CPU side
# has the same problem, and A76 cores were observed at 408-1200 MHz against a
# 2400 MHz maximum.
#
# Cost: roughly +0.6-0.85 W per node at idle. Thermals have room — the fleet
# runs at ~64 C against an 85 C passive trip.
#
# This has to be a unit rather than a one-off sysfs write: /sys resets on every
# boot, and nodes here get wiped and re-provisioned routinely.

section_banner "Step 2b: Pin CPU and DRAM governors to performance"

for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"
    info "Installing performance-governor unit on ${HOST}..."

    # The work lives in a script rather than inline ExecStart= lines on purpose:
    # systemd runs its OWN variable expansion over ExecStart, so a shell "$p"
    # written there is replaced with an empty string before /bin/sh ever sees
    # it, and the loop silently does nothing. A script file sidesteps that
    # entirely and is far easier to read on the node.
    ssh_cmd "$HOST" "$USER" "$KEY" "
        sudo tee /usr/local/bin/aureax-performance.sh > /dev/null <<'SCRIPT'
#!/bin/sh
# Installed by scripts/setup-all.sh. Pins CPU and DRAM to their top OPP.
#
# Every write is best-effort: not all boards expose a dmc domain (the x86
# planner does not) and intel_pstate offers a different governor set, so a node
# missing one knob must still boot and still get the others. Always exits 0.
for p in /sys/devices/system/cpu/cpufreq/policy*; do
    [ -e \"\$p/scaling_available_governors\" ] || continue
    if grep -qw performance \"\$p/scaling_available_governors\"; then
        echo performance > \"\$p/scaling_governor\" 2>/dev/null || true
    fi
done

# Only the DRAM controller, deliberately. Pinning every devfreq domain would
# also peg the Mali GPU and the NPU, which this workload never touches.
if [ -e /sys/class/devfreq/dmc/governor ]; then
    echo performance > /sys/class/devfreq/dmc/governor 2>/dev/null || true
fi
exit 0
SCRIPT
        sudo chmod 0755 /usr/local/bin/aureax-performance.sh
        sudo tee /etc/systemd/system/aureax-performance.service > /dev/null <<'UNIT'
[Unit]
Description=Aureax: pin CPU and DRAM to their maximum OPP
# Ordered late so the cpufreq and devfreq drivers have registered their sysfs
# nodes by the time this runs.
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/aureax-performance.sh

[Install]
WantedBy=multi-user.target
UNIT
        sudo systemctl daemon-reload
        sudo systemctl enable --now aureax-performance.service >/dev/null 2>&1
    " || { warn "${HOST}: could not install performance-governor unit"; continue; }

    # Read back what actually took effect. A unit that installed cleanly but
    # left the clock at its minimum is the failure mode worth catching here —
    # it is invisible later and presents only as "inference is slow".
    #
    # Degrades exactly as the install three lines above does, and for the same
    # reason: a bare assignment made a transient SSH failure fatal under
    # `set -e` + pipefail, aborting the whole provisioning run over a
    # diagnostic about clock governors. `n/a` for the CPU governor is already
    # treated as a WARN below, which is the right report for a failed read.
    GOV_STATE=$(ssh_cmd "$HOST" "$USER" "$KEY" \
        'printf "%s %s %s" \
            "$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null || echo n/a)" \
            "$(cat /sys/class/devfreq/dmc/governor 2>/dev/null || echo n/a)" \
            "$(( $(cat /sys/class/devfreq/dmc/cur_freq 2>/dev/null || echo 0) / 1000000 ))"') \
        || GOV_STATE="n/a n/a 0"

    CPU_GOV=$(echo "$GOV_STATE" | cut -d' ' -f1)
    DRAM_GOV=$(echo "$GOV_STATE" | cut -d' ' -f2)
    DRAM_MHZ=$(echo "$GOV_STATE" | cut -d' ' -f3)

    # BOTH domains are checked. Reading back only the DRAM side reported [OK]
    # for a node still on `ondemand` on the CPU side — the case the rationale
    # above names explicitly (A76 cores observed at 408-1200 MHz against a
    # 2400 MHz ceiling), and the one the installed script skips silently when
    # `grep -qw performance` finds no such governor in its kernel.
    #
    # `n/a` is a pass for DRAM only. Not every board has a dmc domain — the x86
    # planner does not — but every board has cpufreq policies, so an absent CPU
    # governor means the read itself failed and is worth a WARN.
    if [ "$CPU_GOV" = "performance" ] && { [ "$DRAM_GOV" = "performance" ] || [ "$DRAM_GOV" = "n/a" ]; }; then
        ok "${HOST}: cpu=${CPU_GOV} dram=${DRAM_GOV} (${DRAM_MHZ} MHz)"
    else
        warn "${HOST}: governors are cpu=${CPU_GOV} dram=${DRAM_GOV} (${DRAM_MHZ} MHz), expected 'performance' — inference will be slow on this node"
    fi
done

# ===================== STEP 3: K3S CLUSTER ON MAC =====================

section_banner "Step 3: Create or start K3s cluster (k3d on Mac)"

# The cluster's name, API port and server count come from the config, not from
# literals. All three used to be spelled out here — `grep -q "aureax"`,
# `docker exec k3d-aureax-server-0`, `:6443`, and a node-readiness target of
# NODE_COUNT + 1 — while the Python path read kubernetes.cluster_name and
# kubernetes.k3d.* from the YAML. Editing either value therefore broke this
# script only, and did it several steps downstream of the edit: `docker exec`
# against a container name that no longer exists.
if ! load_network_env k3d; then
    fail "config" "parse-config.py k3d failed — see the error above"
    exit 1
fi

# Three possible states:
#   - Cluster doesn't exist in k3d → create it via setup-k3d.sh
#   - Cluster exists and its servers are all up → already running, no-op
#   - Cluster exists but servers are down → containers stopped (e.g. Docker
#     Desktop restart or Mac sleep). Just start it; if we proceed without
#     this, the next step's `docker exec k3d-<cluster>-server-0 ...` will fail
#     with "container is not running".
#
# Filtered by name rather than `grep -q` over the whole listing: that matched
# any substring and treated the name as a regex, so with an unrelated
# `aureax-dev` cluster present and no `aureax`, this took the "exists" branch
# and the filtered query one line down then failed outright — killing the run
# under pipefail instead of creating the cluster. Same test as setup-k3d.sh.
if k3d cluster list "$K3D_CLUSTER_NAME" --no-headers 2>/dev/null | grep -q .; then
    SERVER_STATUS=$(k3d cluster list "$K3D_CLUSTER_NAME" --no-headers 2>/dev/null | awk '{print $2}')
    if [ "$SERVER_STATUS" = "${K3D_SERVERS}/${K3D_SERVERS}" ]; then
        ok "Cluster '${K3D_CLUSTER_NAME}' already running"
    else
        info "Cluster '${K3D_CLUSTER_NAME}' exists but stopped (${SERVER_STATUS:-?} servers up) — starting..."
        k3d cluster start "$K3D_CLUSTER_NAME"
        # Give containers a few seconds to actually be ready for docker exec.
        sleep 3
        ok "Cluster '${K3D_CLUSTER_NAME}' started"
    fi
else
    bash scripts/setup-k3d.sh "$K3D_CLUSTER_NAME" "$K3D_API_PORT" "$K3D_SERVERS"
    ok "Cluster created"
fi

# Pin the context before the first kubectl call.
#
# Only the create path above ever set one, as a side effect of `k3d cluster
# create`. On the already-exists and the stopped-then-started paths nothing
# touched kubeconfig, so every kubectl below — the label pass, both readiness
# waits, the final verification — ran against whatever context the operator
# happened to leave current. Labelling nodes in someone else's cluster is not a
# failure this script could report; it succeeds.
if ! kubectl config use-context "k3d-${K3D_CLUSTER_NAME}" >/dev/null 2>&1; then
    fail "kubectl context" "no context named 'k3d-${K3D_CLUSTER_NAME}' — is the cluster registered in your kubeconfig?"
    echo "         Check: kubectl config get-contexts"
    exit 1
fi
ok "kubectl context: k3d-${K3D_CLUSTER_NAME}"

# Asks the routing table which interface actually reaches the nodes, honouring
# $MAC_IP when the caller (lifecycle, for the dashboard) already resolved it.
#
# This replaces an `ipconfig getifaddr en0 || en1` scan with an interactive
# `read -p` fallback, and both halves were wrong. Interface names do not
# identify the link — en0 is Wi-Fi on some Macs and Ethernet on others — and
# this address is written into every agent's K3S_URL (Step 4) and stamped into
# every generated config as ${orchestrator_ip} (Step 7), so the wrong one gives
# a cluster that works today and breaks on the next DHCP renewal, with
# `aureax status` reporting stale K3S_URLs right after a clean setup. The
# prompt was worse: the dashboard runs this script non-interactively, where
# `read` had no terminal and hung the run forever.
MAC_IP="$(detect_mac_ip)" || exit 1
ok "Mac LAN IP: ${MAC_IP}"

# One spelling of the container name and the token path, in common.sh, shared
# with whatever else needs it — this was written out here and again in
# setup-k3d.sh, in two different styles.
K3S_TOKEN=$(k3s_token_from_cluster "$K3D_CLUSTER_NAME" || echo "")
if [ -z "$K3S_TOKEN" ]; then
    fail "K3s token" "Could not extract from k3d container"
    exit 1
fi
ok "K3s token extracted"

# ===================== STEP 4: JOIN EDGE NODES TO CLUSTER =====================

section_banner "Step 4: Join edge nodes to K3s cluster"

K3S_URL="https://${MAC_IP}:${K3D_API_PORT}"

# Every agent is pinned to the K3s version the SERVER is already running.
# get.k3s.io installs latest stable, while the k3d server stays on the image
# bundled with the installed k3d release — so nodes joined weeks apart end up on
# different builds, usually with a kubelet NEWER than the API server, which is
# outside the supported skew and surfaces much later as an unexplained NotReady
# node. Read rather than hardcoded, so there is no constant to keep in step:
# `/version`'s gitVersion is the server's own answer and is spelled exactly the
# way INSTALL_K3S_VERSION wants it (v1.31.5+k3s1).
#
# `|| true` and an `if`, not `[ ... ] && break`: this runs under `set -e` with
# pipefail, so a kubectl that has not finished coming up would otherwise take
# the whole run down over a diagnostic. Retried for the same reason — the
# cluster may have been started rather than created a few lines above.
K3S_SERVER_VERSION=""
for _ in 1 2 3 4 5; do
    K3S_SERVER_VERSION="$(kubectl get --raw /version 2>/dev/null \
        | sed -n 's/.*"gitVersion" *: *"\([^"]*\)".*/\1/p')" || K3S_SERVER_VERSION=""
    if [ -n "$K3S_SERVER_VERSION" ]; then
        break
    fi
    sleep 2
done
K3S_VERSION_ARG=""
case "$K3S_SERVER_VERSION" in
    v[0-9]*+k3s*) K3S_VERSION_ARG="--k3s-version ${K3S_SERVER_VERSION}"
                  ok "Pinning K3s agents to the server's version (${K3S_SERVER_VERSION})" ;;
    # Not fatal: join-node.sh installs latest stable and says so. A cluster that
    # cannot answer /version has larger problems than version skew, and the
    # readiness wait below is where they surface.
    *) warn "Could not read the server's K3s version — agents will install latest stable, which may be newer than the server" ;;
esac

# Joins run CONCURRENTLY, on the same bounded fan-out as the builds in Step 6.
# Nothing in the loop depends on another node, and each node independently
# fetches and installs its own k3s agent — tens of seconds, minutes more on a
# GPU node, where join-node.sh also apt-installs the nvidia container toolkit —
# so serially the fleet paid the sum of all of them.
#
# Failure semantics are preserved rather than relaxed: a serial join aborted the
# run through `set -e`, and the drain below still exits non-zero, just after
# every other node has had its turn.
fanout_start "join" "join.log" "K3s agent active"
info "Joining ${NODE_COUNT} node(s), ${FANOUT_CONCURRENCY} at a time — logs in ${FANOUT_LOG_DIR}/"
for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"

    fanout_room
    info "Joining ${HOST} -> $(fanout_log "$HOST")"

    # THE TOKEN GOES IN ON STDIN, never as an argv word. As `$2` it is visible
    # in `ps` on both machines for the length of the install and written into
    # the node's sudo log, where it outlives the run — and it is the credential
    # that lets any machine join this cluster. join-node.sh --token-stdin reads
    # it with a single `read`; a pipe rather than a here-string so the token
    # never touches a temporary file either.
    #
    # The script itself is staged under a random name in a 0700 directory
    # (push_join_script) and removed by the command that runs it: this one is
    # handed to `sudo`, so a predictable path is a standing invitation for
    # another local account to choose what root executes. `exit $rc` preserves
    # join-node.sh's own status, including its `exit 100` reboot-needed code.
    (
        AGENT_STATUS=$(ssh_cmd "$HOST" "$USER" "$KEY" "sudo systemctl is-active k3s-agent 2>/dev/null || echo 'not-running'")
        if [ "$AGENT_STATUS" = "active" ]; then
            echo "K3s agent already running"
            exit 0
        fi
        JOIN_SCRIPT="$(push_join_script "$HOST" "$USER" "$KEY")"
        printf '%s\n' "$K3S_TOKEN" | ssh_cmd "$HOST" "$USER" "$KEY" \
            "chmod +x ~/${JOIN_SCRIPT} && sudo ~/${JOIN_SCRIPT} '${K3S_URL}' --token-stdin ${K3S_VERSION_ARG}; rc=\$?; rm -f ~/${JOIN_SCRIPT}; exit \$rc"
    ) > "$(fanout_log "$HOST")" 2>&1 &
    fanout_track "$!" "$HOST"
done
fanout_drain

if [ "$FANOUT_FAILURES" -gt 0 ]; then
    fail "K3s join" "${FANOUT_FAILURES}/${NODE_COUNT} node(s) failed to join"
    echo "         An unjoined node is never labelled and never gets a pod, and"
    echo "         a head then burns its full peer timeout waiting for a worker"
    echo "         that is never coming. Fix and re-run."
    exit 1
fi

# Budget scales with fleet size: each K3s agent takes a moment to register after
# its install finishes. A fixed 60s was already tight at 4 nodes and would be
# reached routinely at 8.
#
# This wait is load-bearing, not cosmetic. The labelling pass below acts on the
# API server's node list, so a node that has not registered yet is never
# labelled, never gets a pod, and the head then burns its full 300s peer
# timeout and exits 1 — with nothing in the setup output saying why.
NODE_READY_WAIT=$(( 30 * NODE_COUNT ))
info "Waiting for ${NODE_COUNT} node(s) to register and be Ready (up to ${NODE_READY_WAIT}s)..."
NODES_READY=0
for _ in $(seq 1 $(( NODE_READY_WAIT / 5 ))); do
    # `kubectl get nodes` counts the k3d server(s) too, hence + K3D_SERVERS
    # rather than the + 1 this used to assume: a multi-server cluster would
    # otherwise report Ready one server too early, every time.
    #
    # The leading space in " Ready" is what excludes NotReady rows; without it
    # every poll passes on the first tick and the label pass below races nodes
    # that have not registered.
    READY_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | grep -c " Ready" || true)
    if [ "$READY_COUNT" -ge $((NODE_COUNT + K3D_SERVERS)) ]; then
        NODES_READY=1
        break
    fi
    sleep 5
done
if [ "$NODES_READY" -eq 0 ]; then
    warn "Only $((READY_COUNT - K3D_SERVERS))/${NODE_COUNT} node(s) Ready after ${NODE_READY_WAIT}s — labelling below will skip whatever has not registered"
fi

echo
kubectl get nodes
echo

# NODE_HOSTNAMES was filled by the Step 0 preflight, which already asked every
# node its hostname; this pass used to ask a second time. Only edge_executor
# nodes get the label here so the DaemonSet doesn't schedule the planner pod
# until its remote MCP servers (the edges) are up.
LABEL_FAILURES=0
for i in $(seq 0 $((NODE_COUNT - 1))); do
    NODE_HOSTNAME="${NODE_HOSTNAMES[$i]}"
    if [ "${NODE_ROLES[$i]}" = "$PLANNER_ROLE" ]; then
        info "Deferring label for ${NODE_HOSTNAME} (${PLANNER_ROLE} — scheduled after edges are Ready)"
        continue
    fi
    # `|| true` on these used to hide the one failure that matters: labelling a
    # node the API server has never seen fails, and an unlabelled worker gets no
    # pod, which the head only reports 300s later as "peer unreachable". Report
    # it here, where the cause is still obvious.
    if [ "${NODE_ROLES[$i]}" = "$SHARD_WORKER_ROLE" ]; then
        # A separate label key drives the aureax-shard DaemonSet. Labelling a
        # worker as 'edge' too would also schedule it a llama-server pod with
        # no model file, which crash-loops against the very group it serves.
        if kubectl label node "$NODE_HOSTNAME" aureax.io/shard=worker --overwrite >/dev/null 2>&1; then
            ok "Labeled ${NODE_HOSTNAME} as shard worker"
        else
            fail "${NODE_HOSTNAME}" "could not apply aureax.io/shard=worker — node not registered? it will get no pod and the head will time out waiting for it"
            LABEL_FAILURES=$((LABEL_FAILURES + 1))
        fi
        continue
    fi
    if kubectl label node "$NODE_HOSTNAME" aureax.io/role=edge --overwrite >/dev/null 2>&1; then
        ok "Labeled ${NODE_HOSTNAME} as edge"
    else
        fail "${NODE_HOSTNAME}" "could not apply aureax.io/role=edge — node not registered?"
        LABEL_FAILURES=$((LABEL_FAILURES + 1))
    fi
done

if [ "${LABEL_FAILURES:-0}" -gt 0 ]; then
    warn "${LABEL_FAILURES} node(s) could not be labelled — re-run this script once they have joined"
fi

# ===================== STEP 5: DOWNLOAD MODEL =====================

section_banner "Step 5: Download LLM model (once to Mac, then SCP to each node)"

bash scripts/download-model.sh
ok "Model downloaded and distributed"

# ===================== STEP 6: BUILD DOCKER IMAGE ON EACH NODE =====================

section_banner "Step 6: Build client Docker image on each node (per-arch / GPU-aware)"

# Builds run CONCURRENTLY, on the fan-out defined in HELPERS. Each node compiles
# llama.cpp from source locally, so the work is fully independent — the Mac only
# holds SSH sessions — while serially the fleet pays N x (5-15 min).
#
# The concurrency cap is not about the Mac's CPU: the source tree rsync from
# this Mac is the one shared resource, so don't start every node at once on a
# slow uplink.
fanout_start "build" "log" "image built and imported"
info "Building on ${NODE_COUNT} node(s), ${FANOUT_CONCURRENCY} at a time — logs in ${FANOUT_LOG_DIR}/"

for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"

    fanout_room
    info "Building on ${HOST} (5-15 min on first run) -> $(fanout_log "$HOST")"
    # The port goes with the other three: build-on-node.sh takes its target on
    # argv and reads no config, so without it every build dials 22 while Steps
    # 0, 2 and 4 honour the target's `port` — a node on a non-22 port passes
    # every earlier step and then fails here, blaming the build.
    bash scripts/build-on-node.sh "$HOST" "$USER" "$KEY" "${NODE_PORTS[$i]}" > "$(fanout_log "$HOST")" 2>&1 &
    fanout_track "$!" "$HOST"
done
fanout_drain

if [ "$FANOUT_FAILURES" -gt 0 ]; then
    fail "image build" "${FANOUT_FAILURES}/${NODE_COUNT} node(s) failed to build"
    echo "         A node without aureax-client:dev stays Pending forever"
    echo "         (imagePullPolicy: Never), and a shard group is only as"
    echo "         complete as its worst member. Fix and re-run."
    exit 1
fi

# ===================== STEP 7: GENERATE + DEPLOY CONFIGS + APPLY MANIFESTS =====================

section_banner "Step 7: Generate configs, push to nodes, and apply K8s manifests"

# Generate per-node configs (resolved placeholders, deployment metadata stamped)
# and SSH-push each to /etc/aureax/aureax-edge.yaml on its host. The DaemonSet
# mounts /etc/aureax via hostPath, so each pod sees its own config.
uv run aureax deploy --orchestrator-ip "$MAC_IP"
ok "Per-node configs generated, pushed via SSH, and K8s manifests applied"

# --- Phase A: wait for edge pods to become Ready ----------------------------
# Edges are labeled (Step 4), so the DaemonSet has already scheduled their pods.
# The planner stays unscheduled until Phase B below.
EDGE_COUNT=0
WORKER_COUNT=0
for i in $(seq 0 $((NODE_COUNT - 1))); do
    case "${NODE_ROLES[$i]}" in
        "$PLANNER_ROLE") ;;
        "$SHARD_WORKER_ROLE") WORKER_COUNT=$((WORKER_COUNT + 1)) ;;
        *) EDGE_COUNT=$((EDGE_COUNT + 1)) ;;
    esac
done

# --- Phase A0: shard workers must be Ready before any head loads ------------
# A head's llama-server exits immediately if an RPC peer is unreachable when it
# loads, so waiting here is what keeps the group's first start from crash-looping.
if [ "$WORKER_COUNT" -gt 0 ]; then
    # On a fresh deploy the DaemonSet schedules every worker at once, so this is
    # bounded by the slowest single pod (startupProbe budget 5 + 24x5 = 125s),
    # not by their sum. The floor still scales with the fleet so that image
    # unpack and kubelet churn on a large group don't push it over.
    WORKER_READY_WAIT=$(( 180 + 20 * WORKER_COUNT ))
    info "Waiting for ${WORKER_COUNT} shard worker pod(s) to become Ready (up to ${WORKER_READY_WAIT}s)..."
    DEADLINE=$(( $(date +%s) + WORKER_READY_WAIT ))
    LAST_WREADY=""
    while true; do
        # `|| true` because a bare assignment makes one transient kubectl
        # failure fatal: under `set -e` + pipefail the whole run dies mid-poll,
        # after minutes of waiting. awk still prints 0 on a failed pass, so the
        # loop just retries on the next tick.
        WREADY=$(kubectl -n aureax get pods -l app=aureax-shard --no-headers 2>/dev/null | awk '$2 == "1/1" {n++} END {print n+0}' || true)
        if [ "$WREADY" -ge "$WORKER_COUNT" ]; then
            ok "All shard workers Ready (${WREADY}/${WORKER_COUNT})"
            break
        fi
        if [ "$(date +%s)" -gt "$DEADLINE" ]; then
            # Proceeding is deliberate — the head waits for its own peers and
            # will report exactly which are missing — but say plainly that the
            # group is going to come up short, rather than logging it as info.
            warn "Only ${WREADY}/${WORKER_COUNT} shard workers Ready after ${WORKER_READY_WAIT}s — the head will wait on the rest and exit if they never arrive"
            echo "         Check: kubectl -n aureax get pods -l app=aureax-shard -o wide"
            break
        fi
        # Printed only when the count moves. Firing every tick buried the one
        # signal that matters — the count changing — under hundreds of
        # identical lines in the streamed dashboard log. Same guard as the
        # planner loop below.
        if [ "$WREADY" != "$LAST_WREADY" ]; then
            info "  ... ${WREADY}/${WORKER_COUNT} shard workers Ready"
            LAST_WREADY="$WREADY"
        fi
        sleep 5
    done
fi

if [ "$WORKER_COUNT" -gt 0 ]; then
    info "Waiting for ${EDGE_COUNT} edge pod(s) to become Ready (sharded head streams every"
    info "  tensor to its workers over the LAN on first load — minutes, not seconds)..."
else
    info "Waiting for ${EDGE_COUNT} edge pod(s) to become Ready (model load ~60-90s)..."
fi
# A sharded head waits for its RPC peers and THEN streams every tensor of a
# multi-tens-of-GB model to them before /health answers; 5 minutes is a
# local-model figure and would time out every time.
#
# Derived from the head's own config rather than hardcoded. A fixed 1200s was
# already shorter than the head's configured budget after the quant swap, so
# this loop gave up while the pod was still legitimately loading — and the
# operator saw a setup failure for a deploy that was fine. Same two values the
# daemonset-edge startupProbe budget is sized against.
if [ "$WORKER_COUNT" -gt 0 ]; then
    # worker_wait normally lives on the group (the head template carries only
    # the ${worker_wait_timeout_seconds} placeholder); health_timeout is a
    # literal in the head's own config. The two are summed PER GROUP and the
    # largest sum wins, because that is the sequence one head actually performs.
    #
    # A head template that HARDCODES a number for worker_wait overrides the
    # group value when it is larger, because the supervisor reads that number
    # out of the resolved config — the same rule tests/test_probe_budget.py
    # enforces for the startupProbe budget these two values also size.
    #
    # The head's config file is resolved through its target's `config_file`
    # field, never a literal path. This block used to max worker_wait across
    # every group and then read health_timeout from a hardcoded
    # configs/aureax-shard-head.yaml, under a bare `except Exception` with
    # stderr discarded — so renaming that file, or adding a second group whose
    # head has a different budget, silently fell back to the 2700 literal
    # below. The literal happens to equal today's correct value, which is
    # exactly why the breakage would have gone unnoticed until someone raised
    # health_timeout and this loop started giving up mid-load.
    if ! EDGE_WAIT_SECS=$(uv run python -c "
import sys
from pathlib import Path

from client.config.parser import load_edge_config_raw
from orchestrator.config.loader import load_network_config

CONFIG = 'configs/aureax-network.yaml'
net = load_network_config(CONFIG)
config_file_by_id = {t.node_id: t.config_file for t in net.ssh.targets}

budgets = []
for g in net.node_groups:
    head_config = config_file_by_id.get(g.head)
    if not head_config:
        sys.exit(f'group {g.name}: head {g.head} has no config_file in ssh.targets')
    path = Path(CONFIG).parent / head_config
    head = load_edge_config_raw(str(path))
    node = head.get('node') or {}
    wait = g.worker_wait_timeout_seconds
    try:
        wait = max(wait, int(str(node.get('worker_wait_timeout_seconds')).strip()))
    except (TypeError, ValueError):
        pass  # the usual case: an unresolved template placeholder, not a number
    load = int(((head.get('model') or {}).get('server') or {}).get('health_timeout_seconds', 900))
    budgets.append(wait + load)

if not budgets:
    sys.exit('shard worker pods exist but node_groups is empty')
print(max(budgets))
"); then
        # Named, not swallowed: the message above says which group and why.
        warn "could not derive the edge readiness budget from config — falling back to 2700s"
        EDGE_WAIT_SECS=2700
    fi
else
    EDGE_WAIT_SECS=300
fi
info "  (edge readiness budget: ${EDGE_WAIT_SECS}s)"
DEADLINE=$(( $(date +%s) + EDGE_WAIT_SECS ))
LAST_READY=""
while true; do
    # ONE query per tick, both numbers derived from it: this ran the identical
    # `kubectl get pods` twice every 5s for up to EDGE_WAIT_SECS. `|| true` for
    # the same reason as the worker loop above — a transient kubectl failure
    # must cost one tick, not the run.
    EDGE_PODS=$(kubectl -n aureax get pods -l app=aureax-edge --no-headers 2>/dev/null || true)
    READY=$(printf '%s\n' "$EDGE_PODS" | awk '$2 == "1/1" {n++} END {print n+0}')
    TOTAL=$(printf '%s\n' "$EDGE_PODS" | grep -c . || true)
    if [ "$READY" -ge "$EDGE_COUNT" ]; then
        ok "All edge pods Ready (${READY}/${TOTAL})"
        break
    fi
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
        # WARN, like the worker and planner timeouts, because this one means no
        # model server ever came up after the full budget — the classifier maps
        # `[....]` to ordinary progress, and the script still exits 0.
        warn "Edge pods not all Ready after $((EDGE_WAIT_SECS / 60)) min (${READY}/${TOTAL}) — proceeding anyway"
        echo "         Check: kubectl -n aureax get pods -l app=aureax-edge -o wide"
        break
    fi
    if [ "$READY" != "$LAST_READY" ]; then
        info "  ... ${READY}/${EDGE_COUNT} edge pods Ready"
        LAST_READY="$READY"
    fi
    sleep 5
done

# --- Phase B: schedule the planner, then watch it come up -------------------
PLANNER_HOSTNAME=""
PLANNER_HOST=""
for i in $(seq 0 $((NODE_COUNT - 1))); do
    if [ "${NODE_ROLES[$i]}" = "$PLANNER_ROLE" ]; then
        PLANNER_HOSTNAME="${NODE_HOSTNAMES[$i]}"
        PLANNER_HOST="${NODE_HOSTS[$i]}"
        break
    fi
done

if [ -n "$PLANNER_HOSTNAME" ]; then
    echo
    info "=== Cognitive planner phase ==="
    info "Scheduling planner pod on ${PLANNER_HOSTNAME} (${PLANNER_HOST})."
    # The model is whatever the planner's own config_file names, so this line
    # does not spell one: a banner that states the shipped template's model as
    # fact is wrong for every operator who points the planner at another.
    info "Planner runs its configured model on GPU."
    info "First start typically: ~10-30s CUDA kernel autotune, ~30-60s model load,"
    info "then MCP host connects to ${EDGE_COUNT} edge MCP server(s)."
    # Checked, like the edge and shard branches in Step 4 and for the same
    # reason: `|| true` here hid the one failure that matters. Labelling a node
    # the API server has never seen fails, and this branch then printed [OK] and
    # blocked for ten minutes watching for a pod that was never going to be
    # scheduled — ending on an `info` line, so the run still exited 0.
    if kubectl label node "$PLANNER_HOSTNAME" aureax.io/role=edge --overwrite >/dev/null 2>&1; then
        ok "Labeled ${PLANNER_HOSTNAME} as edge — planner pod will be scheduled"
        PLANNER_LABELLED=1
    else
        fail "${PLANNER_HOSTNAME}" "could not apply aureax.io/role=edge — node not registered? no planner pod will be scheduled"
        PLANNER_LABELLED=0
    fi
fi

# Watched only if the label actually landed. Without the label the DaemonSet
# never schedules a planner pod, so the loop below would spend its full ten
# minutes confirming something already known.
if [ -n "$PLANNER_HOSTNAME" ] && [ "${PLANNER_LABELLED:-0}" = "1" ]; then
    PLANNER_DEADLINE=$(( $(date +%s) + 600 ))
    LAST_PHASE=""
    while true; do
        # Filter to the planner pod by node selector. `|| true` because both a
        # failed kubectl and `head -1` closing the pipe early yield a non-zero
        # pipeline, which under `set -e` + pipefail would end the run mid-poll;
        # the empty-pod branch below already handles "nothing came back".
        PLANNER_POD=$(kubectl -n aureax get pods -l app=aureax-edge --field-selector "spec.nodeName=${PLANNER_HOSTNAME}" --no-headers 2>/dev/null | head -1 || true)
        if [ -z "$PLANNER_POD" ]; then
            PHASE="not yet scheduled"
        else
            PSTATUS=$(echo "$PLANNER_POD" | awk '{print $3}')
            PREADY=$(echo "$PLANNER_POD" | awk '{print $2}')
            if [ "$PREADY" = "1/1" ]; then
                ok "Planner pod Ready (${PREADY}) on ${PLANNER_HOSTNAME}"
                break
            fi
            PHASE="${PSTATUS}, ready=${PREADY}"
        fi
        if [ "$PHASE" != "$LAST_PHASE" ]; then
            info "  Planner: ${PHASE}"
            LAST_PHASE="$PHASE"
        fi
        if [ "$(date +%s)" -gt "$PLANNER_DEADLINE" ]; then
            warn "Planner not Ready after 10 min (last seen: ${LAST_PHASE:-no pod})"
            echo "         Check: kubectl -n aureax describe pod --field-selector spec.nodeName=${PLANNER_HOSTNAME}"
            break
        fi
        sleep 5
    done
fi

echo
kubectl -n aureax get pods -o wide

# ===================== STEP 8: VERIFY =====================

section_banner "Step 8: Verify deployment"

echo "  K3s nodes:"
kubectl get nodes
echo

echo "  Aureax pods:"
kubectl -n aureax get pods -o wide
echo

echo "  MCP endpoints:"
# Remembered here for the closing hint below, which needs a host that actually
# answers on the MCP NodePort.
MCP_HINT_HOST=""
for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    # Shard workers and the planner serve no MCP endpoint; curling them would
    # report a false failure for a node that is behaving exactly as designed.
    if [ "${NODE_HAS_MCP_SERVER[$i]}" != "true" ]; then
        echo "    ${HOST}: n/a (${NODE_ROLES[$i]:-no MCP server})"
        continue
    fi
    [ -n "$MCP_HINT_HOST" ] || MCP_HINT_HOST="$HOST"
    # Budget 0: one probe, no wait. This is a report at the end of setup, not
    # a readiness gate — update-all.sh's is the one that polls to a deadline.
    if mcp_health_wait "$HOST" "$MCP_NODE_PORT" 0; then
        ok "MCP on ${HOST}:${MCP_NODE_PORT} — healthy"
    else
        fail "MCP on ${HOST}:${MCP_NODE_PORT}" "Not responding (pod may still be starting)"
    fi
done

# ===================== DONE =====================

section_banner "Setup Complete!"

echo "Your Aureax prototype is running:"
echo
echo "  Cluster:    kubectl get nodes"
echo "  Pods:       kubectl -n aureax get pods"
echo "  Dashboard:  uv run aureax dashboard"
# No URL spelled here. The dashboard serves the same app on two listeners and
# the command prints both: a plain-HTTP loopback one for a browser on this Mac,
# which needs no certificate at all, and an HTTPS one on the LAN address the
# fleet reaches, whose certificate the fleet CA signs. Which of the two an
# operator wants depends on where the browser is, and the port and the address
# both come from config — a fixed line here would be wrong for one of them.
echo "              (prints the URL to open, and the fleet CA to trust for the"
echo "               LAN one)"
echo "  Status:     uv run aureax status --verbose"
# The first node that actually serves MCP, not NODE_HOSTS[0]. Index 0 is the
# group head only by accident of ordering in aureax-network.yaml — disable the
# head, reorder the targets or front the planner and this line handed the
# operator a command guaranteed to fail against a node with no MCP server, the
# exact case the verification loop above skips on purpose.
if [ -n "$MCP_HINT_HOST" ]; then
    echo "  MCP:        curl --cacert ~/.aureax/tls/ca.pem https://${MCP_HINT_HOST}:${MCP_NODE_PORT}/health"
fi
echo
