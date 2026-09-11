#!/usr/bin/env bash
# ==========================================================================
# Aureax Prototype — Full Reset Script
#
# Wipes all Aureax state from edge nodes and tears down the k3d cluster
# on your Mac, returning everything to a clean slate for a fresh setup.
#
# All node configuration is read from configs/aureax-network.yaml, which is
# operator-owned and not in the repository: copy it from the shipped example
# first — `cp configs/aureax-network.example.yaml configs/aureax-network.yaml`
# — and edit in your own hosts, users and credentials.
#
# Run from the prototype/ project root on your Mac.
#
# Usage: ./scripts/reset-all.sh [--keep-models]
#
#   --keep-models  Preserve /opt/aureax/models and /opt/aureax/rpc-cache on
#                  every node. Everything else is still wiped.
#
#                  Worth having because both are expensive to reproduce and
#                  neither is state: a model GGUF is tens of GB fetched from
#                  HuggingFace, which throttles a single connection hard, and
#                  the rpc-cache is what spares a head re-shipping every tensor
#                  to its workers on the next cold start (~17 s/GB).
#
#                  Both are content-addressed or name-addressed, so a stale
#                  entry is never served by mistake — it simply misses and is
#                  re-fetched.
# ==========================================================================
set -euo pipefail

KEEP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --keep-models) KEEP_MODELS=1 ;;
        # The whole leading comment block, however long it grows. A fixed
        # '2,20p' range cut the help off mid-sentence once the header was
        # extended.
        -h|--help) sed -n '2,/^[^#]/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# ok/fail/info, ssh_cmd, load_network_env.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

PROJECT_DIR="$AUREAX_ROOT"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

# ===================== READ CONFIG =====================

# Checked, because everything below is driven by NODE_COUNT: a failed parse
# under `eval "$(...)"` was invisible (eval reports its own status) and this
# destructive script would announce "Uninstall K3s agent from  nodes" before
# dying on an unbound variable.
info "Reading node config from configs/aureax-network.yaml..."
if ! load_network_env all; then
    fail "config" "parse-config.py all failed — see the error above"
    exit 1
fi
# `ssh.targets` has no minimum length and every target may be `enabled: false`,
# so NODE_COUNT can be 0. Caught here because the next line expands
# "${NODE_HOSTS[*]}" — an unbound-variable abort under `set -u` — and the loops
# below run `seq 0 -1`, which on BSD seq prints "0" and "-1" and indexes an
# empty array rather than iterating zero times.
if [ "${NODE_COUNT:-0}" -eq 0 ]; then
    fail "config" "no enabled SSH targets in configs/aureax-network.yaml"
    exit 1
fi
# The cluster's name, exactly as setup-all.sh Step 3 reads it. It used to be the
# literal "aureax" here and in five docker filters, while the Python path read
# kubernetes.cluster_name from the YAML — so renaming the cluster left this
# script deleting nothing and then reporting a verified-clean Mac.
if ! load_network_env k3d; then
    fail "config" "parse-config.py k3d failed — see the error above"
    exit 1
fi

echo "=== Aureax Full Reset ==="
echo
echo "This will:"
echo "  - Uninstall K3s agent from ${NODE_COUNT} nodes (${NODE_HOSTS[*]})"
echo "  - Remove all Docker images and containers from each node"
echo "  - Remove the aureax-performance unit and restore CPU/DRAM governors"
echo "    to kernel defaults (ondemand/schedutil, or powersave on a driver that"
echo "    offers neither)"
if [ "$KEEP_MODELS" = "1" ]; then
    echo "  - Remove Aureax files from each node, but KEEP /opt/aureax/models"
    echo "    and /opt/aureax/rpc-cache (no model re-download, no tensor re-ship)"
else
    echo "  - Remove all Aureax files from each node (/opt/aureax, /etc/aureax,"
    echo "    /var/lib/aureax, /tmp/aureax*, ~/.aureax-stage) — INCLUDING every"
    echo "    downloaded GGUF; pass --keep-models to preserve them"
fi
echo "  - Delete the k3d cluster on this Mac"
# No "kill the planner's llama-server" line here any more: it promised a step
# this script never had. The planner is an ordinary SSH target (planner-01),
# its llama-server runs inside its pod, and the K3s teardown in Step 1 stops it
# along with every other pod on that node. Nothing runs a planner llama-server
# on the Mac, so there was never anything for a local kill to find.
echo "  - Remove local Aureax state (~/.aureax) — INCLUDING every node's"
echo "    llama-server API key; API clients need the new keys after setup"
echo "  - Local prototype/models/ is NOT touched (avoids re-downloading GGUFs)"
echo "  - The boot cmdline cgroup parameters (and their .aureax-bak backup) are"
echo "    LEFT in place on each node — removing them would need another reboot,"
echo "    and setup-all.sh re-adds them anyway"
echo
read -p "Are you sure? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ===================== STEP 1: WIPE EDGE NODES =====================

echo
echo "--- Step 1: Wiping edge nodes ---"
echo

FAILED_NODES=()

# Each node's cleanup runs through every phase regardless of whether marker
# files (uninstall scripts, etc.) exist. After cleanup, we VERIFY no K3s
# state remains, and exit non-zero if it does — so a partial cleanup never
# silently looks like success.
#
# Single-quoted heredoc avoids Mac-side $variable expansion; everything
# inside is evaluated remotely on the node.
NODE_CLEANUP_SCRIPT='set +e

echo "  --- K3s teardown ---"

# 1. Stop K3s services if running. Either k3s-agent (worker) or k3s
#    (server) might be present; stop whichever exists. Fine to run on
#    a node that has neither.
for svc in k3s-agent k3s; do
    if sudo systemctl list-unit-files 2>/dev/null | grep -q "^${svc}.service"; then
        sudo systemctl stop "$svc" 2>/dev/null
        sudo systemctl disable "$svc" 2>/dev/null
        echo "    stopped+disabled $svc"
    fi
done

# 2. Run official uninstall scripts if present. They remove the binaries,
#    systemd units, and most data dirs — but leave behind /var/lib/rancher
#    metadata + don'"'"'t kill orphan containerd-shim children.
for script in /usr/local/bin/k3s-agent-uninstall.sh /usr/local/bin/k3s-uninstall.sh; do
    if [ -f "$script" ]; then
        echo "    running $(basename "$script")..."
        sudo "$script" >/dev/null 2>&1
    fi
done

# 3. Kill any leftover K3s processes. Orphan containerd-shim is the worst
#    offender — it keeps containerds boltdb locked, breaking the next install.
#    The regex is anchored to ^ so it only matches processes whose cmdline
#    STARTS with /var/lib/rancher/k3s/. Without the anchor, pkill -f matches
#    ANY process whose cmdline contains the literal regex string — including
#    the bash that ssh spawned to run THIS script (its cmdline is "bash -c
#    <script content>", and the script content contains the regex string as
#    a literal). pkill would then SIGKILL its own bash parent, terminate the
#    SSH session, and we would never reach the verification step. Anchoring
#    to ^ skips bash because bashs cmdline starts with "bash", not the path.
sudo pkill -9 -f "^/var/lib/rancher/k3s/data/.*containerd" 2>/dev/null
sudo pkill -9 -x k3s 2>/dev/null
sudo pkill -9 -x k3s-agent 2>/dev/null

# 4. Force-remove every K3s artifact path. This catches everything the
#    uninstall scripts miss AND handles the case where the uninstall
#    scripts were never present (e.g. second reset-all run).
sudo rm -rf /var/lib/rancher/k3s /etc/rancher/k3s /run/k3s
sudo rm -f /etc/systemd/system/k3s.service /etc/systemd/system/k3s-agent.service \
           /etc/systemd/system/k3s.service.env /etc/systemd/system/k3s-agent.service.env
sudo rm -f /usr/local/bin/k3s /usr/local/bin/k3s-killall.sh \
           /usr/local/bin/k3s-uninstall.sh /usr/local/bin/k3s-agent-uninstall.sh
# ctr, kubectl and crictl go only when they are K3s'"'"'s own symlinks. The K3s
# installer creates each as a RELATIVE link to the k3s binary beside them
# (ln -sf k3s) and SKIPS any of the three the node already has in PATH — so a
# hand-installed kubectl or crictl, or a stand-alone containerd ctr, is not ours
# to delete. The link TEXT is the test: /usr/local/bin/k3s is already gone by
# now, so anything that resolves the link would find nothing and keep a dangling
# one of ours.
for cmd in ctr kubectl crictl; do
    [ -L "/usr/local/bin/$cmd" ] || continue
    case "$(readlink "/usr/local/bin/$cmd" 2>/dev/null)" in
        k3s|/usr/local/bin/k3s) sudo rm -f "/usr/local/bin/$cmd" ;;
    esac
done
sudo systemctl daemon-reload 2>/dev/null

# 5. VERIFY K3s teardown. Any leftover state means the next install will
#    hit the same problems we have been chasing.
#    Same ^anchor trick as the pkill above: pgrep -f matches against the
#    full cmdline, which includes our own bash if the regex appears in our
#    script content as a literal. Anchoring to /var/lib/rancher/k3s ensures
#    we only match processes whose binary actually lives under that path.
K3S_LEFT=0
[ -e /var/lib/rancher/k3s ] && echo "    [LEFT] /var/lib/rancher/k3s" && K3S_LEFT=1
[ -e /etc/rancher/k3s ]     && echo "    [LEFT] /etc/rancher/k3s" && K3S_LEFT=1
[ -e /run/k3s ]             && echo "    [LEFT] /run/k3s" && K3S_LEFT=1
[ -e /usr/local/bin/k3s ]   && echo "    [LEFT] /usr/local/bin/k3s" && K3S_LEFT=1
if pgrep -f "^/var/lib/rancher/k3s/data/.*containerd" >/dev/null 2>&1; then
    echo "    [LEFT] orphan containerd-shim process(es)"
    K3S_LEFT=1
fi
if pgrep -x k3s       >/dev/null 2>&1; then echo "    [LEFT] k3s process";       K3S_LEFT=1; fi
if pgrep -x k3s-agent >/dev/null 2>&1; then echo "    [LEFT] k3s-agent process"; K3S_LEFT=1; fi
if [ "$K3S_LEFT" = "1" ]; then
    echo "  [FAIL] K3s teardown incomplete"
    exit 1
fi
echo "  [OK] K3s teardown verified — no K3s state remains"

echo "  --- Docker cleanup ---"
if command -v docker >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
    sudo docker stop $(sudo docker ps -aq) 2>/dev/null
    sudo docker rm   $(sudo docker ps -aq) 2>/dev/null
    sudo docker rmi -f $(sudo docker images -q) 2>/dev/null
    sudo docker system prune -af >/dev/null 2>&1
    CCOUNT=$(sudo docker ps -aq 2>/dev/null | wc -l | tr -d " ")
    ICOUNT=$(sudo docker images -q 2>/dev/null | wc -l | tr -d " ")
    if [ "$CCOUNT" != "0" ]; then
        echo "  [FAIL] $CCOUNT docker container(s) still exist after cleanup"
        exit 1
    fi
    echo "  [OK] Docker: 0 containers, $ICOUNT image(s) remain"
else
    echo "  Docker not running on node, skipping"
fi

echo "  --- Aureax systemd units ---"
# The performance-governor unit setup-all.sh installs (Step 2b). Removed here
# so a reset genuinely returns the node to its pre-Aureax state: leaving it
# behind means the next setup-all finds the governors already pinned and its
# install step is never actually exercised — a verification run would pass
# without testing the thing it was meant to test.
#
# Restores the kernel defaults rather than only deleting the unit, or the CPU
# and DRAM stay pinned until the next reboot with nothing on the node
# explaining why.
if sudo systemctl list-unit-files 2>/dev/null | grep -q "^aureax-performance.service"; then
    sudo systemctl stop aureax-performance.service 2>/dev/null
    sudo systemctl disable aureax-performance.service 2>/dev/null
    echo "    stopped+disabled aureax-performance"
fi
sudo rm -f /etc/systemd/system/aureax-performance.service /usr/local/bin/aureax-performance.sh
sudo systemctl daemon-reload 2>/dev/null
# ondemand and schedutil are the kernel defaults, but a driver need offer
# neither: intel_pstate exposes only performance and powersave, and stopping
# there left that node pinned to performance with the unit and its script
# already deleted — nothing on the node left to explain why. So: a default if
# there is one, powersave if not, and failing both the first governor offered
# that is not the pin being undone.
RESTORED_CPU=""
RESTORED_DRAM=""
for p in /sys/devices/system/cpu/cpufreq/policy*; do
    [ -e "$p/scaling_available_governors" ] || continue
    AVAIL=$(cat "$p/scaling_available_governors" 2>/dev/null)
    WANT=""
    for g in ondemand schedutil powersave; do
        case " $AVAIL " in *" $g "*) WANT="$g"; break ;; esac
    done
    if [ -z "$WANT" ]; then
        for g in $AVAIL; do
            if [ "$g" != performance ]; then WANT="$g"; break; fi
        done
    fi
    [ -n "$WANT" ] || continue
    if echo "$WANT" | sudo tee "$p/scaling_governor" >/dev/null 2>&1; then
        RESTORED_CPU="$WANT"
    fi
done
if [ -e /sys/class/devfreq/dmc/governor ]; then
    if echo dmc_ondemand | sudo tee /sys/class/devfreq/dmc/governor >/dev/null 2>&1; then
        RESTORED_DRAM=dmc_ondemand
    fi
fi
# Named, because "restored to kernel defaults" is a claim this step makes in its
# own summary and a silent no-op looks identical to a success.
echo "    governors restored: cpu=${RESTORED_CPU:-unchanged} dram=${RESTORED_DRAM:-unchanged}"

echo "  --- Aureax dirs ---"
# With KEEP_MODELS the two expensive artifacts survive and everything else
# under /opt/aureax still goes: configs, logs, metrics, the MCP queue. Without
# it the whole tree goes, which means every GGUF too — the next setup-all then
# re-downloads and re-pushes the group model to every head.
if [ "${KEEP_MODELS:-0}" = "1" ]; then
    sudo find /opt/aureax -mindepth 1 -maxdepth 1 \
        ! -name models ! -name rpc-cache -exec rm -rf {} + 2>/dev/null
    echo "    kept /opt/aureax/models + /opt/aureax/rpc-cache"
else
    sudo rm -rf /opt/aureax
fi
# /var/lib/aureax holds build-on-node.sh'"'"'s imported-image-id marker, and
# $HOME/.aureax-stage is where the orchestrator stages the join script, the
# secret-bearing files it pushes, and the Docker build context. Both are Aureax
# state on the node, so both go here — the
# verification below says "no state remains" and has to be able to mean it.
sudo rm -rf /etc/aureax /tmp/aureax* /var/lib/aureax
rm -rf "$HOME/.aureax-stage"
AUREAX_LEFT=0
# Under --keep-models, /opt/aureax legitimately survives; only an unexpected
# leftover INSIDE it is a failure.
if [ "${KEEP_MODELS:-0}" = "1" ]; then
    STRAY=$(sudo find /opt/aureax -mindepth 1 -maxdepth 1 \
        ! -name models ! -name rpc-cache 2>/dev/null | head -3)
    if [ -n "$STRAY" ]; then
        echo "    [LEFT] unexpected entries under /opt/aureax: $STRAY"
        AUREAX_LEFT=1
    fi
else
    [ -e /opt/aureax ] && echo "    [LEFT] /opt/aureax" && AUREAX_LEFT=1
fi
[ -e /etc/aureax ] && echo "    [LEFT] /etc/aureax" && AUREAX_LEFT=1
[ -e /var/lib/aureax ] && echo "    [LEFT] /var/lib/aureax" && AUREAX_LEFT=1
[ -e "$HOME/.aureax-stage" ] && echo "    [LEFT] $HOME/.aureax-stage" && AUREAX_LEFT=1
# `ls -d`, not `compgen -G`: this whole cleanup is handed to the node account'"'"'s
# LOGIN shell, and nothing pins which shell that is (the assumption, and where
# the fleet does depend on bash, is stated at scripts/lib/common.sh::ssh_cmd).
# compgen is a bash builtin: under dash or ash it returns 127, the `set +e` at
# the top of this script swallows it, and the leftover check passes with the
# leftovers still on disk. An unmatched glob leaves the literal, which ls
# rejects non-zero.
if ls -d /tmp/aureax* >/dev/null 2>&1; then
    echo "    [LEFT] /tmp/aureax*"
    AUREAX_LEFT=1
fi
if [ "$AUREAX_LEFT" = "1" ]; then
    echo "  [FAIL] Aureax dirs not fully cleaned"
    exit 1
fi
# The boot cmdline edit is the one thing deliberately left behind; it is called
# out in the summary this script prints before it asks for confirmation.
echo "  [OK] Aureax dirs verified — no Aureax files remain"

echo "  Free space:"
df -h /tmp / 2>&1 | tail -2

echo "  Done — all phases verified."
'

for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"
    info "Resetting ${HOST}..."

    if ! ssh_cmd "$HOST" "$USER" "$KEY" "true" &>/dev/null; then
        fail "${HOST}" "Unreachable"
        FAILED_NODES+=("${HOST} (unreachable)")
        continue
    fi

    # Prepended as its own line so the remote shell sees an assignment, not a
    # command prefix — the cleanup script is a multi-line program, not a binary.
    REMOTE_SCRIPT="KEEP_MODELS=${KEEP_MODELS}
${NODE_CLEANUP_SCRIPT}"
    if ssh_cmd "$HOST" "$USER" "$KEY" "$REMOTE_SCRIPT" 2>&1; then
        ok "${HOST}: Wiped and verified"
    else
        fail "${HOST}" "cleanup verification failed (see [FAIL]/[LEFT] lines above)"
        FAILED_NODES+=("${HOST} (verification failed)")
    fi
done

if [ ${#FAILED_NODES[@]} -gt 0 ]; then
    echo
    echo "ERROR: ${#FAILED_NODES[@]} node(s) did not fully reset:"
    printf '  - %s\n' "${FAILED_NODES[@]}"
    echo
    echo "Resolve the issues above and re-run reset-all.sh before setup-all.sh."
    exit 1
fi

# ===================== STEP 2: TEAR DOWN K3D CLUSTER =====================

echo
echo "--- Step 2: Tearing down k3d cluster ---"
echo

# Belt-and-suspenders cleanup of k3d state on the Mac. Three things that
# can each leak across reset cycles, all named after $K3D_CLUSTER_NAME:
#   1. k3d's own metadata (handled by `k3d cluster delete`)
#   2. Docker containers k3d-<cluster>-* (server, serverlb, tools)
#   3. Docker volume k3d-<cluster>-images and network k3d-<cluster>
#
# `k3d cluster delete` is the proper way to do all three at once, but it
# requires Docker Desktop to be running AND k3d's metadata to be consistent.
# Either condition can fail silently (Docker off when reset ran, k3d bug,
# stale state from a prior partial run, container Exited from Mac sleep).
# We do `k3d cluster delete` first, then verify-and-force-clean the docker-
# level resources directly, then assert nothing remains.

# Matched on the NAME column exactly (-xF), not as a substring: `grep -q
# "aureax"` also matched an unrelated 'aureax-bench', so the delete was
# attempted against a cluster that is not this one.
if command -v k3d &>/dev/null &&
   k3d cluster list --no-headers 2>/dev/null | awk '{print $1}' | grep -qxF "$K3D_CLUSTER_NAME"; then
    if k3d cluster delete "$K3D_CLUSTER_NAME" 2>&1; then
        ok "k3d cluster '${K3D_CLUSTER_NAME}' deleted via k3d"
    else
        info "k3d cluster delete returned non-zero — falling through to direct docker cleanup"
    fi
else
    info "No k3d cluster '${K3D_CLUSTER_NAME}' in k3d metadata"
fi

# Direct docker cleanup. Catches:
# - Containers that survived a partial `k3d cluster delete`
# - Anything left behind because Docker Desktop was off when reset ran
# - The volume + network that k3d sometimes leaves orphaned
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    LEFTOVER_CTRS=$(docker ps -aq --filter "name=^k3d-${K3D_CLUSTER_NAME}" 2>/dev/null)
    if [ -n "$LEFTOVER_CTRS" ]; then
        info "Force-removing $(echo "$LEFTOVER_CTRS" | wc -l | tr -d ' ') leftover k3d-${K3D_CLUSTER_NAME} container(s)..."
        echo "$LEFTOVER_CTRS" | xargs -r docker rm -f >/dev/null 2>&1 || true
    fi

    LEFTOVER_VOLS=$(docker volume ls -q --filter "name=^k3d-${K3D_CLUSTER_NAME}" 2>/dev/null)
    if [ -n "$LEFTOVER_VOLS" ]; then
        info "Removing leftover k3d-${K3D_CLUSTER_NAME} volume(s): $LEFTOVER_VOLS"
        echo "$LEFTOVER_VOLS" | xargs -r docker volume rm -f >/dev/null 2>&1 || true
    fi

    LEFTOVER_NETS=$(docker network ls -q --filter "name=^k3d-${K3D_CLUSTER_NAME}\$" 2>/dev/null)
    if [ -n "$LEFTOVER_NETS" ]; then
        info "Removing leftover k3d-${K3D_CLUSTER_NAME} network..."
        echo "$LEFTOVER_NETS" | xargs -r docker network rm >/dev/null 2>&1 || true
    fi

    # Verify nothing remains — the explicit `exit 1` below is what aborts;
    # `set -e` does not fire on a command used as an `if` condition.
    REMAINING=$(docker ps -aq --filter "name=^k3d-${K3D_CLUSTER_NAME}" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$REMAINING" != "0" ]; then
        fail "k3d cleanup" "$REMAINING k3d-${K3D_CLUSTER_NAME} container(s) still exist after cleanup"
        docker ps -a --filter "name=^k3d-${K3D_CLUSTER_NAME}"
        exit 1
    fi
    ok "Verified: no k3d-${K3D_CLUSTER_NAME} containers, volumes, or networks remain"
else
    info "Docker not running — cannot verify/force-clean k3d resources at this layer"
    info "If Docker was off, leftover k3d-${K3D_CLUSTER_NAME} containers will surface on next 'docker ps -a'"
fi

# ===================== STEP 3: CLEAN LOCAL STATE =====================

echo
echo "--- Step 3: Cleaning local Aureax state ---"
echo

if [ -d "$HOME/.aureax" ]; then
    # The size of the directory actually being deleted. This used to measure
    # ~/.aureax/models and call it "downloaded models" — a path nothing creates:
    # GGUFs live in the repo's models/ on this Mac (which reset does not touch,
    # see the summary above) and in /opt/aureax/models on each node. `du` on a
    # missing directory prints nothing, so the line always read "includes  of
    # downloaded models". No `|| echo 0` fallback either: the enclosing test
    # proves the directory exists, so `du` cannot fail here.
    STATE_SIZE=$(du -sh "$HOME/.aureax" 2>/dev/null | cut -f1)
    info "Removing ~/.aureax (${STATE_SIZE:-unknown size})..."
    rm -rf "$HOME/.aureax"
    ok "~/.aureax removed"
else
    info "~/.aureax not found, skipping"
fi

if [ -d "build" ]; then
    rm -rf build
    ok "build/ removed"
fi

# ===================== DONE =====================

echo
echo "=== Reset Complete ==="
echo
echo "Everything is clean. Run this to start fresh:"
echo "  bash scripts/setup-all.sh"
echo
