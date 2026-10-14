#!/usr/bin/env bash
# ==========================================================================
# Aureax Prototype — Fast Code Update
#
# Pushes Python code changes to all edge nodes without a full rebuild, then
# redeploys and restarts them. Three steps, not one:
#
#   1. rsync the project tree to each node and rebuild the image there. The
#      llama.cpp build stage is Docker-cached, so only the Python source layers
#      are rebuilt (~30-60s per node vs 5-15 min). Nodes build concurrently.
#   2. `aureax deploy` — regenerates EVERY per-node config, SSH-pushes each to
#      /etc/aureax/aureax-edge.yaml on its host, and reapplies every manifest.
#   3. `aureax apply --all` — restarts each head/standalone node's pod onto the
#      new image and config, then this script waits for MCP to answer.
#
# A run on a two-group fleet can therefore take ~30-45 minutes, most of it in
# step 3: a sharded head streams every tensor to its peers before it is Ready.
#
# Use this for iterative development instead of re-running setup-all.sh.
# Assumes setup-all.sh has already been run at least once (Docker, K3s,
# and the initial image exist on each node).
#
# All config is read from configs/aureax-network.yaml, which is operator-owned
# and not in the repository: copy it from the shipped example first — `cp
# configs/aureax-network.example.yaml configs/aureax-network.yaml` — and edit
# in your own hosts, users and credentials.
#
# Usage: ./scripts/update-all.sh
# ==========================================================================
set -euo pipefail

# The helpers in scripts/lib/common.sh — ok/fail/info, ssh_cmd,
# load_network_env, detect_mac_ip, rsync_usable, mcp_health_wait — including the
# four this script and build-on-node.sh must not answer differently:
# project_tar/sync_project_tree (what reaches a node) and
# detect_node_arch/pick_dockerfile (which Dockerfile builds it). They live in
# one place because on a version-locked RPC fabric, two machines answering the
# arch question differently is a handshake that HANGS rather than errors.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

PROJECT_DIR="$AUREAX_ROOT"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

# The tar+scp fallback's archive, built at most ONCE per run (before the sync
# loop) and reclaimed on the way out. Its content is the whole project tree
# under a fixed exclude list, identical for every node, so building it per node
# would recompress byte-identical content N times.
#
# mktemp, not a fixed /tmp/aureax-update.tar.gz: that path owned by another user
# makes `tar czf` fail outright and abort the run, and two concurrent
# update-all.sh runs would overwrite each other's archive mid-scp and then
# delete it from the other's EXIT trap. Same convention as test-all.sh's error
# file.
UPDATE_TAR=""
_cleanup_update_tar() {
    [ -n "$UPDATE_TAR" ] && rm -f "$UPDATE_TAR"
    UPDATE_TAR=""
}

# The concurrent build fan-out's live children (see Step 1). Declared HERE,
# above the traps, so the handler never expands it unset under `set -u`.
declare -a BUILD_PIDS
BUILD_PIDS=()
_kill_builds() {
    # Ctrl-C reaches this shell but NOT its asynchronous children: bash sets
    # SIGINT to ignored for background jobs of a non-interactive shell, so
    # without this the script exits while up to BUILD_CONCURRENCY remote syncs
    # and `docker build`s keep running unseen — and the next update-all.sh run
    # then races a second concurrent build for the same aureax-client:dev tag on
    # the same node. Killing the local pid does not stop a build the node's
    # daemon has already started; it does stop this orchestrator from starting
    # more of them and from reporting on jobs it no longer owns. Same handling
    # as setup-all.sh's fan-out.
    if [ "${#BUILD_PIDS[@]}" -gt 0 ]; then
        kill "${BUILD_PIDS[@]}" 2>/dev/null || true
    fi
}
# Split traps: a handler that only cleans up does not stop the script — bash
# runs it and RESUMES, so the signal handlers must exit explicitly or a Ctrl-C
# during the sync loop goes on to build and import on every remaining node.
# 130/143 are 128+SIGINT / 128+SIGTERM. The cleanup nulls UPDATE_TAR, so the
# EXIT trap running afterwards is a no-op.
trap '_kill_builds; _cleanup_update_tar' EXIT
trap '_kill_builds; _cleanup_update_tar; exit 130' INT
trap '_kill_builds; _cleanup_update_tar; exit 143' TERM

# --- Read config from YAML ---
if ! load_network_env all; then
    fail "config" "parse-config.py all failed — see the error above"
    exit 1
fi
# `ssh.targets` has no minimum length and every target may be `enabled: false`,
# so NODE_COUNT can be 0. Caught here because the next line expands
# "${NODE_HOSTS[*]}" — an unbound-variable abort under `set -u` — and BSD
# `seq 0 -1` prints "0" and "-1" rather than iterating zero times.
if [ "${NODE_COUNT:-0}" -eq 0 ]; then
    fail "config" "no enabled SSH targets in configs/aureax-network.yaml"
    exit 1
fi

# Group topology, so the health wait below knows whether a sharded head is in
# play — its load takes minutes, not seconds.
#
# NOT a soft-fail. node_groups defaults to [] in
# orchestrator/config/schema.py, so a groupless fleet already emits
# GROUP_COUNT=0 and exits 0 all by itself; a non-zero exit here can only be a
# real read failure — a traceback, a schema change, an unreadable per-node
# config — and must never be rewritten as "no groups". Same explicit-flag shape
# as test-all.sh's group section, and the error file is mktemp'd because a fixed
# /tmp path owned by another user makes the redirect itself fail.
GROUPS_ERRFILE="$(mktemp "${TMPDIR:-/tmp}/aureax-groups-err.XXXXXX")"
if ! load_network_env groups 2>"$GROUPS_ERRFILE"; then
    fail "config" "parse-config.py groups failed"
    sed 's/^/         /' "$GROUPS_ERRFILE"
    rm -f "$GROUPS_ERRFILE"
    exit 1
fi
rm -f "$GROUPS_ERRFILE"

echo "=== Aureax Code Update ==="
echo "Nodes: ${NODE_HOSTS[*]}"
echo

# --- Step 1: Rebuild image on each node using Docker cache ---

# What reaches a node, and what does not, is common.sh::sync_project_tree's —
# one exclude list for both transfer branches and for build-on-node.sh, which
# ships the same tree. models/ and configs/ are the two that matter: a
# half-finished download leaves multi-GB chunks under models/.pfetch.<pid>/,
# and configs/aureax-network.yaml holds the SSH password for EVERY target.
#
# The tar fallback's archive is built HERE, once, rather than lazily inside the
# loop: the loop bodies run in subshells, where an assignment to $UPDATE_TAR
# would be invisible to the parent — every node would build its own archive and
# the EXIT trap would reclaim none of them. sync_project_tree reuses a
# non-empty archive it is handed, so passing it below costs no rebuild.
NEED_TAR=0
for i in $(seq 0 $((NODE_COUNT - 1))); do
    rsync_usable "${NODE_KEY_FILES[$i]}" || NEED_TAR=1
done
if [ "$NEED_TAR" -eq 1 ]; then
    UPDATE_TAR="$(mktemp "${TMPDIR:-/tmp}/aureax-update.XXXXXX")"
    info "rsync unavailable for at least one node — building one tar fallback archive"
    project_tar "$UPDATE_TAR"
fi

sync_and_build_node() {
    # Everything one node needs, start to finish. Runs in a subshell (see the
    # fan-out below), which is what keeps detect_node_arch's NODE_ARCH /
    # NODE_HAS_NVIDIA globals from being clobbered by a concurrent node.
    local host="$1" user="$2" key="$3"
    local dockerfile remote_dir

    info "[${host}] Syncing project files..."

    # The build context is the node account's own 0700 staging directory, never
    # a fixed path under /tmp: the `sudo docker build` below reads whatever is
    # in it as root, so a directory another local account can pre-create or
    # write into chooses what goes into the image. remote_build_dir fails
    # closed on a directory that is not ours.
    remote_dir=$(remote_build_dir "$host" "$user" "$key") || return 1

    # sync_project_tree picks rsync or the tar+scp fallback, and both branches
    # converge on the SAME tree: the rsync branch deletes excluded paths as
    # well as removed ones, and the fallback extracts into <dir>.new and swaps.
    # Merging into the existing directory instead would leave a module deleted
    # from the repo — or a stale configs/ — inside the `docker build` context
    # below. $UPDATE_TAR is empty unless some node needed the fallback, in
    # which case it was built once above and is reused here.
    sync_project_tree "$host" "$user" "$key" "$remote_dir" "$UPDATE_TAR"

    info "[${host}] Building Docker image (cached — Python layers only)..."

    # Pick the same Dockerfile build-on-node.sh would, through the same helper.
    # Hardcoding the CPU one silently rebuilds the wrong image on a GPU node —
    # and with a version-locked RPC fabric, one node on a different build hangs
    # its whole group at the handshake. detect_node_arch asks both facts in one
    # round trip and parses (never evals) the answer; the names it sets are
    # globals, which is safe here because this function runs in its own subshell
    # per node.
    detect_node_arch "$host" "$user" "$key" || return 1
    dockerfile=$(pick_dockerfile "$NODE_ARCH" "$NODE_HAS_NVIDIA" "$host") || return 1
    info "[${host}] ${NODE_ARCH} (NVIDIA: ${NODE_HAS_NVIDIA}) — using ${dockerfile}"

    # Rebuild. Docker cache skips the llama.cpp stage entirely.
    #
    # The flags are set INSIDE the remote shell on purpose: this script's own
    # `set -euo pipefail` governs the local ssh process, not the command ssh
    # runs. Without pipefail the pipeline exits with tail's status, so a failed
    # build reports success and the import step below silently re-pushes the
    # stale image — on a version-locked RPC fabric that is one group member left
    # on a different llama.cpp build, which hangs at the handshake. -e is what
    # makes a failed `cd ${remote_dir}` stop the block instead of building
    # whatever tree the login shell landed in.
    ssh_cmd "$host" "$user" "$key" "
        set -eo pipefail
        cd ${remote_dir}
        sudo docker build -f ${dockerfile} -t aureax-client:dev . 2>&1 | tail -5
    "

    info "[${host}] Importing into K3s containerd..."

    # Each ssh_cmd is a SEPARATE remote shell, so the flags above do not carry
    # here — this block needs its own, for the same reason and in the same form
    # build-on-node.sh uses for the byte-identical pipeline. pipefail is the
    # half that catches a `docker save` failing into a `ctr images import` that
    # succeeds on the truncated stream.
    ssh_cmd "$host" "$user" "$key" "
        set -eo pipefail
        sudo k3s ctr images rm docker.io/library/aureax-client:dev 2>/dev/null || true
        sudo docker save aureax-client:dev | sudo k3s ctr images import -
    "

    # The context has done its job, so it does not stay on the node between
    # runs: it is a full copy of the project tree, and build-on-node.sh
    # reclaims its own for the same reason. Best-effort — a failure here has
    # nothing to do with whether the image was built and imported.
    ssh_cmd "$host" "$user" "$key" "rm -rf ${remote_dir} ${remote_dir}.tar.gz ${remote_dir}.new" || true
}

# Nodes build CONCURRENTLY, the same fan-out setup-all.sh Step 6 uses for the
# identical work. Every step above is node-local — the Mac only holds SSH
# sessions — and serially the shipped 9 targets spend ~15-20 min queueing behind
# each other, most of it in `docker save | ctr images import` pushing a
# multi-GB image through containerd. Bounded because the rsync of the source
# tree from this Mac is the one shared resource. Output goes to one log per
# node: nine interleaved builds on a shared terminal are unreadable.
BUILD_LOG_DIR="build/logs"
mkdir -p "$BUILD_LOG_DIR"
# BUILD_PIDS is declared with the traps above, which kill whatever it holds.
declare -a BUILD_HOSTS_RUNNING
BUILD_HOSTS_RUNNING=()
BUILD_CONCURRENCY="${BUILD_CONCURRENCY:-4}"
# Validated before the gate below reads it, the same guard setup-all.sh puts on
# the same knob. A cap of 0 makes that gate true against an EMPTY pid array, so
# build_wait_one indexes BUILD_PIDS[0] under `set -u` and the run dies naming an
# internal array. A non-numeric value is worse: `[` failing inside a `while`
# condition is exempt from `set -e`, so the fan-out runs UNBOUNDED — every node
# rsyncs and builds at once, one "integer expression expected" per iteration.
if ! [[ "$BUILD_CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
    fail "BUILD_CONCURRENCY" "must be a positive integer, got '${BUILD_CONCURRENCY}'"
    exit 1
fi
BUILD_FAILURES=0

build_wait_one() {
    # Block on the oldest STARTED build (FIFO — a later build that finishes
    # first still waits its turn); record its failure without aborting the rest,
    # so one bad node does not cost the others their builds.
    local pid="${BUILD_PIDS[0]}" host="${BUILD_HOSTS_RUNNING[0]}"
    BUILD_PIDS=("${BUILD_PIDS[@]:1}")
    BUILD_HOSTS_RUNNING=("${BUILD_HOSTS_RUNNING[@]:1}")
    if wait "$pid"; then
        ok "${host}: Image updated"
    else
        fail "${host}" "sync/build failed — see ${BUILD_LOG_DIR}/${host}.log"
        BUILD_FAILURES=$((BUILD_FAILURES + 1))
    fi
}

info "Rebuilding on ${NODE_COUNT} node(s), ${BUILD_CONCURRENCY} at a time — logs in ${BUILD_LOG_DIR}/"
for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    USER="${NODE_USERS[$i]}"
    KEY="${NODE_KEY_FILES[$i]}"

    while [ "${#BUILD_PIDS[@]}" -ge "$BUILD_CONCURRENCY" ]; do
        build_wait_one
    done

    info "Rebuilding on ${HOST} -> ${BUILD_LOG_DIR}/${HOST}.log"
    ( sync_and_build_node "$HOST" "$USER" "$KEY" ) > "${BUILD_LOG_DIR}/${HOST}.log" 2>&1 &
    BUILD_PIDS+=("$!")
    BUILD_HOSTS_RUNNING+=("$HOST")
done

while [ "${#BUILD_PIDS[@]}" -gt 0 ]; do
    build_wait_one
done

if [ "$BUILD_FAILURES" -gt 0 ]; then
    # Aborts rather than deploying: a node still on the previous image is a
    # group member on a different llama.cpp build, and a version-locked RPC
    # handshake HANGS instead of erroring.
    fail "image update" "${BUILD_FAILURES}/${NODE_COUNT} node(s) failed — not deploying"
    exit 1
fi

# --- Step 2: Regenerate per-node configs and SSH-push them ---

# Pin the context before anything talks to a cluster. `aureax deploy` below
# reapplies every manifest through kubectl and orchestrator/cluster/kubectl.py
# passes no --context either, so this step is the first exposure, not the pod
# listing further down. Without the pin every kubectl in this run went to
# whatever context the operator happened to leave current — and these are
# MUTATING verbs, so an unrelated cluster that happens to have an `aureax`
# namespace gets its workloads restarted while the fleet this was meant to
# update is never touched. That is not a failure this script could report; it
# succeeds. Same pin, same wording, as setup-all.sh Step 3.
if ! load_network_env k3d; then
    fail "config" "parse-config.py k3d failed — see the error above"
    exit 1
fi
if ! kubectl config use-context "k3d-${K3D_CLUSTER_NAME}" >/dev/null 2>&1; then
    fail "kubectl context" "no context named 'k3d-${K3D_CLUSTER_NAME}' — is the cluster registered in your kubeconfig?"
    echo "         Check: kubectl config get-contexts"
    exit 1
fi
ok "kubectl context: k3d-${K3D_CLUSTER_NAME}"

info "Regenerating per-node configs and pushing to nodes..."
# Ask the routing table which interface actually reaches the nodes rather than
# scanning en0/en1 — interface names don't identify the link (en0 is Wi-Fi on
# some Macs, Ethernet on others), and the wrong address ends up baked into every
# node's telemetry endpoint.
#
# ABORTS rather than degrading: never fall back to a loopback or any other
# placeholder here. Whatever this resolves to is baked into every node's
# telemetry endpoint until the next generate, so a silent fallback deploys a
# fleet reporting to an address nothing reads — precisely the outcome the
# paragraph above warns about.
MAC_IP="$(detect_mac_ip)" || exit 1
# `aureax deploy` (without --setup-cluster) regenerates configs, SSH-pushes
# each per-node file to /etc/aureax/aureax-edge.yaml on its host, and reapplies
# manifests. The DaemonSet hostPath-mounts /etc/aureax so each pod sees its
# own config on next pod start.
#
# Output is captured rather than discarded, and re-emitted when the deploy
# fails. Discarding it would abort the script (set -e) having printed nothing
# but `[....] Regenerating per-node configs...` — no error, no hint, for the
# step that pushes every node's config and reapplies every manifest. Kept quiet
# on success: this is the fast iteration path, and the deploy is chatty.
if ! DEPLOY_OUTPUT=$(uv run aureax deploy --orchestrator-ip "$MAC_IP" 2>&1); then
    fail "aureax deploy" "config generation/push failed"
    printf '%s\n' "$DEPLOY_OUTPUT"
    exit 1
fi
ok "Per-node configs regenerated and pushed"

# --- Step 3: Restart pods to pick up new image + config ---
#
# `aureax apply --all`, NOT `kubectl rollout restart`. A DaemonSet rollout is
# serial across the whole fleet, so it reloads every head one after the
# other — minutes each — and tests/test_apply.py pins that the restart path
# never invokes it. apply's primitive is a targeted pod delete per node
# (orchestrator/cluster/kubectl.py::delete_pods_on_node).
#
# apply restarts HEADS AND STANDALONE NODES ONLY. Shard workers are pushed but
# never restarted: rpc-server takes no API key, and bouncing one tears the head
# off its peers and forces a group reload — the ~107s warm one, since the
# content-addressed tensor cache survives the restart, but minutes of outage
# all the same. The consequence for THIS script is that a worker keeps running
# its previous image. That is harmless for a Python-only change — which is
# what this script is for — but if LLAMA_CPP_VERSION moved in either
# Dockerfile the head and its peers are then on different builds, and the RPC
# wire format is version-locked: the group HANGS at the handshake instead of
# reporting an error. Bounce the whole fleet with
# `uv run aureax stop && uv run aureax start` after a llama.cpp version bump.
#
# apply re-runs generation and the config push that Step 2 just did. That is
# cheap and deliberate: it is the one sanctioned restart primitive, and it
# regenerates every node's config as one atomic action with the restart.
#
# Failures are collected rather than swallowed, and decide the exit status at
# the bottom: `./scripts/update-all.sh && ./scripts/test-all.sh` must not run
# the suite against a fleet whose head never came back, under a printed
# "=== Update Complete ===".
FAILURES=()

WORKER_COUNT=0
for i in $(seq 0 $((NODE_COUNT - 1))); do
    [ "${NODE_ROLES[$i]}" = "$SHARD_WORKER_ROLE" ] && WORKER_COUNT=$((WORKER_COUNT + 1))
done
if [ "$WORKER_COUNT" -gt 0 ]; then
    info "${WORKER_COUNT} shard worker(s) keep their running pod — restarting one forces a whole-group reload"
fi

info "Restarting head/standalone pods onto the new image and config..."
if ! uv run aureax apply --all --orchestrator-ip "$MAC_IP"; then
    FAILURES+=("aureax apply — one or more nodes did not restart (see the per-node outcomes above)")
fi

echo
kubectl -n aureax get pods -o wide
echo

# --- Step 4: Verify ---
#
# This poll IS the readiness wait — there is no `rollout status` above it any
# more, and no bare `sleep 5` either: the MCP server only starts after
# llama-server answers /health, so MCP lags pod Ready by a few seconds and a
# single immediate curl measured nothing but that lag.
#
# EDGE_WAIT is a CEILING on this poll, not a target, and it deliberately does
# not shrink with the model. Two relations bound it:
#
#   - It stays under the startupProbe budget in k8s/daemonset-edge.yaml (3015s).
#     Polling past that would report "MCP not responding" for a pod the kubelet
#     has already begun restarting — a different fault, reported as this one.
#   - It has to outlast a real cold load. Cold load measures ~17 s/GB, so the
#     35B group's 22.02 GB needs ~380s of tensor streaming before /health can
#     answer, and a revert to configs/aureax-shard-head.yaml's 71.7 GB model
#     needs ~1240s.
#
# 1800s therefore exceeds the 35B heads' own give-up point on purpose: those
# heads abandon a load at 300s worker wait + 900s health_timeout_seconds
# (configs/aureax-shard-head-35b.yaml) = 1200s, and the extra headroom is what
# keeps the 122B rollback head — 300s + 2400s — from being called dead while it
# is still loading normally. The loop exits as soon as every node answers, so
# the ceiling only costs wall-clock when something is actually wrong.
#
# ONE deadline for the whole pass, not one per node: the pods were restarted
# together and come up together, so a per-node budget would multiply the worst
# case by NODE_COUNT. Every node still gets at least one probe — the deadline is
# checked after the curl, not before it.
if [ "${GROUP_COUNT:-0}" -gt 0 ]; then
    EDGE_WAIT=1800
else
    EDGE_WAIT=120
fi

info "Verifying health (up to ${EDGE_WAIT}s)..."
DEADLINE=$(( $(date +%s) + EDGE_WAIT ))
for i in $(seq 0 $((NODE_COUNT - 1))); do
    HOST="${NODE_HOSTS[$i]}"
    # Shard workers and the planner serve no MCP endpoint by design. Curling
    # them reports a failure for a node behaving exactly as intended — and
    # printing [OK] instead would claim a check that never ran.
    if [ "${NODE_HAS_MCP_SERVER[$i]}" != "true" ]; then
        info "${HOST}: MCP n/a (${NODE_ROLES[$i]:-no MCP server})"
        continue
    fi
    if mcp_health_wait "$HOST" "$MCP_NODE_PORT" "$(( DEADLINE - $(date +%s) ))"; then
        ok "${HOST}: MCP healthy"
    else
        fail "${HOST}" "MCP not responding within ${EDGE_WAIT}s"
        FAILURES+=("${HOST} — MCP did not answer on :${MCP_NODE_PORT}")
    fi
done

echo
if [ "${#FAILURES[@]}" -gt 0 ]; then
    fail "update" "${#FAILURES[@]} problem(s) — the fleet is NOT fully updated"
    printf '         - %s\n' "${FAILURES[@]}"
    echo "         Check: kubectl -n aureax get pods -o wide"
    echo "                kubectl -n aureax logs -l app=aureax-edge --tail=50"
    exit 1
fi
echo "=== Update Complete ==="
