#!/usr/bin/env bash
# Build the Aureax client Docker image natively on an edge node,
# then import it into K3s containerd.
#
# Usage: ./scripts/build-on-node.sh <node_host> [node_user] [ssh_key] [ssh_port]
#
# Runnable from anywhere: every path below is anchored to $AUREAX_ROOT.
set -euo pipefail

# ssh_cmd/scp_cmd (which quote the key path), detect_node_arch, pick_dockerfile
# and sync_project_tree.
#
# This script takes its target on argv and never reads the network config, so
# the SSH PORT has to arrive the same way — see the one-element NODE_* arrays
# below. A bash array cannot be exported, so setup-all.sh's `all` load does not
# reach here, and without the argument every build would dial 22 while the steps
# that do read the config honour the target's `port`.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

# Anchors the build to the repo regardless of the caller's directory. The
# transfer itself takes its source from $AUREAX_ROOT, so this is belt and
# braces for anything relative added later.
cd "$AUREAX_ROOT"

NODE_HOST="${1:?Usage: $0 <node_host> [node_user] [ssh_key] [ssh_port]}"
# Matches SSHTarget.user's schema default. A default that exists on no node in
# any shipped config makes omitting the argument an auth failure every time.
NODE_USER="${2:-aureax}"
# $HOME, not `~`: tilde expansion does not happen inside double quotes, so a
# documented default of "~/.ssh/id_ed25519" would be that literal string — a
# path that resolves nowhere, which is auth failure every time the argument is
# omitted.
SSH_KEY="${3:-$HOME/.ssh/id_ed25519}"
# The lockstep NODE_* arrays common.sh::node_ssh_port resolves a port through,
# synthesised here for this one target. Exporting a port on its own would not
# work: node_ssh_port looks the host up in NODE_HOSTS and reads the matching
# NODE_PORTS entry, so the arrays are the interface.
NODE_COUNT=1
NODE_HOSTS=("$NODE_HOST")
NODE_PORTS=("${4:-22}")
# Set once the node is reachable (step 2 below), because preparing the staging
# directory is itself an SSH round trip.
REMOTE_DIR=""

echo "=== Build Aureax Client on Node ==="
echo "Host: ${NODE_USER}@${NODE_HOST}:${NODE_PORTS[0]}"
echo "Key:  ${SSH_KEY}"
echo

# 1. Ensure Docker + rsync are installed on the node
# rsync is the preferred transport for step 2; the tar+scp fallback covers a
# node or key path it can't handle. Both ends of an rsync transfer need the
# binary, so we install it here even if it's already present on the
# orchestrator Mac.
echo "[1/5] Checking Docker + rsync on node..."
# Bash arrays in a remote snippet: this path assumes the node account's login
# shell is bash, the assumption stated once at scripts/lib/common.sh::ssh_cmd.
#
# The node account is NOT added to the `docker` group. Membership in it is
# root-equivalent on that host, and nothing needs it: every call that reaches
# the daemon — here, in update-all.sh, in test-all.sh and in
# orchestrator/cluster/buildcheck.py — already goes through sudo, and the
# `docker --version` below never touches the socket.
ssh_cmd "$NODE_HOST" "$NODE_USER" "$SSH_KEY" "
    NEED_INSTALL=()
    command -v docker &>/dev/null || NEED_INSTALL+=(docker.io)
    command -v rsync  &>/dev/null || NEED_INSTALL+=(rsync)
    if [ \${#NEED_INSTALL[@]} -gt 0 ]; then
        echo \"Installing: \${NEED_INSTALL[*]}\"
        sudo apt-get update -qq
        sudo apt-get install -y -qq \"\${NEED_INSTALL[@]}\"
        if printf '%s\n' \"\${NEED_INSTALL[@]}\" | grep -qx 'docker.io'; then
            sudo systemctl enable --now docker
        fi
    else
        echo 'Docker + rsync already installed'
    fi
    docker --version
    rsync --version | head -1
"

# 1b. Detect arch + GPU on the node, pick the right Dockerfile.
#
# Both the probe and the mapping live in scripts/lib/common.sh, shared with
# update-all.sh. They used to be written out twice, once here and once there,
# with a comment in each pointing at the other and asking that they be kept
# identical by hand — and they must be, because setup-all.sh builds through this
# script while update-all.sh builds through its own loop, so a divergence gives
# one machine two different images and a sharded group a version-locked RPC
# handshake that hangs rather than errors.
#
# Both helpers return non-zero (rather than exiting) on an unreachable or
# unsupported machine, so the `|| exit 1`s here are what preserve this script's
# hard failure for the caller that builds one node at a time.
detect_node_arch "$NODE_HOST" "$NODE_USER" "$SSH_KEY" || exit 1
DOCKERFILE=$(pick_dockerfile "$NODE_ARCH" "$NODE_HAS_NVIDIA" "$NODE_HOST") || exit 1
echo "Node is ${NODE_ARCH} (NVIDIA: ${NODE_HAS_NVIDIA}) — using ${DOCKERFILE}"

# 2. Sync project files to node
echo
echo "[2/5] Syncing project files to node..."
# The build context is the node account's own 0700 staging directory, never a
# fixed path under /tmp: step 3 hands it to `sudo docker build`, so a directory
# another local account can pre-create or write into chooses what root's daemon
# builds. remote_build_dir also fails closed on a directory that is not ours.
REMOTE_DIR=$(remote_build_dir "$NODE_HOST" "$NODE_USER" "$SSH_KEY") || exit 1

# ONE transport, in scripts/lib/common.sh, shared with update-all.sh: rsync when
# the key path allows it, tar+scp otherwise, with the same exclude list either
# way, and both branches converge on the same tree — nothing an exclude covers
# survives on the receiver. It also owns the fallback tarball, which it removes
# on both paths, so a failed build leaves no copy of the project tree behind.
sync_project_tree "$NODE_HOST" "$NODE_USER" "$SSH_KEY" "$REMOTE_DIR"

echo "Files synced."

# 3. Build the Docker image on the node
echo
echo "[3/5] Building Docker image on node (this takes a while on first run)..."
ssh_cmd "$NODE_HOST" "$NODE_USER" "$SSH_KEY" "
    cd ${REMOTE_DIR}
    sudo docker build -f ${DOCKERFILE} -t aureax-client:dev .
"

# 4. Import into K3s containerd
echo
echo "[4/5] Importing image into K3s containerd..."
ssh_cmd "$NODE_HOST" "$NODE_USER" "$SSH_KEY" "
    # The remote shell has its own settings — this script's \`set -euo pipefail\`
    # governs the local ssh process only. Without these two, a block's status is
    # its LAST command: a failed save|import followed by an echo exited 0, and
    # setup-all.sh printed '[OK] <host>: image built and imported' for a node
    # with no image, which then stays Pending forever under imagePullPolicy:
    # Never. pipefail is the half that catches a \`docker save\` failing into a
    # \`ctr import\` that succeeds on the truncated stream.
    set -eo pipefail
    if ! command -v k3s >/dev/null 2>&1; then
        # FAILS rather than warning. The image only reaches a pod through K3s's
        # containerd, so 'saved in Docker only' means this node has nothing to
        # run: with imagePullPolicy: Never its pod stays Pending forever, and a
        # shard group is only as complete as its worst member. Exiting 0 here
        # made setup-all.sh report '[OK] <host>: image built and imported' for
        # exactly that node — the caller cannot tell the difference otherwise.
        echo 'ERROR: K3s is not installed on this node — image built but NOT imported.' >&2
        echo '       Join the K3s agent first (setup-all.sh Step 4), then re-run.' >&2
        exit 1
    fi

    # Skip the save|import when containerd already holds exactly this image.
    # setup-all.sh is an idempotent bootstrap that gets re-run, and a cache-hit
    # build still wrote 1 GB+ through save|import on every node, every time.
    #
    # BOTH halves are required, and the marker is written only after an import
    # of that exact id succeeds: an id match alone would skip against a node
    # whose containerd was wiped, and the tag alone proves nothing about the
    # CONTENT. A wrong skip leaves one group member on a different llama.cpp
    # build, which hangs at the RPC handshake instead of erroring.
    IMAGE_ID=\"\$(sudo docker image inspect -f '{{.Id}}' aureax-client:dev)\"
    MARKER=/var/lib/aureax/imported-image-id
    if [ \"\$IMAGE_ID\" = \"\$(cat \"\$MARKER\" 2>/dev/null || true)\" ] &&
       sudo k3s ctr images ls -q | grep -qx 'docker.io/library/aureax-client:dev'; then
        echo 'Image unchanged since the last import — skipping save/import.'
        exit 0
    fi

    echo 'Removing old image from K3s containerd (if any)...'
    sudo k3s ctr images rm docker.io/library/aureax-client:dev 2>/dev/null || true
    echo 'Importing new image...'
    sudo docker save aureax-client:dev | sudo k3s ctr images import -
    sudo mkdir -p \"\$(dirname \"\$MARKER\")\"
    printf '%s\n' \"\$IMAGE_ID\" | sudo tee \"\$MARKER\" >/dev/null
    echo 'Image imported into K3s containerd'
"

# 5. Cleanup
echo
echo "[5/5] Cleaning up build files..."
ssh_cmd "$NODE_HOST" "$NODE_USER" "$SSH_KEY" \
    "rm -rf ${REMOTE_DIR} ${REMOTE_DIR}.tar.gz ${REMOTE_DIR}.new"

echo
echo "=== Done ==="
echo "Image aureax-client:dev is ready on ${NODE_HOST}"
