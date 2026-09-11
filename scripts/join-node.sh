#!/usr/bin/env bash
# Install K3s agent on an edge node and join the cluster.
# This script is meant to be run ON the node (pushed via SSH by the orchestrator).
#
# Handles fresh Linux installs:
#   - Enables cgroups memory (required by K3s, not enabled by default on some Linux distros)
#   - If cgroups were just enabled, exits with code 100 to signal "reboot needed"
#     (consumed by k3s_agent.join_agent, which turns it into a "reboot and
#     re-run setup-all.sh" message instead of a generic join failure)
#   - On a host with a working nvidia-smi: installs nvidia-container-toolkit
#     (which adds NVIDIA's apt repo and GPG keyring), enables
#     --default-runtime=nvidia, and removes the toolkit's minimal containerd
#     template, which otherwise breaks K3s's CRI plugin
#   - Installs K3s agent and joins the cluster
#
# Usage:
#   join-node.sh <k3s_server_url> --token-stdin        token on stdin
#   join-node.sh <k3s_server_url> --token-file <path>  token read from <path>,
#                                                      which the caller must
#                                                      create 0600 and remove
#   join-node.sh --cgroups-only                        the cgroup step alone, exit 0
#
# Any of the join forms also accepts:
#   --k3s-version <v>   install this K3s version instead of latest stable
#                       (also read from $INSTALL_K3S_VERSION)
#
# --cgroups-only exists so setup-all.sh's Step 2 can run EXACTLY this logic
# rather than carrying a second copy of the same sed. The sharing has to run in
# this direction: this script is staged on a node and executed there, where
# scripts/lib/common.sh does not exist, so it can never source the shared lib.
# Two copies drift — one deciding from the cmdline FILE and the other from
# /proc/cmdline is how a node setup-all.sh has just declared healthy still hits
# the `exit 100` path here.
set -euo pipefail

USAGE="Usage: $0 <k3s_server_url> {--token-stdin | --token-file <path>} [--k3s-version <v>]
       $0 --cgroups-only"

# HOW THE TOKEN ARRIVES. --token-stdin and --token-file exist because an argv
# word is public: it is visible in `ps` on both machines for the length of the
# install, and sudo writes the whole command line into the node's auth log,
# where it outlives the run. The token is the one credential that lets any
# machine join this cluster; the far weaker llama-server API key is held to the
# same rule (see ARCHITECTURE.md, "llama-server API-key authentication").
# There is deliberately no positional form that would take the token from argv:
# the two callers are orchestrator/deploy/k3s_agent.py, which stages a
# --token-file, and setup-all.sh, which pipes it on stdin.
CGROUPS_ONLY=0
# The version to install, from the environment unless --k3s-version overrides
# it. Empty means "latest stable", which get.k3s.io picks when
# INSTALL_K3S_VERSION is unset or empty — see the install step below for why
# that is a warning rather than the normal case.
K3S_VERSION="${INSTALL_K3S_VERSION:-}"
if [ "${1:-}" = "--cgroups-only" ]; then
    CGROUPS_ONLY=1
else
    K3S_SERVER_URL="${1:?$USAGE}"
    case "${2:?$USAGE}" in
        --token-stdin)
            # `|| true`: read reports EOF-without-newline as a failure, and the
            # emptiness check below is the real guard.
            IFS= read -r K3S_JOIN_TOKEN || true
            shift 2
            ;;
        --token-file)
            TOKEN_FILE="${3:?$USAGE}"
            IFS= read -r K3S_JOIN_TOKEN < "$TOKEN_FILE" || true
            shift 3
            ;;
        *)
            echo "ERROR: the token must arrive on stdin or in a file, not on the command line" >&2
            echo "$USAGE" >&2
            exit 1
            ;;
    esac
    # Whatever follows the token form. An unrecognised word is rejected rather
    # than ignored: a mistyped flag that silently did nothing would leave the
    # agent unpinned, which is exactly the failure --k3s-version exists to stop.
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --k3s-version)
                K3S_VERSION="${2:?--k3s-version needs a value}"
                shift 2
                ;;
            *)
                echo "ERROR: unrecognised argument '$1'" >&2
                echo "$USAGE" >&2
                exit 1
                ;;
        esac
    done
    if [ -z "${K3S_JOIN_TOKEN:-}" ]; then
        echo "ERROR: empty K3s join token" >&2
        echo "$USAGE" >&2
        exit 1
    fi
fi

cgroup_cmdline_file() {
    # The boot cmdline file this node boots from, or nothing at all when it has
    # none — which is the normal case on a cgroups-v2 boot and means there is
    # nothing here to patch.
    #
    # A function rather than a global because enable_cgroups is called through
    # a command substitution (its stdout is the state token), so anything it
    # assigns dies with that subshell.
    if [ -f /boot/firmware/cmdline.txt ]; then
        echo "/boot/firmware/cmdline.txt"
    elif [ -f /boot/cmdline.txt ]; then
        # Older single-board-computer layout
        echo "/boot/cmdline.txt"
    fi
}

enable_cgroups() {
    # Ensure cgroup memory is enabled on a firmware-cmdline boot. Idempotent.
    #
    # Prints exactly ONE state token on stdout and sends everything else to
    # stderr, so a caller can read it with a plain command substitution:
    #   OK                   cgroup_memory=1 is live in /proc/cmdline
    #   NO_FIRMWARE_CMDLINE  no cmdline.txt at all — a cgroups-v2 boot
    #   FIXED_NEEDS_REBOOT   parameters just added; the node must reboot
    #   STALE_NEEDS_REBOOT   already in the file but not live; must reboot
    #
    # ALWAYS returns 0. "Reboot needed" is the token, not the status: the
    # `exit 100` protocol belongs to the full join path below and has exactly
    # one consumer, so there is no reason to grow a second dialect of it.
    #
    # /proc/cmdline is consulted BEFORE the file, which is setup-all.sh's
    # ordering and the one that matters: if the running kernel already has the
    # parameter, K3s works and there is nothing to fix, whatever the file says.
    local cmdline_file
    cmdline_file="$(cgroup_cmdline_file)"
    if [ -z "$cmdline_file" ]; then
        echo "NO_FIRMWARE_CMDLINE"
        return 0
    fi

    if grep -q 'cgroup_memory=1' /proc/cmdline 2>/dev/null; then
        echo "OK"
        return 0
    fi

    if grep -q 'cgroup_memory=1' "$cmdline_file"; then
        # Written by an earlier run that never rebooted.
        echo "STALE_NEEDS_REBOOT"
        return 0
    fi

    echo "Enabling cgroup memory in ${cmdline_file}..." >&2
    # Anchored to line 1 and backed up, because this is the file that decides
    # whether the node boots at all. A bare `s/$/.../` appends to EVERY line, so
    # a cmdline file that is not exactly one line gets the parameters repeated
    # and its boot arguments malformed — and a node that fails to boot is
    # unreachable over SSH, recoverable only with physical access.
    # <cmdline>.aureax-bak is left beside it so the edit can be undone from a
    # serial console or an offline mount.
    sudo sed -i.aureax-bak '1s/$/ cgroup_memory=1 cgroup_enable=memory/' "$cmdline_file"
    echo "FIXED_NEEDS_REBOOT"
}

if [ "$CGROUPS_ONLY" = "1" ]; then
    enable_cgroups
    exit 0
fi

echo "=== Joining K3s cluster ==="
echo "Server: $K3S_SERVER_URL"

# --- Step 1: Ensure cgroups memory is enabled (firmware-cmdline boots only) ---
CGROUP_STATE="$(enable_cgroups)"
case "$CGROUP_STATE" in
    OK)
        echo "cgroups memory: OK (firmware cmdline)"
        ;;
    NO_FIRMWARE_CMDLINE)
        echo "cgroups: OK (no firmware cmdline.txt — assuming cgroups v2)"
        ;;
    FIXED_NEEDS_REBOOT)
        echo "cgroup parameters added. REBOOT REQUIRED before K3s can start."
        echo "After reboot, re-run this script."
        exit 100
        ;;
    STALE_NEEDS_REBOOT)
        echo "WARNING: cgroup_memory=1 is in $(cgroup_cmdline_file) but not in /proc/cmdline."
        echo "A reboot may still be needed."
        exit 100
        ;;
    *)
        echo "ERROR: unexpected cgroup state '${CGROUP_STATE}'" >&2
        exit 1
        ;;
esac

# --- Step 2: Install nvidia-container-toolkit on GPU hosts ---
# Required for K3s to expose host GPUs to pods (libcuda.so.1 injection,
# nvidia-container-runtime hook). Detected via nvidia-smi presence — the
# host driver is independent of the container toolkit, so a working
# nvidia-smi means the GPU is real but doesn't imply the toolkit is set up.
EXTRA_AGENT_ARGS=""
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "NVIDIA driver detected (host nvidia-smi works)"
    if ! command -v nvidia-ctk >/dev/null 2>&1; then
        echo "  nvidia-container-toolkit not installed — installing now..."
        # Add NVIDIA's official apt repo for the toolkit. The repo URL is
        # arch-agnostic and works on any deb-based distro.
        #
        # `gpg --batch --yes` because BOTH writes must be safe to repeat: the
        # guard below tests the sources LIST, while the first write targets the
        # KEYRING, so a run interrupted between the two (or any earlier nvidia
        # install) leaves a keyring that `--dearmor -o` refuses to overwrite. It
        # tries to prompt, ssh here has no tty, and the join dies at `set -e`.
        #
        # Every fetch is bounded and retried: this whole script runs inside an
        # ssh session the orchestrator is blocking on, with no timeout anywhere
        # in that chain, so a black-holed connect hangs the provisioning run.
        if [ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
            curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 120 \
                https://nvidia.github.io/libnvidia-container/gpgkey \
                | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 120 \
                https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sudo sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        fi
        sudo apt-get update -qq
        sudo apt-get install -y -qq nvidia-container-toolkit
        echo "  nvidia-container-toolkit installed: $(nvidia-ctk --version | head -1)"
    else
        echo "  nvidia-container-toolkit already installed: $(nvidia-ctk --version | head -1)"
    fi
    # K3s detects nvidia-container-runtime on its PATH at install time and
    # registers a 'nvidia' runtime in containerd. --default-runtime=nvidia
    # makes pods use it without needing per-pod runtimeClassName.
    if command -v nvidia-container-runtime >/dev/null 2>&1; then
        EXTRA_AGENT_ARGS="--default-runtime=nvidia"
        echo "  Enabling --default-runtime=nvidia for K3s agent"
    else
        # WARNING at column 0: setup-all.sh redirects this script's output into a
        # per-node log file today, but lifecycle._classify_setup_line matches
        # this spelling as a PREFIX, so an indented line would be misclassified
        # the moment the join output is ever streamed.
        echo "WARNING: nvidia-container-runtime still not on PATH after toolkit install"
        echo "         K3s will start without GPU support; pods scheduled here will not see the GPU."
    fi

    # nvidia-container-toolkit 1.19+'s install hook drops a minimal config
    # at /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.tmpl that
    # only `imports = ["/etc/containerd/conf.d/*.toml"]` (where 99-nvidia.toml
    # contains just the runtime registration). K3s then uses that minimal
    # template verbatim, dropping its own required CRI plugin config —
    # containerd starts without the CRI plugin, kubelet can't talk to it,
    # pods never schedule. Remove the bad template so K3s generates its
    # full default config (which auto-detects the nvidia runtime anyway).
    #
    # SIZE is what distinguishes the two: the toolkit's stub is a couple of
    # `imports = [...]` lines (well under 200 bytes), while a real K3s template
    # carries the whole CRI plugin config and is far larger. 999 is the
    # can't-stat sentinel, chosen to fall on the safe side of the test below —
    # an unreadable file is left alone rather than removed.
    if [ -f /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.tmpl ]; then
        TMPL_SIZE=$(stat -c%s /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.tmpl 2>/dev/null || echo 999)
        if [ "$TMPL_SIZE" -lt 200 ]; then
            echo "  Removing minimal config-v3.toml.tmpl from nvidia-toolkit (would break K3s CRI plugin)"
            sudo rm -f /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.tmpl
        fi
    fi
fi

# --- Step 3: Install K3s agent ---
# Bounded and retried for the reason given at the toolkit fetches above: the
# orchestrator bounds this whole ssh session at Deployer.JOIN_TIMEOUT_SECONDS
# (900 s, an install-sized budget) and reports the expiry as rc 124, so an
# unbounded curl does not hang the provisioning run forever — it burns the
# entire budget and takes the join down with it. A transient DNS or TLS blip
# would otherwise abort the join outright. --max-time as well as
# --connect-timeout, because a connect that succeeds and then stalls (a captive
# portal, a black-holing proxy) never trips a connect timeout — sshd answers the
# ServerAlive probes on curl's behalf, so nothing else in the chain has a
# deadline. It bounds this fetch only: the installer's own download of the
# k3s binary from GitHub has no retry at all, and one refused download there
# is enough to fail a whole provision — which is why the entire invocation is
# retried below, not just this curl.
#
# The token travels in the environment, never in argv — K3S_TOKEN is read by
# the install script, and the variable is not exported beyond this pipeline.
# INSTALL_K3S_VERSION rides along the same way, and an empty value is what
# get.k3s.io reads as "latest stable".
#
# WHAT IS TRUSTED HERE: get.k3s.io is trusted on TLS alone. This is the
# vendor's own installation path, and there is no published checksum or
# signature for the shell script itself to pin against — a hash written down
# here would only pin whatever was fetched the day it was written, and the
# script is expected to change. The k3s BINARY the installer then downloads is
# covered: the installer verifies it against the sha256sum published with that
# release. So the trust is Rancher's release infrastructure plus the CA store
# on this node, and it is exercised on every node of every fresh provision.
# Recorded as a trust-boundary decision in SECURITY.md.
if [ -n "$K3S_VERSION" ]; then
    echo "Installing K3s agent ${K3S_VERSION}"
else
    # Unpinned agents track latest stable while the server stays on the version
    # its k3d image carries, so nodes joined weeks apart end up on different
    # builds — usually with a kubelet newer than the API server, outside the
    # supported skew, and visible only much later as a NotReady node.
    echo "WARNING: no K3s version pinned — installing latest stable, which may be newer than the server"
fi
install_agent() {
    curl -sfL --retry 3 --retry-delay 2 --connect-timeout 10 --max-time 120 https://get.k3s.io \
        | INSTALL_K3S_VERSION="$K3S_VERSION" K3S_URL="$K3S_SERVER_URL" \
          K3S_TOKEN="$K3S_JOIN_TOKEN" sh -s - agent $EXTRA_AGENT_ARGS
}
# Three attempts, 15 s then 30 s apart. The installer is idempotent, so a
# repeat costs only time, and three attempts with their fetch ceilings fit
# inside the 900 s the orchestrator allows one join. A deterministic failure
# (bad token, unreachable server) still fails — three times — and the
# installer's own output above the final line says why.
INSTALL_ATTEMPTS=3
attempt=1
until install_agent; do
    if [ "$attempt" -ge "$INSTALL_ATTEMPTS" ]; then
        echo "ERROR: K3s agent install failed ${INSTALL_ATTEMPTS} times — see the installer output above" >&2
        exit 1
    fi
    wait_s=$((attempt * 15))
    echo "K3s agent install attempt ${attempt} failed (usually the binary download from GitHub); retrying in ${wait_s}s..." >&2
    sleep "$wait_s"
    attempt=$((attempt + 1))
done

echo
echo "K3s agent installed. Checking status..."
sudo systemctl status k3s-agent --no-pager || true
echo
echo "Done. This node should appear in 'kubectl get nodes' on the server."
