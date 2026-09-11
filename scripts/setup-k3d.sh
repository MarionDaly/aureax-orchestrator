#!/usr/bin/env bash
# Setup k3d cluster on Mac for Aureax prototype.
# Prerequisites: Docker Desktop running. k3d itself is installed below if it is
# missing (which needs Homebrew), and kubectl is only used for the post-create
# `get nodes`, which is skipped when it is absent.
#
# Usage: setup-k3d.sh [cluster_name] [api_port] [servers]
#   — setup-all.sh passes all three from configs/aureax-network.yaml.
set -euo pipefail

# detect_mac_ip, and $AUREAX_ROOT. This script used to source nothing and carry
# its own Mac-IP lookup; see the MAC_IP block near the bottom.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
cd "$AUREAX_ROOT"

# Defaults match the schema's (orchestrator/config/schema.py: KubernetesConfig,
# K3dConfig), for a standalone run. setup-all.sh passes all three from the
# config so the two entry points cannot create differently-shaped clusters.
CLUSTER_NAME="${1:-aureax}"
API_PORT="${2:-6443}"
SERVERS="${3:-1}"

echo "=== Aureax K3s Cluster Setup ==="
echo "Cluster: $CLUSTER_NAME"
echo "API Port: $API_PORT"
echo "Servers: $SERVERS"
echo

# Check prerequisites
if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker is not installed or not in PATH."
    echo "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker info &>/dev/null; then
    echo "ERROR: Docker daemon is not running. Start Docker Desktop first."
    exit 1
fi

if ! command -v k3d &>/dev/null; then
    # Homebrew is a prerequisite this script cannot install for the operator. A
    # bare `brew install` on a machine without it dies with "brew: command not
    # found", which names neither the missing tool nor the fix.
    if ! command -v brew &>/dev/null; then
        echo "ERROR: k3d is not installed, and Homebrew is not available to install it."
        echo "Install k3d: https://k3d.io/#installation"
        exit 1
    fi
    echo "k3d not found. Installing via Homebrew..."
    brew install k3d
fi

# Check if cluster already exists.
#
# Filtered by name, not `grep -q "$CLUSTER_NAME"` over the whole listing: that
# matched any substring (and treated the name as a regex), so with an unrelated
# `aureax-dev` cluster present and no `aureax`, this exited 0 having created
# nothing — and setup-all.sh, which reads only the exit status, printed
# "[OK] Cluster created". Kept in step with the identical test in setup-all.sh.
if k3d cluster list "$CLUSTER_NAME" --no-headers 2>/dev/null | grep -q .; then
    echo "Cluster '$CLUSTER_NAME' already exists."
    echo "To recreate: k3d cluster delete $CLUSTER_NAME && $0 $CLUSTER_NAME $API_PORT $SERVERS"
    exit 0
fi

# Create the cluster
echo "Creating k3d cluster..."
# No host port map. This used to publish 9900:9900@server:0 for an in-cluster
# telemetry receiver that does not exist: the receiver is mounted on the
# dashboard app (orchestrator/dashboard/api.py, POST /api/v1/telemetry) and
# every node config reports to port 7001. Nothing has ever listened on 9900, so
# the map only held a host port on the Mac. Kept in step with
# orchestrator/cluster/k3d.py::create_cluster, the other way this cluster gets
# made.
k3d cluster create "$CLUSTER_NAME" \
    --servers "$SERVERS" \
    --api-port "$API_PORT" \
    --k3s-arg "--disable=traefik@server:0"

echo
# The listing is a courtesy, so a missing kubectl must not fail the script AFTER
# the cluster exists: setup-all.sh reads only the exit status here, and would
# report a created cluster as a failed create.
if command -v kubectl &>/dev/null; then
    echo "Cluster created. Verifying..."
    kubectl get nodes
else
    echo "Cluster created. kubectl is not installed, so skipping the node listing."
    echo "Install it to talk to the cluster: brew install kubectl"
fi
echo

# No join token is extracted or written here. Nothing ever read the
# ~/.aureax/k3s-token this used to write: setup-all.sh Step 4 re-reads the token
# from the server container at join time (common.sh::k3s_token_from_cluster),
# and orchestrator/cluster/k3d.py::save_token remains the one writer of that
# path for the Python deploy path. Writing a second copy of the credential that
# lets any machine join the cluster, for no reader, is cost without benefit.

# Get the API server address that edge nodes will use to join.
#
# Through common.sh::detect_mac_ip, the only Mac-IP detector in the shell
# layer. This used to be a local `uv run python -c` calling mac_lan_ip() with
# no probe host, under a comment claiming it "asks the routing table which
# interface actually reaches the nodes" — with no host it asks about the
# DEFAULT route, which is exactly the Wi-Fi-vs-Ethernet case the comment said
# it avoided. It also discarded the reason on stderr, so a failure printed
# `K3S_URL=https://UNKNOWN:6443` for the operator to paste.
#
# UNKNOWN is still the fallback, but detect_mac_ip says why on stderr first —
# and it rejects loopback and link-local, which the old block accepted.
MAC_IP="$(detect_mac_ip)" || MAC_IP="UNKNOWN"
echo
# Both next steps are automated paths. The manual `curl | sh` recipe this used
# to print is exactly what scripts/join-node.sh and setup-all.sh Step 4 do for
# every node in configs/aureax-network.yaml — and every manual step is a bug,
# because nodes here are wiped and re-provisioned routinely. The CLI line is
# spelled the way it actually runs: `uv run`, and --orchestrator-ip, which
# orchestrator/cli.py declares required.
echo "=== Next Steps ==="
echo "1. Note your Mac's LAN IP: $MAC_IP"
echo "2. Join every configured node and deploy: ./scripts/setup-all.sh"
echo "3. Or, cluster included: uv run aureax deploy --setup-cluster --orchestrator-ip ${MAC_IP}"
