#!/usr/bin/env bash
# ==========================================================================
# Aureax Prototype — Stop All Workloads
#
# Suspends edge and shard-worker pods (DaemonSet nodeSelector patch), stops
# the k3d control plane, and kills the dashboard. DaemonSets, Services, node
# labels, node state and model files are all preserved.
#
# To bring everything back: uv run aureax start
# To wipe everything:       ./scripts/reset-all.sh
#
# Usage: ./scripts/stop-all.sh [--node NODE_ID]
#
# Arguments are forwarded verbatim to `aureax stop`, so click both honours
# --node (which suspends that node and the rest of its node_groups entry,
# leaving the cluster up) and rejects anything it does not recognise.
# ==========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export PATH="$HOME/.local/bin:$PATH"

echo "=== Aureax Stop ==="
echo
uv run aureax stop "$@"
echo
echo "Cluster and node state preserved. Run 'uv run aureax start' to restart."
