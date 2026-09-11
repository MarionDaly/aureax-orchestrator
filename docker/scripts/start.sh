#!/usr/bin/env bash
# Aureax edge client container startup script.
#
# Launches the Python process supervisor, which manages llama-server + the MCP
# server on a standalone node or a shard group's head, and rpc-server on a shard
# worker. The same image and CMD serve both DaemonSets; the supervisor picks its
# child from node.role in the mounted config.
set -euo pipefail

# Resolved ONCE, above its first use, so the default is spelled in exactly one
# place and every later mention is this variable. `set -u` is why that matters:
# a bare ${AUREAX_CONFIG_PATH} anywhere below would abort on an unbound
# variable, and the branch it would abort is the one that exists to report the
# failure. Every deployed path sets it — both Dockerfiles ENV it and both
# DaemonSets pass it — so an unset value means a run outside those images,
# which is exactly when the diagnostic is wanted.
CONFIG_PATH="${AUREAX_CONFIG_PATH:-/etc/aureax/aureax-edge.yaml}"
# Must match `model.local_path` in the mounted config — that is the directory
# the supervisor itself loads from (client/config/parser.py). Hardcoded rather
# than taken from an env override: nothing sets one, so a knob here would
# advertise a setting that cannot be turned while the real one lives in the
# YAML.
MODEL_DIR=/opt/aureax/models

echo "=== Aureax Edge Client ==="
# AUREAX_NODE_ID carries the KUBERNETES node name (spec.nodeName, set in both
# DaemonSets), i.e. the host — not the fleet's node.id, which lives in the
# mounted config. Labelled "Host" so the banner does not claim otherwise.
echo "Host:    ${AUREAX_NODE_ID:-unknown}"
echo "Config:  ${CONFIG_PATH}"
echo

# Verify config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found at ${CONFIG_PATH}"
    exit 1
fi

# Verify model file exists — but only for a node that loads one. A shard worker
# holds its share of the weights in RAM, shipped from the head over RPC, and
# never has a local GGUF. Warning there means every worker pod opens its
# `kubectl logs` with a WARNING describing its correct state as a failure.
#
# The "shard_worker" literal mirrors SHARD_WORKER_ROLE in
# orchestrator/config/schema.py — the constant the Python client imports rather
# than redeclaring. This script cannot source scripts/lib/common.sh (that file
# is not in the image) and runs before Python, so the literal is spelled here;
# renaming the role in schema.py means renaming it here too, or every worker pod
# opens its logs with a WARNING about the model it is not supposed to have.
if grep -qE '^[[:space:]]*role:[[:space:]]*"?shard_worker"?' "$CONFIG_PATH"; then
    echo "Role:    shard_worker (rpc-server; no local model by design)"
elif [ -z "$(ls -A "$MODEL_DIR"/*.gguf 2>/dev/null)" ]; then
    echo "WARNING: No .gguf model files found in $MODEL_DIR"
    echo "The llama-server will fail to start without a model."
fi

# This script is PID 1 (both Dockerfiles use CMD ["/start.sh"]). Bash does not
# forward signals to its children, and the kernel gives PID 1 no default signal
# action, so without this trap the kubelet's SIGTERM never reaches the
# supervisor: its graceful child shutdown never runs and the pod is SIGKILLed at
# the end of the grace period. k8s/daemonset-shard.yaml's
# terminationGracePeriodSeconds (45s) > process.shutdown_timeout_seconds (30s)
# contract is entirely dependent on this line.
#
# PID is pre-set and the handler tolerates failure because `set -euo pipefail`
# is in force: a signal arriving in the window between this trap and the
# assignment below would abort the handler on "PID: unbound variable" without
# ever signalling the child, and a `kill` that fails on an already-exited child
# would skip the `wait` that follows it.
PID=
trap 'kill -TERM "${PID:-}" 2>/dev/null || true; wait "${PID:-}" 2>/dev/null || true' TERM INT

# Start the Python process supervisor
python -m client.entrypoint &
PID=$!
wait $PID
