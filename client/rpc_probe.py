"""The shard worker's startupProbe: one TCP connect to what rpc-server binds.

Run by the kubelet as an ``exec`` probe in k8s/daemonset-shard.yaml. A
``tcpSocket`` probe can only dial the pod IP — on a hostNetwork pod, the
node's LAN address — and a worker in a group with ``wireguard.enabled`` binds
its tunnel address instead, so the kubelet's own dial would be refused for the
whole failureThreshold and the worker killed mid-start. Asking
``resolve_rpc_host`` instead makes the probe dial exactly the address the
supervisor's readiness wait and rpc-server's ``-H`` use, with or without the
tunnel: on a LAN-bound worker it is the same address the kubelet dialled.

One connect and an immediate close, like the tcpSocket probe it replaces:
rpc-server serves one client with a listen backlog of 1, which is why the shard
pod carries a startupProbe only (see the manifest).

Exit 0 when the connect succeeds, 1 otherwise — the exec-probe contract.
"""

from __future__ import annotations

import os
import socket
import sys

from client.config.parser import load_edge_config_raw, resolve_rpc_host, resolve_rpc_port

CONNECT_TIMEOUT_SECONDS = 3.0


def main() -> int:
    try:
        config = load_edge_config_raw(os.environ.get("AUREAX_CONFIG_PATH", "/etc/aureax/aureax-edge.yaml"))
        host, port = resolve_rpc_host(config), resolve_rpc_port(config)
    except Exception as e:  # the probe's whole output is its exit code plus this line
        print(f"rpc probe: cannot read the node config: {e}", file=sys.stderr)
        return 1
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_SECONDS):
            return 0
    except OSError as e:
        print(f"rpc probe: {host}:{port} not accepting yet ({e})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
