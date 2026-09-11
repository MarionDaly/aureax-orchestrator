"""Install and verify a worker's host firewall before its RPC process starts.

Run in the shard pod's NET_ADMIN init container, sharing the host network.
Rules are confined to the configured TCP port and committed atomically per
family; the IPv6 pass is skipped only on a host that has no IPv6 address at all.
"""

import ipaddress
import os
import shlex
import subprocess
from pathlib import Path

from ruamel.yaml import YAML

# One line per configured IPv6 address, and the file is absent altogether on a
# host booted with ipv6.disable=1.
_IF_INET6 = Path("/proc/net/if_inet6")


def _host_has_ipv6() -> bool:
    """True if this host carries any IPv6 address."""
    try:
        return bool(_IF_INET6.read_text().strip())
    except OSError:
        return False


def _peer_address(peer: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """One allowlist entry as an IP literal, or a named refusal.

    Never a name lookup. This runs in the worker's init container, which has no
    mDNS resolver and no cluster DNS for a LAN name, so a hostname-shaped head
    would fail to resolve here and crash-loop the container — taking the worker,
    and with it the whole group, down. The orchestrator resolves the head on its
    own network at generate time and writes literals into the worker's config.
    """
    try:
        return ipaddress.ip_address(peer)
    except ValueError:
        raise ValueError(
            f"RPC head allowlist entry {peer!r} is not an IP address, and this container "
            "resolves no names; re-run `aureax generate` and `aureax apply` for this worker"
        ) from None


def install(config: dict, local_ip: str) -> None:
    # ``or {}`` at both levels: a key present with no children parses as None,
    # and an AttributeError here reaches the operator as a bare traceback in
    # `kubectl logs <worker-pod> -c rpc-firewall` instead of one of this
    # module's named refusals.
    if (config.get("node") or {}).get("role") != "shard_worker":
        raise ValueError("RPC firewall requires a shard worker config")
    rpc = (config.get("model") or {}).get("rpc_server") or {}
    port = int(rpc.get("port", 50052))
    if not 1 <= port <= 65535:
        raise ValueError("Invalid RPC port")
    peers = rpc.get("allowed_clients", [])
    if not isinstance(peers, list) or not peers:
        raise ValueError("Missing RPC head allowlist; regenerate and apply the worker config")
    local = ipaddress.ip_address(local_ip)
    allowed = {str(local)} | {str(_peer_address(peer)) for peer in peers}
    chain = f"AUREAX_RPC_{port}"
    for version, tool in ((4, "iptables"), (6, "ip6tables")):
        if version == 6 and local.version == 4 and not _host_has_ipv6():
            # rpc-server binds the IPv4 AUREAX_HOST_IP, so no v6 socket exists
            # to protect, and a host with IPv6 disabled has no ip6 filter table
            # for ip6tables to read — one this container cannot create, since it
            # drops every capability but NET_ADMIN and mounts no /lib/modules.
            # Refusing there would keep every worker on such a host down to
            # close a door the kernel never opened. A host that DOES have IPv6
            # still fails closed below.
            print(f"No IPv6 address on this host and an IPv4 RPC bind — skipping {tool}")
            continue
        rules = [["-i", "lo", "-j", "ACCEPT"]]
        rules += [["-s", ip, "-j", "ACCEPT"] for ip in sorted(allowed) if ipaddress.ip_address(ip).version == version]
        rules.append(["-j", "DROP"])
        jump = ["-p", "tcp", "--dport", str(port), "-j", chain]
        current = subprocess.run([tool, "-w", "10", "-S", "INPUT"], capture_output=True, text=True, timeout=15)
        if current.returncode != 0:
            raise RuntimeError(f"Cannot inspect {tool}; RPC startup refused: {current.stderr.strip()}")
        # -C with a not-yet-created target chain can return 2 instead of 1.
        # Inspect INPUT first so an absent chain is distinct from a tool failure.
        expected = ["-A", "INPUT", *jump]
        def normalized(line):
            return shlex.split(line.replace(" -m tcp", ""))
        jump_exists = any(normalized(line) == expected for line in current.stdout.splitlines())
        lines = ["*filter", f":{chain} - [0:0]", f"-F {chain}"]
        if jump_exists:
            lines.append("-D INPUT " + " ".join(jump))
        lines += [f"-A {chain} " + " ".join(rule) for rule in rules]
        lines += ["-I INPUT 1 " + " ".join(jump), "COMMIT", ""]
        subprocess.run([tool + "-restore", "--wait", "10", "--noflush"], input="\n".join(lines),
                       text=True, check=True, timeout=15, capture_output=True)
        for source, rule in [("INPUT", jump), *((chain, rule) for rule in rules)]:
            subprocess.run([tool, "-w", "10", "-C", source, *rule], check=True, timeout=15, capture_output=True)


def main():
    with open(os.environ.get("AUREAX_CONFIG_PATH", "/etc/aureax/aureax-edge.yaml")) as stream:
        config = YAML(typ="safe").load(stream)
    # Named, like every other refusal in this module: a bare KeyError or
    # ValueError is all the operator gets from `kubectl logs <worker-pod> -c
    # rpc-firewall`, which is where README sends them to diagnose a worker that
    # will not start.
    host_ip = os.environ.get("AUREAX_HOST_IP", "")
    try:
        ipaddress.ip_address(host_ip)
    except ValueError:
        raise RuntimeError(
            f"AUREAX_HOST_IP is {host_ip!r}, not an IP address; this init container takes it "
            "from the downward API field status.hostIP — RPC startup refused"
        ) from None
    install(config, host_ip)


if __name__ == "__main__":
    main()
