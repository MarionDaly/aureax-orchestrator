"""k3d cluster lifecycle management, plus orchestrator LAN-IP detection.

The LAN-IP half (``_iface_ip``, ``_route_interface``, ``mac_lan_ip``,
``get_server_url``) is macOS-only: it shells out to ``ipconfig getifaddr`` and
BSD ``route -n get``, neither of which exists on a Linux orchestrator. Both
helpers report a missing binary as "no address", so on any other platform
detection yields "" and ``get_server_url`` falls back to its loopback failure
signal. Supporting another platform means adding a branch here (``ip route
get <dest>`` plus ``ip -4 addr show <iface>``), not changing the callers.

The cluster half is platform-independent — it only runs ``k3d`` and
``docker``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from orchestrator.cluster._subprocess import CommandError
from orchestrator.cluster._subprocess import run as _run


def _cluster_entry(name: str) -> dict | None:
    """The `k3d cluster list` entry for `name`, or None if there is none.

    A single list-and-parse, shared by cluster_exists and cluster_running so
    the JSON is decoded in one place. Nothing is memoized: every call spawns
    k3d and queries the Docker daemon, so callers that ask both questions back
    to back (Deployer.setup_cluster, lifecycle's start path) pay for two.
    """
    try:
        result = _run(["k3d", "cluster", "list", "-o", "json"], check=False, timeout=30)
    except CommandError:
        # A wedged Docker daemon makes this hang. "Cannot tell" has to read as
        # "not there" rather than block every caller forever — the same shape
        # as the non-zero exit below.
        return None
    if result.returncode != 0:
        return None
    try:
        clusters = json.loads(result.stdout)
    except ValueError:
        # Same rule one line up, applied to the output: a k3d that exits 0 and
        # prints something other than a JSON array (nothing at all, a warning
        # line ahead of the JSON) is another way of not being able to tell.
        return None
    if not isinstance(clusters, list):
        return None
    return next((c for c in clusters if isinstance(c, dict) and c.get("name") == name), None)


def cluster_exists(name: str = "aureax") -> bool:
    """Check if a k3d cluster exists."""
    return _cluster_entry(name) is not None


def create_cluster(name: str = "aureax", api_port: int = 6443, servers: int = 1) -> None:
    """Create a k3d cluster."""
    if cluster_exists(name):
        return

    # No host port map: nothing inside the cluster receives telemetry. The
    # receiver is mounted on the dashboard app (orchestrator.dashboard.api,
    # POST /api/v1/telemetry), which runs on the orchestrator itself and is
    # where every node config points, so a published port here would hold a
    # port on the orchestrator for a listener that does not exist. Kept in step
    # with scripts/setup-k3d.sh, the other way this cluster gets made.
    _run([
        "k3d", "cluster", "create", name,
        "--servers", str(servers),
        "--api-port", str(api_port),
        "--k3s-arg", "--disable=traefik@server:0",
    ])


def start_cluster(name: str = "aureax") -> bool:
    """Start a stopped k3d cluster. Returns True if started or already running.

    No existence guard: `k3d cluster start <missing>` exits non-zero, which is
    already the False below — one Docker round trip instead of two. No
    timeout either: a first start pulls images and legitimately runs for
    minutes.
    """
    result = _run(["k3d", "cluster", "start", name], check=False)
    return result.returncode == 0


def stop_cluster(name: str = "aureax") -> bool:
    """Stop a running k3d cluster (preserves state). Returns True on success.

    Like start_cluster, a missing cluster is reported by k3d's own non-zero
    exit rather than by a preceding list.
    """
    result = _run(["k3d", "cluster", "stop", name], check=False)
    return result.returncode == 0


def cluster_running(name: str = "aureax") -> bool:
    """Check if the k3d cluster's server container is running."""
    # serversRunning > 0 means the server container is up.
    return (_cluster_entry(name) or {}).get("serversRunning", 0) > 0


def get_token(name: str = "aureax") -> str | None:
    """Extract the K3s join token from the k3d server container."""
    container = f"k3d-{name}-server-0"
    try:
        result = _run(
            ["docker", "exec", container, "cat", "/var/lib/rancher/k3s/server/node-token"],
            check=False,
            timeout=30,
        )
    except CommandError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def save_token(name: str = "aureax") -> Path | None:
    """Extract and save the K3s token to ~/.aureax/k3s-token."""
    token = get_token(name)
    if not token:
        return None
    token_path = Path.home() / ".aureax" / "k3s-token"
    # 0600 from the first byte, mirroring keystore._write_tmp: write_text
    # creates the file at the umask default (0644 typically), which left a
    # cluster-join credential world-readable until the chmod landed. The
    # trailing chmod stays so an already-existing 0644 file is tightened too.
    token_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(token)
    token_path.chmod(0o600)
    return token_path


def _iface_ip(iface: str) -> str:
    """IPv4 address of a named interface, or '' if it has none.

    A link-local 169.254.x.x address counts as "none": it means DHCP failed
    and the interface self-assigned. It is routable from nowhere, and
    heal-nodes would happily write it into every node's K3S_URL, breaking the
    whole cluster in the name of repairing it.

    The filter lives here rather than at one call site so it covers BOTH
    branches of ``mac_lan_ip``: ``route -n get`` names the self-assigned
    interface whenever the destination is on-link, and nothing downstream
    would catch the result — ``cli._expected_server_url`` rejects only
    ``https://127.`` and ``https://0.``.
    """
    try:
        # macOS-only. On a platform without `ipconfig` the missing binary is
        # reported as CommandError rc 127, which is the same "no address"
        # answer as a failed lookup — see the module docstring.
        result = _run(["ipconfig", "getifaddr", iface], check=False, timeout=10)
    except CommandError:
        # A missing binary or an expired timeout: no address beats hanging
        # every caller that needs to know where we are reachable.
        return ""
    if result.returncode != 0:
        return ""
    ip = result.stdout.strip()
    return "" if ip.startswith("169.254.") else ip


def _route_interface(dest: str) -> str:
    """Interface macOS would use to reach `dest` ('default' is accepted).

    macOS-only, like ``_iface_ip``: `route -n get` is the BSD spelling.
    """
    try:
        result = _run(["route", "-n", "get", dest], check=False, timeout=10)
    except CommandError:
        return ""
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("interface:"):
            return line.split(":", 1)[1].strip()
    return ""


def mac_lan_ip(probe_host: str | None = None) -> str:
    """The orchestrator's LAN IP that edge nodes can actually reach it on.

    macOS-only detection (see the module docstring); returns "" anywhere else,
    which callers must treat as "could not detect".

    Both ``get_server_url`` and lifecycle's setup-all bootstrap use this to
    pin the K3s server URL / pass MAC_IP to scripts, so picking the wrong
    interface points every agent and every telemetry push at an address that
    may work today and vanish on the next DHCP renewal.

    Interface *names* are not a reliable guide to which link is which — en0
    is Wi-Fi on some machines and Ethernet on others — so ask the routing
    table which interface actually reaches the nodes, preferring a route to a
    real node over the default route. A bare en0/en1 scan is the last resort
    when the routing table answers nothing.
    """
    candidates = [h for h in (probe_host, "default") if h]
    for dest in candidates:
        iface = _route_interface(dest)
        if iface:
            ip = _iface_ip(iface)
            if ip:
                return ip

    for iface in ("en0", "en1"):
        # Link-local addresses are filtered out inside _iface_ip, for both
        # this scan and the routing-table branch above.
        ip = _iface_ip(iface)
        if ip:
            return ip
    return ""


def get_server_url(api_port: int = 6443, probe_host: str | None = None) -> str:
    """Get the K3s server URL that edge node agents will connect to.

    When no LAN IP can be detected this returns ``https://127.0.0.1:<port>`` —
    an address no edge node can reach. That loopback URL is a failure SIGNAL,
    not a sane default: ``cli._expected_server_url``,
    ``lifecycle._start_phase_verify_nodes`` and
    ``dashboard.watchers.DriftChecker`` all refuse on an ``https://127.`` /
    ``https://0.`` prefix, and tests/test_k3d_lan_ip.py pins the value. Do not
    "fix" it by returning "" without updating them.
    """
    ip = mac_lan_ip(probe_host) or "127.0.0.1"
    return f"https://{ip}:{api_port}"


def probe_host(config) -> str | None:
    """A real node address, so the LAN IP is chosen by actual route.

    Interface names are not a reliable guide to which link reaches the nodes
    (en0 is Wi-Fi on some machines, Ethernet on others), so let the routing
    table decide against a destination we actually care about.

    Its result feeds mac_lan_ip/get_server_url, and cli.py, lifecycle.py,
    watchers.py and deployer.py all need the same answer — as does the shell
    layer, which imports this and mac_lan_ip from scripts/lib/common.sh. Hence
    one definition next to the functions that consume it rather than five
    copies.
    """
    targets = config.ssh.enabled_targets
    return targets[0].host if targets else None
