"""Mac LAN IP detection, plus the k3d cluster helpers that share its runner.

Picking the wrong interface hands every K3s agent and every telemetry push an
address on a link the nodes may not be reached over — and Wi-Fi addresses
churn on DHCP renewal, which is the failure `aureax heal-nodes` exists to
clean up. Interface *names* don't identify the link (en0 is Wi-Fi on some
Macs, Ethernet on others), so the routing table is the source of truth.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from orchestrator.cluster import _subprocess, k3d
from tests.conftest import make_network_config, make_targets


def _fake_run(routes: dict[str, str], ips: dict[str, str]):
    """Stub for k3d._run covering `route -n get` and `ipconfig getifaddr`."""

    def run(cmd, check=False, timeout=None):  # signature mirrors the real k3d._run
        if cmd[0] == "route":
            dest = cmd[-1]
            iface = routes.get(dest)
            if iface is None:
                return subprocess.CompletedProcess(cmd, 1, "", "no route")
            return subprocess.CompletedProcess(
                cmd, 0, f"   route to: {dest}\n  interface: {iface}\n", ""
            )
        if cmd[0] == "ipconfig":
            iface = cmd[-1]
            ip = ips.get(iface)
            if ip is None:
                return subprocess.CompletedProcess(cmd, 1, "", "")
            return subprocess.CompletedProcess(cmd, 0, ip + "\n", "")
        raise AssertionError(f"unexpected command: {cmd}")

    return run


def test_prefers_interface_that_routes_to_the_node(monkeypatch) -> None:
    """en0 (Wi-Fi) has an IP, but the node is reached via en8 (Ethernet)."""
    monkeypatch.setattr(k3d, "_run", _fake_run(
        routes={"192.0.2.11": "en8", "default": "en8"},
        ips={"en0": "192.0.2.99", "en8": "192.0.2.1"},
    ))
    assert k3d.mac_lan_ip("192.0.2.11") == "192.0.2.1"
    assert k3d.get_server_url(6443, "192.0.2.11") == "https://192.0.2.1:6443"


def test_falls_back_to_default_route_without_a_probe_host(monkeypatch) -> None:
    monkeypatch.setattr(k3d, "_run", _fake_run(
        routes={"default": "en8"},
        ips={"en0": "192.0.2.99", "en8": "192.0.2.1"},
    ))
    assert k3d.mac_lan_ip() == "192.0.2.1"


def test_falls_back_to_interface_scan_when_route_is_unavailable(monkeypatch) -> None:
    """Preserves the historical behaviour where `route` tells us nothing."""
    monkeypatch.setattr(k3d, "_run", _fake_run(
        routes={},
        ips={"en0": "192.0.2.99"},
    ))
    assert k3d.mac_lan_ip("192.0.2.11") == "192.0.2.99"


def test_link_local_on_the_route_path_is_not_returned(monkeypatch) -> None:
    """169.254.x.x means DHCP failed and the interface self-assigned.

    The guard used to live only in the en0/en1 fallback, so the routing-table
    branch could still return one — `route -n get` names the self-assigned
    interface whenever the destination is on-link — and nothing downstream
    catches it: cli.py's heal-nodes preflight rejects only `https://127.` and
    `https://0.`, so an unroutable address would be written into every node's
    K3S_URL in the name of repairing the cluster.
    """
    monkeypatch.setattr(k3d, "_run", _fake_run(
        routes={"192.0.2.11": "en8", "default": "en8"},
        ips={"en8": "169.254.13.7", "en0": "192.0.2.99"},
    ))
    assert k3d.mac_lan_ip("192.0.2.11") == "192.0.2.99"


def test_returns_empty_when_every_candidate_is_link_local(monkeypatch) -> None:
    """No address at all beats an unroutable one — callers must be able to
    tell that detection failed."""
    monkeypatch.setattr(k3d, "_run", _fake_run(
        routes={"default": "en0"},
        ips={"en0": "169.254.13.7", "en1": "169.254.200.1"},
    ))
    assert k3d.mac_lan_ip() == ""


def test_returns_empty_when_nothing_is_up(monkeypatch) -> None:
    monkeypatch.setattr(k3d, "_run", _fake_run(routes={}, ips={}))
    assert k3d.mac_lan_ip() == ""
    # get_server_url must not emit a bare empty host.
    assert k3d.get_server_url(6443) == "https://127.0.0.1:6443"


def test_probe_host_skips_disabled_targets() -> None:
    """The probe destination is what every LAN-IP decision routes against.

    A disabled node is typically powered off, so probing it makes `route -n
    get` fall back to the default interface — which may not be the link that
    reaches the fleet.
    """
    cfg = make_network_config(make_targets("a", "b", disabled=("a",)))
    assert k3d.probe_host(cfg) == "10.77.0.2"


def test_probe_host_is_none_without_any_target() -> None:
    """mac_lan_ip then routes against 'default' alone, which is the intent."""
    assert k3d.probe_host(make_network_config([])) is None


# --- cluster helpers ---------------------------------------------------------
# Every one of these spawns k3d and queries the Docker daemon, so what matters
# is both the answer and how many times it asks.


def _cluster_run(entries: list[dict], action_rc: int = 0, calls: list | None = None):
    """Stub for k3d._run covering `k3d cluster list|start|stop`."""

    def run(cmd, check=False, timeout=None):
        if calls is not None:
            calls.append(cmd)
        if cmd[:3] == ["k3d", "cluster", "list"]:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(entries), "")
        return subprocess.CompletedProcess(cmd, action_rc, "", "")

    return run


def test_cluster_running_reads_the_server_count(monkeypatch) -> None:
    monkeypatch.setattr(k3d, "_run", _cluster_run([{"name": "aureax", "serversRunning": 1}]))
    assert k3d.cluster_exists("aureax") is True
    assert k3d.cluster_running("aureax") is True
    assert k3d.cluster_running("other") is False, "another cluster's entry must not answer"


def test_a_stopped_cluster_exists_but_is_not_running(monkeypatch) -> None:
    """The distinction setup_cluster branches on: a stopped cluster left every
    later step talking to an unreachable API server."""
    monkeypatch.setattr(k3d, "_run", _cluster_run([{"name": "aureax", "serversRunning": 0}]))
    assert k3d.cluster_exists("aureax") is True
    assert k3d.cluster_running("aureax") is False


@pytest.mark.parametrize(
    "stdout", ["", "null", 'WARN[0000] no config file found\n[{"name": "aureax"}]', "{}"],
    ids=["empty", "null", "preamble", "object"],
)
def test_output_that_is_not_a_cluster_array_reads_as_no_cluster(monkeypatch, stdout) -> None:
    """"Cannot tell" has to read as "not there" for the OUTPUT too.

    cluster_exists and cluster_running are the first calls both `aureax stop`
    and the start path make, so a k3d that exits 0 and prints something else
    would hand the operator a JSONDecodeError instead of the degradation the
    non-zero-exit and wedged-daemon paths already give them.
    """
    monkeypatch.setattr(
        k3d, "_run",
        lambda cmd, check=False, timeout=None: subprocess.CompletedProcess(cmd, 0, stdout, ""),
    )

    assert k3d.cluster_exists("aureax") is False
    assert k3d.cluster_running("aureax") is False


def test_start_cluster_does_not_pre_list(monkeypatch) -> None:
    """`k3d cluster start <missing>` already exits non-zero, so the existence
    guard was a second Docker round trip for an answer k3d gives anyway."""
    calls: list[list[str]] = []
    monkeypatch.setattr(k3d, "_run", _cluster_run([{"name": "aureax"}], calls=calls))

    assert k3d.start_cluster("aureax") is True
    assert [c[:3] for c in calls] == [["k3d", "cluster", "start"]]


def test_start_and_stop_report_a_missing_cluster_as_false(monkeypatch) -> None:
    monkeypatch.setattr(k3d, "_run", _cluster_run([], action_rc=1))
    assert k3d.start_cluster("nope") is False
    assert k3d.stop_cluster("nope") is False


def test_the_saved_k3s_token_is_never_world_readable(monkeypatch, tmp_aureax_home) -> None:
    """It is a cluster-join credential; write_text created it at the umask
    default and only chmodded afterwards."""
    monkeypatch.setattr(k3d, "get_token", lambda _n: "K10secret")

    path = k3d.save_token("aureax")
    assert path is not None
    assert path.read_text() == "K10secret"
    assert path.stat().st_mode & 0o777 == 0o600

    # A file an older build left at 0644 is tightened, not inherited.
    path.chmod(0o644)
    k3d.save_token("aureax")
    assert path.stat().st_mode & 0o777 == 0o600


def test_a_timed_out_cluster_command_fails_instead_of_hanging() -> None:
    """No blanket ceiling — `k3d cluster create` legitimately runs for minutes
    on a first image pull — but the calls that must be quick need one, and a
    command that never returned has no result for a caller to inspect."""
    with pytest.raises(_subprocess.CommandError) as excinfo:
        _subprocess.run(["sleep", "5"], check=False, timeout=0.2)

    assert excinfo.value.returncode == 124  # coreutils timeout(1)'s convention


def test_a_missing_binary_fails_like_any_other_command() -> None:
    """k3d/kubectl/docker are external tools an orchestrator may not have.

    Every caller in the cluster package handles ``CommandError``; a bare
    ``FileNotFoundError`` escapes `aureax stop`/`start` — whose handlers list
    ``CommandError`` and ``RuntimeError`` — as a traceback instead of a line
    naming the tool to install.
    """
    with pytest.raises(_subprocess.CommandError) as excinfo:
        _subprocess.run(["aureax-no-such-binary-xyz", "list"], check=False, timeout=5)

    assert excinfo.value.returncode == 127  # the shell's convention
    assert "not found on PATH" in str(excinfo.value)
