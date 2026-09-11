"""`rpc-server` on :50052 is reachable only from its head, or it does not start.

That port speaks a bespoke binary protocol with no TLS and no authentication at
any layer, so the host firewall is the entire control: `client/rpc_firewall.py`
runs in the shard pod's NET_ADMIN init container and installs the chain, for
every IP family the host actually has, before the RPC process exists. Order
inside the chain is the semantics — the ACCEPTs are worthless below the DROP —
and every failure path must refuse startup rather than leave the port open. See
ARCHITECTURE.md, "Security".
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from client import rpc_firewall

CHAIN = "AUREAX_RPC_50052"
LOCAL_IP = "192.0.2.2"
WORKER = {
    "node": {"role": "shard_worker"},
    "model": {"rpc_server": {"port": 50052, "allowed_clients": ["192.0.2.1"]}},
}


def _chain_rules(blob: str) -> list[str]:
    """The chain's own rules, in the order iptables-restore will evaluate them."""
    prefix = f"-A {CHAIN} "
    return [line[len(prefix):] for line in blob.splitlines() if line.startswith(prefix)]


def _restores(run: Mock) -> list:
    return [call for call in run.call_args_list if call.args[0][0].endswith("-restore")]


@pytest.fixture
def host_with_ipv6(monkeypatch):
    """A host that carries an IPv6 address, whatever the test machine has.

    The v6 pass is skipped on a host with no IPv6 and an IPv4 RPC bind, so
    without this every both-families assertion here would pass vacuously —
    or fail — depending on the machine running the suite.
    """
    monkeypatch.setattr(rpc_firewall, "_host_has_ipv6", lambda: True)


def test_rpc_firewall_requires_a_head_before_touching_rules(monkeypatch):
    """An empty allowlist would produce a chain that accepts nothing but drops
    everything, which reads as a working firewall and is a dead worker."""
    run = Mock()
    monkeypatch.setattr(rpc_firewall.subprocess, "run", run)
    with pytest.raises(ValueError, match="allowlist"):
        rpc_firewall.install({"node": {"role": "shard_worker"}}, LOCAL_IP)
    run.assert_not_called()


def test_rpc_firewall_installs_both_families_with_drop_after_head_and_local(
    monkeypatch, host_with_ipv6
):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(rpc_firewall.subprocess, "run", run)

    rpc_firewall.install(WORKER, LOCAL_IP)

    restores = _restores(run)
    assert [call.args[0][0] for call in restores] == ["iptables-restore", "ip6tables-restore"]
    for call in restores:
        rules = _chain_rules(call.kwargs["input"])
        # Membership alone cannot see this: a DROP built with insert(0, ...)
        # instead of append leaves every rule below present in the blob and
        # blackholes the head's RPC traffic.
        assert rules[0] == "-i lo -j ACCEPT", "the loopback exemption must precede the DROP"
        assert rules.index("-j DROP") == len(rules) - 1, "DROP is the last rule or it is a black hole"
        assert f"-I INPUT 1 -p tcp --dport 50052 -j {CHAIN}" in call.kwargs["input"]
        assert call.kwargs["check"]
    # The head's address and the worker's own, and only in the family they
    # belong to — a v4 allowlist under ip6tables would silently accept nothing.
    assert _chain_rules(restores[0].kwargs["input"]) == [
        "-i lo -j ACCEPT",
        "-s 192.0.2.1 -j ACCEPT",
        "-s 192.0.2.2 -j ACCEPT",
        "-j DROP",
    ]
    assert _chain_rules(restores[1].kwargs["input"]) == ["-i lo -j ACCEPT", "-j DROP"]


def test_rpc_firewall_failure_blocks_init_completion(monkeypatch):
    """The failure has to come from the `-restore` call, which is the one that
    carries `check=True`. A blanket side_effect fires on the read-only
    `-S INPUT` probe instead — a call production inspects itself — so it would
    stay green with the restore's `check=True` deleted."""
    def run(argv, **_kwargs):
        if argv[0].endswith("-restore"):
            raise subprocess.CalledProcessError(1, argv[0])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(rpc_firewall.subprocess, "run", Mock(side_effect=run))
    with pytest.raises(subprocess.CalledProcessError) as exc:
        rpc_firewall.install(WORKER, LOCAL_IP)
    assert exc.value.cmd == "iptables-restore", "the probe, not the commit, would be the wrong failure"


def test_an_unreadable_rule_table_refuses_rpc_startup(monkeypatch):
    """The probe deliberately runs without `check=True` so an absent chain is
    distinguishable from a broken tool, which puts the refusal on the
    returncode: unreadable rules must abort the init container, not install an
    unverifiable chain."""
    run = Mock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="permission denied"))
    monkeypatch.setattr(rpc_firewall.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="Cannot inspect iptables"):
        rpc_firewall.install(WORKER, LOCAL_IP)

    assert _restores(run) == [], "no rules may be written when the current ones cannot be read"


def test_a_host_with_ipv6_still_fails_closed_on_the_v6_pass(monkeypatch, host_with_ipv6):
    """The v6 skip is for hosts that have no IPv6 at all.

    Where a v6 stack exists, rpc-server's port is reachable over it, so an
    ip6tables that cannot be read must still refuse startup rather than leave
    that door open.
    """
    def run(argv, **_kwargs):
        ok = SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="no ip6 table") \
            if argv[0].startswith("ip6tables") else ok

    monkeypatch.setattr(rpc_firewall.subprocess, "run", Mock(side_effect=run))
    with pytest.raises(RuntimeError, match="Cannot inspect ip6tables"):
        rpc_firewall.install(WORKER, LOCAL_IP)


def test_a_host_without_ipv6_installs_the_v4_policy_and_skips_the_v6_pass(monkeypatch, capsys):
    """A node booted with ipv6.disable=1 has no ip6 filter table.

    The init container cannot create one — it drops every capability but
    NET_ADMIN and mounts no /lib/modules — so failing closed there kept every
    worker on such a host permanently down to protect a socket the kernel never
    opened. The IPv4 policy, which is the one that matters, still installs.
    """
    monkeypatch.setattr(rpc_firewall, "_host_has_ipv6", lambda: False)
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(rpc_firewall.subprocess, "run", run)

    rpc_firewall.install(WORKER, LOCAL_IP)

    assert [call.args[0][0] for call in _restores(run)] == ["iptables-restore"]
    assert not any(call.args[0][0].startswith("ip6tables") for call in run.call_args_list)
    # The skip is announced: `kubectl logs <worker-pod> -c rpc-firewall` is
    # where an operator finds out which families were actually installed.
    assert "skipping ip6tables" in capsys.readouterr().out


def test_a_hostname_in_the_allowlist_is_refused_by_name(monkeypatch):
    """No resolver exists in this container, so a name can only be a dead end.

    Calling getaddrinfo here crash-looped the init container on a fleet whose
    head is hostname-shaped; the orchestrator resolves the head on its own
    network at generate time instead.
    """
    run = Mock()
    monkeypatch.setattr(rpc_firewall.subprocess, "run", run)
    config = {
        "node": {"role": "shard_worker"},
        "model": {"rpc_server": {"port": 50052, "allowed_clients": ["shard-head.local"]}},
    }

    with pytest.raises(ValueError, match="aureax generate"):
        rpc_firewall.install(config, LOCAL_IP)
    run.assert_not_called()


@pytest.mark.parametrize("config", [
    {"node": None, "model": {"rpc_server": {"allowed_clients": ["192.0.2.1"]}}},
    {"node": {"role": "shard_worker"}, "model": None},
    {"node": {"role": "shard_worker"}, "model": {"rpc_server": None}},
])
def test_an_empty_config_key_reaches_a_named_refusal(monkeypatch, config):
    """A YAML key present with no children parses as None, not {}.

    An AttributeError out of this init container is a bare traceback in the
    place README sends operators to diagnose a worker that will not start.
    """
    monkeypatch.setattr(rpc_firewall.subprocess, "run", Mock())
    with pytest.raises(ValueError):
        rpc_firewall.install(config, LOCAL_IP)


@pytest.mark.parametrize("host_ip", [None, "", "not-an-ip"])
def test_a_missing_or_unusable_host_ip_is_named_rather_than_raw(monkeypatch, tmp_path, host_ip):
    """AUREAX_HOST_IP comes from the downward API, so a bad value is a manifest
    problem — and the message has to say so, since the traceback it replaced
    named nothing an operator could act on."""
    config = tmp_path / "aureax-edge.yaml"
    config.write_text(
        "node:\n  role: shard_worker\n"
        "model:\n  rpc_server:\n    port: 50052\n    allowed_clients: ['192.0.2.1']\n"
    )
    monkeypatch.setenv("AUREAX_CONFIG_PATH", str(config))
    monkeypatch.delenv("AUREAX_HOST_IP", raising=False)
    if host_ip is not None:
        monkeypatch.setenv("AUREAX_HOST_IP", host_ip)
    monkeypatch.setattr(rpc_firewall.subprocess, "run", Mock())

    with pytest.raises(RuntimeError, match=r"status\.hostIP"):
        rpc_firewall.main()
