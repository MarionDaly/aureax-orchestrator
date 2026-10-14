"""The shard worker's startupProbe dials exactly what rpc-server binds.

k8s/daemonset-shard.yaml runs ``python -m client.rpc_probe`` instead of a
``tcpSocket`` probe, because a tcpSocket probe can only dial the pod IP (the
node's LAN address) while a worker in a group with ``wireguard.enabled`` binds
its tunnel address. These tests pin that the module asks
``resolve_rpc_host`` — so on a LAN-bound worker it dials the very address the
tcpSocket probe used to — and that its exit code is the probe's verdict.
"""

from __future__ import annotations

import socket

import pytest

from client import rpc_probe


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    def write(text: str):
        path = tmp_path / "aureax-edge.yaml"
        path.write_text(text)
        monkeypatch.setenv("AUREAX_CONFIG_PATH", str(path))
        return path

    return write


def _record_connects(monkeypatch, *, succeed: bool) -> list[tuple]:
    dialled: list[tuple] = []

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def connect(addr, timeout=None):
        dialled.append((addr, timeout))
        if not succeed:
            raise ConnectionRefusedError(111, "Connection refused")
        return Conn()

    monkeypatch.setattr(rpc_probe.socket, "create_connection", connect)
    return dialled


def test_a_lan_bound_worker_is_probed_on_its_host_ip(config_file, monkeypatch) -> None:
    """The address the kubelet's tcpSocket probe dialled (pod IP == host IP)."""
    config_file("node: {role: shard_worker}\nmodel: {rpc_server: {host: auto, port: 50052}}\n")
    monkeypatch.setenv("AUREAX_HOST_IP", "192.0.2.20")
    dialled = _record_connects(monkeypatch, succeed=True)

    assert rpc_probe.main() == 0
    assert dialled == [(("192.0.2.20", 50052), rpc_probe.CONNECT_TIMEOUT_SECONDS)]


def test_a_wireguard_worker_is_probed_on_its_tunnel_address(config_file, monkeypatch) -> None:
    config_file(
        "node: {role: shard_worker, wireguard: {interface: wg0, address: 10.254.0.3/24}}\n"
        "model: {rpc_server: {port: 50052}}\n"
    )
    monkeypatch.setenv("AUREAX_HOST_IP", "192.0.2.20")
    dialled = _record_connects(monkeypatch, succeed=True)

    assert rpc_probe.main() == 0
    assert dialled[0][0] == ("10.254.0.3", 50052)


def test_a_refused_connect_fails_the_probe(config_file, monkeypatch, capsys) -> None:
    config_file("node: {role: shard_worker}\nmodel: {rpc_server: {port: 50052}}\n")
    monkeypatch.setenv("AUREAX_HOST_IP", "192.0.2.20")
    _record_connects(monkeypatch, succeed=False)

    assert rpc_probe.main() == 1
    assert "192.0.2.20:50052" in capsys.readouterr().err


def test_an_unreadable_config_fails_the_probe_with_a_line(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("AUREAX_CONFIG_PATH", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("dialled anyway"))

    assert rpc_probe.main() == 1
    assert "cannot read the node config" in capsys.readouterr().err
