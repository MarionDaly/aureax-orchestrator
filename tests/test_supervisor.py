"""Edge supervisor behaviours that only bite in production.

Each case here is one that looked fine in review and misbehaved on a node: an
exit code nobody recognises, a shutdown that stalls for its whole timeout, a
telemetry buffer that grows forever, or a peer list that costs five minutes to
reject.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

import httpx
import pytest

from client.entrypoint import AureaxClient, _exit_status, _is_host_port

# --- TLS scaffolding ----------------------------------------------------------
#
# TLS is mandatory fleet-wide: start() refuses to run without the three files
# the deployer installs (client/tls.py), and every in-pod httpx client verifies
# against the fleet CA. So a supervisor test that reaches either needs REAL
# material — ssl.create_default_context parses the CA, and _require_tls_files
# reads a byte from each file, so paths to nothing exercise only the error
# paths.


@pytest.fixture(scope="session")
def fleet_tls(tmp_path_factory) -> dict:
    """A CA and one node leaf on disk, minted by the orchestrator's own store.

    Session-scoped because minting is the slow part and nothing here mutates
    the files. Uses orchestrator/config/tlsstore.py rather than hand-rolled
    PEMs so the material is exactly what `aureax generate` produces and
    `aureax apply` installs at these paths.
    """
    from orchestrator.config import tlsstore

    home = tmp_path_factory.mktemp("aureax-home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HOME", str(home))
        tlsstore.ensure_ca()
        cert, key = tlsstore.ensure_leaf("test-node", ["IP:127.0.0.1", "DNS:localhost"])
        ca = tlsstore.ca_cert_path()
    return {"ca_file": str(ca), "cert_file": str(cert), "key_file": str(key)}


# --- exit-status mapping -----------------------------------------------------

@pytest.mark.parametrize(
    ("rc", "expected"),
    [
        (0, 0),
        (1, 1),
        (-signal.SIGKILL, 137),
        (-signal.SIGTERM, 143),
        (None, 0),
    ],
)
def test_signal_deaths_map_to_conventional_exit_codes(rc, expected) -> None:
    """Popen reports a signal death as negative, and sys.exit(-9) wraps to 247.

    `kubectl describe` then shows an exit code nobody recognises instead of the
    137 everyone reads as OOM/SIGKILL.
    """
    assert _exit_status(rc) == expected


# --- RPC peer validation -----------------------------------------------------

@pytest.mark.parametrize(
    "peer",
    ["10.77.0.2:50052", "worker-1:50052", "10.77.0.2:1"],
)
def test_valid_rpc_peers_are_accepted(peer) -> None:
    assert _is_host_port(peer)


@pytest.mark.parametrize(
    "peer",
    [
        "10.77.0.2",        # no port
        ":50052",          # no host
        "10.77.0.2:abc",    # non-numeric port
        "10.77.0.2:0",      # port 0
        "10.77.0.2:70000",  # out of range
        "${rpc_servers}",  # unresolved placeholder, char by char
        "$",
        "{",
    ],
)
def test_malformed_rpc_peers_are_rejected(peer) -> None:
    """A malformed peer can never become reachable.

    Treating one as a slow node costs the head its whole 300s peer timeout
    before it says anything useful.
    """
    assert not _is_host_port(peer)


def test_unresolved_placeholder_peers_fail_fast(bare_client, monkeypatch) -> None:
    """An unresolved ${rpc_servers} used to burn the full timeout, one char at a time."""
    client = bare_client()
    client.config = {
        "node": {"role": "edge_executor", "worker_wait_timeout_seconds": 300},
        "model": {"server": {"rpc_servers": list("${rpc_servers}")}},
    }
    client._shutdown = False

    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    with pytest.raises(SystemExit) as exc:
        client._wait_for_rpc_workers()

    assert exc.value.code == 1
    assert slept == [], "must not enter the wait loop at all"


# --- port probing ------------------------------------------------------------

def test_port_probe_detects_a_listener_on_localhost() -> None:
    """The probe is how a shard worker learns its own rpc-server came up."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        # backlog 5, not 1: with a backlog of 1 the first unaccepted probe fills
        # the queue and the second is dropped. That is exactly why the dashboard
        # must not poll a worker's rpc-server port (see daemonset-shard.yaml),
        # but it is not what this test is checking.
        srv.listen(5)
        port = srv.getsockname()[1]

        assert AureaxClient._port_in_use("127.0.0.1", port)
        # A bind address of 0.0.0.0 is probed via loopback.
        assert AureaxClient._port_in_use("0.0.0.0", port)

    # Closed again — the probe must not report a stale success.
    assert not AureaxClient._port_in_use("127.0.0.1", port)


# --- CPU affinity ------------------------------------------------------------

def _client_with_affinity(bare_client, cpus) -> AureaxClient:
    client = bare_client()
    client.config = {"node": {"cpu_affinity": cpus}} if cpus is not None else {"node": {}}
    return client


def test_no_affinity_configured_leaves_the_spawn_path_untouched(bare_client) -> None:
    """Absent or empty means None, so Popen gets no preexec_fn at all.

    Every node that does not opt in must spawn exactly as it did before this
    feature existed.
    """
    assert _client_with_affinity(bare_client, None)._affinity_preexec() is None
    assert _client_with_affinity(bare_client, [])._affinity_preexec() is None
    client = bare_client()
    client.config = {}
    assert client._affinity_preexec() is None


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"),
    reason="sched_setaffinity is Linux-only",
)
def test_affinity_pins_the_child_to_the_requested_cores(bare_client) -> None:
    """The hook must actually change the child's mask, not just be callable."""
    available = sorted(os.sched_getaffinity(0))
    target = available[: min(2, len(available))]

    preexec = _client_with_affinity(bare_client, target)._affinity_preexec()
    assert preexec is not None

    pid = os.fork()
    if pid == 0:  # child
        try:
            preexec()
            os._exit(0 if sorted(os.sched_getaffinity(0)) == target else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert os.WEXITSTATUS(status) == 0


def test_unavailable_cores_decline_rather_than_break_the_spawn(bare_client) -> None:
    """Pinning to cores the process cannot use makes exec fail outright.

    On a node with fewer cores than the config assumes — or under a cpuset that
    excludes them — that would take the pod down on every restart. Declining to
    pin keeps the node serving, slower, which is the right failure direction.
    """
    assert _client_with_affinity(bare_client, [9001, 9002])._affinity_preexec() is None


# --- shutdown ----------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Popen/terminate timing; the project targets macOS + Linux",
)
def test_stop_reaps_a_live_child_promptly(bare_client) -> None:
    """stop() reaps a live child promptly instead of spinning its timeout.

    What this pins is the observable outcome: the child is reaped and stop()
    returns in well under `shutdown_timeout_seconds`.

    Why that outcome is fragile, and why the supervisor polls rather than
    calling Popen.wait(): a blocking wait() holds CPython's non-reentrant
    _waitpid_lock across os.waitpid, so a stop() running from the signal
    handler sees poll() return None and spins the full timeout before logging a
    SIGKILL for a child that is already dead. This test does not reconstruct
    that concurrency by hand — doing so reliably means racing an interpreter
    lock — so treat it as a guard on the timeout budget, not proof that the
    reentrant case is handled.
    """
    client = bare_client()
    client.config = {"process": {"shutdown_timeout_seconds": 30}}
    client._shutdown = False
    client._child_name = "rpc-server"
    client._child_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    started = time.monotonic()
    client._shutdown = True
    client.stop()
    elapsed = time.monotonic() - started

    assert client._child_proc.poll() is not None, "child must be reaped"
    assert elapsed < 5, (
        f"stop() took {elapsed:.1f}s — it should terminate promptly, not spin "
        "out shutdown_timeout_seconds on an already-dead child"
    )


def test_stop_is_safe_on_a_partially_populated_client() -> None:
    """The one site that must NOT use the shared `bare_client` factory.

    stop() reads every child-handle attribute through getattr with a default,
    and entrypoint.py's docstring says why: _exit_reporting funnels every
    failure path through stop(), including paths reached before __init__
    finished populating the object. The factory sets the full attribute set, so
    building through it here would make the test pass no matter what stop()
    did — the attributes whose ABSENCE is the subject would always be present.
    """
    client = AureaxClient.__new__(AureaxClient)
    client.config = {}
    client._child_proc = None
    # _child_name, _child_log_thread and _child_log_file are deliberately never
    # set: their absence is the case under test.
    client.stop()  # must not raise


# --- telemetry retention -----------------------------------------------------

def _telemetry_client(bare_client, tmp_path) -> AureaxClient:
    """A client scaffolded for _telemetry_loop, with a real on-disk buffer."""
    from client.telemetry.storage import TelemetryStorage

    return bare_client(
        node_id="n1", telemetry_storage=TelemetryStorage(str(tmp_path / "t.db"))
    )


def _stop_after(client, passes: int = 1):
    """A ``time.sleep`` stand-in that ends the loop after ``passes`` iterations.

    The loop is `while not self._shutdown`, so every test here needs some way
    to make it terminate; hand-rolling one per test meant three different
    spellings of the same stopper. The seconds it was asked to sleep for
    accumulate on ``.slept``, which is what pins the loop's own interval floor.
    """

    def sleeper(seconds) -> None:
        sleeper.slept.append(seconds)
        if len(sleeper.slept) >= passes:
            client._shutdown = True

    sleeper.slept = []
    return sleeper


def _record_rotations(monkeypatch) -> list[int]:
    """Record the retention_days every TelemetryStorage.rotate call is given."""
    from client.telemetry.storage import TelemetryStorage

    rotated: list[int] = []
    monkeypatch.setattr(
        TelemetryStorage, "rotate", lambda _s, days: rotated.append(days) or 0
    )
    return rotated


def test_telemetry_loop_prunes_the_local_buffer(bare_client, tmp_path, monkeypatch) -> None:
    """rotate() had no callers, so retention_days did nothing and the DB grew forever."""
    client = _telemetry_client(bare_client, tmp_path)
    rotated = _record_rotations(monkeypatch)
    monkeypatch.setattr("client.entrypoint.collect_all", lambda *_a, **_k: [])
    # One pass, then stop: sample_interval=1 makes the tail loop exit at once.
    monkeypatch.setattr(time, "sleep", _stop_after(client))

    client._telemetry_loop(None, 1, 60, "", reporter=None, retention_days=3)
    assert rotated == [3]


def test_telemetry_loop_skips_pruning_when_retention_is_disabled(
    bare_client, tmp_path, monkeypatch
) -> None:
    client = _telemetry_client(bare_client, tmp_path)
    rotated = _record_rotations(monkeypatch)
    monkeypatch.setattr("client.entrypoint.collect_all", lambda *_a, **_k: [])
    monkeypatch.setattr(time, "sleep", _stop_after(client))

    client._telemetry_loop(None, 1, 60, "", reporter=None, retention_days=0)
    assert rotated == []


def test_telemetry_loop_pushes_a_backlog_without_stretching_the_sample_interval(
    bare_client, tmp_path, monkeypatch
) -> None:
    """The push arguments are the whole contract between the loop and the reporter.

    ``attempts=1`` because the rows are already durably buffered: the
    reporter's own 3 x 30s backoff bought no durability and stopped SAMPLING
    for ~3 minutes per cycle, punching holes in the buffer during exactly the
    outage it exists to cover. ``max_batches=10`` because one batch per push
    drains a backlog at (batch_size - production rate) rows per interval, so a
    day offline needed hours of catch-up with the dashboard stale throughout.
    """
    client = _telemetry_client(bare_client, tmp_path)

    pushes: list[dict] = []
    reporter = types.SimpleNamespace(push=lambda **kw: pushes.append(kw))

    monkeypatch.setattr("client.entrypoint.collect_all", lambda *_a, **_k: [])
    monkeypatch.setattr(time, "sleep", _stop_after(client))

    # A non-empty endpoint is what arms the push branch at all.
    client._telemetry_loop(None, 1, 0, "https://mac:7001/api/v1/telemetry", reporter=reporter,
                           retention_days=0)
    assert pushes == [{"max_batches": 10, "attempts": 1}]


def test_telemetry_loop_does_not_push_without_an_endpoint(
    bare_client, tmp_path, monkeypatch
) -> None:
    """An endpoint-less node buffers locally; there is nowhere to push to."""
    client = _telemetry_client(bare_client, tmp_path)

    pushes: list[dict] = []
    reporter = types.SimpleNamespace(push=lambda **kw: pushes.append(kw))

    monkeypatch.setattr("client.entrypoint.collect_all", lambda *_a, **_k: [])
    monkeypatch.setattr(time, "sleep", _stop_after(client))

    client._telemetry_loop(None, 1, 0, "", reporter=reporter, retention_days=0)
    assert pushes == []


def test_telemetry_loop_survives_a_storage_error(bare_client, tmp_path, monkeypatch) -> None:
    """One SQLite failure must not end telemetry for the life of the pod.

    collect_all/insert sit at the top of a daemon thread with no handler above
    them, so an escaping sqlite3.OperationalError ("disk is full", "database is
    locked") silently stopped both telemetry and — when it rode the same loop —
    log shipping, with the pod still reporting healthy.
    """
    client = _telemetry_client(bare_client, tmp_path)

    calls: list[int] = []

    def exploding_collect(*_a, **_k):
        calls.append(1)
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr("client.entrypoint.collect_all", exploding_collect)
    monkeypatch.setattr(time, "sleep", _stop_after(client, passes=2))

    # Must return normally rather than propagating out of the thread body.
    client._telemetry_loop(None, 1, 60, "", reporter=None, retention_days=0)
    assert len(calls) >= 2, "loop stopped after the first failure"


# --- start() ordering --------------------------------------------------------

def test_reaper_runs_before_the_port_probe(bare_client, monkeypatch, fleet_tls) -> None:
    """The reaper must run before start() decides to attach to a live port.

    An orphaned llama-server from a previous pod is precisely what holds the
    port, and llama.cpp accepts connections while still loading. Reaping only
    inside _spawn_child meant the attach branch was taken first and the
    mitigation never ran in the scenario it was written for.
    """
    client = bare_client()
    client.node_id = "head"
    client.config = {
        "tls": fleet_tls,
        "node": {"role": "edge_executor"},
        "mcp": {"server": {"enabled": False}},
    }
    client._shutdown = False

    order: list[str] = []
    monkeypatch.setattr(AureaxClient, "_shard_role", lambda _s: "edge_executor")
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: None)
    monkeypatch.setattr(
        AureaxClient, "_reap_stale_children", lambda _s, b: order.append(f"reap:{b}")
    )
    monkeypatch.setattr(
        AureaxClient, "_llama_port_in_use", lambda _s: order.append("probe") or True
    )
    monkeypatch.setattr(AureaxClient, "_llama_port", lambda _s: 8080)
    # The attach branch health-checks the server it attached to; without this
    # the real wait would poll httpx for its whole timeout and then exit.
    monkeypatch.setattr(
        AureaxClient, "_wait_for_llama_health",
        lambda _s, *_a, **_k: order.append("wait_health"),
    )
    monkeypatch.setattr(
        AureaxClient, "_refuse_unauthenticated_attach",
        lambda _s: order.append("auth_probe"),
    )
    monkeypatch.setattr(signal, "signal", lambda *_a: None)
    # With no MCP runtime configured, start() now supervises the child forever
    # instead of returning (PID-1-exits-0 fix); this test only cares about the
    # reap/probe ordering on the way there.
    monkeypatch.setattr(AureaxClient, "_block_on_child", lambda _s: None)

    client.start()

    assert order[0].startswith("reap:"), f"probe ran before reap: {order}"
    assert "llama-server" in order[0]
    assert "probe" in order
    # Attaching to whatever holds the port without checking it meant a wedged
    # or half-loaded server was fronted until the 3015s startupProbe budget.
    assert "wait_health" in order, "the attached server was never health-checked"
    # And only then the auth check: it reads /props off a server that answers.
    assert order[-1] == "auth_probe", f"the attach branch skipped the auth check: {order}"


def test_a_surviving_orphan_ends_the_start_before_the_peer_wait(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """A recorded child still alive holds :8080 under hostNetwork.

    Neither branch can then work: attaching fronts a process this supervisor has
    no handle on, and spawning cannot bind. So start() must refuse immediately —
    and BEFORE the peer wait, which would otherwise spend
    node.worker_wait_timeout_seconds (300s on a head) reaching the same failure.
    The exit goes through _exit_reporting, the only path that ships the crash
    context, so the pod restart carries a reason.
    """
    client = bare_client()
    client.node_id = "head"
    client.config = {
        "tls": fleet_tls,
        "node": {"role": "edge_executor"},
        "mcp": {"server": {"enabled": False}},
    }
    client._shutdown = False

    order: list[str] = []
    monkeypatch.setattr(AureaxClient, "_shard_role", lambda _s: "edge_executor")
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_reap_stale_children", lambda _s, _b: [4242])
    monkeypatch.setattr(
        AureaxClient, "_llama_port_in_use", lambda _s: order.append("probe") or True
    )
    monkeypatch.setattr(AureaxClient, "_llama_port", lambda _s: 8080)
    monkeypatch.setattr(
        AureaxClient, "_wait_for_rpc_workers", lambda _s: order.append("wait_workers")
    )
    monkeypatch.setattr(
        AureaxClient, "_start_llama_server", lambda _s: order.append("start_llama")
    )
    monkeypatch.setattr(
        AureaxClient, "_wait_for_llama_health",
        lambda _s, *_a, **_k: order.append("wait_health"),
    )
    monkeypatch.setattr(signal, "signal", lambda *_a: None)
    monkeypatch.setattr(AureaxClient, "_block_on_child", lambda _s: None)

    reported: list[str] = []
    monkeypatch.setattr(
        AureaxClient, "_flush_crash_context", lambda _s, _rc: reported.append("flush")
    )
    monkeypatch.setattr(AureaxClient, "_ship_logs_now", lambda _s: reported.append("ship"))

    with pytest.raises(SystemExit) as exc:
        client.start()

    assert exc.value.code == 1
    assert order == [], f"start() got past the survivor check: {order}"
    assert reported == ["flush", "ship"], "the refusal skipped the reporting exit path"


def test_head_waits_for_its_rpc_peers_before_spawning_llama_server(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """On the spawn path, peers must be up before llama-server is started.

    This is the second half of the ordering the deploy phases enforce, and the
    only half that covers a pod restart or a node reboot — a head whose peers
    are not listening exits immediately, and the DaemonSet then restarts it
    into the same race forever. Swapping the two calls breaks nothing that any
    other test can see.

    The reap/probe pair is re-checked here because this is the branch the other
    test cannot reach: it pins `_llama_port_in_use` True to exercise the attach
    branch, which returns before either of the calls below.
    """
    client = bare_client()
    client.node_id = "head"
    client.config = {
        "tls": fleet_tls,
        "node": {"role": "edge_executor"},
        "mcp": {"server": {"enabled": False}},
    }
    client._shutdown = False

    order: list[str] = []

    def _probe(_self):
        order.append("probe")
        return False

    monkeypatch.setattr(AureaxClient, "_shard_role", lambda _s: "edge_executor")
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: None)
    monkeypatch.setattr(
        AureaxClient, "_reap_stale_children", lambda _s, _b: order.append("reap")
    )
    monkeypatch.setattr(AureaxClient, "_llama_port_in_use", _probe)
    monkeypatch.setattr(AureaxClient, "_llama_port", lambda _s: 8080)
    monkeypatch.setattr(
        AureaxClient, "_wait_for_rpc_workers", lambda _s: order.append("wait_workers")
    )
    monkeypatch.setattr(
        AureaxClient, "_start_llama_server", lambda _s: order.append("start_llama")
    )
    # The real signature takes timeout/interval keywords.
    monkeypatch.setattr(
        AureaxClient, "_wait_for_llama_health",
        lambda _s, *_a, **_k: order.append("wait_health"),
    )
    monkeypatch.setattr(signal, "signal", lambda *_a: None)
    monkeypatch.setattr(AureaxClient, "_block_on_child", lambda _s: None)

    client.start()

    assert order == ["reap", "probe", "wait_workers", "start_llama", "wait_health"], order


# --- attach branch: the port's auth state ------------------------------------

LLAMA_KEY = "aureax_sk_" + "l" * 32


def _stub_httpx(monkeypatch, status: int) -> dict:
    """Point httpx.Client at a stub answering `status`; returns what it saw."""
    seen: dict = {}

    class _Stub:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def get(self, url, headers=None):
            seen["url"] = url
            seen["headers"] = headers
            return types.SimpleNamespace(status_code=status)

    monkeypatch.setattr(httpx, "Client", _Stub)
    return seen


@pytest.mark.parametrize("status", [401, 403, 404])
def test_an_attached_server_that_enforces_a_key_is_attached_to(
    bare_client, monkeypatch, fleet_tls, status
) -> None:
    """Anything but a 2xx leaves the attach alone.

    The check acts on positive evidence that the port is ungated, never on the
    absence of evidence: a build that moved /props, or a transient error, must
    not crash-loop a head whose server is enforcing its key perfectly.
    """
    client = bare_client()
    client.config = {
        "tls": fleet_tls,
        "model": {"server": {"port": 8080, "api_key": LLAMA_KEY}},
    }
    seen = _stub_httpx(monkeypatch, status)

    client._refuse_unauthenticated_attach()

    from client.tls import ssl_context

    assert seen["url"] == "https://127.0.0.1:8080/props"
    # /health is auth-exempt, so probing it would prove nothing; and the probe
    # must carry NO key, or a gated server answers 200 and reads as ungated.
    assert seen["headers"] is None, "the probe must send no Authorization header"
    assert seen["verify"] is ssl_context(client.config), "the fleet CA context was not used"


def test_an_attached_server_with_no_key_is_refused_rather_than_fronted(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """A foreign llama-server on :8080 with auth off would be served ungated.

    Nothing else catches it: the reaper only knows recorded children, /health
    answers 200 either way, so the pod goes Ready and the node's :30800
    NodePort serves inference with no bearer gate while its own config says
    auth is on. Crash-looping is the visible outcome; fronting it is the silent
    one.
    """
    client = bare_client()
    client.config = {
        "tls": fleet_tls,
        "model": {"server": {"port": 8080, "api_key": LLAMA_KEY}},
        "process": {"shutdown_timeout_seconds": 1},
    }
    _stub_httpx(monkeypatch, 200)

    reported: list[str] = []
    monkeypatch.setattr(
        AureaxClient, "_flush_crash_context", lambda _s, _rc: reported.append("flush")
    )
    monkeypatch.setattr(AureaxClient, "_ship_logs_now", lambda _s: reported.append("ship"))

    with pytest.raises(SystemExit) as exc:
        client._refuse_unauthenticated_attach()

    assert exc.value.code == 1
    assert reported == ["flush", "ship"], "the refusal must leave through the reporting exit"


def test_a_fleet_running_without_auth_makes_no_probe_at_all(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """With no key configured there is nothing to enforce, so nothing is asked."""
    client = bare_client()
    client.config = {"tls": fleet_tls, "model": {"server": {"port": 8080}}}

    def _never(**_kwargs):
        raise AssertionError("an auth-off node must not probe the attached server")

    monkeypatch.setattr(httpx, "Client", _never)

    client._refuse_unauthenticated_attach()


def test_a_probe_that_cannot_be_made_does_not_take_the_node_down(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """A server that just answered /health can still drop this connection.

    Exiting on a transport error would turn one dropped socket into a pod
    restart on a node that is serving correctly.
    """
    client = bare_client()
    client.config = {
        "tls": fleet_tls,
        "model": {"server": {"port": 8080, "api_key": LLAMA_KEY}},
    }

    class _Broken:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def get(self, *_a, **_kw):
            raise httpx.ConnectError("connection reset")

    monkeypatch.setattr(httpx, "Client", _Broken)

    client._refuse_unauthenticated_attach()


# --- planner MCP host resilience ---------------------------------------------

def _planner_client(bare_client, monkeypatch, pools: list, fleet_tls) -> AureaxClient:
    """A planner client whose MCPPool is a recording stub.

    The config carries real TLS paths because _run_planner_host builds the
    fleet-CA context the pool dials every remote under.
    """
    client = bare_client()
    client.config = {"tls": fleet_tls}
    client._shutdown = False
    client._child_proc = None

    class _StubPool:
        def __init__(self, verify=None) -> None:
            self.closed = False
            self.verify = verify
            self.connects: list[tuple] = []
            pools.append(self)

        async def connect_url(self, name, url, api_key=None):
            self.connects.append((name, url, api_key))
            return True

        @property
        def connected_nodes(self):
            return []

        async def close(self):
            self.closed = True

    import client.mcp.pool as pool_mod
    monkeypatch.setattr(pool_mod, "MCPPool", _StubPool)
    return client


def test_planner_survives_a_remote_session_cancellation(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """A remote MCP server going away must not kill the planner pod.

    connect_url enters the MCP SDK's streamable_http_client in the planner's own
    task, so a dead remote fires an anyio cancel scope whose CancelledError
    surfaces at the planner's next await. Propagating it made the supervisor
    exit(1) and CrashLoopBackOff every time an edge node restarted — observed
    10 restarts from a single head outage.
    """
    import asyncio as aio

    pools: list = []
    client = _planner_client(bare_client, monkeypatch, pools, fleet_tls)

    calls = {"n": 0}
    real_sleep = aio.sleep

    async def flaky_sleep(_secs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise aio.CancelledError("Cancelled via cancel scope")
        if calls["n"] >= 3:
            client._shutdown = True
        await real_sleep(0)

    monkeypatch.setattr(aio, "sleep", flaky_sleep)

    aio.run(client._run_planner_host([{"name": "edge-01", "url": "https://x/mcp"}]))

    # Survived the cancellation and kept looping rather than exiting.
    assert calls["n"] >= 3, "planner stopped looping after the cancellation"
    # The torn-down pool was replaced, not reused.
    assert len(pools) == 2, f"expected the pool to be rebuilt, saw {len(pools)}"
    assert pools[0].closed, "the dead pool must be closed"


def test_planner_still_propagates_a_real_shutdown(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """Recovery must not swallow the CancelledError that stop() relies on."""
    import asyncio as aio

    pools: list = []
    client = _planner_client(bare_client, monkeypatch, pools, fleet_tls)

    async def shutdown_sleep(_secs):
        client._shutdown = True
        raise aio.CancelledError("Cancelled via cancel scope")

    monkeypatch.setattr(aio, "sleep", shutdown_sleep)

    with pytest.raises(aio.CancelledError):
        aio.run(client._run_planner_host([{"name": "edge-01", "url": "https://x/mcp"}]))


# --- orphan reaping ----------------------------------------------------------

def _reaper_client(bare_client, reap: bool = True) -> AureaxClient:
    client = bare_client()
    client.config = {"node": {"reap_stale_children": reap}}
    return client


def _unique_victim(tmp_path: Path) -> Path:
    """Use a unique binary for each bounded orphan-process test."""
    src = shutil.which("sleep")
    assert src, "no sleep binary on PATH"
    victim = tmp_path / f"axreap-{uuid.uuid4().hex[:8]}"
    shutil.copy(src, victim)
    victim.chmod(0o755)
    return victim


def _spawn_orphan(binary: Path, seconds: int = 300) -> int:
    """Start a process that is NOT a direct child of this one. Returns its pid.

    This matters. ``_reap_stale_children`` skips any process whose ppid is our
    own — "a live child of ours is not stale" — so a plain ``Popen`` victim can
    never be reaped, and a test built on one passes for the wrong reason no
    matter what the reaper does. Double-forking through a shell that exits
    immediately re-parents the grandchild to init, which is exactly what an
    orphan from a previous pod looks like in /proc.
    """
    proc = subprocess.run(
        ["sh", "-c", f'{shlex.quote(str(binary))} {seconds} >/dev/null 2>&1 & echo $!'],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _register_test_orphan(pid, binary):
    import json

    from client import process_ownership
    # This record refers only to the unique process created by this test.
    record = {"child": process_ownership._identity(pid),
              "owner": {"pid": 2147483647, "start": "0", "boot": "expired-test-owner"}}
    process_ownership._record_path("test-node", str(binary)).write_text(json.dumps(record))


def _reap_orphan(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)


def _is_zombie(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    """Poll until `pid` disappears. It is not our child, so waitpid is out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_reaper_kills_an_orphaned_inference_process(bare_client, tmp_path: Path) -> None:
    """An orphan holds the group's RPC slots and wedges every replacement pod.

    rpc-server serves exactly ONE client, so a llama-server that outlives its
    pod keeps its RPC sockets ESTABLISHED and the new pod waits on peers it
    can never reach — reporting only "still loading". Observed wedging the
    group for ~40 minutes with no error surfaced anywhere.
    """
    victim = _unique_victim(tmp_path)
    pid = _spawn_orphan(victim)
    _register_test_orphan(pid, victim)
    try:
        assert _alive(pid), "the victim did not start"
        _reaper_client(bare_client)._reap_stale_children(f"/usr/local/bin/{victim.name}")
        assert _wait_gone(pid) or _is_zombie(pid), "the orphan should have exited"
    finally:
        _reap_orphan(pid)


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_reaper_waits_for_the_kill_to_land(bare_client, tmp_path: Path) -> None:
    """os.kill only QUEUES the signal; the port stays held until the pid is gone.

    start() probes the port microseconds after this returns, so returning while
    the orphan is still unmapping its model put the pod straight back on the
    attach branch — in front of the corpse it had just killed.
    """
    victim = _unique_victim(tmp_path)
    pid = _spawn_orphan(victim)
    _register_test_orphan(pid, victim)
    try:
        survivors = _reaper_client(bare_client)._reap_stale_children(f"/usr/local/bin/{victim.name}")
        assert survivors == [], f"reaper returned before the kill landed: {survivors}"
        assert not _alive(pid) or _is_zombie(pid), "the reaper returned while the orphan was still running"
    finally:
        _reap_orphan(pid)


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_reaper_leaves_unrelated_binaries_alone(bare_client, tmp_path: Path) -> None:
    """Only the binary THIS node manages is a candidate.

    A node hosting both roles must never have one supervisor kill the other's
    child, so the match is on the exact process name. The bystander is an
    orphan too, so the name check is the only thing that can spare it.
    """
    pid = _spawn_orphan(_unique_victim(tmp_path), seconds=5)
    try:
        _reaper_client(bare_client)._reap_stale_children("/usr/local/bin/llama-server")
        assert _alive(pid), "an unrelated process was killed"
    finally:
        _reap_orphan(pid)


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_reaper_can_be_disabled(bare_client, tmp_path: Path) -> None:
    """node.reap_stale_children: false must actually disable the scan.

    The victim is an orphan matching the target binary, so with reaping enabled
    it WOULD be killed — which is what makes this test capable of failing, and
    why the victim's name must not be shared with anything else on the host.
    """
    victim = _unique_victim(tmp_path)
    pid = _spawn_orphan(victim, seconds=5)
    try:
        _reaper_client(bare_client, reap=False)._reap_stale_children(f"/usr/local/bin/{victim.name}")
        assert _alive(pid), "reaping ran despite reap_stale_children: false"
    finally:
        _reap_orphan(pid)


def test_the_spawn_path_refuses_a_survivor_through_the_reporting_exit(
    bare_client, monkeypatch, tmp_path: Path
) -> None:
    """The last line of defence, reached under the spawn lock.

    Raising here ended PID 1 with a traceback out of main(), which has no
    handler — skipping the flush and the log ship that are the only way the
    reason ever leaves the node.
    """
    client = bare_client()
    client.config = {"process": {"shutdown_timeout_seconds": 1}}
    monkeypatch.setattr(AureaxClient, "_reap_stale_children", lambda _s, _b: [4242])

    reported: list[str] = []
    monkeypatch.setattr(
        AureaxClient, "_flush_crash_context", lambda _s, _rc: reported.append("flush")
    )
    monkeypatch.setattr(AureaxClient, "_ship_logs_now", lambda _s: reported.append("ship"))

    with pytest.raises(SystemExit) as exc:
        client._spawn_child("llama-server", ["sh", "-c", "sleep 30"], tmp_path / "child.log")

    assert exc.value.code == 1
    assert reported == ["flush", "ship"]
    assert client._child_proc is None, "a second inference process was spawned over a live one"


# --- fatal-exit hardening -----------------------------------------------------

class _FakeChild:
    """Popen stand-in whose lifecycle the tests can observe."""

    def __init__(self, rc: int | None = None) -> None:
        self.rc = rc
        self.pid = 4242
        self.terminated = False

    def poll(self):
        return self.rc

    @property
    def returncode(self):
        return self.rc

    def terminate(self) -> None:
        self.terminated = True
        if self.rc is None:
            self.rc = -15

    def wait(self, timeout=None):
        return self.rc

    def kill(self) -> None:
        self.rc = -9


def test_health_timeout_terminates_the_child_and_ships_the_report(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """The health-timeout exit used to leave a hostPID llama-server running for
    the next pod's reaper, and bypassed _exit_reporting — so the crash context
    explaining the failed load never left the node.
    """
    client = bare_client()
    client.config = {"tls": fleet_tls, "process": {"shutdown_timeout_seconds": 1}}
    client._shutdown = False
    child = _FakeChild(rc=None)
    client._child_proc = child
    client._child_name = "llama-server"

    reported: list = []
    monkeypatch.setattr(
        AureaxClient, "_flush_crash_context", lambda _s, rc: reported.append(("flush", rc))
    )
    monkeypatch.setattr(
        AureaxClient, "_ship_logs_now", lambda _s: reported.append(("ship", None))
    )

    with pytest.raises(SystemExit) as exc:
        client._wait_for_llama_health(timeout=0)

    assert exc.value.code == 1
    assert child.terminated, "the still-running child must be torn down on the way out"
    assert reported == [("flush", None), ("ship", None)]


def test_planner_llama_crash_exits_with_the_wrapped_status(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """A SIGKILLed llama-server must surface as 137 in `kubectl describe`, not
    the 247 that sys.exit(-9) wraps to — and must flush/ship like every other
    fatal path (this one used to do neither).
    """
    import asyncio as aio

    pools: list = []
    client = _planner_client(bare_client, monkeypatch, pools, fleet_tls)
    client.config = {"tls": fleet_tls, "process": {"shutdown_timeout_seconds": 1}}
    client._child_proc = _FakeChild(rc=-9)
    client._child_name = "llama-server"

    reported: list = []
    monkeypatch.setattr(
        AureaxClient, "_flush_crash_context", lambda _s, rc: reported.append(rc)
    )
    monkeypatch.setattr(AureaxClient, "_ship_logs_now", lambda _s: reported.append("ship"))

    with pytest.raises(SystemExit) as exc:
        aio.run(client._run_planner_host([]))

    assert exc.value.code == 137
    assert reported == [-9, "ship"]


def test_capped_log_file_rotates_once_at_the_cap(tmp_path) -> None:
    """The child log lives on a hostPath volume; before the cap it grew for the
    lifetime of the node. Rotation keeps the newest output — the part that
    explains a crash — while bounding the disk at ~2x the cap.
    """
    from client.entrypoint import _CappedLogFile

    path = tmp_path / "child.log"
    f = _CappedLogFile(path, max_bytes=100)
    f.write("a" * 60)
    f.write("b" * 60)  # crosses the cap -> rotate
    f.write("tail")
    f.flush()
    f.close()

    rotated = tmp_path / "child.log.1"
    assert rotated.exists()
    assert "a" * 60 in rotated.read_text() and "b" * 60 in rotated.read_text()
    assert path.read_text() == "tail"


def test_a_failed_rotation_degrades_the_sink_instead_of_killing_it(tmp_path, monkeypatch) -> None:
    """The cap is hit when the disk is most likely full, which is where the
    reopen fails. The exception escaped into _pump_child_output's suppress, so
    llama-server.log silently stopped recording for the rest of the pod's life
    — and _size stayed at the cap, so no write ever retried.
    """
    from client.entrypoint import _CappedLogFile

    path = tmp_path / "child.log"
    f = _CappedLogFile(path, max_bytes=100)

    real_open = Path.open
    failing = {"on": True}

    def flaky_open(self, *a, **kw):
        if failing["on"] and self == path:
            raise OSError(28, "No space left on device")
        return real_open(self, *a, **kw)

    monkeypatch.setattr(Path, "open", flaky_open)
    f.write("a" * 60)
    f.write("b" * 60)  # crosses the cap -> rotate -> reopen raises ENOSPC

    failing["on"] = False
    f.write("recovered\n")
    f.flush()
    f.close()

    assert path.read_text() == "recovered\n"
    # Not re-rotated on the retry: _size must be reset, or every later line
    # would replace the .1 archive with a one-line file.
    assert (tmp_path / "child.log.1").read_text().startswith("a" * 60)


def test_a_write_error_degrades_the_sink_loudly_and_recovers(tmp_path, monkeypatch, caplog) -> None:
    """A full disk hits write() long before it hits the rotation cap.

    _pump_child_output suppresses OSError, so an unguarded write left the sink
    dead for the pod's life with nothing logged: _size never advanced, so the
    rotation — the only other path that warns — was never reached. That is the
    exact state a node whose hostPath filled up is in.
    """
    import logging

    from client.entrypoint import _CappedLogFile

    path = tmp_path / "child.log"
    f = _CappedLogFile(path, max_bytes=10_000)

    class _FullDisk:
        """The open file object ENOSPC turns into: every write raises."""

        def write(self, _s):
            raise OSError(28, "No space left on device")

        def tell(self):
            return 0

        def flush(self):
            pass

        def close(self):
            pass

    real_open = Path.open
    failing = {"on": True}

    def flaky_open(self, *a, **kw):
        # The retry has to fail too, or the second line reopens the sink and
        # the one-shot warning cannot be tested at all.
        if failing["on"] and self == path:
            raise OSError(28, "No space left on device")
        return real_open(self, *a, **kw)

    f._f = _FullDisk()
    monkeypatch.setattr(Path, "open", flaky_open)
    with caplog.at_level(logging.WARNING, logger="aureax.client"):
        f.write("lost\n")
        f.write("also lost\n")

    degraded = [r for r in caplog.records if "sink degraded" in r.message]
    assert len(degraded) == 1, "one warning per failed line would be its own outage"

    failing["on"] = False
    f.write("recovered\n")
    f.flush()
    f.close()

    assert path.read_text() == "recovered\n", "the sink never retried through _reopen()"


# --- child supervision --------------------------------------------------------

def test_block_on_child_idles_when_there_is_no_child(bare_client, monkeypatch) -> None:
    """On the attach branch _child_proc is None, and exiting would take down a
    pod fronting a llama-server that is serving fine.
    """
    client = bare_client()
    client.config = {}
    client._shutdown = False
    client._child_proc = None

    exits: list = []
    monkeypatch.setattr(
        AureaxClient, "_exit_reporting", lambda _s, *a: exits.append(a)
    )
    monkeypatch.setattr(time, "sleep", lambda _s: setattr(client, "_shutdown", True))

    client._block_on_child()

    assert exits == [], "idling with no child must not exit the pod"


@pytest.mark.parametrize(("rc", "status"), [(0, 1), (-signal.SIGKILL, 137), (2, 2)])
def test_block_on_child_reports_the_wrapped_status(bare_client, rc, status, monkeypatch) -> None:
    """A child exiting 0 is still a failure for PID 1 — K8s would otherwise tear
    down the pod as a clean completion — and a signal death must reach
    `kubectl describe` as 137, not the 247 sys.exit(-9) wraps to.
    """
    client = bare_client()
    client.config = {}
    client._shutdown = False
    client._child_proc = _FakeChild(rc=rc)

    exits: list = []

    def record_and_exit(_self, *args):
        exits.append(args)
        raise SystemExit(args[0])

    monkeypatch.setattr(AureaxClient, "_exit_reporting", record_and_exit)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit):
        client._block_on_child()

    assert exits == [(status, rc)]


def test_the_mcp_branch_watcher_turns_a_child_crash_into_a_process_exit(bare_client, monkeypatch) -> None:
    """mcp.run() blocks forever, so nothing polled the child on this branch: a
    llama-server that OOMed after startup was noticed only by the kubelet's
    liveness probe, with the crash context discarded. sys.exit from the watcher
    thread would end only that thread, hence os._exit.
    """
    client = bare_client()
    client.config = {"process": {"shutdown_timeout_seconds": 1}}
    client._shutdown = False
    client._child_name = "llama-server"
    client._child_proc = _FakeChild(rc=-9)

    reported: list = []
    codes: list[int] = []
    monkeypatch.setattr(
        AureaxClient, "_flush_crash_context", lambda _s, rc: reported.append(rc)
    )
    monkeypatch.setattr(AureaxClient, "_ship_logs_now", lambda _s: reported.append("ship"))
    monkeypatch.setattr(os, "_exit", lambda code: codes.append(code))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    client._watch_child_in_background().join(timeout=5)

    assert codes == [137], f"watcher did not exit the process: {codes}"
    assert reported == [-9, "ship"]


# --- shard worker startup -----------------------------------------------------

def test_rpc_listen_wait_reports_a_child_that_died_before_listening(bare_client, monkeypatch) -> None:
    """A worker has no HTTP server and no reachable kubelet, so _exit_reporting
    is the ONLY way the run-up to its death reaches the orchestrator.
    """
    client = bare_client()
    client.config = {}
    client._shutdown = False
    client._child_proc = _FakeChild(rc=1)

    exits: list = []

    def record_and_exit(_self, *args):
        exits.append(args)
        raise SystemExit(args[0])

    monkeypatch.setattr(AureaxClient, "_exit_reporting", record_and_exit)
    monkeypatch.setattr(AureaxClient, "_port_in_use", staticmethod(lambda *_a, **_k: False))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit):
        client._wait_for_rpc_listening("127.0.0.1", 50052)

    assert exits == [(1, 1)]


def test_rpc_listen_wait_reports_its_own_timeout(bare_client, monkeypatch) -> None:
    """The 120s budget expiring must ship too — a regression to a bare
    sys.exit(1) drops the crash context from every worker-side load failure.
    """
    client = bare_client()
    client.config = {}
    client._shutdown = False
    client._child_proc = _FakeChild(rc=None)

    exits: list = []

    def record_and_exit(_self, *args):
        exits.append(args)
        raise SystemExit(args[0] if args else 1)

    monkeypatch.setattr(AureaxClient, "_exit_reporting", record_and_exit)
    monkeypatch.setattr(AureaxClient, "_port_in_use", staticmethod(lambda *_a, **_k: False))
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit):
        client._wait_for_rpc_listening("127.0.0.1", 50052, timeout=1)

    assert exits == [(1,)]


def test_peer_wait_names_only_the_unreachable_worker(bare_client, monkeypatch) -> None:
    """With several workers, "it hung" is indistinguishable from "one box is
    off" — and a peer that already answered must not be re-probed.
    """
    client = bare_client()
    client.config = {
        "node": {"role": "edge_executor", "worker_wait_timeout_seconds": 1},
        "model": {"server": {"rpc_servers": ["10.77.0.2:50052", "10.77.0.3:50052"]}},
    }
    client._shutdown = False

    probed: list[str] = []

    def probe(host, port, timeout=1.0):
        probed.append(host)
        return ("ok", (5, 0, 0)) if host == "10.77.0.2" else ("closed", None)

    exits: list = []

    def record_and_exit(_self, *args):
        exits.append(args)
        raise SystemExit(1)

    monkeypatch.setattr(AureaxClient, "_rpc_hello", staticmethod(probe))
    monkeypatch.setattr(AureaxClient, "_exit_reporting", record_and_exit)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit):
        client._wait_for_rpc_workers()

    assert exits == [(1,)], "the head's death must go through _exit_reporting"
    assert probed.count("10.77.0.2") == 1, "a peer that answered was re-probed"
    assert probed.count("10.77.0.3") > 1


# --- the peer wait speaks the RPC HELLO ---------------------------------------

def _fake_rpc_server(reply: bytes | None):
    """A one-shot TCP listener on 127.0.0.1 that records what it receives and
    answers `reply` (or nothing, holding the connection open, when None)."""
    import socket
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    received: list[bytes] = []
    done = threading.Event()

    def serve():
        conn, _ = srv.accept()
        with conn:
            conn.settimeout(3)
            try:
                received.append(conn.recv(64))
                if reply is not None:
                    conn.sendall(reply)
                else:
                    done.wait(3)
            except OSError:
                pass
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1], received, done


def test_rpc_hello_reads_the_servers_protocol_version() -> None:
    """The request is the pinned build's HELLO (1 + 8 + 24 bytes, command 14);
    the reply is a u64 size of 28 then major, minor, patch, pad, 24 caps."""
    import struct

    reply = struct.pack("<Q", 28) + bytes([7, 0, 0, 0]) + bytes(24)
    port, received, _ = _fake_rpc_server(reply)

    state, version = AureaxClient._rpc_hello("127.0.0.1", port, timeout=2.0)

    assert (state, version) == ("ok", (7, 0, 0))
    assert received and received[0] == bytes([14]) + struct.pack("<Q", 24) + bytes(24)


def test_rpc_hello_reports_a_listener_that_does_not_speak_the_protocol() -> None:
    """Accepts, says nothing: a pre-HELLO server or another service on the port."""
    port, _received, done = _fake_rpc_server(None)
    try:
        assert AureaxClient._rpc_hello("127.0.0.1", port, timeout=0.3) == ("silent", None)
    finally:
        done.set()


def test_rpc_hello_reports_a_closed_port() -> None:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert AureaxClient._rpc_hello("127.0.0.1", port, timeout=0.3) == ("closed", None)


def _head_waiting_on(bare_client, monkeypatch, tmp_path, build_json: str | None, answers):
    """A head whose peer wait is driven by `answers`, with the image's build
    record (and so the expected RPC protocol) set to `build_json`."""
    import client.entrypoint as entrypoint_mod

    client = bare_client()
    client.config = {
        "node": {"role": "edge_executor", "worker_wait_timeout_seconds": 1},
        "model": {"server": {"rpc_servers": ["10.77.0.2:50052"]}},
    }
    client._shutdown = False
    record = tmp_path / "llama-build.json"
    if build_json is not None:
        record.write_text(build_json)
    monkeypatch.setattr(entrypoint_mod, "_LLAMA_BUILD_JSON", str(record))
    monkeypatch.setattr(AureaxClient, "_rpc_hello", staticmethod(lambda h, p, timeout=1.0: answers.pop(0)))
    exits: list = []

    def record_and_exit(_self, *args):
        exits.append(args)
        raise SystemExit(1)

    monkeypatch.setattr(AureaxClient, "_exit_reporting", record_and_exit)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    return client, exits


def test_a_worker_on_another_rpc_protocol_fails_the_head_at_once(bare_client, monkeypatch, tmp_path, caplog) -> None:
    """Two llama.cpp builds in one group hang at the RPC handshake. With the
    image's own protocol recorded, the head refuses by name before llama-server
    ever connects, instead of after the 300 s wait plus a hang."""
    client, exits = _head_waiting_on(
        bare_client, monkeypatch, tmp_path,
        '{"version":"b11439","commit":"abc","rpc_proto":"7.0.0"}',
        [("ok", (5, 0, 0))],
    )
    with caplog.at_level("ERROR"), pytest.raises(SystemExit):
        client._wait_for_rpc_workers()
    assert exits == [(1,)]
    assert "5.0.0" in caplog.text and "expects 7.0" in caplog.text
    assert "build-on-node.sh" in caplog.text


def test_a_worker_on_the_same_rpc_protocol_is_accepted(bare_client, monkeypatch, tmp_path, caplog) -> None:
    client, exits = _head_waiting_on(
        bare_client, monkeypatch, tmp_path,
        '{"version":"b11439","commit":"abc","rpc_proto":"7.0.0"}',
        [("ok", (7, 0, 3))],
    )
    with caplog.at_level("INFO"):
        client._wait_for_rpc_workers()
    assert exits == []
    assert "RPC protocol 7.0.3" in caplog.text


def test_an_image_without_a_recorded_protocol_compares_nothing(bare_client, monkeypatch, tmp_path, caplog) -> None:
    """Images built before the key existed must not be refused by a check they
    cannot pass; the worker's version is still logged."""
    client, exits = _head_waiting_on(
        bare_client, monkeypatch, tmp_path,
        '{"version":"b10375","commit":"ba360efe"}',
        [("ok", (5, 0, 0))],
    )
    with caplog.at_level("INFO"):
        client._wait_for_rpc_workers()
    assert exits == []
    assert "RPC protocol 5.0.0" in caplog.text


def test_a_silent_listener_is_warned_about_once_and_waited_for(bare_client, monkeypatch, tmp_path, caplog) -> None:
    client, exits = _head_waiting_on(
        bare_client, monkeypatch, tmp_path, None,
        [("silent", None), ("silent", None), ("ok", (5, 0, 0))],
    )
    with caplog.at_level("WARNING"):
        client._wait_for_rpc_workers()
    assert exits == []
    assert caplog.text.count("did not answer the RPC HELLO") == 1


def test_worker_cache_dir_falls_back_and_is_exported_expanded(bare_client, tmp_path, monkeypatch) -> None:
    """An empty cache_dir with cache on would emit `-c` while leaving
    LLAMA_CACHE unset, so rpc-server cached into its container working
    directory — lost on every restart, restoring the multi-GiB re-pull. The
    exported value must also be the EXPANDED one: llama.cpp does no tilde
    expansion, so a "~" path created $HOME/... and cached into a directory
    literally named "~".
    """
    client = bare_client()
    client.config = {
        "model": {"rpc_server": {"cache": True, "cache_dir": "~/rpc-cache"}},
        "logging": {"llama_log_path": str(tmp_path / "rpc-server.log")},
    }
    client._shutdown = True
    client._child_proc = _FakeChild(rc=0)

    monkeypatch.setenv("HOME", str(tmp_path))
    # setenv first: monkeypatch records an undo only for a name it has
    # observed, so delenv on an already-absent variable records nothing and
    # the value _run_shard_worker writes straight into os.environ would
    # outlive the test. Seeding puts the original state in the ledger.
    monkeypatch.setenv("LLAMA_CACHE", "")
    monkeypatch.delenv("LLAMA_CACHE")
    monkeypatch.setattr(AureaxClient, "_spawn_child", lambda *_a, **_k: None)
    monkeypatch.setattr(AureaxClient, "_wait_for_rpc_listening", lambda *_a, **_k: None)
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: None)

    with pytest.raises(SystemExit):
        client._run_shard_worker()

    assert os.environ["LLAMA_CACHE"] == str(tmp_path / "rpc-cache")
    assert (tmp_path / "rpc-cache").is_dir()


def test_worker_cache_dir_defaults_when_left_empty(bare_client, tmp_path, monkeypatch) -> None:
    """`cache_dir: ""` must land on the default path, not on "no LLAMA_CACHE"."""
    client = bare_client()
    client.config = {
        "model": {"rpc_server": {"cache": True, "cache_dir": ""}},
        "logging": {"llama_log_path": str(tmp_path / "rpc-server.log")},
    }
    client._shutdown = True
    client._child_proc = _FakeChild(rc=0)

    made: list[str] = []
    # Seeded before the delete so monkeypatch has an undo to run; see the
    # note in test_worker_cache_dir_falls_back_and_is_exported_expanded.
    monkeypatch.setenv("LLAMA_CACHE", "")
    monkeypatch.delenv("LLAMA_CACHE")
    # The fallback is an absolute /opt path; creating it on a dev machine is
    # either a permission error or litter, so record instead.
    monkeypatch.setattr(Path, "mkdir", lambda self, **_k: made.append(str(self)))
    monkeypatch.setattr(AureaxClient, "_spawn_child", lambda *_a, **_k: None)
    monkeypatch.setattr(AureaxClient, "_wait_for_rpc_listening", lambda *_a, **_k: None)
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: None)

    with pytest.raises(SystemExit):
        client._run_shard_worker()

    assert os.environ["LLAMA_CACHE"] == "/opt/aureax/rpc-cache"
    assert "/opt/aureax/rpc-cache" in made


# --- head topology check counts devices, not peers ------------------------------

def test_topology_check_compares_against_the_stamped_device_count(bare_client, monkeypatch) -> None:
    """A worker exposing two GPUs is two RPC devices behind one address, so a
    four-entry split over three peers is right when the generator stamped 4."""
    client = bare_client()
    exits: list = []
    monkeypatch.setattr(AureaxClient, "_exit_reporting", lambda _s, *a: exits.append(a) or (_ for _ in ()).throw(SystemExit(1)))
    peers = ["a:1", "b:1", "c:1"]

    client.config = {"model": {"server": {"rpc_servers": peers, "tensor_split": [1, 1, 1, 1], "rpc_device_count": 4}}}
    client._verify_rpc_topology()
    assert exits == []

    client.config = {"model": {"server": {"rpc_servers": peers, "tensor_split": [1, 1, 1], "rpc_device_count": 4}}}
    with pytest.raises(SystemExit):
        client._verify_rpc_topology()
    assert exits == [(1,)]

    # Without a stamp the peer count is the device count, as before.
    client.config = {"model": {"server": {"rpc_servers": peers, "tensor_split": [1, 1, 1]}}}
    client._verify_rpc_topology()


# --- worker tensor-cache audit --------------------------------------------------

def _worker_cache(tmp_path: Path, files: dict[str, int]) -> Path:
    """A fake rpc-server cache at tmp_path/rpc-cache/rpc/ with the given sizes."""
    rpc_dir = tmp_path / "rpc-cache" / "rpc"
    rpc_dir.mkdir(parents=True)
    for name, size in files.items():
        (rpc_dir / name).write_bytes(b"x" * size)
    return tmp_path / "rpc-cache"


def _manifest(path: Path, entries: list[dict]) -> Path:
    import json

    path.write_text(json.dumps({"threshold": 10 * 1024 * 1024, "entries": entries}))
    return path


def test_cache_audit_keeps_manifest_weights_and_removes_the_rest(bare_client, tmp_path, monkeypatch) -> None:
    """With a hashed manifest a file is judged by name AND size.

    A truncated weight (right name, wrong size) is the silent-partial-tensor
    hazard: rpc-server trusts the file's size on a hit and never re-hashes.
    An unknown name is what the pinned build writes for every prefill mask
    over 10 MiB. A dotfile is the seeder's partial export.
    """
    import client.entrypoint as entrypoint_mod

    cache = _worker_cache(tmp_path, {
        "0123456789abcdef": 100,   # known, right size
        "fedcba9876543210": 50,    # known, truncated
        "00000000deadbeef": 70,    # unknown name
        ".0123456789abcdef.tmp": 10,
    })
    manifest = _manifest(tmp_path / "manifest.json", [
        {"name": "blk.0.a", "offset": 0, "nbytes": 100, "hash": "0123456789abcdef"},
        {"name": "blk.0.b", "offset": 100, "nbytes": 200, "hash": "fedcba9876543210"},
    ])
    monkeypatch.setattr(entrypoint_mod, "_RPC_CACHE_MANIFEST", str(manifest))

    client = bare_client()
    client._audit_rpc_cache(str(cache), 0)

    assert sorted(p.name for p in (cache / "rpc").iterdir()) == ["0123456789abcdef"]


def test_cache_audit_with_a_sizes_only_manifest_judges_by_size(bare_client, tmp_path, monkeypatch) -> None:
    """A manifest written without --hash still tells a weight from a mask by size."""
    import client.entrypoint as entrypoint_mod

    cache = _worker_cache(tmp_path, {"aaaaaaaaaaaaaaaa": 100, "bbbbbbbbbbbbbbbb": 77})
    manifest = _manifest(tmp_path / "manifest.json", [
        {"name": "blk.0.a", "offset": 0, "nbytes": 100, "hash": None},
    ])
    monkeypatch.setattr(entrypoint_mod, "_RPC_CACHE_MANIFEST", str(manifest))

    bare_client()._audit_rpc_cache(str(cache), 0)

    assert sorted(p.name for p in (cache / "rpc").iterdir()) == ["aaaaaaaaaaaaaaaa"]


def test_cache_audit_without_a_manifest_deletes_nothing_but_dotfiles(bare_client, tmp_path, monkeypatch) -> None:
    """No manifest means no way to tell weights from masks: report only."""
    import client.entrypoint as entrypoint_mod

    cache = _worker_cache(tmp_path, {"aaaaaaaaaaaaaaaa": 100, "bbbbbbbbbbbbbbbb": 77, ".partial.tmp": 5})
    monkeypatch.setattr(entrypoint_mod, "_RPC_CACHE_MANIFEST", str(tmp_path / "absent.json"))

    bare_client()._audit_rpc_cache(str(cache), 0)

    assert sorted(p.name for p in (cache / "rpc").iterdir()) == ["aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"]


def test_cache_audit_treats_an_unreadable_manifest_as_absent(bare_client, tmp_path, monkeypatch) -> None:
    """A corrupt manifest must not become 'delete everything'."""
    import client.entrypoint as entrypoint_mod

    cache = _worker_cache(tmp_path, {"aaaaaaaaaaaaaaaa": 100})
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json")
    monkeypatch.setattr(entrypoint_mod, "_RPC_CACHE_MANIFEST", str(bad))

    bare_client()._audit_rpc_cache(str(cache), 0)

    assert [p.name for p in (cache / "rpc").iterdir()] == ["aaaaaaaaaaaaaaaa"]


def test_cache_audit_refuses_to_start_below_the_free_disk_floor(bare_client, tmp_path, monkeypatch) -> None:
    """A full cache filesystem takes the kubelet with it, so the worker stops here."""
    import shutil

    import client.entrypoint as entrypoint_mod

    cache = _worker_cache(tmp_path, {})
    monkeypatch.setattr(entrypoint_mod, "_RPC_CACHE_MANIFEST", str(tmp_path / "absent.json"))
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda _p: types.SimpleNamespace(total=100 * 2**30, used=99 * 2**30, free=1 * 2**30),
    )
    exits: list[int] = []

    def fake_exit(self, status, rc=None):
        exits.append(status)
        raise SystemExit(status)

    monkeypatch.setattr(AureaxClient, "_exit_reporting", fake_exit)

    with pytest.raises(SystemExit):
        bare_client()._audit_rpc_cache(str(cache), 4.0)
    assert exits == [1]

    # The floor is a floor, not a trip at any value: 0 disables it.
    bare_client()._audit_rpc_cache(str(cache), 0)


def test_worker_audits_its_cache_after_shipping_starts_and_before_rpc_server(bare_client, tmp_path, monkeypatch) -> None:
    """Order is the contract: the audit's warnings must be shippable, and the
    cache must not be edited under a running rpc-server."""
    client = bare_client()
    client.config = {
        "model": {"rpc_server": {"cache": True, "cache_dir": str(tmp_path / "rpc-cache"), "min_free_disk_gib": 0}},
        "logging": {"llama_log_path": str(tmp_path / "rpc-server.log")},
    }
    client._shutdown = True
    client._child_proc = _FakeChild(rc=0)
    monkeypatch.setenv("LLAMA_CACHE", "")
    monkeypatch.delenv("LLAMA_CACHE")

    order: list[str] = []
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: order.append("ship"))
    monkeypatch.setattr(AureaxClient, "_audit_rpc_cache", lambda _s, d, f: order.append(f"audit:{Path(d).name}:{f}"))
    monkeypatch.setattr(AureaxClient, "_spawn_child", lambda *_a, **_k: order.append("spawn"))
    monkeypatch.setattr(AureaxClient, "_wait_for_rpc_listening", lambda *_a, **_k: None)
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)

    with pytest.raises(SystemExit):
        client._run_shard_worker()

    assert order == ["ship", "audit:rpc-cache:0.0", "spawn"]


# --- telemetry startup --------------------------------------------------------

def test_telemetry_storage_failure_does_not_take_the_node_out_of_service(bare_client, monkeypatch) -> None:
    """_start_telemetry is the first thing start() does, and TelemetryStorage
    mkdirs + CREATE TABLEs on the hostPath data dir. A corrupt telemetry.db or
    a full disk raised straight out of start(), killing PID 1 before
    llama-server was spawned — identically on every restart.
    """
    client = bare_client()
    client.config = {"telemetry": {"enabled": True, "local": {"db_path": "/nope/t.db"}}}
    client._shutdown = False
    client.telemetry_storage = None

    def explode(*_a, **_k):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr("client.entrypoint.TelemetryStorage", explode)

    client._start_telemetry()  # must not raise

    assert client.telemetry_storage is None
    assert getattr(client, "_telemetry_thread", None) is None


def test_telemetry_runs_when_the_config_omits_enabled(
    bare_client, tmp_path, monkeypatch, fleet_tls
) -> None:
    """The client's re-declared default must match the schema's.

    schema.py::TelemetryConfig defaults `enabled` to True, so a node template
    that omits the key validates as telemetry-on. A False default here would
    make that node ship nothing while every orchestrator-side view believes it
    is reporting — and /api/nodes is derived entirely from the telemetry
    database, so the node would simply never appear.
    """
    client = bare_client()
    client.config = {
        "tls": fleet_tls,
        "telemetry": {"local": {"db_path": str(tmp_path / "t.db")}},
    }

    monkeypatch.setattr(AureaxClient, "_telemetry_loop", lambda *_a, **_k: None)

    client._start_telemetry()

    assert client.telemetry_storage is not None, "an omitted `enabled` must not disable telemetry"
    assert client._telemetry_thread is not None


def test_telemetry_loop_does_not_busy_spin_on_a_zero_interval(bare_client, monkeypatch) -> None:
    """`sample_interval_seconds: 0` made the loop run collect_all plus a sqlite
    insert flat out, on a pod that carries no CPU limit. _log_loop already had
    the max(1, ...) floor.
    """
    client = bare_client(node_id="n1")

    monkeypatch.setattr("client.entrypoint.collect_all", lambda *_a, **_k: [])
    sleeper = _stop_after(client)
    monkeypatch.setattr(time, "sleep", sleeper)

    client._telemetry_loop(None, 0, 60, "", reporter=None, retention_days=0)

    assert sleeper.slept == [1], "a zero sample interval must still sleep a second"


def test_a_crash_during_model_load_surfaces_as_the_wrapped_status(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """The head OOMing mid-load is the failure the shard workers' memory limit
    is deliberately tuned to produce. The kernel kills the CHILD, so K8s only
    sees what this propagates — a hardcoded 1 read as "Error, exit code 1"
    instead of the 137 that says OOM.
    """
    client = bare_client()
    client.config = {"tls": fleet_tls, "process": {"shutdown_timeout_seconds": 1}}
    client._shutdown = False
    client._child_proc = _FakeChild(rc=-signal.SIGKILL)
    client._child_name = "llama-server"

    monkeypatch.setattr(AureaxClient, "_flush_crash_context", lambda _s, _rc: None)
    monkeypatch.setattr(AureaxClient, "_ship_logs_now", lambda _s: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit) as exc:
        client._wait_for_llama_health(timeout=5)

    assert exc.value.code == 137


# --- llama URL normalisation --------------------------------------------------

@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "https://127.0.0.1:8080"),
        ("::", "https://127.0.0.1:8080"),
        (None, "https://127.0.0.1:8080"),
        ("fd00::5", "https://[fd00::5]:8080"),
        ("10.77.0.7", "https://10.77.0.7:8080"),
    ],
)
def test_llama_url_is_always_dialable(bare_client, host, expected) -> None:
    """The bind address goes to argv verbatim, but this URL is DIALLED.

    "::" interpolated raw produced https://:::8080, and httpx.InvalidURL is not
    an httpx.HTTPError, so it escaped the health wait's guard and killed PID 1.
    An unbracketed IPv6 literal did the same.

    The scheme is https on every row: llama-server is started with
    --ssl-cert-file, so it answers nothing at all on plain HTTP.
    """
    client = bare_client()
    client.config = {"model": {"server": {"host": host, "port": 8080}}}
    assert client._llama_url() == expected


def test_an_empty_model_key_does_not_crash_the_port_read(bare_client) -> None:
    """`model:` with no children parses as None, which turned every
    .get("model", {}).get(...) read into an AttributeError at startup."""
    client = bare_client()
    client.config = {"model": None}
    assert client._llama_port() == 8080
    assert client._llama_url() == "https://127.0.0.1:8080"
    assert client._llama_api_key() == ""


# --- MCP-server branch teardown ----------------------------------------------

def test_the_mcp_branch_tears_the_child_down_when_the_server_returns(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """uvicorn's capture_signals() replaces our SIGTERM handler for the whole
    serve(), so _handle_signal cannot run and llama-server would be orphaned on
    every pod delete — hostPID leaves no PID namespace to reap it. The
    try/finally around the server is what tears the child down, and the bounded
    timeout_graceful_shutdown is what stops uvicorn waiting forever on the
    long-lived SSE streams every MCP client holds.

    The two waits are SERIAL and share one budget: uvicorn's drain, then
    stop()'s wait on the child. Their sum must stay within
    shutdown_timeout_seconds, which is the single figure
    terminationGracePeriodSeconds is sized from — spending it twice gets the
    child SIGKILLed by the kubelet and orphaned on :8080 under hostPID.
    """
    import uvicorn

    client = bare_client()
    client.node_id = "edge-01"
    client.config = {
        "tls": fleet_tls,
        "mcp": {"server": {"port": 9000}},
        "process": {"shutdown_timeout_seconds": 7},
    }
    client._shutdown = False
    client.telemetry_storage = None
    client._child_name = "llama-server"
    client._child_proc = None

    configs: list = []

    class _FakeServer:
        def __init__(self, config) -> None:
            configs.append(config)

        def run(self) -> None:
            return None  # as if SIGTERM had been handled by uvicorn

    monkeypatch.setattr("client.mcp.server.init", lambda *_a, **_k: None)
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(AureaxClient, "_watch_child_in_background", lambda _s: None)

    stopped: list = []
    monkeypatch.setattr(
        AureaxClient, "stop", lambda _s, timeout=None: stopped.append(timeout)
    )

    client._start_mcp_server()

    assert len(stopped) == 1, "the inference child was left running"
    assert client._shutdown is True, "the watcher would read this as a crash"

    drain = configs[0].timeout_graceful_shutdown
    assert drain > 0, "uvicorn would wait forever on the SSE streams"
    assert drain + stopped[0] <= 7, (
        f"the two serial waits ({drain}s draining + {stopped[0]}s on the child) "
        "must fit inside one shutdown_timeout_seconds"
    )


# --- mandatory TLS ------------------------------------------------------------

@pytest.mark.parametrize("missing", ["ca_file", "cert_file", "key_file"])
def test_start_exits_78_when_a_tls_file_is_missing(
    bare_client, monkeypatch, fleet_tls, tmp_path, missing
) -> None:
    """78 is EX_CONFIG, and it has to fire before anything is started.

    A node whose certificates never arrived cannot serve: llama-server exits on
    a missing --ssl-cert-file, uvicorn raises inside the handshake, and every
    outbound push fails verification. Starting anyway would produce a pod that
    crash-loops with three different errors depending on the role; exiting here
    produces one code `kubectl describe` names, and a message with the remedy.

    All three files are required on every role, including a shard worker: it
    ships telemetry and logs over TLS, and the deployer installs the three in
    one command, so a missing cert means that push was broken.
    """
    client = bare_client()
    tls = dict(fleet_tls)
    tls[missing] = str(tmp_path / "never-installed.pem")
    client.config = {"tls": tls, "node": {"role": "edge_executor"}}

    def unreachable(*_a, **_k):
        raise AssertionError("start() got past the TLS gate")

    monkeypatch.setattr(signal, "signal", unreachable)
    monkeypatch.setattr(AureaxClient, "_start_telemetry", unreachable)
    monkeypatch.setattr(AureaxClient, "_run_shard_worker", unreachable)

    with pytest.raises(SystemExit) as exc:
        client.start()

    assert exc.value.code == 78


def test_a_shard_worker_is_held_to_the_same_tls_requirement(
    bare_client, monkeypatch, tmp_path
) -> None:
    """rpc-server takes no certificate, but the worker still ships over TLS.

    The uniform rule has no branch to get wrong, and a worker whose files are
    missing means the deployer's four-file push failed — worth failing on.
    """
    client = bare_client()
    client.config = {
        "tls": {
            "ca_file": str(tmp_path / "ca.pem"),
            "cert_file": str(tmp_path / "node.pem"),
            "key_file": str(tmp_path / "node.key"),
        },
        "node": {"role": "shard_worker"},
    }
    monkeypatch.setattr(
        AureaxClient, "_run_shard_worker",
        lambda _s: pytest.fail("the worker branch ran without TLS material"),
    )
    monkeypatch.setattr(signal, "signal", lambda *_a: None)

    with pytest.raises(SystemExit) as exc:
        client.start()

    assert exc.value.code == 78


def test_a_config_with_no_tls_block_at_all_exits_78(bare_client, monkeypatch) -> None:
    """An older resolved config on the node predates the injected block.

    That is the upgrade case: the pod restarts with the new image before
    `aureax apply` has pushed the new config, and it must say so rather than
    fall back to plaintext.
    """
    client = bare_client()
    client.config = {"node": {"role": "edge_executor"}}
    monkeypatch.setattr(signal, "signal", lambda *_a: None)
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)

    with pytest.raises(SystemExit) as exc:
        client.start()

    assert exc.value.code == 78


def test_all_three_files_present_lets_start_past_the_check(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """The companion to the exits above: a correctly installed node proceeds.

    Without this, a _require_tls_files that rejected everything would pass
    every test in this section.
    """
    client = bare_client()
    client.config = {
        "tls": fleet_tls,
        "node": {"role": "edge_executor"},
        "mcp": {"server": {"enabled": False}},
    }
    client._shutdown = False

    reached: list[str] = []
    monkeypatch.setattr(signal, "signal", lambda *_a: None)
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: reached.append("telemetry"))
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_reap_stale_children", lambda _s, _b: [])
    monkeypatch.setattr(AureaxClient, "_llama_port_in_use", lambda _s: True)
    monkeypatch.setattr(AureaxClient, "_llama_port", lambda _s: 8080)
    monkeypatch.setattr(AureaxClient, "_wait_for_llama_health", lambda _s, *_a, **_k: None)
    monkeypatch.setattr(AureaxClient, "_block_on_child", lambda _s: None)

    client.start()

    assert reached == ["telemetry"]


def test_the_health_wait_verifies_against_the_fleet_ca(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """The in-pod probe dials https://127.0.0.1:8080 with the node's own leaf.

    Passing no context would fall back to the public trust store, which cannot
    verify a fleet-CA certificate — every probe would fail and every head would
    look like a failed model load. verify=False would "fix" that by trusting
    anything, which is exactly what this fleet has no mode for.
    """
    client = bare_client()
    client.config = {"tls": fleet_tls, "model": {"server": {"port": 8080}}}
    client._shutdown = False

    seen: dict = {}

    class _FakeHttpx:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def get(self, url, headers=None):
            seen["url"] = url
            return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(httpx, "Client", _FakeHttpx)

    client._wait_for_llama_health(timeout=5)

    from client.tls import ssl_context

    assert seen["url"] == "https://127.0.0.1:8080/health"
    assert seen["verify"] is ssl_context(client.config), "the CA context was not reused"


# --- MCP bearer key ------------------------------------------------------------

MCP_KEY = "aureax_sk_" + "m" * 32


def _mcp_branch(client, monkeypatch, captured: dict):
    """Run _start_mcp_server with uvicorn and FastMCP's app stubbed out."""
    import uvicorn

    class _FakeServer:
        def __init__(self, config) -> None:
            captured["config"] = config

        def run(self) -> None:
            return None

    monkeypatch.setattr(
        "client.mcp.server.init",
        lambda *a, **kw: captured.update(init_args=a, init_kwargs=kw),
    )
    monkeypatch.setattr("client.mcp.server.build_app", lambda: captured.setdefault("app", object()))
    monkeypatch.setattr(uvicorn, "Server", _FakeServer)
    monkeypatch.setattr(AureaxClient, "_watch_child_in_background", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "stop", lambda _s, timeout=None: None)
    client._start_mcp_server()


def test_the_mcp_key_reaches_init_and_never_argv_or_the_environment(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """The MCP server runs in THIS process, so the key stays in the config.

    There is no MCP_API_KEY variable on purpose: an env var would be a second
    source of truth for the same secret and would be readable in `kubectl exec
    <pod> -- env` and in /proc/<pid>/environ by every process on the node,
    which runs hostPID.
    """
    client = bare_client()
    client.node_id = "edge-01"
    client.config = {
        "tls": fleet_tls,
        "mcp": {"server": {"port": 9000, "api_key": MCP_KEY}},
        "process": {"shutdown_timeout_seconds": 4},
    }
    client._shutdown = False

    captured: dict = {}
    _mcp_branch(client, monkeypatch, captured)

    assert captured["init_kwargs"]["mcp_api_key"] == MCP_KEY
    assert "MCP_API_KEY" not in os.environ
    assert MCP_KEY not in os.environ.values(), "the key was exported to the child environment"

    # The one argv this node builds is llama-server's, which _spawn_child logs
    # verbatim into container stdout and the shipped crash logs, and which any
    # process on the node can read out of /proc under hostPID.
    from client.config.parser import get_llama_server_args

    assert not any(MCP_KEY in a for a in get_llama_server_args(client.config))


def test_no_key_configured_hands_init_none_rather_than_an_empty_string(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """build_app gates on truthiness, and BearerAuthMiddleware refuses "".

    An empty key would otherwise make compare_digest(b"", b"") true for a
    request with no Authorization header at all — auth silently off.
    """
    client = bare_client()
    client.node_id = "edge-01"
    client.config = {"tls": fleet_tls, "mcp": {"server": {"port": 9000}}}
    client._shutdown = False

    captured: dict = {}
    _mcp_branch(client, monkeypatch, captured)

    assert captured["init_kwargs"]["mcp_api_key"] is None


def test_the_mcp_listener_gets_this_nodes_certificate(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """uvicorn terminates TLS below the ASGI app, so :30900 is https.

    Without the two arguments the MCP port would serve plaintext while
    llama-server on the same node served https — and `mcp_health_wait`, which
    now dials https, would report the node down with the pod perfectly healthy.
    """
    client = bare_client()
    client.node_id = "edge-01"
    client.config = {"tls": fleet_tls, "mcp": {"server": {"port": 9000}}}
    client._shutdown = False

    captured: dict = {}
    _mcp_branch(client, monkeypatch, captured)

    config = captured["config"]
    assert config.ssl_certfile == fleet_tls["cert_file"]
    assert config.ssl_keyfile == fleet_tls["key_file"]
    assert config.is_ssl is True


def test_the_mcp_branch_serves_the_gated_app_not_the_bare_one(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """build_app() is the only enforcement point for both doors into this pod.

    Serving mcp.streamable_http_app() directly is how the port silently loses
    its bearer gate, with the pod, the probes and /health all still green.
    """
    client = bare_client()
    client.node_id = "edge-01"
    client.config = {
        "tls": fleet_tls,
        "mcp": {"server": {"port": 9000, "api_key": MCP_KEY}},
    }
    client._shutdown = False

    captured: dict = {}
    _mcp_branch(client, monkeypatch, captured)

    assert captured["config"].app is captured["app"]


def test_mcp_api_key_helper_reads_the_config_not_the_environment(
    bare_client, monkeypatch
) -> None:
    """Same rule as the llama key: the YAML pushed to /etc/aureax is authoritative."""
    client = bare_client()
    monkeypatch.setenv("MCP_API_KEY", "aureax_sk_environment_value_only")

    client.config = {}
    assert client._mcp_api_key() == ""

    # `mcp:` or `server:` present with no children parses as None, which used
    # to turn this read into an AttributeError at startup.
    client.config = {"mcp": None}
    assert client._mcp_api_key() == ""
    client.config = {"mcp": {"server": None}}
    assert client._mcp_api_key() == ""

    client.config = {"mcp": {"server": {"api_key": MCP_KEY}}}
    assert client._mcp_api_key() == MCP_KEY


# --- planner: one key per remote ----------------------------------------------

def test_the_planner_sends_each_remote_its_own_key(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """Every peer has a different MCP key, so the pool cannot hold just one.

    The planner's remote_servers list is generated with each peer's key beside
    its URL (orchestrator/config/templating.py::mcp_endpoints); sending the
    wrong one 401s against a node that is up.
    """
    import asyncio as aio

    pools: list = []
    client = _planner_client(bare_client, monkeypatch, pools, fleet_tls)
    client._shutdown = True  # one connect pass, then straight to the finally

    aio.run(client._run_planner_host([
        {"name": "edge-01", "url": "https://a:30900/mcp", "api_key": "aureax_sk_" + "a" * 32},
        {"name": "edge-02", "url": "https://b:30900/mcp", "api_key": "aureax_sk_" + "b" * 32},
    ]))

    assert sorted(pools[0].connects) == [
        ("edge-01", "https://a:30900/mcp", "aureax_sk_" + "a" * 32),
        ("edge-02", "https://b:30900/mcp", "aureax_sk_" + "b" * 32),
    ]
    assert pools[0].verify is not None, "the pool must dial peers under the fleet CA"


@pytest.mark.parametrize(
    "value",
    ["${mcp_api_key}", "Bearer ${mcp_api_key}", "", "   ", None, 12345],
)
def test_an_unusable_peer_key_is_dropped_rather_than_sent(
    bare_client, monkeypatch, fleet_tls, value
) -> None:
    """A planner running an ungenerated config falls back to the template.

    Its remote_servers then carry the literal "${mcp_api_key}". Sending that
    turns every connect into a 401 that reads like a wrong key rather than an
    unresolved config, and writes a template placeholder into the peer's
    auth-rejection log.
    """
    import asyncio as aio

    pools: list = []
    client = _planner_client(bare_client, monkeypatch, pools, fleet_tls)
    client._shutdown = True

    aio.run(client._run_planner_host(
        [{"name": "edge-01", "url": "https://a:30900/mcp", "api_key": value}]
    ))

    assert pools[0].connects == [("edge-01", "https://a:30900/mcp", None)]


def test_a_peer_list_that_cannot_be_indexed_is_filtered_not_crashed(
    bare_client, monkeypatch, fleet_tls, caplog
) -> None:
    """_run_planner_host keys its pending map on s["name"].

    An entry with no name is a KeyError out of asyncio.run — a crash-looping
    planner pod — and two entries sharing a name collapse to one, so a peer is
    never connected while the ready count still expects it. Only a hand-edited
    /etc/aureax/aureax-edge.yaml gets here, which is exactly the file no
    generator validated.
    """
    import logging

    pools: list = []
    client = _planner_client(bare_client, monkeypatch, pools, fleet_tls)
    client.config = {
        "tls": fleet_tls,
        "mcp": {"client": {"enabled": True, "remote_servers": [
            {"name": "edge-01", "url": "https://a:30900/mcp"},
            {"url": "https://b:30900/mcp"},                 # no name
            {"name": "edge-03"},                            # no url
            {"name": "edge-01", "url": "https://c:30900/mcp"},  # duplicate name
        ]}},
    }
    client._shutdown = True  # one connect pass, then straight to the finally

    with caplog.at_level(logging.WARNING, logger="aureax.client"):
        client._start_planner()

    assert pools[0].connects == [("edge-01", "https://a:30900/mcp", None)]
    skipped = [r.message for r in caplog.records if "MCP host: skipping" in r.message]
    assert len(skipped) == 3, f"a bad entry was dropped silently: {caplog.messages}"
    assert any("duplicate" in m for m in skipped)


# --- sharded topology ----------------------------------------------------------

def _head_client(bare_client, fleet_tls, monkeypatch, **server) -> AureaxClient:
    """A head whose fatal exits are stripped down to their status code."""
    client = bare_client()
    client.node_id = "shard-head"
    client.config = {"tls": fleet_tls, "model": {"server": server}}
    client._shutdown = False
    monkeypatch.setattr(AureaxClient, "_flush_crash_context", lambda _s, _rc: None)
    monkeypatch.setattr(AureaxClient, "_ship_logs_now", lambda _s: None)
    return client


def test_a_tensor_split_that_outnumbers_the_rpc_peers_exits(
    bare_client, monkeypatch, fleet_tls, caplog
) -> None:
    """A head with a mismatched split loads, answers /health and serves.

    llama.cpp names RPC devices positionally from --rpc order, so a split of
    the wrong length either strands a peer's share or addresses a device that
    does not exist — and the pod is Ready, the MCP endpoint answers and
    `aureax status` reports the group up throughout. validate_groups catches
    this at generate time; this catches the file the node ACTUALLY has, which
    is what a hand edit on the node produces.
    """
    client = _head_client(
        bare_client, fleet_tls, monkeypatch,
        rpc_servers=["10.77.0.2:50052", "10.77.0.3:50052"],
        tensor_split=[1, 1, 1],
    )

    with caplog.at_level("ERROR", logger="aureax.client"), pytest.raises(SystemExit) as exc:
        client._verify_rpc_topology()

    assert exc.value.code == 1
    message = caplog.records[-1].getMessage()
    assert "tensor_split has 3 entries" in message
    assert "2 peer(s) expose 2 RPC device(s)" in message
    assert "aureax apply --node shard-head" in message


def test_a_matching_tensor_split_passes(bare_client, monkeypatch, fleet_tls) -> None:
    """The shipped 35B heads run one split entry per peer."""
    client = _head_client(
        bare_client, fleet_tls, monkeypatch,
        rpc_servers=["10.77.0.2:50052", "10.77.0.3:50052", "10.77.0.4:50052"],
        tensor_split=[1, 1, 1],
    )

    client._verify_rpc_topology()  # must not raise


def test_an_empty_tensor_split_is_llama_cpps_own_split(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """No split means the free-memory split, which is a supported setup.

    parser.get_llama_server_args omits --tensor-split entirely for it, so there
    is nothing to disagree with the peer list.
    """
    client = _head_client(
        bare_client, fleet_tls, monkeypatch,
        rpc_servers=["10.77.0.2:50052", "10.77.0.3:50052"],
    )

    client._verify_rpc_topology()


def test_the_topology_check_is_skipped_entirely_without_rpc_servers(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """A standalone node emits neither --rpc nor --tensor-split.

    A leftover tensor_split in a standalone template is inert on the command
    line, so failing on it here would take a serving node out over a field that
    changes nothing.
    """
    client = _head_client(bare_client, fleet_tls, monkeypatch, tensor_split=[1, 1, 1])

    client._verify_rpc_topology()


def test_start_runs_the_topology_check_after_the_health_wait(
    bare_client, monkeypatch, fleet_tls
) -> None:
    """Order matters: before /health the head has not attached its peers yet.

    Checking earlier would report a mismatch on a config that llama-server was
    still loading, and checking on the MCP branch only would skip the workers'
    own supervisor path.
    """
    client = bare_client()
    client.node_id = "shard-head"
    client.config = {
        "tls": fleet_tls,
        "node": {"role": "edge_executor"},
        "mcp": {"server": {"enabled": False}},
    }
    client._shutdown = False

    order: list[str] = []
    monkeypatch.setattr(signal, "signal", lambda *_a: None)
    monkeypatch.setattr(AureaxClient, "_start_telemetry", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_start_log_shipping", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_reap_stale_children", lambda _s, _b: [])
    monkeypatch.setattr(AureaxClient, "_llama_port_in_use", lambda _s: False)
    monkeypatch.setattr(AureaxClient, "_wait_for_rpc_workers", lambda _s: None)
    monkeypatch.setattr(AureaxClient, "_start_llama_server", lambda _s: None)
    monkeypatch.setattr(
        AureaxClient, "_wait_for_llama_health", lambda _s, *_a, **_k: order.append("health")
    )
    monkeypatch.setattr(
        AureaxClient, "_verify_rpc_topology", lambda _s: order.append("topology")
    )
    monkeypatch.setattr(AureaxClient, "_block_on_child", lambda _s: None)

    client.start()

    assert order == ["health", "topology"]
