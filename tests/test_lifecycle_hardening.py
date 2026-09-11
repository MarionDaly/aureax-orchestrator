"""Control-plane error propagation.

Three rules, one per failure shape the control plane can produce: every wait
has a deadline, every failure is reported rather than swallowed and reported as
success, and an SSH error is classified into a message a caller can act on
instead of escaping as a raw traceback. A test failing here means an operation
is claiming an outcome it did not have.
"""

from __future__ import annotations

import io
import subprocess
import threading

import pytest

from orchestrator import lifecycle
from orchestrator.cluster import k3d, kubectl
from orchestrator.cluster._subprocess import CommandError
from orchestrator.config.schema import NetworkConfig
from orchestrator.deploy import k3s_agent
from orchestrator.deploy import ssh as ssh_mod
from tests.conftest import (
    NETWORK_BASE,
    FakeKubectl,
    FakeLog,
    make_network_config,
    make_targets,
)

# --- node-join wait deadline -------------------------------------------------

def test_node_wait_times_out_instead_of_blocking_forever(monkeypatch) -> None:
    """kubectl.get_nodes() returns [] on ANY API error, so an unreachable API
    server made this wait spin for good — and the CLI passes mgr=None, so it
    had no cancel path either (same defect class the pod wait already fixed).
    """
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    _hostnames(monkeypatch)
    cfg = make_network_config(make_targets("a", "b"))
    log = FakeLog()

    with pytest.raises(RuntimeError, match="timed out"):
        lifecycle._start_phase_wait_k8s_nodes(
            cfg, log, None, FakeKubectl(nodes=[]), timeout_seconds=10
        )
    assert any(lvl == "error" and "timed out" in t for lvl, t in log.lines)


def _hostnames(monkeypatch) -> None:
    """Resolve node_id ``x`` to K8s node name ``host-x`` without SSH."""
    monkeypatch.setattr(
        lifecycle, "_resolve_node_hostname", lambda config, node_id: f"host-{node_id}"
    )


def test_node_wait_still_succeeds_before_the_deadline(monkeypatch) -> None:
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    _hostnames(monkeypatch)
    cfg = make_network_config(make_targets("a", "b"))
    log = FakeLog()
    ready = [
        {"name": "k3d-server-0", "roles": "control-plane", "status": "True"},
        {"name": "host-a", "roles": "worker", "status": "True"},
        {"name": "host-b", "roles": "worker", "status": "True"},
    ]
    lifecycle._start_phase_wait_k8s_nodes(
        cfg, log, None, FakeKubectl(nodes=ready), timeout_seconds=10
    )
    assert any(lvl == "success" for lvl, _t in log.lines)


def test_a_stranger_in_the_cluster_cannot_stand_in_for_a_configured_node(monkeypatch) -> None:
    """The wait is an allow-list, not a count.

    Counting non-control-plane Ready nodes let any machine joined to the
    cluster — a dev box, an `enabled: false` target — satisfy the count for a
    configured node that never joined, so the miss only surfaced when the pod
    wait burned its full hour on a host absent from the cluster.
    """
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    _hostnames(monkeypatch)
    cfg = make_network_config(make_targets("a", "b"))
    log = FakeLog()
    nodes = [
        {"name": "k3d-server-0", "roles": "control-plane", "status": "True"},
        {"name": "host-a", "roles": "worker", "status": "True"},
        {"name": "someone-elses-box", "roles": "worker", "status": "True"},
    ]

    with pytest.raises(RuntimeError, match="never joined the cluster: host-b"):
        lifecycle._start_phase_wait_k8s_nodes(
            cfg, log, None, FakeKubectl(nodes=nodes), timeout_seconds=10
        )


# --- stop: drain before the control plane goes down --------------------------

def _stop_fixture(
    monkeypatch,
    *,
    pods_then_empty: bool,
    stop_ok: bool = True,
    patch_outcomes: dict[str, str] | None = None,
    api_ok: bool = True,
) -> tuple[FakeLog, list]:
    """Wire do_stop's collaborators; returns (log, ordered-call recorder).

    ``patch_outcomes`` overrides the selector-patch result per DaemonSet, and
    ``api_ok`` is what the drain's reachability probe answers.
    """
    order: list[str] = []
    outcomes = patch_outcomes or {}
    monkeypatch.setattr(k3d, "cluster_running", lambda _n: True)
    monkeypatch.setattr(
        k3d, "stop_cluster", lambda _n: order.append("stop_cluster") or stop_ok
    )
    monkeypatch.setattr(
        lifecycle, "_patch_daemonset_selector",
        lambda _kc, _ns, ds, _l, _v, _log=None: (
            order.append(f"patch:{ds}") or outcomes.get(ds, "patched")
        ),
    )
    monkeypatch.setattr(lifecycle, "_sleep_or_cancel", lambda mgr, secs: None)
    # The drain only believes an empty pod list when the API answers, and
    # do_stop uses the real kubectl module — unstubbed this shells out.
    monkeypatch.setattr(kubectl, "api_reachable", lambda *_a, **_k: api_ok)

    polls = {"n": 0}

    def fake_get_pods(_ns: str) -> list:
        order.append("poll_pods")
        polls["n"] += 1
        if pods_then_empty and polls["n"] > 1:
            return []
        return [{"name": "aureax-edge-x", "node": "n1", "ready": False}]

    monkeypatch.setattr(kubectl, "get_pods", fake_get_pods)
    return FakeLog(), order


def test_do_stop_waits_for_pod_termination_before_stopping_the_cluster(monkeypatch) -> None:
    """Pod deletion is delivered by the API server, so the cluster has to
    outlive it. Stopping k3d straight after the patch leaves any node that has
    not yet seen the deletion running llama-server with no control plane left
    to ever deliver it.
    """
    log, order = _stop_fixture(monkeypatch, pods_then_empty=True)
    cfg = make_network_config(make_targets("a"))

    lifecycle.do_stop(cfg, log, skip_self_kill=True)

    assert "poll_pods" in order
    assert order.index("patch:aureax-edge") < order.index("poll_pods") < order.index("stop_cluster")
    assert any("All pods terminated" in t for _l, t in log.lines)


def test_do_stop_proceeds_with_a_warning_when_the_drain_expires(monkeypatch) -> None:
    """A stop must never refuse to stop — an expired drain is a warning."""
    log, order = _stop_fixture(monkeypatch, pods_then_empty=False)
    cfg = make_network_config(make_targets("a"))

    lifecycle.do_stop(cfg, log, skip_self_kill=True)

    assert "stop_cluster" in order, "the cluster stop was skipped"
    assert any(lvl == "warn" and "still terminating" in t for lvl, t in log.lines)


def test_do_stop_reports_a_failed_cluster_stop(monkeypatch) -> None:
    """A failed `k3d cluster stop` must not be logged as 'k3d cluster stopped'."""
    log, _order = _stop_fixture(monkeypatch, pods_then_empty=True, stop_ok=False)
    cfg = make_network_config(make_targets("a"))

    lifecycle.do_stop(cfg, log, skip_self_kill=True)

    assert any(lvl == "warn" and "FAILED" in t for lvl, t in log.lines)
    assert not any("k3d cluster stopped" in t for _l, t in log.lines)


def test_do_stop_asks_the_shard_daemonset_to_terminate_without_node_groups(monkeypatch) -> None:
    """The shard patch is not gated on `node_groups`.

    Start decides a node is a shard worker from that node's OWN `node.role`, so
    a worker config with no matching group still gets rpc-server scheduled.
    Gating the stop on node_groups instead takes the control plane down with
    those pods never asked to terminate — nothing left to deliver the deletion.
    """
    log, order = _stop_fixture(monkeypatch, pods_then_empty=True)
    cfg = make_network_config(make_targets("a"))
    assert not cfg.node_groups, "the fixture must have no groups for this to mean anything"

    lifecycle.do_stop(cfg, log, skip_self_kill=True)

    assert "patch:aureax-shard" in order, "the shard DaemonSet was never asked to terminate"
    assert order.index("patch:aureax-shard") < order.index("stop_cluster")


# --- start: SSH failures are classified, not tracebacked ---------------------

def test_verify_nodes_classifies_an_unreachable_host(monkeypatch) -> None:
    """A raised SSH error must become a clean RuntimeError naming the node —
    NOT fold into `unjoined`, which would trigger a full re-bootstrap on a
    transient blip, and NOT escape as a raw paramiko traceback.
    """
    monkeypatch.setattr(k3d, "get_server_url", lambda _p, _h: "https://10.77.0.99:6443")
    monkeypatch.setattr(k3d, "probe_host", lambda _c: "10.77.0.1")

    def boom(_host, _facts, **_kw):
        raise TimeoutError("no route to host")

    monkeypatch.setattr(ssh_mod, "probe_facts", boom)
    ran_setup = []
    monkeypatch.setattr(lifecycle, "_run_setup_all", lambda *_a, **_k: ran_setup.append(1))

    cfg = make_network_config(make_targets("a", "b"))
    log = FakeLog()
    with pytest.raises(RuntimeError, match="unreachable"):
        lifecycle._start_phase_verify_nodes(cfg, log, None)
    assert not ran_setup, "an unreachable node must not trigger a re-bootstrap"


# --- deploy-side raw-SSH guards ----------------------------------------------

def test_join_agent_reports_an_unreachable_host_instead_of_raising(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise TimeoutError("no route to host")

    monkeypatch.setattr(k3s_agent, "run_command", boom)
    ok, msg = k3s_agent.join_agent("10.77.0.9", "https://s:6443", "tok")
    assert ok is False
    assert "unreachable" in msg


def test_reconcile_refuses_to_rewrite_an_unreadable_env_file(monkeypatch) -> None:
    """The `2>/dev/null || true` read reports a missing env file as exit 0 with
    empty output. Rebuilding the file from that writes a lone K3S_URL line and
    destroys K3S_TOKEN, leaving the agent restarted against a token it does not
    have — so an empty read must refuse instead of rewriting.
    """
    monkeypatch.setattr(k3s_agent, "run_command", lambda *_a, **_k: (0, "", ""))
    ok, msg = k3s_agent.reconcile_server_url("10.77.0.9", "https://s:6443")
    assert ok is False
    assert "K3S_TOKEN" in msg


# --- stop: the shard half, and the outcomes callers must distinguish ---------

def _group_cfg():
    return make_network_config(
        make_targets("h", "w1"), [{"name": "g", "head": "h", "workers": ["w1"]}]
    )


def test_do_stop_suspends_the_shard_daemonset_after_the_edge_one(monkeypatch) -> None:
    """Without the shard patch every rpc-server worker survives `aureax stop`.

    Edge (head) first: the reverse makes the head abort on a vanished RPC peer
    and crash-loop while we are shutting it down.
    """
    log, order = _stop_fixture(monkeypatch, pods_then_empty=True)

    lifecycle.do_stop(_group_cfg(), log, skip_self_kill=True)

    assert (
        order.index("patch:aureax-edge")
        < order.index("patch:aureax-shard")
        < order.index("poll_pods")
        < order.index("stop_cluster")
    )


def test_do_stop_names_the_daemonset_whose_pods_were_not_asked_to_terminate(monkeypatch) -> None:
    """A failed patch is warned about, never raised — but it must be named."""
    log, order = _stop_fixture(
        monkeypatch, pods_then_empty=True, patch_outcomes={"aureax-shard": "failed"}
    )

    lifecycle.do_stop(_group_cfg(), log, skip_self_kill=True)

    assert any(
        lvl == "warn" and "aureax-shard" in t and "NOT asked to terminate" in t
        for lvl, t in log.lines
    )
    assert "stop_cluster" in order, "a stop must never refuse to stop"


def test_drain_does_not_read_an_unreachable_api_as_all_pods_terminated(monkeypatch) -> None:
    """kubectl.get_pods returns [] on ANY API error.

    Claiming the drain finished and stopping k3d leaves any node that never saw
    the deletion running llama-server with no control plane to deliver it.
    """
    log, order = _stop_fixture(monkeypatch, pods_then_empty=True, api_ok=False)

    lifecycle.do_stop(make_network_config(make_targets("a")), log, skip_self_kill=True)

    assert not any("All pods terminated" in t for _l, t in log.lines)
    assert any(lvl == "warn" and "not answering" in t for lvl, t in log.lines)
    assert any(lvl == "warn" and "Could not confirm" in t for lvl, t in log.lines)
    assert "stop_cluster" in order, "a stop must never refuse to stop"


def test_stop_kills_the_port_the_dashboard_config_names(monkeypatch) -> None:
    """`aureax stop` kills whatever listens on server.dashboard.port.

    A hardcoded default here meant changing the config made stop shoot at a
    port the dashboard was never on.
    """
    log, _order = _stop_fixture(monkeypatch, pods_then_empty=True)
    cfg = NetworkConfig.model_validate({
        **NETWORK_BASE,
        "ssh": {"targets": make_targets("a")},
        "server": {"dashboard": {"port": 7999}},
    })
    killed: list[int] = []
    monkeypatch.setattr(
        lifecycle, "_kill_process_on_port",
        lambda port, _log, label="Process": killed.append(port),
    )

    lifecycle.do_stop(cfg, log, skip_self_kill=False)

    assert killed == [7999]


class _PatchKubectl:
    """kubectl stand-in whose `patch` raises whatever the test hands it."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error

    def patch(self, *_a, **_k) -> str:
        if self._error is not None:
            raise self._error
        return ""


def test_a_missing_daemonset_is_absent_not_failed() -> None:
    """"absent" is fine for stop and fatal for start; "failed" is neither."""
    err = subprocess.CalledProcessError(
        1, [], stderr='Error from server (NotFound): daemonsets "aureax-shard" not found'
    )
    log = FakeLog()
    outcome = lifecycle._patch_daemonset_selector(
        _PatchKubectl(err), "aureax", "aureax-shard", "k", "v", log
    )
    assert outcome == "absent"


def test_any_other_patch_error_is_failed_and_logged() -> None:
    err = subprocess.CalledProcessError(1, [], stderr="connection refused")
    log = FakeLog()
    outcome = lifecycle._patch_daemonset_selector(
        _PatchKubectl(err), "aureax", "aureax-edge", "k", "v", log
    )
    assert outcome == "failed"
    assert any(lvl == "error" and "aureax-edge" in t for lvl, t in log.lines)


def test_a_clean_patch_is_patched() -> None:
    outcome = lifecycle._patch_daemonset_selector(
        _PatchKubectl(), "aureax", "aureax-edge", "k", "v", FakeLog()
    )
    assert outcome == "patched"


# --- kubectl reads degrade, they do not raise --------------------------------

def test_the_daemonset_selector_read_goes_through_the_kubectl_wrapper(monkeypatch) -> None:
    """One place owns kubectl's argv, its --request-timeout and its "{} on any
    failure" contract. A private subprocess call here is bounded only by the
    local process timeout, so a hung API request holds the start for as long as
    kubectl is willing to wait — which by default is forever."""
    asked: list[tuple[str, str]] = []
    ds = {"spec": {"template": {"spec": {"nodeSelector": {"aureax.io/role": "edge"}}}}}
    monkeypatch.setattr(
        kubectl, "get",
        lambda resource, namespace=None: asked.append((resource, namespace)) or ds,
    )

    selector = lifecycle._get_daemonset_node_selector("aureax", "aureax.io/role")

    assert selector == "edge"
    assert asked == [("daemonset/aureax-edge", "aureax")]


def test_a_selector_that_cannot_be_read_is_empty_not_a_traceback(monkeypatch) -> None:
    """This read only decides whether a patch is a no-op, so "cannot tell" has
    to read as "unset" — including when kubectl is not installed at all, which
    the shared wrapper reports as a CommandError rather than FileNotFoundError.
    """
    def boom(_resource, namespace=None):
        raise CommandError(127, ["kubectl"], "", "kubectl not found on PATH — install it")

    monkeypatch.setattr(kubectl, "get", boom)

    assert lifecycle._get_daemonset_node_selector("aureax", "aureax.io/role") == ""


def test_a_crash_dump_survives_a_missing_kubectl(monkeypatch) -> None:
    """A diagnostic must never end the operation it is diagnosing: this runs
    inside the pod-readiness wait, on a node that has already crash-looped."""
    def boom(*_a, **_k):
        raise FileNotFoundError("kubectl")

    monkeypatch.setattr(subprocess, "run", boom)
    log = FakeLog()

    ok = lifecycle._log_crashed_container_stdout(
        pod_name="aureax-edge-x", pod_uid="u", container="client",
        namespace="aureax", node_name="host-a", mgr=None, log=log,
    )

    assert ok is False
    assert any(lvl == "warn" for lvl, _t in log.lines), log.lines


# --- start: the cluster phase ------------------------------------------------

def test_start_phase_cluster_raises_when_k3d_start_fails(monkeypatch) -> None:
    """The return was ignored, so a port conflict was logged as "started" and
    the run burned the whole node-join wait blaming the edge nodes."""
    monkeypatch.setattr(k3d, "cluster_running", lambda _n: False)
    monkeypatch.setattr(k3d, "cluster_exists", lambda _n: True)
    monkeypatch.setattr(k3d, "start_cluster", lambda _n: False)
    monkeypatch.setattr(kubectl, "wait_api_ready", lambda *_a, **_k: True)
    log = FakeLog()

    with pytest.raises(RuntimeError, match="failed to start"):
        lifecycle._start_phase_cluster(make_network_config(make_targets("a")), log, None)
    assert not any("k3d cluster started" in t for _l, t in log.lines)


def test_start_phase_cluster_waits_for_the_api_before_continuing(monkeypatch) -> None:
    """`k3d cluster start` can exit 0 before the API server serves."""
    monkeypatch.setattr(k3d, "cluster_running", lambda _n: False)
    monkeypatch.setattr(k3d, "cluster_exists", lambda _n: True)
    monkeypatch.setattr(k3d, "start_cluster", lambda _n: True)
    monkeypatch.setattr(kubectl, "wait_api_ready", lambda *_a, **_k: False)

    with pytest.raises(RuntimeError, match="API server did not become ready"):
        lifecycle._start_phase_cluster(make_network_config(make_targets("a")), FakeLog(), None)


# --- start: sudo failures are not "not joined" -------------------------------

def _verify_facts(monkeypatch, facts: dict) -> list:
    """Stub the node probe with fixed facts; returns the setup-all.sh recorder."""
    monkeypatch.setattr(k3d, "get_server_url", lambda _p, _h: "https://10.77.0.99:6443")
    monkeypatch.setattr(k3d, "probe_host", lambda _c: "10.77.0.1")
    monkeypatch.setattr(ssh_mod, "probe_facts", lambda _h, _f, **_kw: dict(facts))
    ran: list[int] = []
    monkeypatch.setattr(lifecycle, "_run_setup_all", lambda *_a, **_k: ran.append(1))
    return ran


def test_a_sudo_failure_does_not_trigger_a_fleet_re_bootstrap(monkeypatch) -> None:
    """probe_facts wraps every snippet in `2>/dev/null || true`, so a sudo
    failure and a missing env file both come back empty. Reading that as "not
    joined" re-ran setup-all.sh — image builds and model downloads across the
    WHOLE fleet — because one node's sudo stopped working.
    """
    ran = _verify_facts(monkeypatch, {
        "k3s_env": "", "agent_env_present": "yes", "image": "", "hostname": "node-a",
    })

    with pytest.raises(RuntimeError, match="refused sudo"):
        lifecycle._start_phase_verify_nodes(make_network_config(make_targets("a")), FakeLog(), None)
    assert not ran, "a sudo failure must not bootstrap"


def test_a_genuinely_unjoined_node_still_bootstraps(monkeypatch) -> None:
    """No env file present is the first-time-bootstrap case, and must stay one."""
    ran = _verify_facts(monkeypatch, {
        "k3s_env": "", "agent_env_present": "", "image": "", "hostname": "node-a",
    })

    cont = lifecycle._start_phase_verify_nodes(
        make_network_config(make_targets("a")), FakeLog(), None
    )
    assert cont is False and ran == [1]


def test_verify_nodes_seeds_the_hostname_cache_from_the_same_round_trip(monkeypatch) -> None:
    """The very next phase resolves every node's hostname over SSH — that was a
    second full handshake per node, on top of this one."""
    _verify_facts(monkeypatch, {
        "k3s_env": "K3S_URL='https://10.77.0.99:6443'",
        "agent_env_present": "yes",
        "image": "yes",
        "hostname": "node-a",
    })

    assert lifecycle._start_phase_verify_nodes(
        make_network_config(make_targets("a")), FakeLog(), None
    ) is True
    assert lifecycle._hostname_cache["a"][0] == "node-a"


# --- start: the setup-all.sh subprocess --------------------------------------

class _CancelledMgr:
    """Enough of LifecycleManager for _run_setup_all, already cancelled."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self.cancel_event.set()
        self.subprocess = None


def test_setup_all_is_not_spawned_when_the_run_is_already_cancelled(monkeypatch) -> None:
    """A stop landing anywhere in the pre-Popen path (docker wait, IP
    detection) must still cancel the run; otherwise it is lost and the 5-15 min
    bootstrap runs to completion first."""
    monkeypatch.setattr(lifecycle, "_ensure_docker_running", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle, "detect_mac_lan_ip", lambda _c=None: "10.77.0.1")
    spawned: list[tuple] = []
    monkeypatch.setattr(
        lifecycle.subprocess, "Popen", lambda *a, **k: spawned.append((a, k))
    )

    with pytest.raises(lifecycle.CancelledOp):
        lifecycle._run_setup_all(FakeLog(), mgr=_CancelledMgr(), config=None)
    assert not spawned


def test_setup_all_decodes_leniently_and_never_inherits_stdin(monkeypatch) -> None:
    """One undecodable byte raised out of the read loop, and proc.wait() then
    blocked forever on a pipe nobody was draining."""
    monkeypatch.setattr(lifecycle, "_ensure_docker_running", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle, "detect_mac_lan_ip", lambda _c=None: "10.77.0.1")
    captured: dict = {}

    class _Proc:
        def __init__(self) -> None:
            self.stdout = io.StringIO("[OK] done\n")

        def wait(self) -> int:
            return 0

    def fake_popen(_argv, **kwargs):
        captured.update(kwargs)
        return _Proc()

    monkeypatch.setattr(lifecycle.subprocess, "Popen", fake_popen)

    lifecycle._run_setup_all(FakeLog(), mgr=None, config=None)

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["stdin"] is subprocess.DEVNULL
    assert captured["start_new_session"] is True, "cancel() kills the process GROUP"
