"""K3s agent join and env-file reconciliation.

Everything here is decided from one text blob or one command string, and every
failure mode is silent: a node that keeps running against a stale K3S_URL while
`heal-nodes` calls it healthy, a stopped agent that reads as joined, or a
cluster credential written into a command line every local account can read
with `ps`.

Nothing touches SSH: run_command and upload_file are stubbed, so what is under
test is the command that WOULD be sent.
"""

from __future__ import annotations

import pytest

from orchestrator.config.schema import SSHTarget
from orchestrator.deploy import k3s_agent
from orchestrator.deploy.ssh import conn_kwargs

ENV_FILE = (
    "K3S_URL='https://10.77.0.98:6443'\n"
    "K3S_TOKEN='K10ffeecafedeadbeef::server:supersecret'\n"
)


class _Calls:
    """Recorder for the stubbed SSH primitives."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: list[dict] = []


@pytest.fixture
def ssh_calls(monkeypatch):
    calls = _Calls()

    def run_command(_host, command, **_kw):
        calls.commands.append(command)
        if command.startswith("sudo cat "):
            return 0, ENV_FILE, ""
        if "is-active" in command:
            return 0, "not-running\n", ""
        return 0, "", ""

    def upload_file(_host, local_path, remote_path, **kw):
        # Read it here: the local staging file is unlinked as soon as the real
        # upload returns, so the content is only observable during the call.
        with open(local_path) as f:
            content = f.read()
        calls.uploads.append({
            "local": str(local_path), "remote": remote_path, "content": content, **kw,
        })

    monkeypatch.setattr(k3s_agent, "run_command", run_command)
    monkeypatch.setattr(k3s_agent, "upload_file", upload_file)
    return calls


# --- _parse_k3s_url -----------------------------------------------------------


def test_the_last_k3s_url_assignment_wins() -> None:
    """systemd's EnvironmentFile applies the LAST assignment.

    Stopping at the first made reconcile_server_url compare a stale line
    against the expected URL, answer "already correct", and leave the agent
    pointing elsewhere — a node heal-nodes then reported healthy.
    """
    text = "K3S_URL='https://10.77.0.1:6443'\nK3S_TOKEN='K10x'\nK3S_URL=https://10.77.0.99:6443\n"
    assert k3s_agent._parse_k3s_url(text) == "https://10.77.0.99:6443"


def test_a_single_quoted_url_is_unwrapped() -> None:
    assert k3s_agent._parse_k3s_url(ENV_FILE) == "https://10.77.0.98:6443"


def test_no_k3s_url_line_is_none() -> None:
    assert k3s_agent._parse_k3s_url("K3S_TOKEN='K10x'\n") is None


# --- reconcile_server_url -----------------------------------------------------


def test_the_env_file_never_reaches_a_command_line(ssh_calls) -> None:
    """That file holds K3S_TOKEN (which is why an empty read is refused), and a
    remote shell's argv is readable via `ps` by every local account on the node
    — for the sake of changing one non-secret URL line."""
    ok, msg = k3s_agent.reconcile_server_url("10.77.0.9", "https://10.77.0.99:6443")

    assert ok is True, msg
    for command in ssh_calls.commands:
        assert "K3S_TOKEN" not in command, command
        assert "supersecret" not in command, command


def test_the_rewritten_file_is_uploaded_and_installed_as_root(ssh_calls) -> None:
    k3s_agent.reconcile_server_url("10.77.0.9", "https://10.77.0.99:6443")

    upload = ssh_calls.uploads[0]
    assert upload["mode"] == 0o600, "the staged file carries K3S_TOKEN"
    assert upload["parent_mode"] == 0o700
    assert not upload["remote"].startswith("/tmp/"), upload["remote"]
    staged = upload["content"]
    assert "K3S_URL='https://10.77.0.99:6443'" in staged
    assert "K3S_TOKEN='K10ffeecafedeadbeef::server:supersecret'" in staged, "token was dropped"

    install = ssh_calls.commands[-1]
    assert "install -m 600 -o root -g root" in install
    assert k3s_agent.AGENT_ENV_PATH in install
    assert "systemctl restart k3s-agent" in install
    assert install.rstrip().endswith("exit $rc"), "the staged token file must always be removed"
    assert "rm -f" in install


def test_a_matching_url_changes_nothing(ssh_calls) -> None:
    ok, msg = k3s_agent.reconcile_server_url("10.77.0.9", "https://10.77.0.98:6443")

    assert ok is True
    assert "already correct" in msg
    assert ssh_calls.uploads == []


# --- join_agent ---------------------------------------------------------------


def test_the_join_script_is_staged_outside_tmp_and_removed(ssh_calls) -> None:
    """It is executed as root moments after landing, so it must not be
    writable by anyone else, and it must not persist as a root-run script in a
    shared directory."""
    k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")

    upload = ssh_calls.uploads[0]
    assert upload["local"].endswith("join-node.sh")
    assert not upload["remote"].startswith("/tmp/"), upload["remote"]
    assert upload["mode"] == 0o700
    assert upload["parent_mode"] == 0o700

    join = ssh_calls.commands[-1]
    assert upload["remote"] in join
    assert "rm -f" in join
    # `exit 100` (reboot needed) is consumed by the caller, so the script's own
    # status has to survive the cleanup.
    assert join.rstrip().endswith("exit $rc")


def test_the_agent_version_is_pinned_when_the_server_reports_one(ssh_calls) -> None:
    """Unpinned, get.k3s.io installs latest stable while the k3d server stays
    on its bundled image, so nodes joined weeks apart end up with kubelets
    newer than the API server — outside the supported skew, and surfacing much
    later as an unexplained NotReady node.

    None keeps the unpinned install: a cluster that cannot answer /version has
    larger problems than skew, and join-node.sh warns on that path.
    """
    k3s_agent.join_agent(
        "10.77.0.9", "https://10.77.0.99:6443", "K10secret", k3s_version="v1.31.5+k3s1",
    )
    assert "--k3s-version v1.31.5+k3s1" in ssh_calls.commands[-1]

    ssh_calls.commands.clear()
    k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")
    assert "--k3s-version" not in ssh_calls.commands[-1]


def test_only_a_well_formed_k3s_version_is_passed_on(monkeypatch) -> None:
    """A malformed pin makes the installer fail outright, where an absent one
    merely installs latest stable — so anything not spelled the way
    INSTALL_K3S_VERSION wants is reported as unknown rather than forwarded."""
    from types import SimpleNamespace

    from orchestrator.cluster import kubectl

    def answer(rc: int, out: str):
        monkeypatch.setattr(
            kubectl, "_run",
            lambda _args, check=True: SimpleNamespace(returncode=rc, stdout=out),
        )

    answer(0, '{"gitVersion": "v1.31.5+k3s1"}')
    assert kubectl.server_version() == "v1.31.5+k3s1"
    answer(0, '{"gitVersion": "v1.31.5"}')          # upstream k8s, not a k3s build
    assert kubectl.server_version() is None
    answer(0, "not json at all")
    assert kubectl.server_version() is None
    answer(1, "")                                    # API server unreachable
    assert kubectl.server_version() is None


def test_the_join_token_never_reaches_a_command_line(ssh_calls) -> None:
    """It is the one credential that lets any machine join this cluster.

    An argv word is public twice over: /proc/<pid>/cmdline is world-readable
    (both DaemonSets set hostPID: true) and sudo copies the whole command line
    into the node's auth log, where it outlives the join. join-node.sh grew
    --token-file for exactly this; the token used to travel as $2.
    """
    ok, msg = k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")

    assert ok is True, msg
    for command in ssh_calls.commands:
        assert "K10secret" not in command, command
    join = ssh_calls.commands[-1]
    assert "--token-file" in join, join

    token_upload = next(u for u in ssh_calls.uploads if u["content"].startswith("K10secret"))
    assert token_upload["mode"] == 0o600
    assert token_upload["parent_mode"] == 0o700
    assert not token_upload["remote"].startswith("/tmp/"), token_upload["remote"]
    # join-node.sh reads it with `IFS= read -r`, which wants a terminated line.
    assert token_upload["content"] == "K10secret\n"
    # The path handed to --token-file is the one that was uploaded (the uploads
    # record SFTP paths, the command the "$HOME"/-prefixed shell words for the
    # same names), and the same command removes BOTH stages whatever the join
    # exited with — a failed join must not leave the token behind either.
    assert token_upload["remote"] in join
    rm = join.split("rm -f ", 1)[1]
    assert ssh_calls.uploads[0]["remote"] in rm and token_upload["remote"] in rm, join


def test_a_staged_token_is_swept_when_the_join_never_runs(monkeypatch) -> None:
    """The join command carries its own `rm -f`, so it cleans up on every exit
    code — but a run_command that RAISES (a node that stops answering
    mid-install) never reaches it, and what is left behind is the cluster
    credential sitting in the login user's home."""
    commands: list[str] = []
    uploaded: list[str] = []

    def run_command(_host, command, **_kw):
        commands.append(command)
        if "is-active" in command:
            return 3, "not-running\n", ""
        if command.startswith("rm -f "):
            return 0, "", ""
        raise TimeoutError("node stopped answering")

    monkeypatch.setattr(k3s_agent, "run_command", run_command)
    monkeypatch.setattr(
        k3s_agent, "upload_file",
        lambda _h, _local, remote, **_kw: uploaded.append(remote),
    )

    ok, msg = k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")

    assert ok is False and "TimeoutError" in msg, msg
    sweep = commands[-1]
    assert sweep.startswith("rm -f "), commands
    for remote in uploaded:
        # `uploaded` holds the SFTP paths; the sweep uses the "$HOME"/-prefixed
        # shell words for the same names.
        assert remote in sweep, sweep


def test_a_staged_token_is_swept_when_the_join_times_out(monkeypatch) -> None:
    """The failing path is a RETURN, not a raise.

    ``ssh.run_command`` reports its own expiry as rc 124 with the message on
    stderr — it does not raise — so the sweep has to trigger on that status.
    A join that expired never reached its own trailing `rm -f`, and what is
    left behind is the cluster credential in the login user's home.
    """
    commands: list[str] = []
    uploaded: list[str] = []

    def run_command(_host, command, **_kw):
        commands.append(command)
        if "is-active" in command:
            return 3, "not-running\n", ""
        if command.startswith("rm -f "):
            return 0, "", ""
        return k3s_agent._TIMEOUT_RC, "", "command timed out after 120s: sudo"

    monkeypatch.setattr(k3s_agent, "run_command", run_command)
    monkeypatch.setattr(
        k3s_agent, "upload_file",
        lambda _h, _local, remote, **_kw: uploaded.append(remote),
    )

    ok, _msg = k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")

    assert ok is False
    sweep = commands[-1]
    assert sweep.startswith("rm -f "), commands
    assert len(uploaded) == 2, uploaded  # the join script and the token
    for remote in uploaded:
        assert remote in sweep, sweep


def test_the_env_stage_is_swept_when_the_rewrite_times_out(monkeypatch) -> None:
    """Same rc-124 contract on the reconcile path.

    The staged file here is the rebuilt k3s-agent env, which carries K3S_TOKEN
    — leaving it behind after an expired install is the same credential leak.
    """
    commands: list[str] = []
    uploaded: list[str] = []

    def run_command(_host, command, **_kw):
        commands.append(command)
        if command.startswith("sudo cat "):
            return 0, ENV_FILE, ""
        if command.startswith("rm -f "):
            return 0, "", ""
        return k3s_agent._TIMEOUT_RC, "", "command timed out after 60s: sudo"

    monkeypatch.setattr(k3s_agent, "run_command", run_command)
    monkeypatch.setattr(
        k3s_agent, "upload_file",
        lambda _h, _local, remote, **_kw: uploaded.append(remote),
    )

    ok, _msg = k3s_agent.reconcile_server_url("10.77.0.9", "https://10.77.0.99:6443")

    assert ok is False
    sweep = commands[-1]
    assert sweep.startswith("rm -f "), commands
    assert uploaded and uploaded[0] in sweep, sweep


def test_an_unanswered_env_read_raises_but_an_empty_one_is_none(monkeypatch) -> None:
    """"Could not read" and "there is nothing to read" must stay distinct.

    The remote command ends in ``|| true``, so a read that COMPLETED always
    exits 0. Collapsing an expired read into the same None as "no env file"
    lets the drift checker score an unreachable node as clean and announce
    "Drift cleared".
    """
    def timed_out(_host, _command, **_kw):
        return k3s_agent._TIMEOUT_RC, "", "command timed out after 15s: sudo"

    monkeypatch.setattr(k3s_agent, "run_command", timed_out)
    with pytest.raises(TimeoutError):
        k3s_agent.get_configured_server_url("10.77.0.9")

    monkeypatch.setattr(k3s_agent, "run_command", lambda *_a, **_kw: (0, "  \n", ""))
    assert k3s_agent.get_configured_server_url("10.77.0.9") is None


def test_a_stopped_agent_is_not_read_as_joined(monkeypatch, ssh_calls) -> None:
    """`systemctl is-active` prints inactive/activating/deactivating, all of
    which contain the substring "active" — the `not-running` sentinel from the
    `||` branch is what makes the substring test safe."""
    seen: list[str] = []

    def run_command(_host, command, **_kw):
        seen.append(command)
        if "is-active" in command:
            return 3, "inactive\nnot-running\n", ""
        return 0, "", ""

    monkeypatch.setattr(k3s_agent, "run_command", run_command)

    ok, msg = k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")

    assert ok is True
    assert "already running" not in msg
    assert any("is-active" not in c for c in seen), "the join never ran"


def test_a_running_agent_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(k3s_agent, "run_command", lambda *_a, **_k: (0, "active\n", ""))
    ok, msg = k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")
    assert (ok, msg) == (True, "K3s agent already running")


def test_the_reboot_exit_code_becomes_an_actionable_message(monkeypatch, ssh_calls) -> None:
    def run_command(_host, command, **_kw):
        if "is-active" in command:
            return 3, "not-running\n", ""
        return 100, "", ""

    monkeypatch.setattr(k3s_agent, "run_command", run_command)

    ok, msg = k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret")
    assert ok is False
    assert "reboot" in msg and "setup-all.sh" in msg


# --- one connection-kwargs dict, every helper ---------------------------------


def test_every_helper_accepts_the_shared_connection_kwargs(ssh_calls) -> None:
    """conn_kwargs adds a `timeout` key whenever it is given one, so a helper
    without that parameter turns `**self._conn(target, timeout=30)` into a
    TypeError at the call site — the asymmetry was unmarked and untested."""
    conn = conn_kwargs(SSHTarget(node_id="n", host="10.77.0.9"), "~/.ssh/", timeout=30)

    k3s_agent.join_agent("10.77.0.9", "https://10.77.0.99:6443", "K10secret", **conn)
    k3s_agent.get_configured_server_url("10.77.0.9", **conn)
    k3s_agent.reconcile_server_url("10.77.0.9", "https://10.77.0.99:6443", **conn)
