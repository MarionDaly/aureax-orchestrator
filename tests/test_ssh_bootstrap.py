"""Tests for SSH key bootstrapping and password auth.

The authorized_keys command is exercised by actually running it under `sh`
against a temporary HOME, because both bugs it guards against are shell-level
and produce a node that looks bootstrapped while refusing every key.
"""

from __future__ import annotations

import subprocess
import threading
import time
from types import SimpleNamespace
from typing import ClassVar

import paramiko
import pytest

from orchestrator.config.schema import SSHConfig, SSHTarget
from orchestrator.deploy.ssh import authorized_keys_command, conn_kwargs

# Synthetic bodies, not keys: only the one-line public-key SHAPE is exercised,
# so the base64 after the type prefix is filler that decodes to nothing usable.
KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleFixtureKeyNotARealKey0000000000000 orchestrator@example.com"
EXISTING = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQExampleFixtureRsaNotARealKey node@example.com"
# An operator's own key file, whose comment field is free-form: ensure_key_installed
# checks only that the key is one line of public-key shape, so a shell
# metacharacter reaches the command builder unchanged.
AWKWARD_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleFixtureKeyNotARealKey0000000000000 user@alice's-$(id)-mac"


@pytest.fixture(autouse=True)
def _tmp_home(monkeypatch, tmp_path):
    """Redirect $HOME for every test in this module.

    `get_ssh_client` reads and rewrites ``~/.aureax/known_hosts`` on any path
    that reaches a connect, and `ensure_key_installed` and `test_connectivity`
    both go through it. Redirecting per test made that a thing each new case
    had to remember, and a case that forgets writes into the operator's own
    host-key store.
    """
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def home(tmp_path):
    """A tmp $HOME at the default mode, created once per test.

    Shared so that the ONE test needing a group-writable HOME
    (test_home_directory_group_write_bit_is_cleared, the StrictModes case) is
    the only one that spells a chmod out, instead of looking identical to the
    cases that must run at the default mode.
    """
    path = tmp_path / "home"
    path.mkdir()
    return path


def _keys(home) -> list[str]:
    """The non-blank lines of the node's authorized_keys."""
    text = (home / ".ssh" / "authorized_keys").read_text()
    return [ln for ln in text.splitlines() if ln.strip()]


def _run(home, seed: str | None, key: str = KEY) -> str:
    ssh_dir = home / ".ssh"
    if seed is not None:
        ssh_dir.mkdir(parents=True, exist_ok=True)
        (ssh_dir / "authorized_keys").write_text(seed)
    result = subprocess.run(
        ["sh", "-c", authorized_keys_command(key)],
        cwd=home, env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_appends_to_file_without_trailing_newline(home) -> None:
    """The splice bug: a plain append fuses our key onto the last entry.

    That corrupts the existing key AND ours, and sshd then rejects both with
    nothing in the client output to explain why.
    """
    out = _run(home, EXISTING)  # deliberately no trailing newline

    assert out == "INSTALLED"
    lines = _keys(home)
    assert lines == [EXISTING, KEY], "each key must remain on its own line"


def test_appends_to_file_with_trailing_newline(home) -> None:
    out = _run(home, EXISTING + "\n")

    assert out == "INSTALLED"
    lines = _keys(home)
    assert lines == [EXISTING, KEY]


def test_creates_ssh_dir_when_absent(home) -> None:
    out = _run(home, None)

    assert out == "INSTALLED"
    lines = _keys(home)
    assert lines == [KEY]


def test_second_run_is_idempotent(home) -> None:
    assert _run(home, EXISTING) == "INSTALLED"
    assert _run(home, None) == "ALREADY_PRESENT"

    lines = _keys(home)
    assert lines.count(KEY) == 1


def test_spliced_key_is_not_reported_as_present(home) -> None:
    """A whole-line match is required — substring matching masks corruption."""
    # Exactly the corruption a newline-less append produces.
    out = _run(home, EXISTING + KEY + "\n")

    assert out == "INSTALLED", "a spliced key must not count as installed"
    assert KEY in _keys(home)


def test_a_key_carrying_shell_metacharacters_is_installed_verbatim(home) -> None:
    """The key is spliced into a single-quoted word in a command run on every
    node, so the apostrophe in a comment field is the whole escaping contract.
    An unescaped one either breaks the command or lets the rest of the comment
    run as shell — and the grep that decides "already present" quotes it the
    same way, so both halves are checked here."""
    assert _run(home, None, AWKWARD_KEY) == "INSTALLED"
    assert _keys(home) == [AWKWARD_KEY], "the key must land byte-for-byte, unexpanded"

    assert _run(home, None, AWKWARD_KEY) == "ALREADY_PRESENT"
    assert _keys(home) == [AWKWARD_KEY]


def test_home_directory_group_write_bit_is_cleared(home) -> None:
    """sshd StrictModes refuses keys when HOME is group- or world-writable.

    The chmod is the whole premise, hence explicit here and nowhere else.
    """
    home.chmod(0o775)
    _run(home, None)

    assert home.stat().st_mode & 0o022 == 0, "HOME must not be group/other writable"
    assert (home / ".ssh").stat().st_mode & 0o777 == 0o700
    assert (home / ".ssh" / "authorized_keys").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("target_kwargs", "expected"),
    [
        ({"password": "pw"}, "pw"),
        ({}, None),
    ],
)
def test_conn_kwargs_carries_password(target_kwargs, expected) -> None:
    target = SSHTarget(node_id="n", host="10.77.0.1", **target_kwargs)
    assert conn_kwargs(target, "~/.ssh/")["password"] == expected


def test_password_env_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("AUREAX_TEST_PW", "from-env")
    target = SSHTarget(
        node_id="n", host="10.77.0.1", password="literal", password_env="AUREAX_TEST_PW"
    )
    assert target.resolve_password() == "from-env"

    # Unset env var must not silently fall back to a stale literal.
    monkeypatch.delenv("AUREAX_TEST_PW")
    assert target.resolve_password() is None


class _RecordingClient:
    """Stub paramiko.SSHClient that records the kwargs `connect` was given.

    The host-key methods are no-ops: get_ssh_client remembers every node's key
    in ~/.aureax/known_hosts, and a stub without them would push that side
    effect into whatever HOME the test happens to run under.
    """

    last_kwargs: ClassVar[dict] = {}

    def set_missing_host_key_policy(self, _policy) -> None:
        pass

    def load_host_keys(self, _filename) -> None:
        pass

    def get_host_keys(self) -> dict:
        return {}

    def save_host_keys(self, _filename) -> None:
        pass

    def get_transport(self):
        return None

    def connect(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs

    def close(self) -> None:
        pass

    # run_command/upload_file use `with get_ssh_client(...)`; the real client is
    # a paramiko ClosingContextManager.
    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


@pytest.fixture
def recorded_connect(monkeypatch):
    """Return a callable that connects with a stub and yields the kwargs used."""
    from orchestrator.deploy import ssh

    def go(**target_kwargs) -> dict:
        monkeypatch.setattr(ssh.paramiko, "SSHClient", _RecordingClient)
        _RecordingClient.last_kwargs = {}
        ssh.get_ssh_client("10.77.0.1", **target_kwargs)
        return _RecordingClient.last_kwargs

    return go


def test_password_only_target_suppresses_key_offers(recorded_connect) -> None:
    """Paramiko otherwise offers every agent/default key before the password.

    sshd's MaxAuthTries then disconnects, so a factory-password node fails to
    authenticate with the correct password in hand — and the failure reads as
    "wrong password" rather than "too many keys tried first".
    """
    kwargs = recorded_connect(password="pw")

    assert kwargs["password"] == "pw"
    assert kwargs["look_for_keys"] is False
    assert kwargs["allow_agent"] is False
    assert "key_filename" not in kwargs


def test_existing_key_plus_password_offers_the_key_first(recorded_connect, tmp_path) -> None:
    """Key first, password as fallback — what makes bootstrapping transparent."""
    key = tmp_path / "id_ed25519"
    key.write_text("x")

    kwargs = recorded_connect(key_file=str(key), password="pw")

    assert kwargs["key_filename"] == str(key)
    assert kwargs["password"] == "pw"
    # The suppression is only for the password-only case; with a usable key the
    # defaults must stay, or a key in the agent stops working.
    assert "look_for_keys" not in kwargs
    assert "allow_agent" not in kwargs


def test_missing_key_with_a_password_falls_back_instead_of_raising(
    recorded_connect, tmp_path
) -> None:
    """First connect to a fresh node: the key isn't installed there yet."""
    kwargs = recorded_connect(key_file=str(tmp_path / "absent"), password="pw")

    assert "key_filename" not in kwargs
    assert kwargs["look_for_keys"] is False


def test_missing_key_with_no_password_still_raises(monkeypatch, tmp_path) -> None:
    """A key that is the ONLY credential must fail loudly, not silently retry."""
    from orchestrator.deploy import ssh

    monkeypatch.setattr(ssh.paramiko, "SSHClient", _RecordingClient)
    with pytest.raises(FileNotFoundError, match="SSH key not found"):
        ssh.get_ssh_client("10.77.0.1", key_file=str(tmp_path / "absent"))


def test_the_client_is_closed_when_connect_fails(monkeypatch) -> None:
    """An unclosed client leaks a Transport thread and socket per attempt.

    The `with` block never runs when connect() raises, and every SSH pass walks
    the whole fleet — so one unreachable or wrong-password node leaks once per
    node per pass on the long-lived dashboard process.
    """
    from orchestrator.deploy import ssh

    closed: list[int] = []

    class _FailingClient(_RecordingClient):
        def connect(self, **_kwargs) -> None:
            raise paramiko.AuthenticationException("bad password")

        def close(self) -> None:
            closed.append(1)

    monkeypatch.setattr(ssh.paramiko, "SSHClient", _FailingClient)
    with pytest.raises(paramiko.AuthenticationException):
        ssh.get_ssh_client("10.77.0.1", password="pw")
    assert closed == [1]


def test_a_changed_host_key_is_logged_replaced_and_retried(monkeypatch, caplog) -> None:
    """Nodes are re-imaged as routine, so a changed key is loud but not fatal:
    one ERROR naming both fingerprints, the store rewritten, one reconnect."""
    from orchestrator.deploy import ssh
    old, new = paramiko.ECDSAKey.generate(), paramiko.ECDSAKey.generate()
    ssh._record_key("10.77.0.1", 22, old)
    attempts = []

    class Changed(_RecordingClient):
        def connect(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise paramiko.BadHostKeyException("10.77.0.1", new, old)

    monkeypatch.setattr(ssh.paramiko, "SSHClient", Changed)
    with caplog.at_level("ERROR", logger="aureax.ssh"):
        ssh.get_ssh_client("10.77.0.1", password="pw")
    assert len(attempts) == 2
    store = paramiko.HostKeys(str(ssh._known_hosts_path()))
    assert store.check("10.77.0.1", new) and not store.check("10.77.0.1", old)
    message = next(r.getMessage() for r in caplog.records if "CHANGED" in r.getMessage())
    assert ssh._fingerprint(old) in message and ssh._fingerprint(new) in message


def test_a_key_that_changes_again_on_the_retry_is_not_retried_forever(monkeypatch) -> None:
    from orchestrator.deploy import ssh
    old, new = paramiko.ECDSAKey.generate(), paramiko.ECDSAKey.generate()
    attempts = []

    class Changed(_RecordingClient):
        def connect(self, **kwargs):
            attempts.append(kwargs)
            raise paramiko.BadHostKeyException("10.77.0.1", new, old)

    monkeypatch.setattr(ssh.paramiko, "SSHClient", Changed)
    with pytest.raises(paramiko.BadHostKeyException):
        ssh.get_ssh_client("10.77.0.1", password="pw")
    assert len(attempts) == 2


def test_a_first_contact_is_announced_and_recorded(caplog) -> None:
    from orchestrator.deploy import ssh
    key = paramiko.RSAKey.generate(2048)
    client = paramiko.SSHClient()
    with caplog.at_level("WARNING", logger="aureax.ssh"):
        ssh._FirstContactPolicy().missing_host_key(client, "[10.77.0.1]:2222", key)
    path = ssh._known_hosts_path()
    assert paramiko.HostKeys(str(path)).check("[10.77.0.1]:2222", key)
    assert path.stat().st_mode & 0o777 == 0o600
    assert any("First contact" in r.getMessage() for r in caplog.records)


# --- ensure_key_installed: never raises, and never ships a private key --------


def test_ensure_key_installed_reports_a_missing_public_key(tmp_path) -> None:
    from orchestrator.deploy import ssh

    ok, msg = ssh.ensure_key_installed("10.77.0.1", str(tmp_path / "absent.pub"))
    assert ok is False
    assert "public key not found" in msg


def test_ensure_key_installed_reports_an_empty_public_key(tmp_path) -> None:
    from orchestrator.deploy import ssh

    pub = tmp_path / "id.pub"
    pub.write_text("   \n")
    ok, msg = ssh.ensure_key_installed("10.77.0.1", str(pub))
    assert ok is False
    assert "empty" in msg


def test_ensure_key_installed_refuses_a_private_key(tmp_path) -> None:
    """bootstrap_public_key is a free-form path: dropping the `.pub` appends the
    PRIVATE key to every node's authorized_keys and still reports success,
    because publickey auth keeps working via the real key."""
    from orchestrator.deploy import ssh

    priv = tmp_path / "id_ed25519"
    priv.write_text(
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA\n"
        "-----END OPENSSH PRIVATE KEY-----\n"
    )
    ok, msg = ssh.ensure_key_installed("10.77.0.1", str(priv))
    assert ok is False
    assert "not an OpenSSH public key" in msg


def test_ensure_key_installed_never_raises_on_a_connect_failure(monkeypatch, tmp_path) -> None:
    """It runs per node inside join_nodes; a raise here aborts the whole deploy
    on the first box that happens to be off."""
    from orchestrator.deploy import ssh

    pub = tmp_path / "id.pub"
    pub.write_text(KEY + "\n")

    def boom(*_a, **_k):
        raise OSError("no route to host")

    monkeypatch.setattr(ssh, "get_ssh_client", boom)
    ok, msg = ssh.ensure_key_installed("10.77.0.1", str(pub))
    assert ok is False
    assert msg.startswith("OSError: ")


# --- run_command: the exec-request handshake and the timeout path -------------


def test_ssh_exec_acknowledgement_has_a_deadline() -> None:
    """Paramiko waits forever for the server's exec-request acknowledgement, so
    `timeout` alone bounds the connect and the output loop but not the step in
    between — a node that accepts the TCP connection and then stops answering
    parks the whole deploy pass on one host. Closing the client is what unblocks
    the channel event wait."""
    from orchestrator.deploy.ssh import _exec_with_deadline

    released = threading.Event()
    client = SimpleNamespace(exec_command=lambda *a, **k: released.wait(), close=released.set)
    started = time.monotonic()

    with pytest.raises(TimeoutError):
        _exec_with_deadline(client, "synthetic-command", started + 0.05)

    assert released.is_set(), "the hung worker must be released, not left waiting"
    # A loose ceiling on purpose: the proof is the raise and the release, and
    # the bound only has to separate "returns promptly" from "waits forever" —
    # a thread spawn and two queue wake-ups are all schedulable.
    assert time.monotonic() - started < 2


# --- run_command: the timeout path ------------------------------------------


class _HangingStream:
    def __init__(self) -> None:
        self.channel = paramiko.Channel(1)
        # EOF in both streams without an exit status must still time out.
        self.channel.in_buffer.close()
        self.channel.in_stderr_buffer.close()


class _HangingClient(_RecordingClient):
    def exec_command(self, _command, timeout=None):
        return None, _HangingStream(), _HangingStream()


def test_run_command_reports_a_command_timeout_as_124(monkeypatch) -> None:
    """A command that never exits must fail, not hang the deploy step."""
    from orchestrator.deploy import ssh

    monkeypatch.setattr(ssh.paramiko, "SSHClient", _HangingClient)

    rc, out, err = ssh.run_command("10.77.0.1", "sudo join.sh https://s:6443 K10secret", timeout=0.05)

    assert (rc, out) == (124, "")
    assert "timed out after 0.05s" in err


def test_the_timeout_message_does_not_echo_the_command_body(monkeypatch) -> None:
    """join_agent embeds this stderr in its message and join_nodes logs it, so a
    command line carrying the K3s token would put the token in the log."""
    from orchestrator.deploy import ssh

    monkeypatch.setattr(ssh.paramiko, "SSHClient", _HangingClient)

    _rc, _out, err = ssh.run_command("10.77.0.1", "printf '%s' 'K3S_TOKEN=K10secret'", timeout=0.05)

    assert "K10secret" not in err
    assert "timed out" in err


# --- probe_facts: one round trip, N answers ----------------------------------


# probe_facts salts its delimiter per call; pin the salt so a transcript can
# be written down. The marker is what the node would have to forge.
_TOKEN = "0" * 32
_MARK = f"===AUREAX_FACT_{_TOKEN}:"


def _facts_transcript(monkeypatch, transcript: str, rc: int = 0, err: str = "") -> None:
    from orchestrator.deploy import ssh

    monkeypatch.setattr(ssh.secrets, "token_hex", lambda _n=16: _TOKEN)
    monkeypatch.setattr(ssh, "run_command", lambda *_a, **_k: (rc, transcript, err))


def test_probe_facts_maps_each_marker_to_its_snippet_output(monkeypatch) -> None:
    """Names absent from the output must stay "", and multi-line values must
    survive intact — a mis-parse reports healthy nodes as unreachable."""
    from orchestrator.deploy import ssh

    _facts_transcript(monkeypatch, "\n".join([
        f"{_MARK}hostname===",
        "edge-01",
        f"{_MARK}k3s_env===",
        "K3S_URL='https://10.77.0.99:6443'",
        "K3S_TOKEN='K10xxx'",
        f"{_MARK}silent===",
    ]))

    facts = ssh.probe_facts(
        "10.77.0.1", {"hostname": "hostname", "k3s_env": "cat env", "silent": "true", "absent": "x"}
    )

    assert facts["hostname"] == "edge-01"
    assert facts["k3s_env"] == "K3S_URL='https://10.77.0.99:6443'\nK3S_TOKEN='K10xxx'"
    assert facts["silent"] == ""
    assert facts["absent"] == "", "a fact whose marker never appeared must not shift"


def test_probe_facts_ignores_a_marker_the_caller_never_asked_for(monkeypatch) -> None:
    """Snippets cat node-controlled files, so the output is a trust boundary:
    an echoed marker must not truncate a value or invent a key."""
    from orchestrator.deploy import ssh

    _facts_transcript(monkeypatch, "\n".join([
        f"{_MARK}k3s_env===",
        "K3S_URL='https://10.77.0.99:6443'",
        f"{_MARK}gpu===",
        "injected",
    ]))

    facts = ssh.probe_facts("10.77.0.1", {"k3s_env": "cat env"})

    assert set(facts) == {"k3s_env"}
    assert facts["k3s_env"].startswith("K3S_URL=")
    assert "injected" in facts["k3s_env"], "the forged marker must be data, not a boundary"


def test_probe_facts_raises_when_the_probe_itself_timed_out(monkeypatch) -> None:
    """Empty facts read as "this node never joined" in lifecycle's start path,
    which re-runs setup-all.sh (5-15 min) against a node that was merely slow.
    Raising instead lands in the per-node `except Exception` that classifies the
    node unreachable and does NOT re-bootstrap."""
    from orchestrator.deploy import ssh

    _facts_transcript(monkeypatch, "", rc=124, err="command timed out after 15s: sudo")

    with pytest.raises(TimeoutError, match="timed out"):
        ssh.probe_facts("10.77.0.1", {"k3s_env": "cat env"})


def test_probe_facts_raises_on_any_other_nonzero_exit(monkeypatch) -> None:
    """recv_exit_status() also returns -1 for a channel closed without a status."""
    from orchestrator.deploy import ssh

    _facts_transcript(monkeypatch, "", rc=-1, err="")

    with pytest.raises(RuntimeError, match="rc=-1"):
        ssh.probe_facts("10.77.0.1", {"k3s_env": "cat env"})


# --- upload_file -------------------------------------------------------------


class _RecordingSFTP:
    """Records what upload_file asks the SFTP subsystem to do."""

    def __init__(self, existing_dirs: tuple[str, ...] = ()) -> None:
        self.existing = set(existing_dirs)
        self.calls: list[tuple] = []
        self.channel_timeout: float | None = None

    def get_channel(self):
        sftp = self

        class _Channel:
            def settimeout(self, value) -> None:
                sftp.channel_timeout = value

        return _Channel()

    def mkdir(self, path, mode) -> None:
        self.calls.append(("mkdir", path, mode))
        if path in self.existing:
            raise OSError("already exists")

    def chmod(self, path, mode) -> None:
        self.calls.append(("chmod", path, mode))

    def put(self, local, remote) -> None:
        self.calls.append(("put", local, remote))

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        pass


def _upload_client(monkeypatch, sftp) -> None:
    from orchestrator.deploy import ssh

    class _SFTPClient(_RecordingClient):
        def open_sftp(self):
            return sftp

    monkeypatch.setattr(ssh.paramiko, "SSHClient", _SFTPClient)


def test_upload_file_stages_under_a_private_directory(monkeypatch, tmp_path) -> None:
    """A secret-bearing file cannot be staged at a fixed world-writable path:
    any local account can pre-create it, and the put then writes into a file
    they own and can read."""
    from orchestrator.deploy import ssh

    sftp = _RecordingSFTP()
    _upload_client(monkeypatch, sftp)
    local = tmp_path / "cfg.yaml"
    local.write_text("x")

    ssh.upload_file("10.77.0.1", local, ".aureax-stage/cfg.yaml", mode=0o600, parent_mode=0o700)

    assert sftp.calls[0] == ("mkdir", ".aureax-stage", 0o700)
    assert ("put", str(local), ".aureax-stage/cfg.yaml") in sftp.calls
    assert sftp.calls[-1] == ("chmod", ".aureax-stage/cfg.yaml", 0o600)
    # The transfer is bounded: `timeout` otherwise covers the connect only, and
    # a node that stops responding mid-put blocks the deploy forever.
    assert sftp.channel_timeout == 15


def test_upload_file_tightens_an_existing_staging_directory(monkeypatch, tmp_path) -> None:
    """Whatever mode (or owner) an earlier run left it at is not to be trusted."""
    from orchestrator.deploy import ssh

    sftp = _RecordingSFTP(existing_dirs=(".aureax-stage",))
    _upload_client(monkeypatch, sftp)
    local = tmp_path / "cfg.yaml"
    local.write_text("x")

    ssh.upload_file("10.77.0.1", local, ".aureax-stage/cfg.yaml", mode=0o600, parent_mode=0o700)

    assert ("chmod", ".aureax-stage", 0o700) in sftp.calls


# --- test_connectivity classification ----------------------------------------


def test_a_missing_key_file_is_not_reported_as_a_network_error(
    monkeypatch, tmp_path
) -> None:
    """FileNotFoundError subclasses OSError, so `aureax status` used to print
    "network error: SSH key not found" for a perfectly reachable node and send
    the operator after cabling and firewalls."""
    from orchestrator.deploy import ssh

    monkeypatch.setattr(ssh.paramiko, "SSHClient", _RecordingClient)

    ok, msg = ssh.test_connectivity("10.77.0.1", key_file=str(tmp_path / "absent"))

    assert ok is False
    assert "network error" not in msg
    assert "SSH key not found" in msg


def test_enabled_targets_filters_disabled() -> None:
    cfg = SSHConfig(targets=[
        SSHTarget(node_id="a", host="10.77.0.1"),
        SSHTarget(node_id="b", host="10.77.0.2", enabled=False),
        SSHTarget(node_id="c", host="10.77.0.3"),
    ])
    assert [t.node_id for t in cfg.enabled_targets] == ["a", "c"]
