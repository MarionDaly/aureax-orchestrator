"""SSH operations for reaching edge nodes."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import logging
import queue
import secrets
import shlex
import socket
import threading
import time
from pathlib import Path, PurePosixPath

import paramiko

from orchestrator.config.atomic import write_private_file

logger = logging.getLogger("aureax.ssh")

# Mirrors orchestrator/config/schema.py::SSHConfig.key_store — change both
# together. A divergence is silent: callers that omit key_store (tests, direct
# resolve_key_path uses) keep resolving relative key names against the old
# directory, and the symptom is "key installed, still denied", not an error.
DEFAULT_KEY_STORE = "~/.aureax/ssh-keys/"

# Staging lives under the login user's home, one unique name per push. /tmp is
# world-writable, so a fixed name there can be pre-created by any local account
# on the node — and what gets staged is either executed as root (the join
# script) or holds a secret (K3S_TOKEN, a resolved config's llama API key). The
# path is relative on purpose: SFTP resolves it against the login user's home,
# which is what "$HOME" expands to in the remote command.
STAGE_DIR = ".aureax-stage"


def stage_path(prefix: str, suffix: str = "") -> tuple[str, str]:
    """(SFTP path, shell word) for a fresh per-push staging file.

    The single home for the staging convention — directory, unique name and
    the ``"$HOME"/`` spelling — so every caller staging a secret-bearing file
    lands in the same 0700 directory instead of re-deriving the path and
    drifting from it.
    """
    rel = f"{STAGE_DIR}/{prefix}-{secrets.token_hex(8)}{suffix}"
    return rel, f'"$HOME"/{rel}'


def conn_kwargs(target, key_store: str = DEFAULT_KEY_STORE, timeout: int | None = None) -> dict:
    """Connection kwargs for an ``orchestrator.config.schema.SSHTarget``.

    Single place that decides how a target authenticates, so adding a
    credential kind doesn't mean auditing every call site again.
    """
    kwargs: dict = {
        "port": target.port,
        "user": target.user,
        "key_file": resolve_key_path(target.key_file, key_store),
        "password": target.resolve_password(),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def resolve_key_path(key_file: str | None, key_store: str = DEFAULT_KEY_STORE) -> str | None:
    """Resolve a key_file path. If relative, prepend key_store directory."""
    if not key_file:
        return None
    path = Path(key_file).expanduser()
    if path.is_absolute():
        return str(path)
    # Relative path — resolve against key_store
    store = Path(key_store).expanduser()
    return str(store / key_file)


def _known_hosts_path() -> Path:
    """Where node host keys are remembered.

    Resolved per call, not at import: tests redirect ``$HOME``, and a Path
    bound at import time would read and write the developer's real ~/.aureax.
    """
    return Path.home() / ".aureax" / "known_hosts"


def _load_known_hosts(client: paramiko.SSHClient) -> None:
    """Point ``client`` at ~/.aureax/known_hosts when the file exists.

    An unreadable store propagates (OSError, SSHException): silently skipping
    the load would turn every later connect into an unverified one.
    """
    path = _known_hosts_path()
    if path.exists():
        client.load_host_keys(str(path))


def _fingerprint(key: paramiko.PKey) -> str:
    """OpenSSH's ``SHA256:`` spelling of a host key, as ``ssh-keygen -lf`` prints it."""
    return "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")


def _host_entry(host: str, port: int) -> str:
    """known_hosts spelling of a host: bare name, or [host]:port off 22."""
    return host if port == 22 else f"[{host}]:{port}"


def _record_key(host: str, port: int, key: paramiko.PKey, *, replace: bool = False) -> None:
    path = _known_hosts_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Serialize read/merge/replace across concurrent CLI and dashboard processes.
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        keys = paramiko.HostKeys(str(path)) if path.exists() else paramiko.HostKeys()
        entry = _host_entry(host, port)
        if entry in keys and not replace and not keys.check(entry, key):
            raise paramiko.SSHException("Host identity changed while enrolling it")
        if replace and entry in keys:
            del keys[entry]
        keys.add(entry, key.get_name(), key)
        data = "".join(f"{name} {kind} {value.get_base64()}\n"
                       for name in keys for kind, value in keys[name].items())
        write_private_file(path, data)


class _FirstContactPolicy(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client, hostname, key):
        # Paramiko supplies the known_hosts spelling, including non-default port.
        _record_key(hostname, 22, key)
        client.get_host_keys().add(hostname, key.get_name(), key)
        logger.warning("First contact: recorded SSH host identity for %s", hostname)


def enroll_host_key(host: str, fingerprint: str, *, port: int = 22, timeout: int = 15) -> None:
    """Replace a host key only when it matches an independently obtained SHA256 fingerprint.

    This handshake sends no password, client key, or application command.
    """
    with socket.create_connection((host, port), timeout=timeout) as sock, paramiko.Transport(sock) as transport:
        transport.start_client(timeout=timeout)
        key = transport.get_remote_server_key()
        if not hmac.compare_digest(_fingerprint(key), fingerprint):
            raise ValueError("SSH fingerprint does not match; trust store unchanged")
        _record_key(host, port, key, replace=True)


def get_ssh_client(
    host: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    timeout: int = 15,
    password: str | None = None,
) -> paramiko.SSHClient:
    """Create and connect an SSH client to a remote host.

    The returned client is a ``ClosingContextManager`` — use it with ``with``
    so close() runs automatically on success or failure.

    When a key is configured it is tried first and the password acts as a
    fallback, which is what makes key bootstrapping transparent: the first
    connect to a factory-password node uses the password, installs our public
    key, and every connect afterwards authenticates with the key.

    Host keys are trusted on first use and recorded in ~/.aureax/known_hosts.
    A key that later CHANGES is logged at ERROR with both fingerprints, replaced
    in the store and connected to once more: nodes are wiped and re-imaged as
    routine, and setup has to stay automated (ARCHITECTURE.md, "SSH
    authentication"), so refusing would mean a hand edit after every wipe. The
    loud line is the deliverable, not a veto. ``enroll_host_key`` is the
    optional out-of-band check for an operator who wants the fingerprint
    verified before any connection is made.
    """
    connect_kwargs: dict = {
        "hostname": host,
        "port": port,
        "username": user,
        "timeout": timeout,
        "auth_timeout": timeout,
        "banner_timeout": timeout,
        "disabled_algorithms": {"keys": ["ssh-rsa"], "pubkeys": ["ssh-rsa"]},
    }

    key_path = Path(key_file).expanduser() if key_file else None
    key_usable = key_path is not None and key_path.exists()

    if key_path is not None and not key_usable and not password:
        # A configured key that does not exist is a misconfiguration, not a
        # fallback — fail loudly unless a password is also available.
        raise FileNotFoundError(f"SSH key not found: {key_path}")

    if key_usable:
        connect_kwargs["key_filename"] = str(key_path)

    if password:
        connect_kwargs["password"] = password
        if not key_usable:
            # Without these, Paramiko offers every agent/default key first and
            # the server can hit MaxAuthTries and disconnect before the
            # password is ever tried.
            connect_kwargs["look_for_keys"] = False
            connect_kwargs["allow_agent"] = False

    def attempt() -> paramiko.SSHClient:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(_FirstContactPolicy())
        _load_known_hosts(client)
        try:
            client.connect(**connect_kwargs)
        except BaseException:
            # Callers use `with get_ssh_client(...)`, but the with-block is
            # never entered when connect() raises — and once paramiko has
            # built the Transport (auth failures, host-key rejection), only
            # close() stops its thread and socket.
            client.close()
            raise
        return client

    try:
        client = attempt()
    except paramiko.BadHostKeyException as e:
        # The node presents a different host key than the one recorded.
        # Replace it and connect once more; a second mismatch propagates.
        logger.error(
            "host key for %s CHANGED (recorded %s, presented %s) — trusting the new one. "
            "Expected after a re-image; if this node was not re-imaged, stop and investigate.",
            _host_entry(host, port), _fingerprint(e.expected_key), _fingerprint(e.key),
        )
        _record_key(host, port, e.key, replace=True)
        client = attempt()

    transport = client.get_transport()
    if transport is not None:
        # `timeout` covers the connect only, and a blocking read on a silent
        # socket has nothing to time out against — SFTP transfers in
        # particular (upload_file) would otherwise wait on a dead peer forever.
        transport.set_keepalive(15)
    return client


def authorized_keys_command(pub_key: str) -> str:
    """Build the remote shell command that installs `pub_key` idempotently.

    Split out from ensure_key_installed so the shell logic — which has two
    non-obvious failure modes, both of which produce a node that *looks*
    bootstrapped and still refuses every key — can be tested without a node.
    """
    # Single-quoted in the remote shell, so only ' needs escaping.
    escaped = pub_key.replace("'", "'\\''")
    return (
        # sshd runs StrictModes by default: it walks up from authorized_keys
        # to the home directory and refuses the key if ANY component is group-
        # or world-writable. Images that ship with a 0775 home therefore accept
        # the key file happily and still reject every publickey login, with
        # nothing in the client output to say why.
        'chmod go-w "$HOME" && '
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        "touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
        # -x (whole-line), not bare -F: a substring match also matches a key
        # that was spliced onto another line, reporting "already present" for
        # an entry sshd cannot parse.
        f"(grep -qxF '{escaped}' ~/.ssh/authorized_keys && echo ALREADY_PRESENT "
        # Leading \n guards a file with no trailing newline — a plain append
        # would splice our key onto the end of the last entry, destroying that
        # key and ours. sshd ignores blank lines.
        f"|| (printf '\\n%s\\n' '{escaped}' >> ~/.ssh/authorized_keys && echo INSTALLED))"
    )


def ensure_key_installed(
    host: str,
    public_key_path: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    timeout: int = 15,
    password: str | None = None,
) -> tuple[bool, str]:
    """Append our public key to the node's authorized_keys, idempotently.

    Nodes that ship with only a factory password still have to be reachable by
    the bash scripts (setup/update/download-model/reset), and those only know
    ``ssh -i <key>``. Rather than making every script password-aware, we use
    the password once to install the key and let everything downstream stay
    key-based.

    Returns (ok, message). Never raises — a bootstrap failure should degrade
    to "keep using the password", not abort the caller.
    """
    pub_path = Path(public_key_path).expanduser()
    if not pub_path.exists():
        return False, f"public key not found: {pub_path}"

    try:
        pub_key = pub_path.read_text().strip()
    except OSError as e:
        return False, f"could not read {pub_path}: {e}"

    if not pub_key:
        return False, f"public key is empty: {pub_path}"

    # Shape check before it reaches a node. bootstrap_public_key is a free-form
    # config string: pointing it one character wrong (~/.ssh/id_ed25519 instead
    # of the .pub) appends the PRIVATE key to the authorized_keys of every
    # enabled target and still reports "installed", because publickey auth goes
    # on working via the real key and nothing else ever looks.
    single_line = len(pub_key.splitlines()) == 1
    looks_public = pub_key.startswith(("ssh-", "ecdsa-", "sk-ssh-", "sk-ecdsa-"))
    if not single_line or not looks_public or "PRIVATE KEY" in pub_key:
        return False, f"not an OpenSSH public key: {pub_path}"

    command = authorized_keys_command(pub_key)

    try:
        # run_command, not a raw exec_command: it reads the streams BEFORE
        # recv_exit_status(), which is what makes `timeout` bound the command
        # rather than just the connect. This runs inside Deployer.join_nodes,
        # once per node, so a bounded command is what keeps one node wedged on
        # `chmod go-w "$HOME"` from stalling the whole deploy.
        exit_code, out, err = run_command(host, command, port, user, key_file, timeout, password)
    except Exception as e:  # bootstrap is best-effort — never abort the caller
        return False, f"{type(e).__name__}: {e}"

    if exit_code != 0:
        return False, err.strip() or f"exit {exit_code}"
    return True, "already present" if "ALREADY_PRESENT" in out else "installed"


def test_connectivity(
    host: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    timeout: int = 15,
    password: str | None = None,
) -> tuple[bool, str]:
    """Test SSH connectivity to a host. Returns (reachable, message)."""
    try:
        with get_ssh_client(host, port, user, key_file, timeout, password) as client:
            _, stdout, _ = client.exec_command("hostname", timeout=10)
            hostname = stdout.read().decode().strip()
        return True, hostname
    except paramiko.AuthenticationException:
        return False, "authentication failed"
    except paramiko.SSHException as e:
        return False, f"SSH error: {e}"
    except TimeoutError:
        return False, "connection timed out"
    except FileNotFoundError as e:
        # get_ssh_client raises this when a key is the only credential and the
        # file is absent. FileNotFoundError subclasses OSError; catch it first
        # so a key_store/key_file misconfiguration is reported as configuration,
        # not as a network error that sends the operator after cabling.
        return False, str(e)
    except OSError as e:
        return False, f"network error: {e}"


def _exec_with_deadline(client, command: str, deadline: float):
    """Bound the SSH exec-request acknowledgement, which Paramiko otherwise waits for indefinitely."""
    result = queue.Queue(maxsize=1)
    def execute():
        try:
            result.put((True, client.exec_command(command, timeout=max(0.001, deadline - time.monotonic()))))
        except BaseException as exc:
            result.put((False, exc))
    worker = threading.Thread(target=execute, daemon=True, name="ssh-exec-request")
    worker.start()
    try:
        ok, value = result.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty as exc:
        client.close()  # Unblocks Paramiko's channel event wait.
        raise TimeoutError from exc
    if not ok:
        raise value
    return value


def run_command(
    host: str,
    command: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    timeout: int = 60,
    password: str | None = None,
) -> tuple[int, str, str]:
    """Drain both streams under one command deadline, retaining at most 8 MiB each.

    Connection/authentication have their own bounded timeouts. Once connected,
    channel opening, output and exit status share one monotonic deadline.
    A timeout closes the channel and returns 124 without echoing command secrets.
    """
    with get_ssh_client(host, port, user, key_file, timeout, password) as client:
        deadline = time.monotonic() + timeout
        channel = None
        output = [bytearray(), bytearray()]
        truncated = [False, False]
        limit = 8 * 1024 * 1024
        try:
            _, stdout, _ = _exec_with_deadline(client, command, deadline)
            channel = stdout.channel
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                drained = False
                for index, ready, recv in ((0, channel.recv_ready, channel.recv),
                                           (1, channel.recv_stderr_ready, channel.recv_stderr)):
                    if ready():
                        chunk = recv(65536)
                        drained = True
                        remaining = limit - len(output[index])
                        output[index].extend(chunk[:remaining])
                        truncated[index] |= len(chunk) > remaining
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                    code = channel.recv_exit_status()
                    break
                if not drained:
                    time.sleep(0.01)
        except TimeoutError:
            code = 124
            output[1].extend(f"\nSSH command timed out after {timeout}s".encode())
        finally:
            if channel is not None:
                channel.close()
        streams = [data.decode(errors="replace") + ("\n[output truncated]" if cut else "")
                   for data, cut in zip(output, truncated, strict=True)]
        return code, streams[0], streams[1]


# Prefix of the per-call fact delimiter; the rest is a random token, so a
# node-controlled file a snippet cats cannot contain the marker verbatim.
_FACT_MARKER_PREFIX = "===AUREAX_FACT_"


def probe_facts(
    host: str,
    facts: dict[str, str],
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    timeout: int = 60,
    password: str | None = None,
) -> dict[str, str]:
    """Run several probe snippets over ONE SSH connection.

    ``facts`` maps names to shell snippets; the result maps the same names to
    each snippet's stripped stdout ("" when that snippet printed nothing —
    presence probes should echo a sentinel).

    RAISES on connection failure exactly like run_command, and also when the
    probe command itself fails or times out. Returning empty facts there is
    not survivable: lifecycle.py reads an empty ``k3s_env`` as "this node never
    joined" and re-runs setup-all.sh (5-15 min) against a node that was merely
    slow, while cli.status reports the same node as healthy. Both callers wrap
    this in `except Exception` and classify a raise as *unreachable*.

    A single snippet's own failure is still invisible: each is wrapped in
    ``|| true``, so a denied `sudo cat` returns rc 0 with no output.

    Exists because an SSH handshake against an edge node's small CPU costs
    real time, and callers needing two or three facts per node were paying it
    per fact, per node, serially across the fleet.
    """
    marker = f"{_FACT_MARKER_PREFIX}{secrets.token_hex(16)}:"
    script = "; ".join(
        f"printf '%s\\n' {shlex.quote(marker + key + '===')}; "
        f"{{ {snippet} ; }} 2>/dev/null || true"
        for key, snippet in facts.items()
    )
    rc, out, err = run_command(
        host, script, port=port, user=user, key_file=key_file,
        timeout=timeout, password=password,
    )
    if rc != 0:
        # 124 is run_command's timeout; anything else non-zero (including the
        # -1 recv_exit_status() reports for a channel closed without a status)
        # is a transport failure. Both are "we did not get an answer", never
        # "the answer was empty".
        detail = err.strip() or f"rc={rc}"
        if rc == 124:
            raise TimeoutError(f"fact probe of {host} failed: {detail}")
        raise RuntimeError(f"fact probe of {host} failed: {detail}")

    results = dict.fromkeys(facts, "")
    current: str | None = None
    lines: list[str] = []
    for line in out.splitlines():
        name = None
        if line.startswith(marker) and line.endswith("==="):
            name = line[len(marker):-3]
        # The marker carries a per-call random token, so node-controlled
        # content (cli.status and lifecycle._start_phase_verify_nodes both cat
        # the k3s env file this way) cannot spell it. The request-set check is
        # defence in depth: only a marker naming a fact the caller ASKED for
        # opens a new value; anything else is data.
        if name is not None and name in results:
            if current is not None:
                results[current] = "\n".join(lines).strip()
            current = name
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        results[current] = "\n".join(lines).strip()
    return results


def upload_file(
    host: str,
    local_path: Path,
    remote_path: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    timeout: int = 15,
    password: str | None = None,
    mode: int | None = None,
    parent_mode: int | None = None,
) -> None:
    """Upload a file to a remote host via SFTP.

    ``mode`` tightens the remote file's permissions right after the put. There
    is a sub-second window where the file sits at the SFTP-default mode — for a
    secret-bearing file the caller should also keep its staging life short
    (Deployer.deploy_configs installs it and deletes the stage in the next
    command).

    ``parent_mode`` creates ``remote_path``'s parent directory at that mode
    first, and tightens it if it already exists. A relative ``remote_path``
    resolves against the SFTP start directory, i.e. the login user's home. That
    is the point for a secret-bearing file: /tmp is world-writable, so a fixed
    name there can be pre-created by any local account, and the put then writes
    into a file somebody else owns and can read.

    ``timeout`` bounds each channel read as well as the connect — without it a
    node that stops responding mid-put blocks the caller forever. It is a
    per-read budget, not a whole-transfer one; the 15s default suits the small
    config and script files this pushes, not a model-sized transfer.
    """
    with (
        get_ssh_client(host, port, user, key_file, timeout, password) as client,
        client.open_sftp() as sftp,
    ):
        sftp.get_channel().settimeout(timeout)
        _sftp_put(sftp, local_path, remote_path, mode, parent_mode)


def upload_files(
    host: str,
    items,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    timeout: int = 15,
    password: str | None = None,
) -> None:
    """Upload several files over ONE connection and ONE SFTP channel.

    ``items`` is an iterable of ``(local_path, remote_path, mode, parent_mode)``
    tuples with upload_file's meaning for each field. Exists because an SSH
    handshake against an edge node's small CPU costs real time and
    Deployer.deploy_configs pushes four files per node; ``timeout`` is the same
    per-read budget as upload_file's.
    """
    with (
        get_ssh_client(host, port, user, key_file, timeout, password) as client,
        client.open_sftp() as sftp,
    ):
        sftp.get_channel().settimeout(timeout)
        for local_path, remote_path, mode, parent_mode in items:
            _sftp_put(sftp, local_path, remote_path, mode, parent_mode)


def _sftp_put(sftp, local_path: Path, remote_path: str, mode: int | None, parent_mode: int | None) -> None:
    if parent_mode is not None:
        parent = str(PurePosixPath(remote_path).parent)
        if parent not in ("", ".", "/"):
            try:
                sftp.mkdir(parent, parent_mode)
            except OSError:
                # Already there. Tighten it rather than trusting whatever
                # mode (or owner) an earlier run or another user left.
                sftp.chmod(parent, parent_mode)
    sftp.put(str(local_path), remote_path)
    if mode is not None:
        sftp.chmod(remote_path, mode)


