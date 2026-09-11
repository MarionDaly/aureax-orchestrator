"""K3s agent lifecycle over SSH: joining a node, and reading/reconciling the
K3S_URL its k3s-agent is configured with."""

from __future__ import annotations

import contextlib
import os
import shlex
import tempfile
from pathlib import Path

from orchestrator.deploy.ssh import run_command, stage_path, upload_file

AGENT_ENV_PATH = "/etc/systemd/system/k3s-agent.service.env"

# run_command reports its own timeout as coreutils' 124 rather than raising,
# and 124 is the one status no remote shell here produces. It is therefore the
# marker for "the command never finished", which means its trailing `rm -f`
# never ran and the stage file it was meant to remove is still on the node.
_TIMEOUT_RC = 124

# Floor for the short steps here — the env-file read, the stage upload. It is
# also the schema default for ``ssh.connect_timeout_seconds``
# (orchestrator/config/schema.py::SSHConfig), which is the value a caller
# raises for a slow fleet and passes back in as ``connect_timeout``: paramiko
# and run_command take ONE number for the connect leg and the command, so these
# short budgets are what decide whether a node is reachable at all.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 15


def _stage_secret(host: str, content: str, prefix: str, conn: dict) -> str:
    """Upload ``content`` as a fresh 0600 stage file; return its shell word.

    The local temp file is mkstemp'd (0600) and unlinked as soon as the put
    returns, so the secret is never on this machine's disk at the umask
    default and never outlives the upload. Removing the REMOTE copy is the
    caller's job — see ``_sweep_stage``.
    """
    stage, remote = stage_path(prefix)
    fd, local_tmp = tempfile.mkstemp(prefix="aureax-stage-")  # mkstemp is 0600
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        upload_file(host, Path(local_tmp), stage, mode=0o600, parent_mode=0o700, **conn)
    finally:
        os.unlink(local_tmp)
    return remote


def _sweep_stage(host: str, remotes: list[str], conn: dict) -> None:
    """Best-effort removal of stage files a failed run left behind.

    The happy path removes them inside the same command that consumes them
    (``rm -f`` after the run, so an `exit 100` still gets cleaned up); this
    covers the paths where that command never ran — an upload that failed
    after an earlier one landed, or a run_command that timed out. Never
    raises: it runs from a ``finally`` whose job is to preserve the real
    error, and a stage file that outlives one failed join is a smaller
    problem than losing the reason the join failed.
    """
    if not remotes:
        return
    with contextlib.suppress(Exception):
        run_command(host, f"rm -f {' '.join(remotes)}", timeout=15, **conn)


def join_agent(
    host: str,
    server_url: str,
    token: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    password: str | None = None,
    timeout: int = 120,
    k3s_version: str | None = None,
) -> tuple[bool, str]:
    """Install K3s agent on a node and join it to the cluster.

    Returns (success, message) and never raises: Deployer.join_nodes loops
    over the fleet reporting per-node results, so a connect error escaping
    here aborted the whole deployment on the first unreachable node.

    ``timeout`` bounds the join itself — the install-and-join is the long call
    here; the is-active probe keeps its own short budget. Accepting the kwarg
    at all is what lets a caller spread one ``ssh.conn_kwargs(..., timeout=N)``
    dict over every helper in this package without a TypeError.

    ``k3s_version`` pins the agent to the version the SERVER already runs
    (``orchestrator/cluster/kubectl.py::server_version``). Without it the
    installer takes latest stable, so nodes joined weeks apart end up with
    kubelets newer than the API server — outside the supported skew, and
    surfacing much later as an unexplained NotReady node. None keeps the
    unpinned install, which is what a cluster that cannot answer /version gets;
    join-node.sh warns on that path.
    """
    conn = {"port": port, "user": user, "key_file": key_file, "password": password}
    # Remote stage files this call is still responsible for. Emptied once the
    # join command — which removes them itself — has demonstrably run; whatever
    # is left is swept in the `finally`.
    staged: list[str] = []

    try:
        # `is-active` prints inactive/activating/deactivating, all of which
        # contain the substring "active"; those states exit non-zero, so the
        # `not-running` sentinel from the `||` branch is what makes the
        # substring test safe. Do not drop either half.
        _, stdout, _ = run_command(
            host,
            "sudo systemctl is-active k3s-agent 2>/dev/null || echo 'not-running'",
            timeout=15,
            **conn,
        )
        if "active" in stdout and "not-running" not in stdout:
            return True, "K3s agent already running"

        # Upload the join script
        from orchestrator.resources import resource_root
        join_script = resource_root() / "scripts" / "join-node.sh"
        if not join_script.exists():
            return False, f"Join script not found: {join_script}"

        stage, remote = stage_path("join-node")
        # 0700 from the first byte: this file is executed as root moments
        # later, so it must never be writable by anyone but the login user.
        upload_file(host, join_script, stage, mode=0o700, parent_mode=0o700, **conn)
        staged.append(remote)

        # THE TOKEN IS A FILE, NEVER AN ARGV WORD. It is the one credential
        # that lets any machine join this cluster, and an argv word is public
        # twice over: /proc/<pid>/cmdline is world-readable (both DaemonSets
        # set hostPID: true) and sudo copies the whole command line into the
        # node's auth log, where it outlives the join. join-node.sh's
        # --token-file reads the path given here; the trailing newline is what
        # its `IFS= read -r` consumes.
        token_remote = _stage_secret(host, token + "\n", "k3s-token", conn)
        staged.append(token_remote)

        # Run the join script, then remove BOTH stages whatever it exited with
        # — `exit 100` (reboot needed) is load-bearing, so the rc is preserved,
        # and the token file must not survive a failed join.
        version_arg = f" --k3s-version {shlex.quote(k3s_version)}" if k3s_version else ""
        cmd = (
            f"sudo {remote} {shlex.quote(server_url)} --token-file {token_remote}"
            f"{version_arg}; "
            f"rc=$?; rm -f {remote} {token_remote}; exit $rc"
        )
        exit_code, stdout, stderr = run_command(host, cmd, timeout=timeout, **conn)
        if exit_code != _TIMEOUT_RC:
            # The remote shell ran its own `rm -f` for every status it can
            # return. Only run_command's synthetic 124 means the command never
            # reached that `rm -f`, so on that path the stages stay on the list
            # for the `finally` sweep — otherwise the token file outlives the
            # join in the login user's home.
            staged.clear()
    except Exception as e:
        return False, f"unreachable: {type(e).__name__}: {e}"
    finally:
        # Reached on the early returns and on the `except` above (a return
        # inside except still runs this), so a token staged by a run that then
        # timed out does not sit in the login user's home until the next join.
        _sweep_stage(host, staged, conn)

    if exit_code == 100:
        # exit 100 is join-node.sh's documented "reboot needed" status: cgroup
        # parameters were added to the boot config but are not live yet. This
        # is the only consumer of that code, so translate it here or it reads
        # as a generic failure.
        return False, (
            "node needs a reboot to enable cgroups — run ./scripts/setup-all.sh, "
            "which handles the reboot-and-rejoin flow"
        )
    if exit_code != 0:
        return False, f"Join failed (exit {exit_code}): {stderr}"

    return True, "K3s agent joined successfully"


def _parse_k3s_url(env_text: str) -> str | None:
    """Extract the K3S_URL value from the k3s-agent env file text.

    Handles lines like:  K3S_URL='https://192.0.2.1:6443'
                   or:   K3S_URL="https://192.0.2.1:6443"
                   or:   K3S_URL=https://192.0.2.1:6443
    Returns the URL without surrounding quotes, or None if not found.

    The LAST assignment wins, because that is the one systemd's EnvironmentFile
    applies. Reading an earlier line instead makes reconcile_server_url compare
    a stale value to the expected URL, report "already correct" and leave the
    agent pointing somewhere else — a node `heal-nodes` then calls healthy. (The
    file this module writes cannot get divergent duplicates: _rewrite_env_text
    replaces every K3S_URL line. A hand-edited file can.)
    """
    found: str | None = None
    for raw in env_text.splitlines():
        line = raw.strip()
        if not line.startswith("K3S_URL="):
            continue
        value = line[len("K3S_URL="):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        found = value
    return found


def get_configured_server_url(
    host: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    password: str | None = None,
    timeout: int = 15,
) -> str | None:
    """Read the K3S_URL currently configured on a node's k3s-agent.

    Returns the configured URL string, or None when the env file is missing or
    unparseable. RAISES on SSH failure and on a probe that never answered —
    callers that tolerate an unreachable host must wrap this in try/except (see
    the drift loop in orchestrator/dashboard/watchers.py, which catches per node
    so one unreachable box does not abort the scan). cli.status and
    lifecycle._start_phase_verify_nodes deliberately bypass this helper: they
    batch several facts into one connection with probe_facts + _parse_k3s_url.
    """
    exit_code, stdout, stderr = run_command(
        host,
        f"sudo cat {AGENT_ENV_PATH} 2>/dev/null || true",
        port=port,
        user=user,
        key_file=key_file,
        password=password,
        timeout=timeout,
    )
    if exit_code != 0:
        # The remote command ends in `|| true`, so a read that COMPLETED always
        # exits 0 — a non-zero status here means no answer, not an empty one.
        # `sudo` routinely stalls tens of seconds on a node with incomplete DNS,
        # and collapsing that into the same None as "no env file" lets the drift
        # checker score an unread node as clean and announce "Drift cleared".
        detail = stderr.strip() or f"exit {exit_code}"
        message = f"could not read {AGENT_ENV_PATH} on {host}: {detail}"
        if exit_code == _TIMEOUT_RC:
            raise TimeoutError(message)
        raise RuntimeError(message)
    if not stdout.strip():
        return None
    return _parse_k3s_url(stdout)


def _rewrite_env_text(env_text: str, new_url: str) -> str:
    """Return the env file text with the K3S_URL line replaced.

    Preserves all other lines. If no K3S_URL line is present, appends one.
    Always quotes the new URL with single quotes.
    """
    new_line = f"K3S_URL='{new_url}'"
    lines = env_text.splitlines()
    replaced = False
    out = []
    for raw in lines:
        if raw.strip().startswith("K3S_URL="):
            out.append(new_line)
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        out.append(new_line)
    # Preserve trailing newline if original had one
    trailing = "\n" if env_text.endswith("\n") else ""
    return "\n".join(out) + trailing


def reconcile_server_url(
    host: str,
    expected_url: str,
    port: int = 22,
    user: str = "aureax",
    key_file: str | None = None,
    password: str | None = None,
    timeout: int = 60,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> tuple[bool, str]:
    """Ensure k3s-agent on `host` points at `expected_url`.

    Reads the current env file, rebuilds it in Python (avoids sed quoting
    pitfalls), SFTPs it back and installs it as root, then daemon-reload +
    restart k3s-agent.

    Returns (ok, message) and never raises. `heal-nodes` exists precisely for
    the case where nodes are unreachable, and it loops over every target
    counting failures — letting a connect error escape would abort that loop on
    the first bad node and leave the rest unhealed.

    ``timeout`` bounds the install-and-restart, which is the long call.
    ``connect_timeout`` is the floor for the short steps before it — the env
    read and the stage upload — and is where ``ssh.connect_timeout_seconds``
    lands: those steps carry this call's connect leg, and heal-nodes runs
    exactly when nodes are slow to answer, so a raised connect budget has to
    reach them or the node is reported unreachable at the default.
    """
    conn = {"port": port, "user": user, "key_file": key_file, "password": password}
    short = max(DEFAULT_CONNECT_TIMEOUT_SECONDS, connect_timeout)

    # 1. Read the current env file
    try:
        exit_code, stdout, stderr = run_command(
            host,
            f"sudo cat {AGENT_ENV_PATH} 2>/dev/null || true",
            timeout=short,
            **conn,
        )
    except Exception as e:
        return False, f"unreachable: {type(e).__name__}: {e}"
    if exit_code != 0:
        return False, f"could not read {AGENT_ENV_PATH}: {stderr.strip() or 'exit ' + str(exit_code)}"
    if not stdout.strip():
        # The `2>/dev/null || true` read reports a missing or unreadable file
        # as exit 0 with empty output. Rebuilding from "" would write a lone
        # K3S_URL line — destroying K3S_TOKEN — and then restart the agent.
        return False, (
            f"{AGENT_ENV_PATH} is empty or unreadable — refusing to rewrite "
            "(rebuilding from an empty read would drop K3S_TOKEN); "
            "the node likely needs a fresh join via setup-all.sh"
        )
    current = _parse_k3s_url(stdout)
    if current == expected_url:
        return True, f"already correct ({expected_url})"

    # 2. Build the new contents
    new_text = _rewrite_env_text(stdout, expected_url)

    # 3. SFTP the rebuilt file up and install it as root. NOT `printf '%s'
    #    '<text>' | sudo tee`: this file holds K3S_TOKEN (see the guard above),
    #    and that puts the whole cluster credential into the remote shell's
    #    argv, readable via `ps` by every local account on the node — to change
    #    one non-secret URL line. A sed on K3S_URL would leak nothing but
    #    silently drops the append-when-absent case _rewrite_env_text handles.
    try:
        remote = _stage_secret(host, new_text, "k3s-agent-env", {**conn, "timeout": short})
    except Exception as e:
        return False, f"could not stage {AGENT_ENV_PATH}: {type(e).__name__}: {e}"

    cmd = (
        f"sudo install -m 600 -o root -g root {remote} {AGENT_ENV_PATH} "
        f"&& sudo systemctl daemon-reload "
        f"&& sudo systemctl restart k3s-agent"
    )
    # The stage carries K3S_TOKEN, so it is removed whatever the install did.
    cmd = f"{cmd}; rc=$?; rm -f {remote}; exit $rc"
    staged = [remote]
    try:
        exit_code, _, stderr = run_command(host, cmd, timeout=timeout, **conn)
        if exit_code != _TIMEOUT_RC:
            staged.clear()  # the remote `rm -f` ran for every status a shell returns
    except Exception as e:
        return False, f"rewrite/restart failed: {type(e).__name__}: {e}"
    finally:
        # A run_command that timed out (rc 124) or raised never reached its own
        # `rm -f`, and this stage holds K3S_TOKEN.
        _sweep_stage(host, staged, conn)
    if exit_code != 0:
        return False, f"rewrite/restart failed: {stderr.strip() or 'exit ' + str(exit_code)}"

    prev = current if current else "(unset)"
    return True, f"rewrote K3S_URL: {prev} -> {expected_url} and restarted k3s-agent"
