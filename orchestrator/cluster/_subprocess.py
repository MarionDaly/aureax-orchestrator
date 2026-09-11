"""Shared subprocess wrapper for k3d/kubectl modules."""

from __future__ import annotations

import subprocess


class CommandError(subprocess.CalledProcessError):
    """CalledProcessError whose message carries the captured stderr.

    Every kubectl/k3d call here captures stderr, and the stock exception
    prints only the argv and exit status — which reduces a check=True failure
    to an exit code, discarding the line that says what kubectl objected to.
    Subclassing keeps every `except CalledProcessError` handler working.
    """

    def __str__(self) -> str:
        detail = (self.stderr or "").strip()
        base = super().__str__()
        return f"{base} stderr: {detail}" if detail else base


def run(
    args: list[str], check: bool = True, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Run a CLI command, capture stdout/stderr as text, return the result.

    With check=True (the default) a non-zero exit raises ``CommandError``, a
    ``CalledProcessError`` subclass whose str() carries the captured stderr;
    check=False returns the result for the caller to inspect.

    ``timeout`` (seconds) is opt-in per call rather than a module default:
    ``k3d cluster create``/``start`` routinely run for minutes on a first image
    pull, so a blanket ceiling would abort exactly the calls that legitimately
    take longest. Pass it on the calls that must be quick (a cluster list, a
    docker exec, a route lookup). Exceeding it raises ``CommandError`` with rc
    124 — coreutils timeout(1)'s convention, matching ssh.run_command —
    whatever ``check`` says, because a command that never returned has no
    result to inspect.

    A binary that is not on PATH raises ``CommandError`` with rc 127 — the
    shell's convention — for the same reason: ``k3d``, ``kubectl`` and
    ``docker`` are external tools an orchestrator may simply not have
    installed, and every caller here already handles ``CommandError`` while a
    bare ``FileNotFoundError`` escapes `aureax stop`/`start` as a traceback.
    """
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise CommandError(124, args, e.stdout or "", f"timed out after {timeout}s") from e
    except FileNotFoundError as e:
        raise CommandError(
            127, args, "", f"{args[0]} not found on PATH — install it"
        ) from e
    if check and result.returncode != 0:
        raise CommandError(result.returncode, args, result.stdout, result.stderr)
    return result
