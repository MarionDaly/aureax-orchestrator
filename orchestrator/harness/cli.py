"""Aureax MCP Testing Harness — lightweight CLI for testing edge node MCP servers.

Connects directly to a node's MCP server over streamable-http and lets you
browse tools, call them with custom arguments, send inference prompts, and
see the full output. No MCP host or LLM needed on the orchestrator side —
just a pure MCP client connection.

Every node speaks HTTPS with a certificate the fleet CA signed, and every
MCP-serving node may require a bearer key, so each command takes an endpoint
(a hostname/IP **or** an enabled node id from the fleet config), a CA bundle
and a key. NODE below is either spelling.

Usage:
    aureax-harness ping <NODE>
    aureax-harness tools <NODE>
    aureax-harness call <NODE> get_health
    aureax-harness call <NODE> read_sensor --arg sensor_type=temperature
    aureax-harness infer <NODE> "What is 2+2?"
    aureax-harness sensor <NODE> --type all
    aureax-harness connect <NODE>

Given a node id the harness reads that node's host from the fleet config and
its MCP key from the orchestrator keystore, so the common case is a bare
``aureax-harness ping shard-head``. See _resolve_endpoint for the precedence.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import signal
import ssl
import sys
import threading
import time
from pathlib import Path

import click
import httpx
from ruamel.yaml.error import YAMLError

from orchestrator.config.cache import load_network_config_cached
from orchestrator.config.loader import DEFAULT_NETWORK_CONFIG, ConfigError
from orchestrator.config.schema import MCP_NODE_PORT

# The tree's single IPv6-bracketing rule for a URL authority — see _make_url.
from orchestrator.config.templating import _url_host

DEFAULT_PORT = MCP_NODE_PORT

# Seconds of SILENCE tolerated on the MCP stream — see _make_http_client.
DEFAULT_READ_TIMEOUT = 300.0

# The MCP bearer key, read from the environment so it never becomes an argv
# word: argv is readable by any local process for the life of the run, which is
# the disclosure channel this codebase closes everywhere else (--api-key exists
# for one-offs and says so in its own help).
ENV_API_KEY = "AUREAX_MCP_API_KEY"

# The CA bundle that must have signed the node's certificate. Same variable
# scripts/lib/common.sh exports for curl_tls, so one export covers both.
ENV_CA = "AUREAX_CA"

# What counts as "the endpoint is unreachable" when deciding whether the
# endpoint hint applies. httpx.TransportError covers ConnectError,
# ConnectTimeout, ReadTimeout and the rest of the transport family; OSError
# covers a socket that fails before httpx wraps it.
#
# A TLS verification failure is BOTH of those and is not one of these cases —
# see _tls_failure, which _is_connection_failure defers to.
_CONNECTION_ERRORS = (httpx.TransportError, OSError)

# What the last resolved endpoint was, for the failure hints in `main`. `main`
# is the single renderer for exceptions raised anywhere in six commands, and a
# TLS hint that cannot name the CA it verified against or the node to re-apply
# is not actionable; the alternative is six copies of the same try/except.
_CONTEXT: dict[str, str] = {}


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flatten an ExceptionGroup into the real exceptions it carries.

    The MCP SDK runs its transport inside a task group, so a refused
    connection, a DNS failure and an HTTP 404 on /mcp all arrive as
    ``ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)`` — a
    message that names no cause and makes the three indistinguishable. The
    cause lives only in ``.exceptions``, so every error path in this file
    formats the leaves rather than the group.
    """
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaf_exceptions(sub)]
    return [exc]


def _describe(exc: BaseException) -> str:
    """Render an exception — or an ExceptionGroup's leaves — as `Type: message`."""
    return "; ".join(f"{type(e).__name__}: {e}" for e in _leaf_exceptions(exc))


def _tls_failure(exc: BaseException) -> ssl.SSLError | None:
    """The ssl error behind a transport failure, if there is one.

    httpcore maps ``ssl.SSLError`` (an OSError) onto its own ``ConnectError``
    with ``raise to_exc(exc) from exc``, and httpx re-wraps that the same way,
    so a certificate the fleet CA did not sign arrives in exactly the shape a
    refused connection does: ``ExceptionGroup(httpx.ConnectError)``. Measured
    against httpx 0.28.1: ``httpx.ConnectError -> httpcore.ConnectError ->
    ssl.SSLCertVerificationError``, two ``__cause__`` links down.

    Without this walk the harness answers a node that DID answer — and whose
    only problem is an untrusted certificate — with "MCP server not reachable"
    plus advice about pods still loading, sending the operator to look at the
    node instead of at the CA on this Mac.
    """
    for leaf in _leaf_exceptions(exc):
        seen: set[int] = set()
        cur: BaseException | None = leaf
        # __cause__ then __context__, guarded against a cycle: an exception
        # re-raised inside its own handler can point back at itself, and this
        # runs on the failure path where a hang is worst.
        while cur is not None and id(cur) not in seen:
            if isinstance(cur, ssl.SSLError):
                return cur
            seen.add(id(cur))
            cur = cur.__cause__ or cur.__context__
    return None


def _auth_failure(exc: BaseException) -> httpx.HTTPStatusError | None:
    """The 401 behind a failed MCP call, if there is one.

    The MCP port takes a bearer key, and a key the node rejects arrives as an
    ``httpx.HTTPStatusError`` leaf inside the SDK's task-group ExceptionGroup —
    a shape that is neither a transport error nor a TLS error, so without this
    it fell through to "MCP server not reachable". The server was reached and
    answered; it simply refused the key, and the remedies (wait for the pod /
    supply the right key) are unrelated.

    Leaf-only, unlike _tls_failure: httpx raises this from
    ``response.raise_for_status()`` and nothing re-wraps it, so there is no
    ``__cause__`` chain to walk.
    """
    for leaf in _leaf_exceptions(exc):
        if isinstance(leaf, httpx.HTTPStatusError) and leaf.response.status_code == 401:
            return leaf
    return None


def _is_connection_failure(exc: BaseException) -> bool:
    """True when any leaf is a transport failure, i.e. the endpoint hint applies.

    TLS verification failures are excluded even though they are transport
    errors: they have their own hint, and the two remedies are opposites (wait
    for the pod / fix the CA on this Mac).
    """
    if _tls_failure(exc) is not None:
        return False
    return any(isinstance(e, _CONNECTION_ERRORS) for e in _leaf_exceptions(exc))


def _tls_hint() -> str:
    """The remedy line for a certificate the configured CA did not sign."""
    ca = _CONTEXT.get("ca", "the configured CA")
    host = _CONTEXT.get("host", "the node")
    node = _CONTEXT.get("node_id") or "<node-id>"
    # --orchestrator-ip is spelled out because `aureax apply` requires it: a
    # remedy line printed at the moment verification fails has to be a command
    # that runs, not one Click rejects with exit 2.
    return (
        f"TLS verification failed against {ca} for {host} — run "
        f"`uv run aureax apply --node {node} --orchestrator-ip <orchestrator-ip>` to push the "
        f"current CA and certificate to the node, or point --ca / ${ENV_CA} at the CA that "
        f"signed the one it is serving."
    )


def _auth_hint() -> str:
    """The remedy line for an MCP key the node rejected."""
    return (
        "Pass the node id instead of its address so the key is read from "
        f"~/.aureax/keys/<node_id>.mcp.key, or give --api-key / --key-file / ${ENV_API_KEY}."
    )


class _Spinner:
    """Simple elapsed-time spinner for long-running operations.

    Shows a cycling animation with elapsed time so the user knows the
    harness isn't hung. Runs in a background thread to avoid blocking
    the async event loop.

    Silent when stderr is not a terminal. The animation repaints one line with
    `\\r` and clears it with an ANSI escape, neither of which a redirected
    stderr collapses: generation on the sharded fleet runs for minutes, so a
    1024-token generation would leave several thousand literal frames in the
    file the caller redirected to.

    Usage:
        with _Spinner("Generating"):
            await long_operation()
    """

    FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, message: str = "Working") -> None:
        """Set the spinner message displayed during the operation."""
        self._message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _Spinner:
        """Start the background spinner thread, unless stderr is not a terminal."""
        if not sys.stderr.isatty():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop the spinner and clear the line it was repainting."""
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join()
        self._thread = None
        # Clear the spinner line
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def _spin(self) -> None:
        """Animate the spinner until stopped."""
        start = time.time()
        i = 0
        while not self._stop.wait(0.15):
            elapsed = int(time.time() - start)
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stderr.write(f"\r  {frame} {self._message}... {elapsed}s")
            sys.stderr.flush()
            i += 1


# ---------------------------------------------------------------------------
# Async helpers — these do the actual MCP client work
# ---------------------------------------------------------------------------

def _make_http_client(
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    api_key: str | None = None,
    verify: ssl.SSLContext | None = None,
) -> httpx.AsyncClient:
    """Create an httpx client with timeouts sized for LLM inference.

    The MCP SDK's own 30s default is far too short for run_inference on an
    edge node: generation on the shipped sharded fleet runs at a few tokens
    per second at best, so `infer`'s 1024-token default is a generation of
    many minutes.

    ``read`` bounds ONE read, not the whole call, so a streaming response
    keeps resetting it — this is a silence budget (long enough to cover
    prefill on a cold head), not a total-duration budget. Raise it with
    --timeout on a fleet that goes quiet for longer.

    ``connect`` is 5s, matching client/mcp/pool.py: an unready MCP pod behind
    the NodePort DROPs the SYN rather than refusing it, so the connect leg
    always runs to its full budget and the SDK's 30s default makes every dial
    of a cold node a 30s wait. A LAN handshake is sub-second and 5s still
    covers two SYN retransmits. write/pool stay at 30s.

    This is also the ONE place the Authorization header and the CA bundle are
    applied: ``streamable_http_client`` takes no headers of its own, so a
    pre-built client is the SDK's supported path for both, and every command
    reaches the network through here.

    ``verify=None`` falls back to httpx's own default (the system trust
    store), which the fleet's private CA is not in — so it fails verification
    rather than connecting in the clear. Every command passes a context built
    from --ca; the fallback exists only for callers that build a client
    directly.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0, read=read_timeout),
        follow_redirects=True,
        verify=verify if verify is not None else True,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
    )


@contextlib.asynccontextmanager
async def _mcp_session(
    url: str,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    api_key: str | None = None,
    verify: ssl.SSLContext | None = None,
):
    """Open an initialized MCP session against `url` and yield it.

    The `async with` nesting is load-bearing: it keeps the MCP SDK's
    background tasks (GET stream, post writer) alive for the full duration of
    every call made through the session. Flattening it tears them down
    mid-call, which run_inference — minutes long against a sharded head —
    hits every time.

    One place, so the bearer header, the CA context, a terminate_on_close flip
    or a retry is one edit rather than five.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with (
        _make_http_client(read_timeout, api_key, verify) as http_client,
        streamable_http_client(
            url, http_client=http_client, terminate_on_close=False
        ) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def _unwrap(result) -> tuple[bool, str]:
    """(is_error, text) from an MCP tool result.

    The single unwrap point: `_call_tool`, which returns a dict, and
    `_print_tool_result`, which renders, both go through it, so the two cannot
    disagree about what a failed tool call looks like.
    """
    text = result.content[0].text if result.content else ""
    if result.isError:
        return True, text or "unknown error"
    return False, text


def _tool_failed(result) -> bool:
    """True when `_call_tool` returned a failure shape.

    Two shapes arrive here: ``{"error": True, "content": <failure text>}`` for
    a call the server marked isError, and ``{"error": "<message>"}`` for a tool
    that caught its own exception and returned it as an ordinary result —
    run_inference does exactly that whenever llama-server is not answering.
    Both have to set a non-zero exit status, because scripts/test-all.sh grades
    harness invocations on exit status alone: a failure reported only as red
    text on stdout is counted as a pass.
    """
    return isinstance(result, dict) and bool(result.get("error"))


async def _ping(url: str, api_key: str | None = None,
                verify: ssl.SSLContext | None = None) -> BaseException | None:
    """Ping a node's MCP server; return None on success, else the failure.

    The exception is returned rather than reduced to a bool because the
    failures are not interchangeable: a shard worker serves no MCP at all and
    never will, a head is unreachable for minutes while it loads its model,
    and a node whose certificate this CA did not sign answered fine. `ping`
    prints the leaf cause under its FAILED line so the three can be told apart.
    """
    try:
        async with _mcp_session(url, api_key=api_key, verify=verify) as session:
            await session.send_ping()
            return None
    except Exception as e:
        return e


async def _list_tools(url: str, api_key: str | None = None,
                      verify: ssl.SSLContext | None = None) -> list[dict]:
    """List all tools on a node with their schemas."""
    async with _mcp_session(url, api_key=api_key, verify=verify) as session:
        result = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in result.tools
        ]


async def _call_tool(url: str, tool_name: str, args: dict,
                     read_timeout: float = DEFAULT_READ_TIMEOUT,
                     api_key: str | None = None,
                     verify: ssl.SSLContext | None = None) -> dict:
    """Call a tool on a node and return the parsed result."""
    async with _mcp_session(url, read_timeout, api_key, verify) as session:
        is_error, text = _unwrap(await session.call_tool(tool_name, args))
        if is_error:
            return {"error": True, "content": text}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}


def _exit_on_interrupt() -> None:
    """End the REPL on Ctrl-C without waiting on the prompt thread."""
    sys.stderr.write("\nDisconnected.\n")
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(130)


async def _interactive(url: str, read_timeout: float = DEFAULT_READ_TIMEOUT,
                       api_key: str | None = None,
                       verify: ssl.SSLContext | None = None,
                       label: str | None = None) -> None:
    """Run an interactive REPL session connected to a node.

    Opens a single persistent MCP session for the duration of the REPL; every
    command reuses that one session.

    Ctrl-C is handled here, on the event loop, and it ends the process
    outright. The prompt runs in a worker thread blocked in `input()`, and a
    thread in that state can be neither cancelled nor interrupted: both
    `asyncio.Runner.close` and the interpreter's own ThreadPoolExecutor
    shutdown hook JOIN it, so an ordinary KeyboardInterrupt unwind waits for a
    line the user has already given up on typing. `os._exit` is the one exit
    that does not join it.
    """
    loop = asyncio.get_running_loop()
    # Signal handlers are not available on every event loop; without one,
    # Ctrl-C falls back to the KeyboardInterrupt path in `connect`.
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, _exit_on_interrupt)
    async with _mcp_session(url, read_timeout, api_key, verify) as session:
        await _repl_loop(session, label or url)


async def _repl_loop(session, label: str) -> None:
    """Inner REPL loop — separated from _interactive to keep indentation clean.

    ``label`` is display only (the URL, plus the node id when the positional
    resolved to one); every call goes through the already-open ``session``.
    """
    tools_result = await session.list_tools()
    tools = {t.name: t for t in tools_result.tools}

    click.echo(click.style(f"Connected to {label}", fg="green"))
    click.echo(f"{len(tools)} tools available. Type 'help' for commands.\n")

    while True:
        try:
            # Off the loop. This REPL holds ONE MCP session open for its whole
            # life, and the SDK keeps that session alive with background tasks
            # (GET stream, post writer) that only run when the loop does —
            # blocking it at the prompt is blocking it for as long as the user
            # takes to type, which the server eventually reads as a dead client.
            line = (await asyncio.to_thread(input, "aureax> ")).strip()
        except EOFError:
            # Ctrl-D. Ctrl-C never reaches here: it is delivered to the main
            # thread, never to the worker blocked in input(), and
            # _interactive's loop-level handler ends the process on it.
            click.echo("\nDisconnected.")
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            click.echo("Disconnected.")
            break
        elif cmd == "help":
            _print_help()
        elif cmd == "tools":
            for t in tools.values():
                desc = t.description or ""
                if len(desc) > 60:
                    desc = desc[:57] + "..."
                click.echo(f"  {click.style(t.name, fg='cyan'):30s} {desc}")
        elif cmd == "schema":
            if not rest:
                click.echo("Usage: schema <tool_name>")
                continue
            tool = tools.get(rest.strip())
            if not tool:
                click.echo(f"Unknown tool: {rest}")
                continue
            click.echo(json.dumps(tool.inputSchema, indent=2))
        elif cmd == "ping":
            try:
                await session.send_ping()
                click.echo(click.style("PONG", fg="green"))
            except Exception as e:
                click.echo(click.style(f"FAILED: {_describe(e)}", fg="red"))
        elif cmd == "call":
            await _repl_call(session, tools, rest)
        elif cmd == "infer":
            if not rest:
                click.echo("Usage: infer <your prompt here>")
                continue
            await _repl_infer(session, rest)
        elif cmd == "sensor":
            sensor_type = rest.strip() if rest else "all"
            await _repl_call_tool(session, "read_sensor", {"sensor_type": sensor_type})
        else:
            click.echo(f"Unknown command: {cmd}. Type 'help' for commands.")


async def _repl_call(session, tools: dict, rest: str) -> None:
    """Handle the 'call' REPL command: call <tool> [key=value ...].

    Split with shlex, not str.split, so a quoted value survives as one
    argument. In the documented example

        call report_anomaly severity=high description="spike detected"

    whitespace splitting yields `description="spike` plus a stray `detected"`,
    truncating the description to one word with nothing reported.
    """
    try:
        parts = shlex.split(rest)
    except ValueError as e:
        # Unbalanced quotes. Say so rather than falling back to a split that
        # would quietly mangle the arguments.
        click.echo(click.style(f"Could not parse arguments: {e}", fg="red"))
        return
    if not parts:
        click.echo("Usage: call <tool_name> [key=value ...]")
        return

    tool_name = parts[0]
    if tool_name not in tools:
        click.echo(f"Unknown tool: {tool_name}")
        click.echo(f"Available: {', '.join(tools.keys())}")
        return

    args, rejected = _parse_kv_args(parts[1:])
    if rejected:
        # Warn rather than raise — an unusable token must not tear down the
        # one MCP session this REPL holds open.
        click.echo(click.style(
            f"Not key=value: {', '.join(rejected)} — nothing called.", fg="yellow"
        ))
        return
    await _repl_call_tool(session, tool_name, args)


async def _repl_infer(session, prompt: str) -> None:
    """Handle the 'infer' REPL command — shortcut for run_inference.

    The spinner is stopped BEFORE the response is printed, as in the `infer`
    command: the spinner repaints a single line and clears it on exit, so a
    response echoed while it still animates is overwritten mid-print.
    """
    try:
        with _Spinner("Generating"):
            result = await session.call_tool("run_inference", {"prompt": prompt})
        _print_tool_result(result)
    except Exception as e:
        # Rendering is inside the try with the call: an unexpected content
        # shape must not take down the one MCP session this REPL holds open.
        click.echo(click.style(f"Call failed: {_describe(e)}", fg="red"))


async def _repl_call_tool(session, tool_name: str, args: dict) -> None:
    """Call a tool and pretty-print the result."""
    try:
        _print_tool_result(await session.call_tool(tool_name, args))
    except Exception as e:
        click.echo(click.style(f"Call failed: {_describe(e)}", fg="red"))


def _print_tool_result(result) -> None:
    """Pretty-print one MCP tool result, as JSON where the text parses as JSON."""
    is_error, text = _unwrap(result)
    if is_error:
        click.echo(click.style(f"Error: {text}", fg="red"))
        return
    try:
        click.echo(json.dumps(json.loads(text), indent=2))
    except json.JSONDecodeError:
        click.echo(text)


def _parse_kv_args(parts: list[str]) -> tuple[dict, list[str]]:
    """Parse key=value pairs into a dict, coercing types.

    Handles: strings, integers, floats, booleans.
    Examples: sensor_type=temperature  max_tokens=512  temperature=0.3

    Returns ``(args, rejected)``, where `rejected` holds every token with no
    '='. Callers report them rather than dropping them: a dropped token issues
    the tool call with that argument missing, and the node then answers with a
    schema error naming nothing the user typed.
    """
    args = {}
    rejected = []
    for part in parts:
        if "=" not in part:
            rejected.append(part)
            continue
        key, value = part.split("=", 1)

        # Try to coerce to appropriate type
        if value.lower() in ("true", "false"):
            args[key] = value.lower() == "true"
        else:
            try:
                args[key] = int(value)
            except ValueError:
                try:
                    args[key] = float(value)
                except ValueError:
                    args[key] = value
    return args, rejected


def _print_help() -> None:
    """Print REPL help text."""
    click.echo("""
Commands:
  tools                          List all available tools
  schema <tool>                  Show input schema for a tool
  call <tool> [key=value ...]    Call a tool with arguments
  infer <prompt>                 Send a prompt to the node's LLM
  sensor [type]                  Read sensor data (default: all)
  ping                           Test connectivity
  help                           Show this help
  quit                           Disconnect and exit
""")


def _make_url(host: str, port: int) -> str:
    """Build the MCP endpoint URL from host and port.

    https only. Every node's MCP server is served with the certificate the
    fleet CA issued for it (ARCHITECTURE.md, "TLS"), and there is no plaintext
    listener to fall back to — an http:// dial would simply hang on a
    TLS handshake the server is waiting to start.

    The host goes through the tree's single IPv6-bracketing rule: a bare
    literal's own colons read as the port separator, and httpx raises
    InvalidURL on the result.
    """
    return f"https://{_url_host(host)}:{port}/mcp"


# ---------------------------------------------------------------------------
# Endpoint, key and CA resolution
# ---------------------------------------------------------------------------

def _default_ca_path() -> Path:
    """Where `aureax generate` writes the fleet CA.

    Resolved per call rather than at import so a redirected ``$HOME`` (the
    tests, a per-fleet HOME) is honoured, matching
    ``orchestrator.config.tlsstore.store_dir``. Spelled out here rather than
    imported from tlsstore so the harness keeps starting without cryptography.
    """
    return Path.home() / ".aureax" / "tls" / "ca.pem"


def _ca_path(explicit: str | None) -> Path:
    """--ca, else $AUREAX_CA, else the store default. Highest wins."""
    if explicit:
        return Path(explicit)
    from_env = os.environ.get(ENV_CA)
    if from_env:
        return Path(from_env)
    return _default_ca_path()


# Keyed on (path, mtime, size), like tlsstore.ca_ssl_context: `aureax tls
# regenerate --ca` inside a long-lived REPL must not leave the session
# verifying against a CA that is no longer on disk.
_ssl_contexts: dict[tuple[str, int, int], ssl.SSLContext] = {}


def _ca_context(explicit: str | None) -> ssl.SSLContext:
    """The SSLContext every request verifies against, from --ca / env / default.

    A missing or unreadable bundle is a ClickException naming the path, not a
    bare OSError: OSError is in _CONNECTION_ERRORS, so letting it reach `main`
    would diagnose a CA this Mac never generated as an unreachable node.
    """
    path = _ca_path(explicit)
    _CONTEXT["ca"] = str(path)
    try:
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        ctx = _ssl_contexts.get(key)
        if ctx is None:
            ctx = ssl.create_default_context(cafile=str(path))
            _ssl_contexts[key] = ctx
        return ctx
    except OSError as e:
        raise click.ClickException(
            f"cannot verify TLS against {path}: {e}. Run "
            f"`uv run aureax generate --orchestrator-ip <ip>` to create the fleet CA, or point "
            f"--ca / ${ENV_CA} at the bundle that signed the node's certificate."
        ) from e


def _lookup_target(host: str, fleet_config: str | None):
    """The enabled SSH target whose node_id is `host`, or None for a literal host.

    An unknown id is not an error: the positional has always taken an address,
    and a fleet file is optional on a machine that only ever talks to one node.
    An unreadable file the operator NAMED is an error, though — silently
    falling back to a literal host there would report a typo'd --config as an
    unreachable node.
    """
    path = Path(fleet_config) if fleet_config else DEFAULT_NETWORK_CONFIG
    try:
        config = load_network_config_cached(path)
    except (ConfigError, YAMLError, OSError, ValueError) as e:
        # YAMLError separately from ConfigError, which only wraps schema
        # failures: the parse itself happens outside the loader's try, so one
        # mistyped line in the fleet file arrives here as a ruamel error.
        if fleet_config:
            raise click.ClickException(f"--config {path}: {e}") from e
        return None
    return next((t for t in config.ssh.targets if t.enabled and t.node_id == host), None)


def _read_key_file(path: str) -> str | None:
    """The key in `path`, stripped. An unreadable file is a ClickException.

    Same reason _ca_context refuses to let an OSError escape: `main` would
    attach the unreachable-endpoint hint to a permissions problem on this Mac.
    """
    try:
        return Path(path).read_text().strip() or None
    except OSError as e:
        raise click.ClickException(f"--key-file {path}: {e}") from e


def _resolve_key(api_key: str | None, key_file: str | None, node_id: str | None) -> str | None:
    """The MCP bearer key to send, highest-precedence source first.

    --api-key > --key-file > $AUREAX_MCP_API_KEY > the orchestrator keystore
    (only when the positional resolved to a node id, since the keystore is
    keyed by node) > none, which is correct for a node whose template carries
    no ``api_key`` line.

    Explicit beats implicit all the way down so a rotation caught mid-apply can
    be worked around from the command line: the node still enforces the old key
    while the new one is already in the keystore.
    """
    if api_key:
        return api_key
    if key_file:
        return _read_key_file(key_file)
    from_env = (os.environ.get(ENV_API_KEY) or "").strip()
    if from_env:
        return from_env
    if node_id:
        from orchestrator.config import keystore

        return keystore.read_key(node_id, kind=keystore.KIND_MCP)
    return None


def _resolve_endpoint(host: str, port: int, api_key: str | None = None,
                      key_file: str | None = None,
                      fleet_config: str | None = None) -> tuple[str, str | None, str | None]:
    """(url, key, node_id) for a positional that may be a host OR a node id.

    A node id is resolved to that target's address and remembers the id, which
    is what lets the key come from the keystore with nothing typed. Anything
    else is used verbatim, so an address still works on a machine with no fleet
    file at all.

    The returned key is never echoed: the commands print the URL (and the node
    id when one resolved) so a mistyped id is visible as a literal host, but
    the key only ever reaches the Authorization header.
    """
    target = _lookup_target(host, fleet_config)
    node_id = target.node_id if target is not None else None
    url = _make_url(target.host if target is not None else host, port)
    _CONTEXT["host"] = host
    _CONTEXT["node_id"] = node_id or ""
    return url, _resolve_key(api_key, key_file, node_id), node_id


def _endpoint_label(url: str, node_id: str | None) -> str:
    """`url`, naming the node id when the positional resolved to one."""
    return f"{url} (node {node_id})" if node_id else url


# ---------------------------------------------------------------------------
# Click CLI commands
# ---------------------------------------------------------------------------

# The endpoint every command takes. NODE rather than HOST because it accepts
# either spelling — an address, or an enabled node id resolved through the
# fleet config (see _resolve_endpoint). A separate --node option would force
# the positional to become optional, and `call NODE TOOL` / `infer NODE
# PROMPT...` already have a second positional, so an optional leading one is
# ambiguous to parse and to read.
_host_argument = click.argument("host", metavar="NODE")

# Declared once: six commands take the same port, and show_default renders it
# from DEFAULT_PORT. A literal retyped into six help strings would go stale in
# all six the moment MCP_NODE_PORT moves.
_port_option = click.option(
    "--port", "-p", default=DEFAULT_PORT, show_default=True, help="MCP NodePort"
)
_timeout_option = click.option(
    "--timeout", "read_timeout", default=DEFAULT_READ_TIMEOUT, show_default=True,
    help="Seconds of silence tolerated on the MCP stream before giving up",
)


def _compose(*options):
    """Apply several click options as one decorator."""
    def wrap(f):
        for option in reversed(options):
            f = option(f)
        return f
    return wrap


# One block rather than four separate decorators repeated six times: every
# command needs all four to reach a node, and a command that grew three of them
# would fail in a way that reads as "this node is broken".
_endpoint_options = _compose(
    click.option(
        "--api-key", default=None,
        help=f"MCP bearer key. Lands in your shell history and in `ps` for the life of the "
             f"run — prefer --key-file or ${ENV_API_KEY}.",
    ),
    click.option(
        "--key-file", type=click.Path(exists=True, dir_okay=False), default=None,
        help="File holding the MCP bearer key (e.g. ~/.aureax/keys/<node-id>.mcp.key).",
    ),
    click.option(
        "--config", "fleet_config", type=click.Path(dir_okay=False), default=None,
        help=f"Fleet config used to resolve a node id given as NODE [default: {DEFAULT_NETWORK_CONFIG}]",
    ),
    click.option(
        "--ca", "ca_file", type=click.Path(dir_okay=False), default=None,
        help=f"CA bundle the node's certificate must chain to "
             f"[default: ${ENV_CA}, else ~/.aureax/tls/ca.pem]",
    ),
)


@click.group()
def cli():
    """Aureax MCP Testing Harness — test edge node MCP servers.

    NODE is an address or an enabled node id from the fleet config; a node id
    also supplies that node's MCP key from the orchestrator key store.
    """


@cli.command()
@_host_argument
@_port_option
@_endpoint_options
def ping(host: str, port: int, api_key: str | None, key_file: str | None,
         fleet_config: str | None, ca_file: str | None) -> None:
    """Check if a node's MCP server is reachable."""
    url, key, node_id = _resolve_endpoint(host, port, api_key, key_file, fleet_config)
    verify = _ca_context(ca_file)
    click.echo(f"Pinging {_endpoint_label(url, node_id)}...")
    failure = asyncio.run(_ping(url, key, verify))
    if failure is None:
        click.echo(click.style("PONG — MCP server is reachable", fg="green"))
    else:
        # A node whose certificate this CA did not sign ANSWERED, so reporting
        # it as unreachable sends the operator to the node instead of the CA.
        if _tls_failure(failure) is not None:
            click.echo(click.style("FAILED — TLS verification failed", fg="red"))
            click.echo(f"  {_tls_hint()}")
        elif _auth_failure(failure) is not None:
            # Same reasoning one branch up: a node that answered 401 is running
            # and reachable, and reporting it as unreachable sends the operator
            # to the pod instead of to the key.
            click.echo(click.style("FAILED — MCP key rejected (401 Unauthorized)", fg="red"))
            click.echo(f"  {_auth_hint()}")
        else:
            click.echo(click.style("FAILED — MCP server not reachable", fg="red"))
        # The cause, not just the verdict: a refused connection, an
        # unpublished NodePort and a head still loading its model are three
        # different problems behind the same one-line verdict.
        click.echo(f"  {_describe(failure)}")
        sys.exit(1)


@cli.command()
@_host_argument
@_port_option
@_endpoint_options
def tools(host: str, port: int, api_key: str | None, key_file: str | None,
          fleet_config: str | None, ca_file: str | None) -> None:
    """List all tools on a node with their descriptions and schemas."""
    url, key, node_id = _resolve_endpoint(host, port, api_key, key_file, fleet_config)
    tool_list = asyncio.run(_list_tools(url, key, _ca_context(ca_file)))
    click.echo(f"Tools on {_endpoint_label(url, node_id)} ({len(tool_list)} available):\n")
    for t in tool_list:
        click.echo(f"  {click.style(t['name'], fg='cyan')}")
        click.echo(f"    {t['description']}")
        # One --arg line per input; required ones marked.
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if props:
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                click.echo(f"    --arg {pname}=<{ptype}>{req}")
        click.echo()


@cli.command()
@_host_argument
@click.argument("tool_name")
@click.option("--arg", "-a", multiple=True, help="Tool argument as key=value")
@_port_option
@_timeout_option
@_endpoint_options
def call(host: str, tool_name: str, arg: tuple, port: int, read_timeout: float,
         api_key: str | None, key_file: str | None,
         fleet_config: str | None, ca_file: str | None) -> None:
    """Call a specific tool on a node."""
    url, key, _node_id = _resolve_endpoint(host, port, api_key, key_file, fleet_config)
    args, rejected = _parse_kv_args(list(arg))
    if rejected:
        raise click.UsageError(
            f"--arg takes key=value; these have no '=': {', '.join(rejected)}"
        )
    result = asyncio.run(_call_tool(url, tool_name, args, read_timeout, key, _ca_context(ca_file)))
    click.echo(json.dumps(result, indent=2))
    # Print the result, THEN fail. The JSON is what the operator came for; the
    # exit status is what a wrapper reads — see _tool_failed.
    if _tool_failed(result):
        sys.exit(1)


@cli.command()
@_host_argument
@click.argument("prompt", nargs=-1, required=True)
@_port_option
@_timeout_option
@click.option("--max-tokens", "-m", default=1024, help="Max output tokens")
@click.option("--temperature", "-t", default=0.3, help="Sampling temperature")
@_endpoint_options
def infer(host: str, prompt: tuple, port: int, read_timeout: float,
          max_tokens: int, temperature: float, api_key: str | None,
          key_file: str | None, fleet_config: str | None, ca_file: str | None) -> None:
    """Send a prompt to a node's LLM and see the response."""
    url, key, _node_id = _resolve_endpoint(host, port, api_key, key_file, fleet_config)
    verify = _ca_context(ca_file)
    prompt_text = " ".join(prompt)

    with _Spinner(f"Generating on {host} (up to {max_tokens} tokens)"):
        result = asyncio.run(_call_tool(url, "run_inference", {
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }, read_timeout, key, verify))

    # Error before content. _call_tool reports a failed tool call as
    # {"error": True, "content": <the failure text>}, so testing "content"
    # first would print a failure as a green "Response:" with a fabricated
    # "(? tokens, ?ms)" line under it.
    if _tool_failed(result):
        detail = result.get("content", "") if result["error"] is True else result["error"]
        click.echo(click.style(f"Error: {detail}", fg="red"))
        sys.exit(1)
    elif "content" in result:
        click.echo()
        click.echo(click.style("Response:", fg="green"))
        click.echo(result["content"])
        click.echo()
        tokens = result.get("tokens_predicted", "?")
        time_ms = result.get("generation_time_ms", "?")
        click.echo(click.style(f"({tokens} tokens, {time_ms}ms)", fg="cyan"))
    else:
        click.echo(json.dumps(result, indent=2))


@cli.command()
@_host_argument
@_port_option
@click.option("--type", "sensor_type", default="all", help="Sensor type (temperature, humidity, pressure, light, battery, all)")
@_endpoint_options
def sensor(host: str, port: int, sensor_type: str, api_key: str | None,
           key_file: str | None, fleet_config: str | None, ca_file: str | None) -> None:
    """Read sensor data from a node."""
    url, key, _node_id = _resolve_endpoint(host, port, api_key, key_file, fleet_config)
    result = asyncio.run(_call_tool(
        url, "read_sensor", {"sensor_type": sensor_type},
        DEFAULT_READ_TIMEOUT, key, _ca_context(ca_file),
    ))
    click.echo(json.dumps(result, indent=2))
    if _tool_failed(result):
        sys.exit(1)


@cli.command()
@_host_argument
@_port_option
@_timeout_option
@_endpoint_options
def connect(host: str, port: int, read_timeout: float, api_key: str | None,
            key_file: str | None, fleet_config: str | None, ca_file: str | None) -> None:
    """Open an interactive session to a node's MCP server."""
    url, key, node_id = _resolve_endpoint(host, port, api_key, key_file, fleet_config)
    verify = _ca_context(ca_file)
    try:
        asyncio.run(_interactive(url, read_timeout, key, verify, _endpoint_label(url, node_id)))
    except KeyboardInterrupt:
        # Fallback path only: _interactive installs a loop-level SIGINT handler
        # that ends the process itself, so this covers Ctrl-C in the window
        # before that handler is in place. Caught before Click, which would
        # otherwise render it as "Aborted!".
        click.echo("\nDisconnected.", err=True)
        sys.exit(130)


def main() -> None:
    """Harness CLI entry point registered in pyproject.toml.

    Every command talks to an MCP server, and an unreachable endpoint is the
    common case rather than an exotic one: a shard worker serves no MCP at all
    by design, and a group's head is unreachable for minutes while it loads.
    So this renders the leaf cause (see _leaf_exceptions) instead of an asyncio
    traceback, and attaches the endpoint hint only when the cause really is a
    transport failure — a TypeError in a tool handler gets no network advice,
    and a certificate the CA did not sign, or a key the node rejected with a
    401, gets its own remedy rather than advice about a node that in fact
    answered.

    Set AUREAX_HARNESS_TRACEBACK=1 to re-raise instead and get the traceback.
    """
    try:
        cli()
    except Exception as e:
        if os.environ.get("AUREAX_HARNESS_TRACEBACK"):
            raise
        # ClickException never reaches here — Click's standalone mode renders
        # and exits on it inside cli() itself.
        click.echo(click.style(f"ERROR: {_describe(e)}", fg="red"), err=True)
        if _tls_failure(e) is not None:
            click.echo(f"Hint: {_tls_hint()}", err=True)
        elif _auth_failure(e) is not None:
            # The ERROR line above already carries httpx's own "401
            # Unauthorized" text, so this branch only adds the remedy — the
            # same shape the TLS branch uses.
            click.echo(f"Hint: {_auth_hint()}", err=True)
        elif _is_connection_failure(e):
            click.echo(
                "Hint: the MCP endpoint may be unreachable. A shard worker serves no MCP — "
                "address its group through the head ('uv run aureax group list' prints the URL) — "
                "and a head that is still loading its model will not answer yet.",
                err=True,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
