"""Aureax dashboard web application."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import logging
import resource
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from client.chat.store import SessionStore
from client.log.store import LogStore
from orchestrator.config.loader import DEFAULT_NETWORK_CONFIG, load_network_config
from orchestrator.dashboard.access_api import ApplyRunner
from orchestrator.dashboard.access_api import router as access_router
from orchestrator.dashboard.api import ingest_router, router
from orchestrator.dashboard.browser_boundary import BrowserBoundaryMiddleware
from orchestrator.dashboard.chat_api import router as chat_router
from orchestrator.dashboard.chat_service import ChatService
from orchestrator.dashboard.config_api import router as config_router
from orchestrator.dashboard.lifecycle_api import router as lifecycle_router
from orchestrator.dashboard.watchers import start_all as start_watchers
from orchestrator.lifecycle import LifecycleManager, State, SystemLogger
from orchestrator.telemetry.queries import MetricsQueries
from orchestrator.telemetry.storage import MetricsStorage

STATIC_DIR = Path(__file__).parent / "static"

# Applied to every response, API and static alike. Set from middleware rather
# than a <meta> tag, which would cover neither the JSON routes (GET
# /api/access/{node_id}/key among them) nor frame-ancestors — ignored in meta
# form. X-Frame-Options says the same thing as frame-ancestors for browsers too
# old to read the CSP directive.
_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (
        b"content-security-policy",
        b"default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    ),
    (b"x-frame-options", b"DENY"),
    (b"x-content-type-options", b"nosniff"),
    # Covers this hostname only; descendant hosts may serve other protocols.
    (b"strict-transport-security", b"max-age=31536000"),
)


class SecurityHeadersMiddleware:
    """Add the headers above to every HTTP response.

    Pure ASGI rather than ``@app.middleware("http")``: BaseHTTPMiddleware pumps
    the response body through a queue, and the SSE routes here stream for the
    life of a browser tab. Header injection needs only the response-start
    message, so nothing has to touch the body at all.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), *_SECURITY_HEADERS]
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _raise_fd_limit() -> None:
    """Bump RLIMIT_NOFILE to the hard limit.

    The dashboard holds many fds simultaneously: SSE streams to the browser,
    SQLite handles (sessions / logs / telemetry), watcher poll loops, and the
    setup-all.sh subprocess pipe during first-time bootstrap. macOS defaults
    to a soft limit of 256, which the build subprocess can exhaust while
    streaming Docker output line-by-line. Raising to the hard limit (usually
    >=10240) is enough headroom and is allowed without root.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            logging.getLogger("aureax.dashboard").info(
                "RLIMIT_NOFILE raised: %d -> %d", soft, hard,
            )
    except (ValueError, OSError) as e:
        logging.getLogger("aureax.dashboard").warning(
            "could not raise RLIMIT_NOFILE: %s", e,
        )


_LIFECYCLE_DRAIN_TIMEOUT = 30.0


async def _drain_lifecycle(lc) -> None:
    """Let an in-flight lifecycle op finish (or abort a start) before exit.

    ``_run_start``/``_run_stop`` sit in ``asyncio.to_thread``, i.e. the loop's
    DEFAULT executor, and can wait up to _POD_WAIT_TIMEOUT (3600s). Loop close
    ends in ``shutdown_default_executor``, which joins that worker — so
    shutdown has to tell it to stop, or Ctrl-C during a Start hangs the
    process waiting on it.

    A START is cancelled — ``cancel()`` sets cancel_event, which the worker's
    wait loops poll — but a STOP is only waited on: cancelling mid-stop makes
    ``_wait_pods_terminated`` raise past ``k3d.stop_cluster``, leaving the
    selectors patched and the control plane up. The wait is bounded because the
    worker can be inside a blocking kubectl/SSH call with no cancel point, and
    an unbounded await here would reproduce the hang it exists to fix.
    """
    if lc is None or not lc.busy:
        return
    if lc.state is State.STARTING:
        # NOTE: blocks the loop up to 5s SIGTERMing a tracked subprocess
        # (LifecycleManager.cancel), which is acceptable during shutdown.
        with contextlib.suppress(Exception):
            lc.cancel()
    # LifecycleManager.wait, not lc._task: the manager owns how its run is
    # joined (shielded, so a timeout here does not cancel a stop mid-flight),
    # and reaching for the private attribute made it part of this module's
    # contract. Everything is suppressed — this runs during shutdown, and the
    # operation's own failure is already recorded in last_error and emitted.
    with contextlib.suppress(Exception, asyncio.CancelledError):
        await lc.wait(_LIFECYCLE_DRAIN_TIMEOUT)


def _open_store(log: logging.Logger, what: str, db_path: str, factory):
    """Build one SQLite-backed store, or None with a warning naming the file.

    The constructors run their DDL immediately, so a corrupt or non-SQLite file
    raises here. Unhandled that aborts ``aureax dashboard`` outright, taking
    lifecycle, config and access down with the one broken database — and the
    traceback does not say which file to remove.
    """
    try:
        return factory(db_path)
    except Exception as e:
        log.warning(
            "%s store unavailable — that tab is disabled until %s is repaired "
            "or removed (%s: %s)",
            what, db_path, type(e).__name__, e,
        )
        return None


def create_app(
    metrics_db_path: str = "~/.aureax/telemetry/metrics.db",
    network_config_path: str = str(DEFAULT_NETWORK_CONFIG),
    chat_db_path: str = "~/.aureax/chat/sessions.db",
    log_db_path: str = "~/.aureax/logs/system.db",
    *,
    trusted_hosts: tuple[str, ...] = (),
) -> FastAPI:
    """Create the FastAPI dashboard application.

    ``trusted_hosts`` names the addresses this app answers to beyond loopback
    and this machine's own hostname — ``run()`` passes the address it binds
    HTTPS on, which is the one the orchestrator leaf was minted for and the one
    every node ships telemetry and logs to. It may be a hostname: see
    BrowserBoundaryMiddleware.
    """
    _raise_fd_limit()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = await start_watchers(app, Path(network_config_path))
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                # Watchers were just cancelled; awaiting them raises
                # CancelledError on success, or any Exception they happened
                # to surface. Either way, swallow during shutdown.
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await t

            # ChatService holds an MCP pool per chat TARGET, each with a live
            # streamable-http session and its own httpx client. This is the one
            # caller of close(); without it those remote sessions are reclaimed
            # only when the process exits.
            chat = getattr(app.state, "chat", None)
            if chat is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await chat.close()

            await _drain_lifecycle(getattr(app.state, "lifecycle", None))

    app = FastAPI(title="Aureax Dashboard", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BrowserBoundaryMiddleware, trusted_hosts=trusted_hosts)
    app.add_middleware(SecurityHeadersMiddleware)

    log = logging.getLogger("aureax.dashboard")

    # Each of the three stores below opens a SQLite file and runs its DDL in
    # the constructor, so a truncated or non-SQLite file costs the operator
    # that one feature rather than the whole dashboard — lifecycle, config and
    # access still have to come up, and the warning names the file to remove.
    # Every consumer already tolerates a missing store (api._queries /
    # api._storage, api.receive_logs, chat_api._service).
    metrics_storage = _open_store(log, "telemetry metrics", metrics_db_path, MetricsStorage)
    if metrics_storage is not None:
        # server.telemetry.storage.retention_days is what the store prunes to —
        # hourly, triggered from insert() via MetricsStorage._maybe_rotate.
        # create_app must still build without a readable network config (it is
        # the first-time bootstrap surface), but a fleet must not end up
        # silently retaining at a figure nobody configured, hence the warning.
        try:
            cfg = load_network_config(network_config_path)
            metrics_storage.retention_days = cfg.server.telemetry.storage.retention_days
        except Exception as e:
            log.warning(
                "could not read telemetry retention_days (%s) — using the default %dd",
                e, metrics_storage.retention_days,
            )
    app.state.metrics_storage = metrics_storage
    # MetricsQueries connects lazily, so it would answer /api/nodes with a
    # 500 out of the same unreadable file rather than an empty fleet.
    app.state.metrics_queries = (
        MetricsQueries(metrics_db_path) if metrics_storage is not None else None
    )

    # Persistent system log + shared SystemLogger (fan-out + DB). Without a
    # store the logger keeps its in-memory ring buffer and its SSE fan-out;
    # only durability across a restart is lost.
    log_store = _open_store(log, "system log", log_db_path, LogStore)
    system_log = SystemLogger(store=log_store)
    app.state.log_store = log_store
    app.state.system_log = system_log

    # Lifecycle manager — one per process. Shares the system logger so its
    # start/stop events land in the same stream as watchers + chat + config.
    app.state.lifecycle = LifecycleManager(
        Path(network_config_path),
        logger=system_log,
    )

    # Chat service — one per process. Lazy engine init on first request.
    # No store means no chat: every route persists its turns, so the honest
    # answer is chat_api's 503 rather than a service over a store that is not
    # there.
    chat_store = _open_store(log, "chat sessions", chat_db_path, SessionStore)
    app.state.chat = (
        ChatService(Path(network_config_path), chat_store, system_log=system_log)
        if chat_store is not None
        else None
    )

    # Access tab — endpoint display + API-key auth. One ApplyRunner per
    # process so lifecycle_api's 409 guard has something to check even before
    # the first apply.
    app.state.access_apply = ApplyRunner()

    # Mount API routes (before the static wildcard mount).
    #
    # Nothing here is authenticated: the dashboard is assumed to sit behind NAT
    # with only trusted users (schema.py: server.dashboard.host). ingest_router
    # keeps its own router because the two POSTs edge nodes push to are the
    # reason this server binds 0.0.0.0, and grouping them says so.
    app.include_router(ingest_router)
    app.include_router(router)
    app.include_router(lifecycle_router)
    app.include_router(config_router)
    app.include_router(chat_router)
    app.include_router(access_router)

    # Mount static files (HTML/CSS/JS dashboard)
    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


# --- Two listeners, one process ---------------------------------------------

# Addresses that mean "every interface". A wildcard HTTPS bind cannot coexist
# with the loopback HTTP listener on the same port — the wildcard socket owns
# 127.0.0.1:<port> too, so the second bind gets EADDRINUSE — which is why the
# CLI resolves server.dashboard.host == "0.0.0.0" to the orchestrator's own LAN
# address before calling run() (see the schema comment on server.dashboard.host).
WILDCARD_HOSTS = ("0.0.0.0", "::", "")

# Addresses that are already loopback, so the plain-HTTP listener would be
# asking for the same address and port the HTTPS one just took.
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")

# Where the plaintext listener is offered. Both families, because Firefox
# resolves `localhost` to ::1 first: a refused ::1 does fall back to 127.0.0.1,
# but the fallback costs a visible round trip and some clients (and some
# corporate proxies) simply report the refusal instead. A host without IPv6
# skips ::1 rather than failing — see _loopback_sockets.
_LOOPBACK_BINDS = ("127.0.0.1", "::1")

# errnos that mean "this machine has no such address", as opposed to "somebody
# else is already there". EAFNOSUPPORT is a kernel with IPv6 compiled out;
# EADDRNOTAVAIL is ::1 not configured on the loopback interface.
_NO_SUCH_ADDRESS = (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT)


class LifespanOnlyShim:
    """The same ASGI app, with its lifespan answered here instead of by it.

    The dashboard is served on two listeners, and an ASGI server runs the
    application's lifespan itself — so the second server would start a SECOND
    copy of everything ``create_app``'s lifespan owns: the node/pod/telemetry
    watchers and their kubectl polling, doubled and racing, and a second
    shutdown that cancels watchers the first server is still using. There is
    only one app object, so the state they populate is shared; what must not
    happen twice is the *starting*.

    Startup and shutdown are therefore completed here without ever calling the
    app, and every other scope (http, and websocket if one is ever added) is
    forwarded untouched — so the loopback listener serves exactly the same
    routes, with the same middleware and the same ``app.state``.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "lifespan":
            await self.app(scope, receive, send)
            return
        # uvicorn's LifespanOn sends startup, waits for the reply, then sends
        # shutdown at exit and waits again. Answering both keeps that server's
        # own startup/shutdown sequencing intact.
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return


def _port_holder(port: int) -> str:
    """`` — held by PID 45235`` when something is listening, else ``''``.

    The operator's report for this was "[Errno 48] address already in use" and
    nothing else: uvicorn logs the OSError and exits 1, which does not say that
    the thing on the port is usually their own earlier `aureax dashboard`.
    """
    from orchestrator.lifecycle import listeners_on_port

    pids, lookup_error = listeners_on_port(port)
    if pids:
        return f" — held by PID {', '.join(pids)}"
    if lookup_error is not None:
        return f" — could not ask lsof who holds it ({lookup_error})"
    return ""


def _listen_socket(host: str, port: int) -> socket.socket:
    """One bound TCP socket, or an OSError whose message names the holder.

    Bound here rather than by uvicorn for two reasons: uvicorn answers a failed
    bind with ``logger.error(exc); sys.exit(1)`` from inside the running loop
    (no traceback, no pid, and the other listener left running), and binding
    both listeners up front makes a busy port fail before either one starts
    serving instead of half-serving.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if family is socket.AF_INET6:
        # Explicitly v6-only: a dual-stack ::1 socket would also claim
        # 127.0.0.1:<port> on Linux, so the sibling v4 bind below would fail
        # with EADDRINUSE against this very process.
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    # Same option uvicorn's own Config.bind_socket sets: it allows a rebind
    # while an old connection is in TIME_WAIT, and does NOT let two processes
    # listen on the same address — a busy port still raises below.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as e:
        sock.close()
        if e.errno == errno.EADDRINUSE:
            raise OSError(
                e.errno,
                f"the dashboard cannot bind {host}:{port} — address already in use"
                f"{_port_holder(port)}. Stop it (`uv run aureax stop`, or kill that pid) "
                f"or serve elsewhere with `aureax dashboard --port <n>`.",
            ) from e
        raise
    return sock


def _loopback_sockets(port: int, log: logging.Logger) -> list[socket.socket]:
    """Bound sockets for http://localhost:<port>, 127.0.0.1 first.

    ``::1`` is best-effort: a host with no IPv6 loopback is not a broken
    dashboard, it is a browser that resolves localhost to 127.0.0.1. A port
    already TAKEN on ::1 is a different thing entirely and still raises —
    something else would be answering http://localhost:<port> in every browser
    that tries IPv6 first.
    """
    socks: list[socket.socket] = []
    try:
        for host in _LOOPBACK_BINDS:
            try:
                socks.append(_listen_socket(host, port))
            except OSError as e:
                if host == "127.0.0.1" or e.errno not in _NO_SUCH_ADDRESS:
                    raise
                log.info(
                    "no IPv6 loopback on this host — the plaintext listener is "
                    "127.0.0.1:%d only (%s)", port, e,
                )
    except OSError:
        for sock in socks:
            sock.close()
        raise
    return socks


async def _mirror_shutdown(servers, tasks) -> None:
    """Make one server's exit — a Ctrl-C, an error — end the other.

    uvicorn installs its SIGINT/SIGTERM handler per ``Server``
    (``Server.capture_signals``, uvicorn 0.42) with ``signal.signal``, which
    keeps only the handler installed LAST. So a Ctrl-C sets ``should_exit`` on
    one of the two servers and the other keeps serving, leaving the gather
    below waiting forever on a dashboard the operator believes they stopped.
    Polling at uvicorn's own main-loop cadence propagates the flag whichever
    server saw the signal, and also covers a server that stops for any other
    reason.
    """
    while not any(s.should_exit for s in servers) and not any(t.done() for t in tasks):
        await asyncio.sleep(0.1)
    for server in servers:
        server.should_exit = True


async def _serve_together(pairs) -> None:
    """Run every (server, sockets) pair on this one loop until they all stop."""
    tasks = [asyncio.create_task(s.serve(sockets=socks)) for s, socks in pairs]
    servers = [s for s, _socks in pairs]
    mirror = asyncio.create_task(_mirror_shutdown(servers, tasks))
    try:
        await asyncio.gather(*tasks)
    finally:
        mirror.cancel()
        # A server whose sibling raised (a bind uvicorn rejected, a crash in
        # startup) is still serving at this point; asking it to exit and
        # joining it is what keeps a half-failed start from leaving a listener
        # behind. Every outcome is suppressed: the failure that got us here is
        # the one worth reporting.
        for server in servers:
            server.should_exit = True
        for task in (*tasks, mirror):
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task


def run(
    port: int = 7001,
    metrics_db_path: str = "~/.aureax/telemetry/metrics.db",
    network_config_path: str = str(DEFAULT_NETWORK_CONFIG),
    chat_db_path: str = "~/.aureax/chat/sessions.db",
    log_db_path: str = "~/.aureax/logs/system.db",
    host: str = "0.0.0.0",
    *,
    ssl_certfile: str,
    ssl_keyfile: str,
    loopback_http: bool = True,
) -> None:
    """Serve one dashboard on two listeners, in one process, on one port.

    * **HTTPS on ``host``:``port``** — the fleet's listener, and the only one
      that leaves this machine. ``host`` is the address the orchestrator leaf
      was minted for, i.e. the address in every node's ``${orchestrator_url}``;
      the CLI resolves the ``0.0.0.0`` default of server.dashboard.host to it.
      It is a specific address rather than a wildcard because a wildcard socket
      owns loopback too, leaving nothing for the second listener to bind — so a
      Mac with several LAN interfaces serves the fleet on exactly the one the
      certificate names.
    * **plain HTTP on 127.0.0.1 and ::1, same port** — for a browser on this
      machine. No certificate, no trust store, no padlock: opening
      ``localhost:<port>`` from the orchestrator just works, which is the whole
      point ("the system that runs the dashboard is a trusted system"). It
      never leaves the host — loopback is not routable — so nothing about the
      fleet's mandatory TLS changes. Set ``loopback_http=False`` to serve HTTPS
      alone.

    ``ssl_certfile``/``ssl_keyfile`` are keyword-only and have NO defaults, so
    a caller structurally cannot start the *fleet's* listener in plaintext: TLS
    is mandatory fleet-wide (ARCHITECTURE.md, "TLS") and every node dials
    ``https://<orchestrator>:7001`` to ship telemetry and logs. ``aureax
    dashboard`` resolves both from ``~/.aureax/tls`` and mints the leaf if the
    Mac's LAN address changed.

    TLS is transport security only. It authenticates this server to the fleet
    and encrypts what crosses the LAN; it does NOT authenticate the caller. The
    ingest POSTs and every control route stay unauthenticated by design — see
    the mount comment in create_app and server.dashboard.host in the schema.
    """
    import uvicorn

    log = logging.getLogger("aureax.dashboard")
    if loopback_http and host in WILDCARD_HOSTS:
        raise ValueError(
            f"the dashboard cannot serve HTTPS on the wildcard address {host!r} and plain "
            f"HTTP on loopback at the same port ({port}): the wildcard socket owns "
            "127.0.0.1 too. Pass the orchestrator's own address as `host` (that is what "
            "`aureax dashboard` does with the server.dashboard.host default), or "
            "loopback_http=False."
        )

    pairs: list[tuple[uvicorn.Server, list[socket.socket]]] = []
    bound: list[socket.socket] = []
    try:
        # Every socket is bound BEFORE the app is built, so a busy port is one
        # clear error naming the holder rather than a half-started dashboard —
        # and before three SQLite stores were opened and the fd limit raised.
        https_sockets = [_listen_socket(host, port)]
        bound += https_sockets
        # An explicitly loopback server.dashboard.host (the SSH-tunnel setup
        # the schema describes) IS this listener's address already, so there is
        # no second one to add — asking for it would be an EADDRINUSE against
        # ourselves.
        serve_loopback = loopback_http and host not in LOOPBACK_HOSTS
        loopback_sockets = _loopback_sockets(port, log) if serve_loopback else []
        bound += loopback_sockets

        # ONE app for both listeners, so both see the same LifecycleManager,
        # ChatService, stores and SSE fan-out — two apps would be two fleets'
        # worth of state in one process.
        app = create_app(metrics_db_path, network_config_path, chat_db_path, log_db_path,
                         trusted_hosts=(host,))
        pairs.append((
            uvicorn.Server(uvicorn.Config(
                app, host=host, port=port,
                ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
            )),
            https_sockets,
        ))
        if loopback_sockets:
            pairs.append((
                uvicorn.Server(uvicorn.Config(
                    # The shim, not `app`: a second lifespan would start every
                    # background watcher a second time. See LifespanOnlyShim.
                    LifespanOnlyShim(app), host=_LOOPBACK_BINDS[0], port=port,
                )),
                loopback_sockets,
            ))

        # asyncio.run, not uvicorn.run: the two servers have to share one loop
        # for the single app object and its watcher tasks to stay single. It is
        # the same loop uvicorn would have built — neither uvloop nor httptools
        # is a dependency here, so its "auto" resolves to asyncio and h11.
        asyncio.run(_serve_together(pairs))
    finally:
        # uvicorn closes these itself on a clean shutdown; this covers the
        # paths where it never got them (a second bind that failed, a crash
        # before serving) so a caller in-process is not left holding the port.
        for sock in bound:
            with contextlib.suppress(OSError):
                sock.close()
