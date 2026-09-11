"""Where the node's TLS material lives, and the one context that trusts it.

Every listener and every client on an edge node speaks TLS: llama-server on
:8080, the MCP server on :9000/:30900, and the outbound telemetry and log
pushes to the orchestrator. All of them read the same three paths out of the
resolved config's ``tls:`` block, which the generator stamps into every node's
config (orchestrator/config/templating.py) and the deployer installs under
``/etc/aureax/tls`` — so this module is the single place that knows the shape
of that block.

There is no plaintext fallback and no unverified-TLS escape hatch anywhere: a
node whose certificates never arrived is a node that cannot serve, and the
supervisor exits :data:`EX_CONFIG` rather than starting a listener an operator
would reasonably assume was encrypted.

Stdlib only, deliberately. ``docker/Dockerfile.client`` copies ``client/``
wholesale but only ``orchestrator/config/schema.py`` from the orchestrator
package, so anything else imported here would be a module the edge image does
not contain — an ImportError inside the pod, not at review time.
"""

from __future__ import annotations

import ssl
from pathlib import Path

# EX_CONFIG from sysexits.h. Distinct from the generic 1, from the 137 an
# OOM-killed model load produces, and from every code _exit_status maps a
# child's death onto — so `kubectl describe` names the cause instead of
# folding a missing certificate in with every other failure.
EX_CONFIG = 78

# One context per CA file rather than one per client. httpx builds an SSL
# context on construction (a CA-bundle load), and the supervisor's health wait
# alone dials once every 2s for up to 2700s; the telemetry loop, the MCP
# server's LlamaClient and the log shipper add more. Keyed by path AND stat,
# the way orchestrator/config/tlsstore.py::ca_ssl_context keys its own cache: a
# CA replaced under the same path must not keep serving the old trust anchor.
_contexts: dict[tuple[str, int, int], ssl.SSLContext] = {}


def tls_paths(config: dict) -> tuple[str, str, str]:
    """``(ca_file, cert_file, key_file)`` from the resolved config's ``tls:`` block.

    Raises ValueError when the block or any of its three keys is missing.
    Refusing here rather than defaulting to the well-known paths is the point:
    a config with no ``tls:`` block was not produced by this version of
    ``aureax generate``, and guessing would put a node on whatever happens to
    be lying in /etc/aureax/tls — including a previous node's certificate.
    """
    block = config.get("tls") or {}
    ca_file = block.get("ca_file")
    cert_file = block.get("cert_file")
    key_file = block.get("key_file")
    missing = [
        name
        for name, value in (
            ("tls.ca_file", ca_file),
            ("tls.cert_file", cert_file),
            ("tls.key_file", key_file),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "the resolved config carries no " + " / ".join(missing) + ". TLS is mandatory "
            "fleet-wide (ARCHITECTURE.md, 'TLS'); regenerate and push this node's config "
            "with `aureax generate --orchestrator-ip <ip>` then `aureax apply --node <id>`."
        )
    return str(ca_file), str(cert_file), str(key_file)


def require_tls_files(config: dict) -> None:
    """Raise unless all three files exist and are readable by this process.

    Existence is not enough: ``node.key`` is installed 0600 root:root and the
    container runs as root, but a half-finished push (the deployer stages four
    files and installs them in one command) can leave a path present and
    unreadable. Reading a byte is what tells the two apart, and doing it once
    at startup beats discovering it inside uvicorn's SSL handshake, where the
    failure surfaces as a connection reset with nothing in the pod log.
    """
    for field, path in zip(
        ("tls.ca_file", "tls.cert_file", "tls.key_file"), tls_paths(config), strict=True
    ):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"{field}: {path} is missing")
        try:
            with p.open("rb") as fh:
                fh.read(1)
        except OSError as e:
            raise FileNotFoundError(f"{field}: {path} is not readable ({e})") from e


def ssl_context(config: dict) -> ssl.SSLContext:
    """The shared context every in-pod httpx client passes as ``verify=``.

    Trusts the fleet CA and nothing else: the node's own leaf is signed by it,
    as is the orchestrator's, so this one context covers both directions —
    dialling 127.0.0.1:8080 and pushing telemetry to the dashboard.

    httpx 0.28 deprecates ``verify="<path>"``, so a context is the supported
    spelling as well as the cheaper one. Hostname checking and
    ``CERT_REQUIRED`` stay at ``create_default_context``'s defaults; turning
    either off is what the loopback and node-id SANs in
    ``orchestrator/config/tlsstore.py::node_sans`` exist to make unnecessary.
    """
    ca_file, _cert, _key = tls_paths(config)
    st = Path(ca_file).stat()
    key = (ca_file, st.st_mtime_ns, st.st_size)
    ctx = _contexts.get(key)
    if ctx is None:
        ctx = ssl.create_default_context(cafile=ca_file)
        _contexts[key] = ctx
    return ctx
