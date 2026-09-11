"""The fleet's private CA and every TLS leaf it signs, under ``~/.aureax/tls``.

TLS is mandatory fleet-wide: llama-server, the MCP server and the dashboard all
listen HTTPS, and every client verifies against the one CA created here. There
is no public CA in the picture — the fleet is addressed by LAN IP, which no
public issuer will certify — so the orchestrator is the issuer.

Layout (mirrors ``keystore.py``'s ``<node_id>.key`` convention, so one ``ls``
of either directory reads the same way)::

    ~/.aureax/tls/            0700
    ├── ca.key                0600   EC P-256, no passphrase
    ├── ca.pem                0644   CN="Aureax Fleet CA", CA:TRUE pathlen:0
    ├── orchestrator.key      0600
    ├── orchestrator.pem      0644
    ├── <node_id>.key         0600   one pair per enabled ssh target
    └── <node_id>.pem         0644

Certificates are 0644 on purpose. The directory is 0700 so reachability is
unchanged, and the mode records that a certificate is not a secret: a
``curl --cacert ~/.aureax/tls/ca.pem`` must not need ``sudo`` to read a public
key that a previous ``sudo`` run left owned by root.

Five decisions worth stating, because each prevents a specific failure:

**EC P-256, not RSA-2048.** Leaves are minted on every ``aureax generate`` —
which is on the interactive path of ``deploy`` and ``apply`` — one per enabled
target plus the orchestrator. P-256 keygen is sub-millisecond; RSA-2048 is
100-500 ms each, so a ten-node fleet would pay several seconds per apply for
nothing. Every consumer in this fleet handles P-256: OpenSSL 3 (llama-server
via cpp-httplib), Python's ``ssl`` (uvicorn, httpx), curl, the macOS keychain
and every current browser. There is no legacy stack here to accommodate.

**397-day leaves.** The operator is told to trust ``ca.pem`` in the macOS
keychain, and Safari refuses a certificate whose lifetime exceeds 398 days when
it chains to a keychain-trusted root. A ten-year leaf would work in ``curl``
and fail in the browser with an error that names nothing actionable. 397 is the
largest value that survives every verifier we ship against.

**One hour of backdate on ``notBefore``.** These nodes commonly have no RTC and
step their clock once time sync converges, which can be *after* the pod has
started. A certificate issued at the instant a node's clock is twenty minutes
behind is ``notBefore``-invalid there, and the failure presents as "certificate
not trusted" — pointing at the CA rather than at the clock.

**The CA is never rotated automatically.** :func:`ensure_ca` creates it iff
absent and refuses to replace one that exists; within
``CA_WARN_BEFORE_DAYS`` of expiry it logs loudly and ``aureax tls status`` says
so. Replacing it requires ``aureax tls regenerate --ca``, because a rotation
that happened by itself mid-``apply`` would push the new CA to some nodes and
leave the rest on the old one — the dashboard would then be unable to reach
half its fleet, mid-run, with no way to tell which half.

**The CA carries name constraints.** README tells the operator to add ``ca.pem``
to the macOS keychain as a trusted *root*, and ``ca.key`` is passphrase-less on
a laptop that gets backed up. Without constraints, whoever ends up holding that
one file can mint a padlock-green certificate for any name on the internet, so
losing it would cost the operator their whole browsing session rather than nine
nodes. :func:`fleet_name_constraints` derives the permitted subtrees from the
fleet file — the private/loopback/link-local ranges, plus a host-sized subtree
for any fleet address outside them, plus ``localhost`` and every hostname-shaped
target host — and :func:`ensure_ca` bakes them in **at creation only**, because
the CA is never rotated silently. A verifier rejects a leaf outright when *any*
of its SANs falls outside those subtrees (not just that one name), which is why
``generate_all_edge_configs`` checks every planned SAN set against the CA on
disk before it mints anything.

The CA private key is the single fleet-wide secret: it forges every node's
identity and the dashboard's. It is passphrase-less because every mint runs
unattended from ``generate_all_edge_configs``.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import os
import secrets
import ssl
import time
from pathlib import Path
from typing import TYPE_CHECKING

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from orchestrator.config.atomic import write_private_file

# The same rule SSHConfig._check_names_are_path_segments applies to a node id,
# imported rather than restated so the fleet-file boundary and this path builder
# cannot drift apart. schema imports nothing from here, so there is no cycle.
from orchestrator.config.schema import _PATH_SEGMENT_RE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from orchestrator.config.schema import NetworkConfig, SSHTarget

logger = logging.getLogger("aureax.config.tlsstore")

# Where the node finds its material. Fixed names, not ``<node_id>.pem``: the
# deployer's install command and client/tls.py then each use one literal, and
# renaming a node in aureax-network.yaml cannot strand a file in
# /etc/aureax/tls that no config names any more.
NODE_CA_FILE = "/etc/aureax/tls/ca.pem"
NODE_CERT_FILE = "/etc/aureax/tls/node.pem"
NODE_KEY_FILE = "/etc/aureax/tls/node.key"

CA_VALIDITY_DAYS = 3650
LEAF_VALIDITY_DAYS = 397
LEAF_RENEW_BEFORE_DAYS = 30
CA_WARN_BEFORE_DAYS = 90
BACKDATE = dt.timedelta(hours=1)

CA_COMMON_NAME = "Aureax Fleet CA"
CA_ORGANIZATION = "Aureax"

# The leaf name reserved for the CA itself, so `status()` and `regenerate()`
# never mistake ca.pem for a leaf.
_CA_STEM = "ca"

# The orchestrator's own leaf, used by the dashboard listener.
ORCHESTRATOR_NAME = "orchestrator"

_CERT_MODE = 0o644
_KEY_MODE = 0o600
_DIR_MODE = 0o700


def store_dir() -> Path:
    """The TLS directory. A function, not a module constant: tests monkeypatch
    $HOME (tests/conftest.py tmp_aureax_home), and a Path bound at import time
    would write into the developer's real ~/.aureax."""
    return Path.home() / ".aureax" / "tls"


def ca_cert_path() -> Path:
    return store_dir() / "ca.pem"


def ca_key_path() -> Path:
    return store_dir() / "ca.key"


def _check_leaf_name(name: str, *, allow_reserved: bool) -> str:
    """``name`` unchanged, or ValueError. Guards the two paths we build from it.

    The check lives at the path builder, not only at the fleet-file boundary:
    ``aureax tls regenerate --node <name>`` hands a raw operator string straight
    to :func:`regenerate`, and a "/" or ".." there would write a PEM outside the
    store — or unlink one. Same reasoning as keystore._check_node_id.

    Two names are the store's own. ``ca`` is never a leaf at all. ``orchestrator``
    is the dashboard's pair, so a node with that id would share the dashboard's
    certificate *and private key* — and ``aureax apply`` would push them to that
    node, handing it an identity for the orchestrator's own address. Both
    callers inside this module pass ``allow_reserved=True``; a node id never
    does (SSHConfig._check_names_are_path_segments refuses the two ids as well,
    so this is the second of two fences).

    Both reserved comparisons are case-insensitive. The store lives on the
    orchestrator's own filesystem, whose default macOS volume is
    case-insensitive, so ``CA.pem`` and ``ca.pem`` are one file: an exact-match
    check would let a leaf named ``CA`` overwrite the fleet's issuer, and
    ``Orchestrator`` hand a node the dashboard's private key. The error reports
    the spelling that was passed in.
    """
    if not _PATH_SEGMENT_RE.fullmatch(name or ""):
        raise ValueError(
            f"TLS leaf name {name!r} must be a single path segment: a letter or digit, then "
            f"letters, digits, '.', '_' or '-' (max 63 chars). It is joined straight into "
            f"{store_dir()}/<name>.pem and .key."
        )
    if name.casefold() == _CA_STEM:
        raise ValueError(
            f"TLS leaf name {name!r} is reserved for the fleet CA ({_CA_STEM!r}), not a leaf name"
        )
    if name.casefold() == ORCHESTRATOR_NAME and not allow_reserved:
        raise ValueError(
            f"TLS leaf name {name!r} is reserved for the dashboard's own certificate "
            f"({ORCHESTRATOR_NAME!r}) — a node with that id would be pushed the "
            f"orchestrator's private key"
        )
    return name


def leaf_paths(name: str, *, allow_reserved: bool = False) -> tuple[Path, Path]:
    """``(cert_path, key_path)`` for a leaf named ``name``.

    ``name`` is a node id, or ``"orchestrator"`` for the store's own callers
    (which pass ``allow_reserved=True``). See :func:`_check_leaf_name` for what
    is refused and why.
    """
    _check_leaf_name(name, allow_reserved=allow_reserved)
    d = store_dir()
    return d / f"{name}.pem", d / f"{name}.key"


def _ensure_store_dir() -> Path:
    """The TLS directory, existing and 0700.

    The chmod is not redundant with mkdir's ``mode``: that is honoured only when
    mkdir actually CREATES the leaf, so a ~/.aureax/tls restored from a backup
    or made by a hand-run `mkdir -p` keeps its 0755 and lets any local account
    read every node's private key. Mirrors keystore._ensure_store_dir.
    """
    d = store_dir()
    d.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    os.chmod(d, _DIR_MODE)
    return d


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


# --- SAN spelling -----------------------------------------------------------
# SANs travel as "IP:..." / "DNS:..." strings rather than x509.GeneralName
# objects. They are what reissue_reason compares (set equality on strings is
# trivially correct; on IPv4Address/DNSName objects it needs normalisation
# first), what `aureax tls status` prints, and what the README documents.


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _host_san(host: str) -> str:
    return f"IP:{host}" if _is_ip(host) else f"DNS:{host}"


def _san_entry(spec: str) -> x509.GeneralName:
    """One ``"IP:..."``/``"DNS:..."`` string as an x509 name."""
    kind, _, value = spec.partition(":")
    if kind == "IP":
        return x509.IPAddress(ipaddress.ip_address(value))
    if kind == "DNS":
        return x509.DNSName(value)
    raise ValueError(f"unsupported SAN spec {spec!r} — expected 'IP:<addr>' or 'DNS:<name>'")


def _san_spec(name: x509.GeneralName) -> str:
    """The inverse of :func:`_san_entry`, for reading a cert back."""
    if isinstance(name, x509.IPAddress):
        return f"IP:{name.value}"
    if isinstance(name, x509.DNSName):
        return f"DNS:{name.value}"
    return f"OTHER:{name!r}"


# On every leaf, node and orchestrator alike. Named because `regenerate` resets
# the orchestrator's accumulated SAN union down to exactly these.
_LOOPBACK_SANS = ("IP:127.0.0.1", "IP:::1", "DNS:localhost")


def orchestrator_sans(orchestrator_ip: str) -> list[str]:
    """SANs for the dashboard's leaf.

    ``orchestrator_ip`` is loader.validate_orchestrator_host's output, so it is
    an IP or a hostname. The loopback entries are load-bearing, not decoration:
    the loopback listener is plain HTTP, but a developer who reaches the HTTPS
    listener as ``https://127.0.0.1:7001`` must not be told the certificate is
    for someone else.
    """
    return [_host_san(orchestrator_ip), *_LOOPBACK_SANS]


def node_sans(target: SSHTarget) -> list[str]:
    """SANs for one node's leaf.

    The loopback entries cover every in-pod client: client/entrypoint.py maps a
    wildcard bind to 127.0.0.1, so the supervisor's own health wait, LlamaClient
    and the telemetry collector all dial loopback against this same leaf.

    Deliberately no ``DNS:<node_id>``: nothing in the fleet dials a node by id —
    every URL is built from ``ssh.targets[].host`` (templating.mcp_endpoints,
    the dashboard's chat and access services, the harness) — so that name only
    widened what the CA has to permit, and a node id is not a resolvable name.
    """
    return [_host_san(target.host), *_LOOPBACK_SANS]


# --- name constraints -------------------------------------------------------
# What the fleet CA is allowed to certify. See the module docstring: the
# operator trusts this root system-wide, so an unconstrained CA makes one
# leaked file a universal MITM credential.

# The ranges a LAN fleet actually lives in. Listed rather than derived from each
# host's /24 so that re-addressing a node inside the LAN — routine here, nodes
# are wiped and re-imaged — does not need a CA rotation, which would mean
# re-trusting the root in every browser.
_PRIVATE_IP_SUBTREES = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
)


def _host_subtree(host: str) -> str | None:
    """The IP subtree ``host`` needs beyond :data:`_PRIVATE_IP_SUBTREES`.

    ``None`` when the address is already covered. A fleet may legitimately sit
    on public addresses — the shipped example uses 192.0.2.x — and a CA that
    refused to certify them would make the example fleet ungeneratable, so each
    such address is permitted as itself (/32 or /128) and nothing wider.
    """
    if not _is_ip(host):
        return None
    addr = ipaddress.ip_address(host)
    if any(addr in ipaddress.ip_network(net) for net in _PRIVATE_IP_SUBTREES):
        return None
    return f"{addr}/{addr.max_prefixlen}"


def fleet_name_constraints(
    network_config: NetworkConfig, orchestrator_ip: str
) -> tuple[list[str], list[str]]:
    """``(permitted_ips, permitted_dns)`` for a CA minted for this fleet.

    Walks ``ssh.targets``, not ``enabled_targets``: the constraints are baked
    once and the CA is never rotated automatically, so a node that is disabled
    today and enabled next month must not need a new root.
    """
    permitted_ips = list(_PRIVATE_IP_SUBTREES)
    permitted_dns = ["localhost"]
    for host in [orchestrator_ip, *(t.host for t in network_config.ssh.targets)]:
        if _is_ip(host):
            subtree = _host_subtree(host)
            if subtree and subtree not in permitted_ips:
                permitted_ips.append(subtree)
        elif host and host not in permitted_dns:
            # A hostname-shaped host is a DNS subtree, which also permits names
            # under it — the fleet may add "edge-02.lan" beside "edge-01.lan".
            permitted_dns.append(host)
    return permitted_ips, permitted_dns


def ca_name_constraints(
    ca_cert: x509.Certificate | None = None,
) -> tuple[list[str], list[str]] | None:
    """The ``(permitted_ips, permitted_dns)`` a CA constrains its leaves to.

    ``None`` — meaning "this CA may certify anything" — for a CA minted before
    constraints existed, or one that cannot be read. Reads ``ca.pem`` when no
    certificate is passed, and never creates one: this is a reporting path.
    """
    if ca_cert is None:
        try:
            ca_cert = _load_cert(ca_cert_path())
        except (OSError, ValueError):
            return None
    try:
        nc = ca_cert.extensions.get_extension_for_class(x509.NameConstraints).value
    except x509.ExtensionNotFound:
        return None
    permitted_ips = [str(g.value) for g in nc.permitted_subtrees or [] if isinstance(g, x509.IPAddress)]
    permitted_dns = [g.value for g in nc.permitted_subtrees or [] if isinstance(g, x509.DNSName)]
    return permitted_ips, permitted_dns


def _dns_permitted(name: str, subtree: str) -> bool:
    """RFC 5280 dNSName subtree matching: the name itself, or anything under it."""
    return name == subtree or name.endswith(f".{subtree}")


def san_constraint_violations(sans: list[str], ca_cert: x509.Certificate) -> list[str]:
    """The entries of ``sans`` this CA's name constraints forbid, as sentences.

    Empty for an unconstrained CA, and empty for a name type the CA does not
    constrain at all — RFC 5280 leaves such a type unrestricted, which is why
    :func:`fleet_name_constraints` always returns both lists.

    Worth checking before minting rather than after: a verifier rejects the
    whole certificate when *one* SAN is outside the permitted subtrees, so a
    leaf minted anyway is not a leaf that works for its other names.
    """
    constraints = ca_name_constraints(ca_cert)
    if constraints is None:
        return []
    permitted_ips, permitted_dns = constraints
    violations = []
    for spec in sans:
        kind, _, value = spec.partition(":")
        if kind == "IP" and permitted_ips:
            addr = ipaddress.ip_address(value)
            if not any(addr in ipaddress.ip_network(net) for net in permitted_ips):
                violations.append(
                    f"{spec} is outside the CA's permitted address ranges "
                    f"({', '.join(permitted_ips)})"
                )
        elif kind == "DNS" and permitted_dns:
            if not any(_dns_permitted(value, s) for s in permitted_dns):
                violations.append(
                    f"{spec} is outside the CA's permitted DNS names ({', '.join(permitted_dns)})"
                )
    return violations


# --- reading and writing PEM ------------------------------------------------


def _generate_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _cert_pem(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.PEM)


def _load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _load_key(path: Path) -> ec.EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{path} is not an EC private key")
    return key


def _spki(pub) -> bytes:
    """DER SubjectPublicKeyInfo — the comparable form of a public key.

    Compared as bytes rather than with ``==`` on the key objects so the answer
    does not depend on which cryptography version implements __eq__ for which
    key type: a false "keys differ" here reissues every leaf on every run.
    """
    return pub.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def _write_key(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    write_private_file(path, _key_pem(key), mode=_KEY_MODE, dir_mode=_DIR_MODE)


def _write_cert(path: Path, cert: x509.Certificate) -> None:
    write_private_file(path, _cert_pem(cert), mode=_CERT_MODE, dir_mode=_DIR_MODE)


# --- the CA -----------------------------------------------------------------


def _build_ca_cert(
    key: ec.EllipticCurvePrivateKey,
    permitted_ips: list[str] | None = None,
    permitted_dns: list[str] | None = None,
) -> x509.Certificate:
    now = _now()
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, CA_COMMON_NAME),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, CA_ORGANIZATION),
    ])
    pub = key.public_key()
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(pub)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - BACKDATE)
        .not_valid_after(now + dt.timedelta(days=CA_VALIDITY_DAYS))
        # pathlen 0: this CA signs leaves and nothing else. An intermediate
        # minted from it would be a second issuer for the whole fleet.
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        # keyCertSign is not optional decoration — cryptography's own path
        # validator (and OpenSSL's) refuses a chain whose issuer lacks it.
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(pub), critical=False)
    )
    if permitted_ips or permitted_dns:
        # Critical, as RFC 5280 requires of a conforming CA — and as macOS,
        # Chrome and Firefox need in order to enforce it on a locally trusted
        # root, which is the whole point of the extension here. Both name types
        # are constrained together: an unconstrained type stays unrestricted, so
        # a DNS-only constraint would still let this key sign any IP SAN.
        builder = builder.add_extension(
            x509.NameConstraints(
                permitted_subtrees=(
                    [x509.IPAddress(ipaddress.ip_network(n)) for n in permitted_ips or []]
                    + [x509.DNSName(d) for d in permitted_dns or []]
                ),
                excluded_subtrees=None,
            ),
            critical=True,
        )
    return builder.sign(key, hashes.SHA256())


def _ca_ski(ca_cert: x509.Certificate) -> x509.SubjectKeyIdentifier:
    return ca_cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value


def _warn_if_ca_expiring(cert: x509.Certificate) -> None:
    days = (cert.not_valid_after_utc - _now()).days
    if days <= CA_WARN_BEFORE_DAYS:
        logger.warning(
            "fleet CA %s expires in %d days — plan `aureax tls regenerate --ca`, then "
            "`aureax apply --all` and re-trust the CA in every browser; nothing rotates it "
            "automatically because a rotation mid-apply would split the fleet",
            ca_cert_path(),
            days,
        )


def ensure_ca(
    *, permitted_ips: list[str] | None = None, permitted_dns: list[str] | None = None
) -> Path:
    """The fleet CA certificate path, creating the CA iff it is absent.

    Never replaces an existing CA — see the module docstring. A certificate
    without a key, or a certificate that does not match its key, raises rather
    than being quietly rebuilt: a silently new CA is a fleet that half trusts
    the orchestrator, discovered as TLS failures on whichever nodes were not
    re-applied. A key without a certificate is the opposite case — the creation
    window of a concurrent caller — and is waited out by ``_await_ca_cert()``,
    which raises only if the certificate never appears.

    The name constraints apply at CREATION only, for the same reason: an
    existing CA is returned untouched even when the fleet has since grown names
    it does not permit. ``generate_all_edge_configs`` reports that as a refusal
    naming `aureax tls regenerate --ca`, which is the one place a root is
    replaced deliberately.
    """
    cert_path, key_path = ca_cert_path(), ca_key_path()
    if cert_path.exists() and key_path.exists():
        try:
            cert = _load_cert(cert_path)
            key = _load_key(key_path)
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"fleet CA at {cert_path} / {key_path} cannot be read ({e}). Recover the pair from "
                f"a backup, or run `aureax tls regenerate --ca` and then `aureax apply --all`."
            ) from e
        if _spki(cert.public_key()) != _spki(key.public_key()):
            raise RuntimeError(
                f"fleet CA is inconsistent: {cert_path} was not signed by {key_path}. Every leaf "
                f"signed from here would verify against neither. Run `aureax tls regenerate --ca`, "
                f"then `aureax apply --all` and re-trust the CA in every browser."
            )
        _warn_if_ca_expiring(cert)
        return cert_path
    if key_path.exists():
        # Key without certificate is the creation window, not corruption: the
        # caller that won the os.link below publishes ca.key and only then
        # spends a keygen and a cert build on ca.pem. Raising here told an
        # operator whose `aureax generate` merely overlapped the dashboard's
        # regenerate to delete the CA the other caller was already signing
        # leaves with. _await_ca_cert has its own bounded wait and its own
        # message for a writer that really did die mid-creation.
        return _await_ca_cert()
    if cert_path.exists():
        # The other half. No concurrent writer can produce it — the key is
        # published first — so this one is a genuine restore-from-backup case.
        raise RuntimeError(
            f"fleet CA is half present ({cert_path.exists()=}, {key_path.exists()=}). Restore the "
            f"missing file from a backup, or remove both and run `aureax tls regenerate --ca` "
            f"followed by `aureax apply --all`."
        )

    _ensure_store_dir()
    key = _generate_key()
    # Create-if-absent, not last-writer-wins: `aureax generate` from the CLI and
    # the dashboard's regenerate can overlap and nothing serialises them, and
    # two winners would leave ca.key from one and ca.pem from the other — a
    # fleet whose leaves verify against neither. os.link FAILS when the
    # destination exists, so exactly one caller publishes the key and only that
    # caller writes the matching certificate.
    staged = store_dir() / f"ca.key.{os.getpid()}.{secrets.token_hex(4)}.staging"
    write_private_file(staged, _key_pem(key), mode=_KEY_MODE, dir_mode=_DIR_MODE)
    try:
        os.link(staged, key_path)
    except FileExistsError:
        return _await_ca_cert()
    finally:
        # The link (when it succeeded) keeps the inode; this only drops the
        # second name. On the lost-race path it is the whole cleanup.
        staged.unlink(missing_ok=True)

    if not (permitted_ips or permitted_dns):
        # Reachable from any caller that has no fleet file to derive them from —
        # `aureax dashboard` on a store that has never been generated is the
        # real one. Loud, because the CA is then created once and never rotated
        # silently, and README tells the operator to trust it system-wide.
        logger.warning(
            "creating the fleet CA at %s with NO name constraints: this root can certify any name "
            "on the internet, and it is the root the operator is told to trust in the keychain. "
            "`aureax generate --orchestrator-ip <ip>` mints a constrained one; on an existing "
            "store, `aureax tls regenerate --ca --orchestrator-ip <ip>` replaces it.",
            cert_path,
        )
    _write_cert(cert_path, _build_ca_cert(key, permitted_ips, permitted_dns))
    logger.info(
        "created fleet CA %s (valid %d days, permitted %s %s)",
        cert_path,
        CA_VALIDITY_DAYS,
        permitted_ips or "any address",
        permitted_dns or "any name",
    )
    return cert_path


def _await_ca_cert() -> Path:
    """Wait for the caller that won the ca.key race to publish ca.pem."""
    cert_path = ca_cert_path()
    for _ in range(200):
        if cert_path.exists():
            return cert_path
        time.sleep(0.01)
    raise RuntimeError(
        f"{ca_key_path()} exists but {cert_path} never appeared — a concurrent `aureax generate` "
        f"died mid-creation, or ca.pem was deleted. Restore ca.pem from a backup, or remove both "
        f"files and re-run."
    )


def load_ca_cert() -> x509.Certificate:
    """The loaded CA certificate, creating the CA if absent.

    Named apart from ``reissue_reason``'s ``ca_cert`` parameter on purpose: the
    two would otherwise shadow each other inside the one function that reads
    the CA's SubjectKeyIdentifier."""
    return _load_cert(ensure_ca())


# --- leaves -----------------------------------------------------------------


def reissue_reason(
    cert_path: Path,
    key_path: Path,
    want_sans: list[str],
    ca_cert: x509.Certificate,
) -> str | None:
    """Why the leaf at these paths must be reissued, or ``None`` to keep it.

    Pure: reads two files and decides. Every caller that mints goes through
    here, so "when is a certificate stale" is answered in exactly one place and
    ``aureax tls status`` can show the same sentence the next generate will act
    on.
    """
    try:
        cert = _load_cert(cert_path)
    except FileNotFoundError:
        return "no certificate yet"
    except (OSError, ValueError) as e:
        return f"certificate unreadable ({e})"
    try:
        key = _load_key(key_path)
    except FileNotFoundError:
        return "no private key yet"
    except (OSError, ValueError) as e:
        return f"private key unreadable ({e})"

    try:
        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        have = {_san_spec(n) for n in san_ext}
    except x509.ExtensionNotFound:
        have = set()
    want = set(want_sans)
    if have != want:
        # A node that moved to a new address keeps serving a certificate for the
        # old one, which every client rejects by hostname — not by expiry — so
        # nothing else in this list would ever notice.
        return f"SANs changed: have {sorted(have)}, want {sorted(want)}"

    now = _now()
    remaining = cert.not_valid_after_utc - now
    if remaining < dt.timedelta(days=LEAF_RENEW_BEFORE_DAYS):
        return f"expires in {remaining.days} days"
    if cert.not_valid_before_utc > now + BACKDATE:
        # Issued against a clock that has since stepped backwards. Left alone,
        # nothing accepts this certificate until the original issue time.
        return f"notBefore {cert.not_valid_before_utc.isoformat()} is in the future"

    try:
        aki = cert.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value
        aki_id = aki.key_identifier
    except x509.ExtensionNotFound:
        aki_id = None
    if aki_id != _ca_ski(ca_cert).digest:
        return "signed by a different CA than the current ~/.aureax/tls/ca.pem"

    if _spki(cert.public_key()) != _spki(key.public_key()):
        # There is no two-file atomic rename, so a crash or a lost race between
        # the key write and the cert write leaves a mismatched pair. This is the
        # rule that repairs it; without it the node serves a certificate its own
        # key cannot complete a handshake for.
        return "certificate does not match the private key on disk"
    return None


def _build_leaf_cert(
    name: str,
    sans: list[str],
    public_key,
    ca_cert: x509.Certificate,
    ca_key: ec.EllipticCurvePrivateKey,
) -> x509.Certificate:
    now = _now()
    return (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
        .issuer_name(ca_cert.subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - BACKDATE)
        .not_valid_after(now + dt.timedelta(days=LEAF_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        # digitalSignature alone. keyEncipherment is meaningless for ECDSA (no
        # RSA key transport in any modern cipher suite) and some strict
        # verifiers reject the combination outright.
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        # serverAuth only: there is no mTLS anywhere in this design, and
        # clientAuth would advertise a capability nothing uses.
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.SubjectAlternativeName([_san_entry(s) for s in sans]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(_ca_ski(ca_cert)),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )


def _issue_leaf(name: str, sans: list[str], cert_path: Path, key_path: Path) -> None:
    """Mint one leaf, writing the KEY first and the certificate second.

    The order matters and is the reason reissue_reason has a key-mismatch rule:
    a crash between the two writes leaves the old certificate beside the new
    key, which is detectable, rather than a new certificate beside the old key,
    which llama-server would load and then fail every handshake with.
    """
    ca = load_ca_cert()
    ca_key = _load_key(ca_key_path())
    key = _generate_key()
    _write_key(key_path, key)
    _write_cert(cert_path, _build_leaf_cert(name, sans, key.public_key(), ca, ca_key))


def ensure_leaf(name: str, sans: list[str], *, allow_reserved: bool = False) -> tuple[Path, Path]:
    """``(cert_path, key_path)`` for ``name``, minting or reissuing as needed.

    Idempotent: on a healthy store this reads two files and returns, so the
    serial numbers survive every ``aureax generate`` — which is what makes
    ``apply`` a no-op for TLS on an unchanged fleet instead of a fleet-wide
    certificate change on each run.

    ``allow_reserved`` is :func:`ensure_orchestrator_leaf`'s: every other caller
    names a node, and a node may not claim the store's own file names.
    """
    cert_path, key_path = leaf_paths(name, allow_reserved=allow_reserved)
    _ensure_store_dir()
    ca = load_ca_cert()
    reason: str | None = None
    # Three attempts, not one: two concurrent generates for the same node
    # interleave their key and cert writes, and the loser sees a mismatched
    # pair on re-check. One more mint settles it; a third failure is a real
    # fault (a full disk, a root-owned file) and must not spin forever.
    for _ in range(3):
        reason = reissue_reason(cert_path, key_path, sans, ca)
        if reason is None:
            return cert_path, key_path
        logger.info("issuing TLS certificate for %s (%s)", name, reason)
        _issue_leaf(name, sans, cert_path, key_path)
    reason = reissue_reason(cert_path, key_path, sans, ca)
    if reason is None:
        return cert_path, key_path
    raise RuntimeError(
        f"could not produce a usable TLS certificate for {name!r} under {store_dir()} after 3 "
        f"attempts — last reason: {reason}"
    )


def orchestrator_leaf_sans(orchestrator_ip: str) -> list[str]:
    """The SANs the orchestrator leaf will carry: a UNION, not a replacement.

    This one leaf has two producers with different inputs — ``aureax generate
    --orchestrator-ip <x>`` and ``aureax dashboard``, which mints for the LAN
    address it detects. Taking either verbatim made whichever ran last delete
    the other's name, and since every node's telemetry and log endpoint is
    stamped with the *generate* value, a dashboard start could silently end
    fleet ingest: the dashboard's own URL keeps working, the nodes' pushes all
    fail verification, and the Overview tab just stops receiving.

    The cost is that a Mac which moves between networks accumulates addresses
    on this certificate. That is bounded and visible in ``aureax tls status``,
    and `aureax tls regenerate --node orchestrator` resets it (see
    :func:`regenerate`).
    """
    cert_path, _ = leaf_paths(ORCHESTRATOR_NAME, allow_reserved=True)
    return sorted(set(orchestrator_sans(orchestrator_ip)) | set(_sans_of(cert_path)))


def ensure_orchestrator_leaf(orchestrator_ip: str) -> tuple[Path, Path]:
    """The dashboard's leaf for every address the fleet may reach it by."""
    sans = orchestrator_leaf_sans(orchestrator_ip)
    violations = san_constraint_violations(sans, load_ca_cert())
    if violations:
        # An accumulated name can fall outside a CA minted for a narrower
        # fleet. Minting anyway would produce a certificate no verifier accepts
        # for ANY of its names, so the dashboard would fail to serve the fleet
        # with a hostname error rather than this sentence.
        raise ValueError(
            "the dashboard's certificate cannot be issued under the fleet CA:\n  "
            + "\n  ".join(violations)
            + f"\nRun `aureax tls regenerate --ca --orchestrator-ip {orchestrator_ip}` to mint a "
            "CA that covers today's fleet, then `aureax apply --all` and re-trust the CA."
        )
    return ensure_leaf(ORCHESTRATOR_NAME, sans, allow_reserved=True)


def ensure_node_leaf(target: SSHTarget) -> tuple[Path, Path]:
    """One node's leaf, named by its node id."""
    return ensure_leaf(target.node_id, node_sans(target))


# --- the client side --------------------------------------------------------

# Keyed by (path, mtime_ns, size) rather than by path alone. Building a context
# costs ~5 ms and every httpx client on the Mac side shares this one, but a
# `aureax tls regenerate --ca` inside the same process must not leave a
# long-lived caller verifying against a CA that is no longer on disk.
_ssl_contexts: dict[tuple[str, int, int], ssl.SSLContext] = {}


def ca_ssl_context() -> ssl.SSLContext:
    """The shared ``SSLContext`` that trusts the fleet CA.

    Every orchestrator-side httpx client passes this as ``verify=``. httpx
    0.28 deprecates ``verify="<path>"``, and building a context per client
    would add its cost to each of the dashboard's per-node polls.

    Reads the CA; deliberately does NOT create one. A client path that minted
    the fleet's root of trust as a side effect would produce a CA no node has
    ever been given, so every request would still fail — but with a hostname or
    verification error pointing at the node instead of at the missing store.
    """
    path = ca_cert_path()
    if not path.exists():
        raise RuntimeError(
            f"no fleet CA at {path} — run `aureax generate --orchestrator-ip <ip>` to create the "
            f"CA and every node's certificate, then `aureax apply --all` to push them."
        )
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    ctx = _ssl_contexts.get(key)
    if ctx is None:
        ctx = ssl.create_default_context(cafile=str(path))
        _ssl_contexts[key] = ctx
    return ctx


# --- reporting and forced reissue -------------------------------------------


def _leaf_names() -> list[str]:
    """Every leaf the store holds, from either half of a pair.

    Both halves, so a leaf whose certificate is missing still shows up in
    `aureax tls status` with its reason instead of vanishing from the report.

    Stems that are not valid leaf names are skipped rather than returned. Both
    consumers hand every entry to ``leaf_paths``, which raises for anything
    ``_check_leaf_name`` refuses, so one stray file — a hand-copied
    ``my ca.pem``, a duplicate named ``orchestrator copy.pem`` — would make
    ``aureax tls status`` traceback and would abort ``regenerate(ca=True)``
    partway through the reissue loop, i.e. after the root has already been
    replaced. That is the half-rotated, two-CA fleet this module exists to
    prevent, caused by a file that is not a leaf at all.
    """
    d = store_dir()
    if not d.is_dir():
        return []
    names = {p.stem for p in d.glob("*.pem")} | {p.stem for p in d.glob("*.key")}
    keep: set[str] = set()
    for name in names:
        # casefold, mirroring _check_leaf_name: on a case-insensitive volume
        # "CA.pem" IS the issuer, and reporting it as a leaf would offer to
        # reissue the root as a server certificate.
        if name.casefold() == _CA_STEM:
            continue
        if not _PATH_SEGMENT_RE.fullmatch(name):
            logger.debug("ignoring %s in %s: not a leaf name", name, d)
            continue
        keep.add(name)
    return sorted(keep)


def _sans_of(cert_path: Path) -> list[str]:
    try:
        cert = _load_cert(cert_path)
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except (OSError, ValueError, x509.ExtensionNotFound):
        return []
    return sorted(_san_spec(n) for n in ext)


def status() -> dict:
    """Plain dicts describing the store, for ``aureax tls status``.

    Never mints — the same rule the Access tab's poll follows for API keys, so
    a status command cannot create fleet credentials as a side effect. That is
    also why each leaf's ``reissue_reason`` is computed against the SANs the
    certificate already carries: this function has no fleet config to say what
    the SANs *should* be, so it reports every staleness rule except the SAN
    one. ``aureax generate`` applies the full rule with the real target list.
    """
    cert_path, key_path = ca_cert_path(), ca_key_path()
    ca_info: dict = {
        "path": str(cert_path),
        "key_path": str(key_path),
        "present": False,
        "subject": None,
        "not_after": None,
        "expires_in_days": None,
        "expiring_soon": False,
        # (permitted_ips, permitted_dns), or None for a CA minted before
        # constraints existed — which is worth showing, since such a root can
        # certify any name on the internet while it sits in the keychain.
        "name_constraints": None,
        "problem": None,
    }
    ca: x509.Certificate | None = None
    if cert_path.exists() and key_path.exists():
        try:
            ca = _load_cert(cert_path)
        except (OSError, ValueError) as e:
            ca_info["problem"] = f"certificate unreadable ({e})"
        else:
            days = (ca.not_valid_after_utc - _now()).days
            constraints = ca_name_constraints(ca)
            ca_info.update(
                present=True,
                subject=ca.subject.rfc4514_string(),
                not_after=ca.not_valid_after_utc.isoformat(),
                expires_in_days=days,
                expiring_soon=days <= CA_WARN_BEFORE_DAYS,
                name_constraints=(
                    None
                    if constraints is None
                    else {"permitted_ips": constraints[0], "permitted_dns": constraints[1]}
                ),
            )
    elif cert_path.exists() or key_path.exists():
        ca_info["problem"] = "half present — one of ca.pem / ca.key is missing"
    else:
        ca_info["problem"] = "absent — run `aureax generate` or `aureax tls regenerate --ca`"

    leaves = []
    for name in _leaf_names():
        lcert, lkey = leaf_paths(name, allow_reserved=True)
        sans = _sans_of(lcert)
        entry: dict = {
            "name": name,
            "cert_path": str(lcert),
            "key_path": str(lkey),
            "present": lcert.exists() and lkey.exists(),
            "sans": sans,
            "not_after": None,
            "expires_in_days": None,
            "reissue_reason": None,
        }
        if lcert.exists():
            try:
                cert = _load_cert(lcert)
            except (OSError, ValueError):
                pass
            else:
                entry["not_after"] = cert.not_valid_after_utc.isoformat()
                entry["expires_in_days"] = (cert.not_valid_after_utc - _now()).days
        entry["reissue_reason"] = (
            reissue_reason(lcert, lkey, sans, ca) if ca is not None else "no CA to check against"
        )
        leaves.append(entry)

    return {"store_dir": str(store_dir()), "ca": ca_info, "leaves": leaves}


def regenerate(
    *,
    ca: bool = False,
    names: list[str] | None = None,
    permitted_ips: list[str] | None = None,
    permitted_dns: list[str] | None = None,
) -> None:
    """Force reissue. ``ca=True`` replaces the CA *and* every leaf under it.

    The only way the CA is ever replaced, and the only way its name constraints
    change: pass ``fleet_name_constraints(config, ip)`` alongside ``ca=True`` so
    the new root covers today's fleet. Replacing the CA invalidates every leaf
    (reissue_reason's AKI rule), so the leaves are reissued in the same call —
    otherwise the fleet would be split between the moment the CA changed and
    the next ``aureax generate``, which is exactly the window this module's
    docstring says must not exist.

    A leaf is reissued with the SANs its current certificate carries; one whose
    certificate cannot be read has its pair removed instead, so the next
    ``aureax generate`` mints it from the fleet config, which is the only place
    that knows the intended SANs. The orchestrator's leaf is the exception: its
    SAN set is a running union (see :func:`orchestrator_leaf_sans`), so keeping
    it would preserve exactly the stale addresses this command exists to drop.
    It is reset to the loopback names every orchestrator leaf carries, and the
    next ``aureax dashboard`` or ``aureax generate`` unions today's address back
    in.

    ``names`` reaches here straight from ``aureax tls regenerate --node``, and
    the rest of the target list comes from a directory glob, so the whole list
    is checked as path segments BEFORE anything is unlinked.
    """
    # The full target list is computed and checked BEFORE the CA is touched.
    # Inside the loop, a name the path builder refuses would abort the run after
    # the root had already been replaced and some leaves reissued — the
    # half-rotated fleet this command exists to avoid. _leaf_names() drops
    # anything unnameable already; this is the fence that makes the promise
    # hold whatever it returns. The CA stems are excluded from _leaf_names(), so
    # computing this before the unlink below sees the same set as after it.
    if ca:
        # --ca forces every leaf; an explicit name list only widens it.
        targets = sorted(set(_leaf_names()) | set(names or []))
    else:
        targets = list(names) if names is not None else _leaf_names()
    for name in targets:
        _check_leaf_name(name, allow_reserved=True)

    if ca:
        logger.warning(
            "replacing the fleet CA at %s — every node needs `aureax apply --all` and every "
            "browser needs the new CA trusted again before it can reach the dashboard",
            ca_cert_path(),
        )
        ca_cert_path().unlink(missing_ok=True)
        ca_key_path().unlink(missing_ok=True)
        ensure_ca(permitted_ips=permitted_ips, permitted_dns=permitted_dns)

    if not targets:
        # Nothing to reissue, and load_ca_cert() below would CREATE a CA — a
        # root no node has ever been given, minted by a command that only
        # promises to replace what is already there.
        return
    ca_cert = load_ca_cert()
    for name in targets:
        cert_path, key_path = leaf_paths(name, allow_reserved=True)
        sans = _sans_of(cert_path)
        if name.casefold() == ORCHESTRATOR_NAME:
            sans = [s for s in sans if s in _LOOPBACK_SANS]
        if not sans:
            logger.warning(
                "cannot reissue %s: its certificate is missing or carries no SANs, so the intended "
                "names are unknown here. Removing the pair — `aureax generate` will mint it from "
                "the fleet config.",
                name,
            )
            cert_path.unlink(missing_ok=True)
            key_path.unlink(missing_ok=True)
            continue
        violations = san_constraint_violations(sans, ca_cert)
        if violations:
            # Warn rather than raise: this is the repair command, and aborting
            # halfway through would leave the fleet split across two CAs. The
            # certificate is still written, but nothing will accept it for ANY
            # of its names, so the line has to say what to run next.
            logger.warning(
                "%s carries names the fleet CA does not permit (%s) — no verifier will accept "
                "this certificate. Re-mint the CA with `aureax tls regenerate --ca "
                "--orchestrator-ip <ip>`, then `aureax apply --all`.",
                name,
                "; ".join(violations),
            )
        _issue_leaf(name, sans, cert_path, key_path)
        logger.info("reissued TLS certificate for %s", name)
