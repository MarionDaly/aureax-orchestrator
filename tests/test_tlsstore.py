"""The fleet CA and leaf store under ~/.aureax/tls.

Three properties carry the design, and every test here pins one of them.

*Shape*: the certificates have to satisfy real verifiers — Python's ``ssl``,
OpenSSL 3 behind llama-server, curl and a browser — so the extensions are
asserted against ``cryptography``'s own path validator rather than merely
read back. A CA without ``keyCertSign`` or a leaf without ``serverAuth``
parses perfectly and fails every handshake.

*Idempotence*: leaves are minted on every ``aureax generate``, which is on the
interactive path of ``deploy`` and ``apply``. A generator that reissued each
run would change every node's certificate on every apply, so the serial
numbers surviving a second call is a behaviour, not a detail.

*Repair*: there is no two-file atomic rename, so a crash or a lost race
between the key write and the certificate write leaves a mismatched pair.
``reissue_reason`` has one rule per way that store can go stale, and each of
those rules gets a test.

Everything runs against ``tmp_aureax_home``, which redirects $HOME —
``tlsstore.store_dir()`` is a function, not a module constant, precisely so
this works.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
import logging
import os
import re
import ssl
import stat
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID
from cryptography.x509.verification import PolicyBuilder, Store

from orchestrator.config import tlsstore
from orchestrator.config.schema import SSHTarget

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes only")


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _cert(path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _sans(cert: x509.Certificate) -> set[str]:
    ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    return {
        (f"IP:{n.value}" if isinstance(n, x509.IPAddress) else f"DNS:{n.value}")
        for n in ext
    }


def _foreign_key_pem() -> bytes:
    """A private key that is not the one any certificate here was signed for."""
    return ec.generate_private_key(ec.SECP256R1()).private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _target(node_id: str = "edge-01", host: str = "10.77.0.31") -> SSHTarget:
    return SSHTarget(node_id=node_id, host=host)


# --- the CA -----------------------------------------------------------------


def test_ensure_ca_writes_a_pathlen_zero_signing_ca(tmp_aureax_home) -> None:
    """CA:TRUE with pathlen 0 and keyCertSign.

    pathlen 0 keeps this from ever issuing a second issuer for the fleet, and
    keyCertSign is what a path validator requires of an issuer — without it the
    CA parses fine and every chain built from it is rejected.
    """
    path = tlsstore.ensure_ca()
    cert = _cert(path)

    bc = cert.extensions.get_extension_for_class(x509.BasicConstraints)
    assert bc.value.ca is True
    assert bc.value.path_length == 0
    assert bc.critical is True

    ku = cert.extensions.get_extension_for_class(x509.KeyUsage)
    assert ku.value.key_cert_sign is True
    assert ku.critical is True

    assert "Aureax Fleet CA" in cert.subject.rfc4514_string()
    assert cert.issuer == cert.subject
    assert cert.extensions.get_extension_for_class(x509.SubjectKeyIdentifier) is not None


def test_ca_key_is_p256(tmp_aureax_home) -> None:
    """P-256, not RSA: leaves are minted on every generate, so keygen cost lands
    on the interactive deploy/apply path."""
    tlsstore.ensure_ca()
    key = serialization.load_pem_private_key(tlsstore.ca_key_path().read_bytes(), password=None)

    assert isinstance(key, ec.EllipticCurvePrivateKey)
    assert isinstance(key.curve, ec.SECP256R1)


def test_ca_validity_is_ten_years(tmp_aureax_home) -> None:
    cert = _cert(tlsstore.ensure_ca())
    now = dt.datetime.now(dt.UTC)

    assert (cert.not_valid_after_utc - now).days == pytest.approx(tlsstore.CA_VALIDITY_DAYS, abs=1)


@posix_only
def test_store_modes_are_key_0600_cert_0644_dir_0700(tmp_aureax_home) -> None:
    """The certificate is deliberately world-readable and the key is not.

    ``curl --cacert ~/.aureax/tls/ca.pem`` must never need sudo, and the 0700
    directory is what actually keeps the keys unreachable — so the two modes
    are asserted together, not separately.
    """
    tlsstore.ensure_ca()
    cert_path, key_path = tlsstore.ensure_orchestrator_leaf("10.77.0.2")

    assert _mode(tlsstore.ca_cert_path()) == 0o644
    assert _mode(tlsstore.ca_key_path()) == 0o600
    assert _mode(cert_path) == 0o644
    assert _mode(key_path) == 0o600
    assert _mode(tlsstore.store_dir()) == 0o700


@posix_only
def test_store_dir_is_reclamped_to_0700(tmp_aureax_home) -> None:
    """A ~/.aureax/tls restored from a backup or made by a hand-run `mkdir -p`
    keeps its 0755 and lets any local account read every node's private key."""
    tlsstore.ensure_ca()
    os.chmod(tlsstore.store_dir(), 0o755)

    tlsstore.ensure_leaf("edge-01", tlsstore.node_sans(_target()))

    assert _mode(tlsstore.store_dir()) == 0o700


def test_ensure_ca_never_replaces_an_existing_ca(tmp_aureax_home) -> None:
    """A CA that rotated by itself would push a new root to some nodes and leave
    the rest on the old one, mid-apply, with no way to tell which half."""
    first = _cert(tlsstore.ensure_ca()).serial_number

    for _ in range(3):
        tlsstore.ensure_ca()

    assert _cert(tlsstore.ca_cert_path()).serial_number == first


def test_ensure_ca_refuses_a_half_present_pair(tmp_aureax_home) -> None:
    tlsstore.ensure_ca()
    tlsstore.ca_key_path().unlink()

    with pytest.raises(RuntimeError, match="half present"):
        tlsstore.ensure_ca()


def test_ensure_ca_refuses_a_cert_that_does_not_match_its_key(tmp_aureax_home) -> None:
    """Every leaf signed from a mismatched pair verifies against neither half,
    which is otherwise invisible until the whole fleet fails to hand shake."""
    tlsstore.ensure_ca()
    tlsstore.ca_key_path().write_bytes(_foreign_key_pem())

    with pytest.raises(RuntimeError, match="was not signed by"):
        tlsstore.ensure_ca()


def _widen_the_creation_window(monkeypatch, seconds: float = 0.3) -> None:
    """Hold ca.pem back so every caller lands INSIDE the creation window.

    The window is real but sub-millisecond, so a test that merely races threads
    passes on most runs and fails on some — which is how the half-present bug
    was first seen. Widening it here makes both tests below deterministic.
    """
    real = tlsstore._write_cert

    def slow(path, cert):
        if path == tlsstore.ca_cert_path():
            time.sleep(seconds)
        real(path, cert)

    monkeypatch.setattr(tlsstore, "_write_cert", slow)


def test_concurrent_ensure_ca_converges_on_one_ca(tmp_aureax_home, monkeypatch) -> None:
    """`aureax generate` from the CLI and the dashboard's regenerate can overlap
    and nothing serialises them. Two winners would leave ca.key from one caller
    and ca.pem from the other."""
    _widen_the_creation_window(monkeypatch)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: tlsstore.ensure_ca(), range(8)))

    cert = _cert(tlsstore.ca_cert_path())
    key = serialization.load_pem_private_key(tlsstore.ca_key_path().read_bytes(), password=None)
    spki = serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    assert cert.public_key().public_bytes(*spki) == key.public_key().public_bytes(*spki)
    assert not list(tlsstore.store_dir().glob("*.staging"))


def test_a_caller_inside_the_creation_window_waits_instead_of_crying_corruption(
    tmp_aureax_home, monkeypatch
) -> None:
    """ca.key published, ca.pem not yet, is the winner mid-creation — not a
    half-present store. It used to raise, and the message told the operator to
    delete both files: the CA the other caller was already signing leaves with.
    """
    _widen_the_creation_window(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        winner = pool.submit(tlsstore.ensure_ca)
        while not tlsstore.ca_key_path().exists():
            time.sleep(0.005)
        assert not tlsstore.ca_cert_path().exists(), "the window closed before the test entered it"

        late = tlsstore.ensure_ca()

    assert late == winner.result() == tlsstore.ca_cert_path()
    assert _cert(late).subject.rfc4514_string().endswith(tlsstore.CA_COMMON_NAME)


def test_a_key_with_no_writer_behind_it_still_reports_a_broken_store(
    tmp_aureax_home, monkeypatch
) -> None:
    """The wait is bounded: a writer that really did die leaves ca.key alone
    forever, and that has to end in an error naming both files rather than a
    hang. `time.sleep` is stubbed so the 2 s budget costs nothing here."""
    tlsstore.ensure_ca()
    tlsstore.ca_cert_path().unlink()
    monkeypatch.setattr(tlsstore.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="never appeared"):
        tlsstore.ensure_ca()


# --- SAN spelling -----------------------------------------------------------


def test_orchestrator_sans_for_an_ipv4_address() -> None:
    assert tlsstore.orchestrator_sans("10.77.0.2") == [
        "IP:10.77.0.2",
        "IP:127.0.0.1",
        "IP:::1",
        "DNS:localhost",
    ]


def test_orchestrator_sans_for_an_ipv6_address() -> None:
    """No brackets here — bracketing belongs to URL authorities
    (templating._url_host), and an ``IP:`` SAN carries the bare address."""
    assert tlsstore.orchestrator_sans("fd00::1")[0] == "IP:fd00::1"


def test_orchestrator_sans_for_a_hostname() -> None:
    """validate_orchestrator_host accepts a hostname (an mDNS name is a
    plausible operator input), so the SAN kind has to follow the value."""
    assert tlsstore.orchestrator_sans("mac.local")[0] == "DNS:mac.local"


def test_node_sans_are_the_host_and_loopback_only() -> None:
    """The loopback entries are load-bearing: every in-pod client dials
    127.0.0.1 against this same leaf.

    No ``DNS:<node_id>``. Nothing dials a node by id — every URL in the fleet is
    built from ``ssh.targets[].host`` — so that name bought nothing and widened
    what the CA has to permit, which is now an enforced constraint rather than
    a comment.
    """
    assert tlsstore.node_sans(_target("edge-07", "192.0.2.7")) == [
        "IP:192.0.2.7",
        "IP:127.0.0.1",
        "IP:::1",
        "DNS:localhost",
    ]
    assert "DNS:edge-07" not in tlsstore.node_sans(_target("edge-07", "192.0.2.7"))


# --- names the store owns ----------------------------------------------------


def test_leaf_paths_refuses_the_reserved_ca_name(tmp_aureax_home) -> None:
    """A node id of "ca" would overwrite the fleet's issuer with a leaf."""
    with pytest.raises(ValueError, match="reserved"):
        tlsstore.leaf_paths("ca")
    with pytest.raises(ValueError, match="reserved"):
        tlsstore.leaf_paths("ca", allow_reserved=True)


@pytest.mark.parametrize("name", ["CA", "Ca", "ORCHESTRATOR", "Orchestrator"])
def test_leaf_paths_refuses_a_reserved_name_in_any_case(name, tmp_aureax_home) -> None:
    """The store is on the orchestrator's own filesystem and the default macOS
    volume is case-insensitive, so ~/.aureax/tls/CA.pem IS ca.pem: an
    exact-match check would let a node named "CA" overwrite the fleet's issuer,
    and "Orchestrator" be handed the dashboard's private key."""
    with pytest.raises(ValueError, match="reserved"):
        tlsstore.leaf_paths(name)


def test_leaf_paths_refuses_the_orchestrator_name_for_a_node(tmp_aureax_home) -> None:
    """orchestrator.pem/.key is the dashboard's pair. A node with that id would
    share it — and `aureax apply` pushes leaf_paths(node_id) to the node, so
    that node would be handed the orchestrator's private key and could present
    a fleet-CA-valid certificate for the dashboard's own address."""
    with pytest.raises(ValueError, match="reserved for the dashboard"):
        tlsstore.leaf_paths("orchestrator")

    cert, _ = tlsstore.leaf_paths("orchestrator", allow_reserved=True)
    assert cert == tlsstore.store_dir() / "orchestrator.pem"


def test_ensure_node_leaf_refuses_a_reserved_node_id(tmp_aureax_home) -> None:
    """The mint path, not just the path builder: a fleet file is not the only
    way a target reaches here (the dashboard rewrites it over an
    unauthenticated bind)."""
    tlsstore.ensure_orchestrator_leaf("10.77.0.2")
    before = tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes()

    with pytest.raises(ValueError, match="reserved for the dashboard"):
        tlsstore.ensure_node_leaf(_target("orchestrator", "10.77.0.9"))

    assert tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].read_bytes() == before


@pytest.mark.parametrize("name", ["../../.ssh/id_ed25519", "a/b", "..", "", ".hidden"])
def test_leaf_paths_refuses_anything_that_is_not_one_path_segment(name, tmp_aureax_home) -> None:
    """`aureax tls regenerate --node <name>` hands an operator string straight
    to regenerate, which unlinks or overwrites both files at leaf_paths(name)."""
    with pytest.raises(ValueError, match="single path segment"):
        tlsstore.leaf_paths(name, allow_reserved=True)


# --- name constraints --------------------------------------------------------
# The operator is told to trust ca.pem as a ROOT in the macOS keychain, and
# ca.key is passphrase-less on a laptop. Without these subtrees, that one file
# signs a valid certificate for any name on the internet.


def _fleet_config(hosts: dict[str, str], disabled: tuple[str, ...] = ()):
    from tests.conftest import make_network_config

    return make_network_config([
        {"node_id": nid, "host": host, "enabled": nid not in disabled}
        for nid, host in hosts.items()
    ])


def test_fleet_constraints_cover_the_private_ranges_and_localhost() -> None:
    ips, dns = tlsstore.fleet_name_constraints(
        _fleet_config({"edge-01": "10.77.0.31"}), "192.168.1.20"
    )

    assert set(tlsstore._PRIVATE_IP_SUBTREES) <= set(ips)
    assert dns == ["localhost"]


def test_a_fleet_address_outside_the_private_ranges_is_permitted_as_itself() -> None:
    """The shipped example fleet is 192.0.2.x, which is not private — a CA that
    refused it would make the example ungeneratable. Permitted as a host route,
    never as the surrounding /24."""
    ips, _ = tlsstore.fleet_name_constraints(
        _fleet_config({"edge-01": "192.0.2.7"}), "192.0.2.1"
    )

    assert "192.0.2.7/32" in ips
    assert "192.0.2.1/32" in ips
    assert not any(net.startswith("192.0.2.0/") for net in ips)


def test_an_ipv6_fleet_address_is_permitted_as_a_128(tmp_aureax_home) -> None:
    ips, _ = tlsstore.fleet_name_constraints(
        _fleet_config({"edge-01": "2001:db8::7"}), "10.77.0.2"
    )

    assert "2001:db8::7/128" in ips


def test_hostname_shaped_hosts_become_dns_subtrees() -> None:
    """A target host may be a name — the schema allows it and _host_san emits a
    DNS SAN for it — so the CA has to permit that name too."""
    _, dns = tlsstore.fleet_name_constraints(
        _fleet_config({"edge-01": "edge-01.lan"}), "mac.local"
    )

    assert dns == ["localhost", "mac.local", "edge-01.lan"]


def test_disabled_targets_are_covered_too() -> None:
    """The constraints are baked once and the CA is never rotated
    automatically, so a node disabled today and enabled next month must not
    need a new root — which would mean re-trusting it in every browser."""
    ips, _ = tlsstore.fleet_name_constraints(
        _fleet_config({"a": "192.0.2.7", "b": "192.0.2.8"}, disabled=("b",)), "10.77.0.2"
    )

    assert "192.0.2.8/32" in ips


def test_the_ca_carries_its_constraints_as_a_critical_extension(tmp_aureax_home) -> None:
    """Critical because RFC 5280 requires it of a conforming CA, and because
    macOS, Chrome and Firefox enforce name constraints on a locally trusted
    root — which is the whole point of putting them there."""
    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8"], permitted_dns=["localhost"])

    ext = _cert(tlsstore.ca_cert_path()).extensions.get_extension_for_class(x509.NameConstraints)
    assert ext.critical is True
    assert tlsstore.ca_name_constraints() == (["10.0.0.0/8"], ["localhost"])


def test_creating_an_unconstrained_ca_says_so_loudly(tmp_aureax_home, caplog) -> None:
    """`aureax dashboard` on a store that has never been generated has no fleet
    file to derive subtrees from, and the CA it creates there is then never
    rotated silently — so the one chance to say it is at creation."""
    with caplog.at_level(logging.WARNING, logger="aureax.config.tlsstore"):
        tlsstore.ensure_ca()

    assert any("NO name constraints" in r.getMessage() for r in caplog.records)


def test_a_ca_minted_without_constraints_reports_none(tmp_aureax_home) -> None:
    """A store created before this rule existed still verifies; nothing here
    may treat "no constraints" as "constrained to nothing"."""
    tlsstore.ensure_ca()

    assert tlsstore.ca_name_constraints() is None
    assert tlsstore.san_constraint_violations(["DNS:anything.test"], _cert(tlsstore.ca_cert_path())) == []


def test_constraints_are_applied_at_creation_only(tmp_aureax_home) -> None:
    """The CA is never rotated silently, so a second call with wider
    constraints must leave the existing root alone — `aureax generate` reports
    the mismatch instead."""
    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8"], permitted_dns=["localhost"])
    serial = _cert(tlsstore.ca_cert_path()).serial_number

    tlsstore.ensure_ca(permitted_ips=["0.0.0.0/0"], permitted_dns=["evil.test"])

    assert _cert(tlsstore.ca_cert_path()).serial_number == serial
    assert tlsstore.ca_name_constraints() == (["10.0.0.0/8"], ["localhost"])


@pytest.mark.parametrize(
    ("san", "expected"),
    [
        ("IP:8.8.8.8", "permitted address ranges"),
        ("DNS:login.yourbank.com", "permitted DNS names"),
    ],
)
def test_san_constraint_violations_names_the_offending_entry(san, expected, tmp_aureax_home) -> None:
    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8", "127.0.0.0/8"], permitted_dns=["localhost"])

    problems = tlsstore.san_constraint_violations([san, "IP:10.77.0.5"], _cert(tlsstore.ca_cert_path()))

    assert len(problems) == 1
    assert san in problems[0] and expected in problems[0]


def test_a_subdomain_of_a_permitted_name_is_permitted(tmp_aureax_home) -> None:
    """RFC 5280 dNSName subtrees cover the name and everything under it."""
    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8"], permitted_dns=["lan"])

    ca = _cert(tlsstore.ca_cert_path())
    assert tlsstore.san_constraint_violations(["DNS:edge-01.lan"], ca) == []
    assert tlsstore.san_constraint_violations(["DNS:notlan"], ca) != []


def test_a_fleet_leaf_verifies_under_the_constrained_ca(tmp_aureax_home) -> None:
    """The constraints must not cost the fleet anything: every SAN a node leaf
    carries is inside them, checked with a real path validator."""
    target = _target("edge-01", "192.0.2.7")
    ips, dns = tlsstore.fleet_name_constraints(
        _fleet_config({"edge-01": "192.0.2.7"}), "10.77.0.2"
    )
    tlsstore.ensure_ca(permitted_ips=ips, permitted_dns=dns)
    ca = _cert(tlsstore.ca_cert_path())
    cert_path, _ = tlsstore.ensure_node_leaf(target)

    for subject in (
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("192.0.2.7")),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address("::1")),
    ):
        verifier = PolicyBuilder().store(Store([ca])).build_server_verifier(subject)
        assert verifier.verify(_cert(cert_path), []) is not None


def test_a_leaf_outside_the_constraints_does_not_verify(tmp_aureax_home) -> None:
    """The reason san_constraint_violations is checked BEFORE minting: a
    verifier rejects the certificate, not merely the offending name."""
    from cryptography.x509.verification import VerificationError

    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8", "127.0.0.0/8", "::1/128"],
                       permitted_dns=["localhost"])
    ca = _cert(tlsstore.ca_cert_path())
    cert_path, _ = tlsstore.ensure_leaf("rogue", ["DNS:login.yourbank.com", "DNS:localhost"])

    for subject in (x509.DNSName("login.yourbank.com"), x509.DNSName("localhost")):
        verifier = PolicyBuilder().store(Store([ca])).build_server_verifier(subject)
        with pytest.raises(VerificationError):
            verifier.verify(_cert(cert_path), [])


def _handshake(cert_path, key_path, server_hostname: str) -> None:
    """A real OpenSSL handshake against the fleet CA, in memory.

    cryptography's path validator and OpenSSL's are separate implementations,
    and OpenSSL is the one that actually runs here — uvicorn, httpx and curl all
    sit on it. Name constraints are only worth adding if THAT verifier enforces
    them, and only safe to add if it still accepts every fleet leaf.
    """
    server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server.load_cert_chain(str(cert_path), str(key_path))
    client = ssl.create_default_context(cafile=str(tlsstore.ca_cert_path()))
    s_in, s_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    c_in, c_out = ssl.MemoryBIO(), ssl.MemoryBIO()
    ends = (
        (client.wrap_bio(c_in, c_out, server_hostname=server_hostname), c_out, s_in),
        (server.wrap_bio(s_in, s_out, server_side=True), s_out, c_in),
    )
    done: set[int] = set()
    for _ in range(20):
        for i, (end, out_bio, peer_in) in enumerate(ends):
            if i in done:
                continue
            try:
                end.do_handshake()
                done.add(i)
            except ssl.SSLWantReadError:
                pass
            payload = out_bio.read()
            if payload:
                peer_in.write(payload)
        if len(done) == len(ends):
            return
    raise AssertionError(f"handshake for {server_hostname!r} did not converge")


def test_a_node_leaf_still_handshakes_under_the_constrained_ca(tmp_aureax_home) -> None:
    ips, dns = tlsstore.fleet_name_constraints(_fleet_config({"edge-01": "192.0.2.7"}), "10.77.0.2")
    tlsstore.ensure_ca(permitted_ips=ips, permitted_dns=dns)
    cert_path, key_path = tlsstore.ensure_node_leaf(_target("edge-01", "192.0.2.7"))

    for name in ("127.0.0.1", "localhost", "192.0.2.7"):
        _handshake(cert_path, key_path, name)


def test_openssl_refuses_a_name_the_ca_may_not_certify(tmp_aureax_home) -> None:
    """The blast radius of a leaked ca.key, measured: with the CA in the
    keychain as a trusted root, this is what a forged bank certificate does."""
    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8", "127.0.0.0/8", "::1/128"],
                       permitted_dns=["localhost"])
    cert_path, key_path = tlsstore.ensure_leaf("rogue", ["DNS:login.yourbank.com"])

    with pytest.raises(ssl.SSLError, match="permitted subtree violation"):
        _handshake(cert_path, key_path, "login.yourbank.com")


# --- leaf shape -------------------------------------------------------------


def test_leaf_sans_are_exactly_node_sans(tmp_aureax_home) -> None:
    target = _target()
    cert_path, _ = tlsstore.ensure_node_leaf(target)

    assert _sans(_cert(cert_path)) == set(tlsstore.node_sans(target))


def test_leaf_eku_is_server_auth_only(tmp_aureax_home) -> None:
    """clientAuth would advertise mTLS, which nothing in this design uses."""
    cert_path, _ = tlsstore.ensure_node_leaf(_target())
    eku = _cert(cert_path).extensions.get_extension_for_class(x509.ExtendedKeyUsage).value

    assert list(eku) == [ExtendedKeyUsageOID.SERVER_AUTH]


def test_leaf_key_usage_is_digital_signature_only(tmp_aureax_home) -> None:
    """keyEncipherment is meaningless for ECDSA (no RSA key transport in any
    modern suite) and some strict verifiers reject the combination."""
    cert_path, _ = tlsstore.ensure_node_leaf(_target())
    ku = _cert(cert_path).extensions.get_extension_for_class(x509.KeyUsage)

    assert ku.value.digital_signature is True
    assert ku.value.key_encipherment is False
    assert ku.value.key_cert_sign is False
    assert ku.critical is True


def test_leaf_is_not_a_ca(tmp_aureax_home) -> None:
    cert_path, _ = tlsstore.ensure_node_leaf(_target())

    assert _cert(cert_path).extensions.get_extension_for_class(x509.BasicConstraints).value.ca is False


def test_leaf_aki_matches_the_ca_ski(tmp_aureax_home) -> None:
    """The link reissue_reason uses to notice the CA was regenerated."""
    ca = _cert(tlsstore.ensure_ca())
    cert_path, _ = tlsstore.ensure_node_leaf(_target())

    aki = _cert(cert_path).extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value
    ski = ca.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value
    assert aki.key_identifier == ski.digest


def test_leaf_verifies_against_the_ca_for_every_san(tmp_aureax_home) -> None:
    """Asserted with a real path validator, not by reading extensions back: a
    chain that parses and does not validate is the failure mode this whole
    module exists to avoid."""
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, _ = tlsstore.ensure_node_leaf(target)
    leaf = _cert(cert_path)
    store = Store([ca])

    for subject in (
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address(target.host)),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ):
        verifier = PolicyBuilder().store(store).build_server_verifier(subject)
        assert verifier.verify(leaf, []) is not None


def test_leaf_does_not_verify_for_an_unlisted_name(tmp_aureax_home) -> None:
    from cryptography.x509.verification import VerificationError

    ca = _cert(tlsstore.ensure_ca())
    cert_path, _ = tlsstore.ensure_node_leaf(_target())
    verifier = PolicyBuilder().store(Store([ca])).build_server_verifier(x509.DNSName("elsewhere.test"))

    with pytest.raises(VerificationError):
        verifier.verify(_cert(cert_path), [])


def test_leaf_validity_is_397_days_backdated_one_hour(tmp_aureax_home) -> None:
    """397 because Safari refuses anything over 398 days chaining to a
    keychain-trusted root; backdated because these nodes commonly have no RTC
    and step the clock after the pod has already started."""
    now = dt.datetime.now(dt.UTC)
    cert_path, _ = tlsstore.ensure_node_leaf(_target())
    cert = _cert(cert_path)

    assert cert.not_valid_before_utc <= now - dt.timedelta(minutes=55)
    assert cert.not_valid_before_utc >= now - dt.timedelta(minutes=65)
    span = cert.not_valid_after_utc - cert.not_valid_before_utc
    assert span == dt.timedelta(days=tlsstore.LEAF_VALIDITY_DAYS) + tlsstore.BACKDATE


# --- idempotence ------------------------------------------------------------


def test_ensure_leaf_is_idempotent(tmp_aureax_home) -> None:
    """A generator that reissued on every run would change every node's
    certificate on every `aureax apply`."""
    target = _target()
    cert_path, key_path = tlsstore.ensure_node_leaf(target)
    first_serial = _cert(cert_path).serial_number
    first_key = key_path.read_bytes()

    tlsstore.ensure_node_leaf(target)
    tlsstore.ensure_node_leaf(target)

    assert _cert(cert_path).serial_number == first_serial
    assert key_path.read_bytes() == first_key


def test_ensure_leaf_does_not_touch_the_ca(tmp_aureax_home) -> None:
    ca_serial = _cert(tlsstore.ensure_ca()).serial_number

    tlsstore.ensure_node_leaf(_target())

    assert _cert(tlsstore.ca_cert_path()).serial_number == ca_serial


# --- reissue rules, one test each -------------------------------------------


def test_reissue_when_the_certificate_is_missing(tmp_aureax_home) -> None:
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.leaf_paths("edge-01")

    assert tlsstore.reissue_reason(cert_path, key_path, ["DNS:localhost"], ca) == "no certificate yet"


def test_reissue_when_the_key_is_missing(tmp_aureax_home) -> None:
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.ensure_node_leaf(target)
    key_path.unlink()

    reason = tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), ca)
    assert reason == "no private key yet"


def test_reissue_when_the_certificate_is_unparseable(tmp_aureax_home) -> None:
    """A truncated PEM — the state os.replace exists to prevent, still possible
    if something outside this module writes the file."""
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.ensure_node_leaf(target)
    cert_path.write_bytes(b"-----BEGIN CERTIFICATE-----\nnot a certificate\n")

    reason = tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), ca)
    assert reason is not None
    assert "unreadable" in reason


def test_reissue_when_the_san_set_differs(tmp_aureax_home) -> None:
    """A node that moved to a new address keeps serving a certificate for the
    old one, which clients reject by name and no expiry rule would ever see."""
    ca = _cert(tlsstore.ensure_ca())
    old = _target(host="10.77.0.31")
    cert_path, key_path = tlsstore.ensure_node_leaf(old)

    reason = tlsstore.reissue_reason(
        cert_path, key_path, tlsstore.node_sans(_target(host="10.77.0.99")), ca
    )
    assert reason is not None
    assert reason.startswith("SANs changed")


def test_reissue_within_the_renewal_window(tmp_aureax_home, monkeypatch) -> None:
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.ensure_node_leaf(target)

    almost_expired = dt.datetime.now(dt.UTC) + dt.timedelta(
        days=tlsstore.LEAF_VALIDITY_DAYS - tlsstore.LEAF_RENEW_BEFORE_DAYS + 1
    )
    monkeypatch.setattr(tlsstore, "_now", lambda: almost_expired)

    reason = tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), ca)
    assert reason is not None
    assert reason.startswith("expires in")


def test_reissue_when_not_before_is_in_the_future(tmp_aureax_home, monkeypatch) -> None:
    """The orchestrator's clock stepped backwards after issuing. Left alone,
    nothing accepts the certificate until the original issue time comes round."""
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.ensure_node_leaf(target)

    monkeypatch.setattr(tlsstore, "_now", lambda: dt.datetime.now(dt.UTC) - dt.timedelta(hours=3))

    reason = tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), ca)
    assert reason is not None
    assert "notBefore" in reason


def test_backdate_alone_does_not_trip_the_future_rule(tmp_aureax_home) -> None:
    """The one-hour backdate is inside the tolerance, so a freshly minted leaf
    is never reissued for its own notBefore."""
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.ensure_node_leaf(target)

    assert tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), ca) is None


def test_reissue_when_the_ca_changed(tmp_aureax_home) -> None:
    target = _target()
    tlsstore.ensure_ca()
    cert_path, key_path = tlsstore.ensure_node_leaf(target)

    tlsstore.ca_cert_path().unlink()
    tlsstore.ca_key_path().unlink()
    new_ca = _cert(tlsstore.ensure_ca())

    reason = tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), new_ca)
    assert reason == "signed by a different CA than the current ~/.aureax/tls/ca.pem"


def test_reissue_when_the_key_does_not_match_the_certificate(tmp_aureax_home) -> None:
    """The half-written pair: there is no two-file atomic rename, so a crash
    between the key write and the cert write has to be detectable."""
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.ensure_node_leaf(target)
    key_path.write_bytes(_foreign_key_pem())

    reason = tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), ca)
    assert reason == "certificate does not match the private key on disk"


def test_ensure_leaf_repairs_a_half_written_pair(tmp_aureax_home) -> None:
    target = _target()
    ca = _cert(tlsstore.ensure_ca())
    cert_path, key_path = tlsstore.ensure_node_leaf(target)
    stale_serial = _cert(cert_path).serial_number
    key_path.write_bytes(_foreign_key_pem())

    tlsstore.ensure_node_leaf(target)

    assert tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), ca) is None
    assert _cert(cert_path).serial_number != stale_serial


def test_ensure_leaf_reissues_after_a_san_change(tmp_aureax_home) -> None:
    cert_path, _ = tlsstore.ensure_node_leaf(_target(host="10.77.0.31"))
    moved = _target(host="10.77.0.99")

    tlsstore.ensure_node_leaf(moved)

    assert _sans(_cert(cert_path)) == set(tlsstore.node_sans(moved))


def test_ensure_leaf_raises_when_it_cannot_settle(tmp_aureax_home, monkeypatch) -> None:
    """Three attempts, then a real fault (a full disk, a root-owned file) is
    reported instead of spinning forever."""
    monkeypatch.setattr(tlsstore, "_issue_leaf", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        tlsstore.ensure_node_leaf(_target())


# --- the shared SSLContext --------------------------------------------------


def test_ca_ssl_context_is_cached(tmp_aureax_home) -> None:
    """Every orchestrator-side httpx client passes this as verify=; building one
    per client would add ~5ms to each of the dashboard's per-node polls."""
    tlsstore.ensure_ca()
    first = tlsstore.ca_ssl_context()

    assert tlsstore.ca_ssl_context() is first


def test_ca_ssl_context_trusts_the_fleet_ca(tmp_aureax_home) -> None:
    tlsstore.ensure_ca()
    ctx = tlsstore.ca_ssl_context()
    subjects = {name for cert in ctx.get_ca_certs() for rdns in cert["subject"] for name in rdns}

    assert ("commonName", "Aureax Fleet CA") in subjects


def test_ca_ssl_context_follows_a_regenerated_ca(tmp_aureax_home) -> None:
    """A long-lived caller must not keep verifying against a CA that is no
    longer on disk after `aureax tls regenerate --ca`."""
    tlsstore.ensure_ca()
    first = tlsstore.ca_ssl_context()

    tlsstore.regenerate(ca=True)

    assert tlsstore.ca_ssl_context() is not first


# --- reporting and forced reissue -------------------------------------------


def test_ca_ssl_context_names_the_remedy_when_the_store_is_empty(tmp_aureax_home) -> None:
    """It must not mint: a CA created from a client path is trusted by no node,
    so every request would still fail — with a verification error pointing at
    the node rather than at the empty store."""
    with pytest.raises(RuntimeError, match="aureax generate"):
        tlsstore.ca_ssl_context()

    assert not tlsstore.ca_cert_path().exists()


def test_status_reports_the_ca_and_every_leaf(tmp_aureax_home) -> None:
    tlsstore.ensure_orchestrator_leaf("10.77.0.2")
    tlsstore.ensure_node_leaf(_target())

    report = tlsstore.status()

    assert report["ca"]["present"] is True
    assert report["ca"]["expires_in_days"] == pytest.approx(tlsstore.CA_VALIDITY_DAYS, abs=1)
    assert report["ca"]["expiring_soon"] is False
    assert [leaf["name"] for leaf in report["leaves"]] == ["edge-01", "orchestrator"]
    assert all(leaf["reissue_reason"] is None for leaf in report["leaves"])
    assert "IP:10.77.0.31" in report["leaves"][0]["sans"]


def test_status_mints_nothing(tmp_aureax_home) -> None:
    """Same rule the Access tab's poll follows for API keys: a status command
    must not create fleet credentials as a side effect."""
    report = tlsstore.status()

    assert report["ca"]["present"] is False
    assert report["leaves"] == []
    assert not tlsstore.store_dir().exists()


def test_regenerate_ca_replaces_the_ca_and_re_signs_every_leaf(tmp_aureax_home) -> None:
    """Leaves are reissued in the same call: otherwise the fleet is split
    between the moment the CA changed and the next `aureax generate`."""
    target = _target()
    old_ca = _cert(tlsstore.ensure_ca()).serial_number
    cert_path, key_path = tlsstore.ensure_node_leaf(target)
    old_leaf = _cert(cert_path).serial_number

    tlsstore.regenerate(ca=True)

    new_ca = _cert(tlsstore.ca_cert_path())
    assert new_ca.serial_number != old_ca
    assert _cert(cert_path).serial_number != old_leaf
    assert _sans(_cert(cert_path)) == set(tlsstore.node_sans(target))
    assert tlsstore.reissue_reason(cert_path, key_path, tlsstore.node_sans(target), new_ca) is None


# --- the orchestrator leaf's two producers ------------------------------------


def test_ensure_orchestrator_leaf_unions_rather_than_replaces(tmp_aureax_home) -> None:
    """`aureax generate --orchestrator-ip mac.local` stamps every node's
    telemetry endpoint with that name; `aureax dashboard` then mints for the
    LAN address it detects. Replacing dropped DNS:mac.local, so every node's
    push failed verification against a dashboard whose own URL still worked."""
    tlsstore.ensure_orchestrator_leaf("mac.local")

    cert_path, _ = tlsstore.ensure_orchestrator_leaf("192.168.1.20")

    assert _sans(_cert(cert_path)) == {
        "DNS:mac.local", "DNS:localhost", "IP:127.0.0.1", "IP:::1", "IP:192.168.1.20",
    }


def test_the_union_is_stable_so_the_two_producers_stop_thrashing(tmp_aureax_home) -> None:
    """Once both names are on the certificate, neither caller reissues — which
    is what keeps `apply` a no-op for TLS on an unchanged fleet."""
    tlsstore.ensure_orchestrator_leaf("mac.local")
    cert_path, _ = tlsstore.ensure_orchestrator_leaf("192.168.1.20")
    serial = _cert(cert_path).serial_number

    tlsstore.ensure_orchestrator_leaf("mac.local")
    tlsstore.ensure_orchestrator_leaf("192.168.1.20")

    assert _cert(cert_path).serial_number == serial


def test_a_union_member_the_ca_forbids_is_refused_with_the_remedy(tmp_aureax_home) -> None:
    """Minting anyway would produce a certificate no verifier accepts for ANY
    of its names, so the dashboard would fail with a hostname error instead of
    a sentence naming the command that fixes it."""
    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8", "127.0.0.0/8", "::1/128"],
                       permitted_dns=["localhost"])

    with pytest.raises(ValueError, match=re.escape("regenerate --ca --orchestrator-ip 8.8.8.8")):
        tlsstore.ensure_orchestrator_leaf("8.8.8.8")

    assert not tlsstore.leaf_paths("orchestrator", allow_reserved=True)[0].exists()


def test_regenerate_resets_the_orchestrator_union(tmp_aureax_home) -> None:
    """The union accumulates every address the Mac has ever been reached at, so
    reissuing it with what it carries would preserve exactly what an operator
    running this command wants dropped. The next dashboard start or generate
    unions today's address back in."""
    tlsstore.ensure_orchestrator_leaf("mac.local")
    cert_path, _ = tlsstore.ensure_orchestrator_leaf("192.168.1.20")

    tlsstore.regenerate(names=["orchestrator"])

    assert _sans(_cert(cert_path)) == {"DNS:localhost", "IP:127.0.0.1", "IP:::1"}

    tlsstore.ensure_orchestrator_leaf("192.168.1.20")
    assert _sans(_cert(cert_path)) == {
        "DNS:localhost", "IP:127.0.0.1", "IP:::1", "IP:192.168.1.20",
    }


def test_regenerate_validates_every_name_before_it_unlinks_anything(tmp_aureax_home) -> None:
    """--node values arrive from the operator's shell. A traversal used to
    reach leaf_paths unchecked, where the no-SANs branch unlinks both files."""
    tlsstore.ensure_node_leaf(_target())
    victim = tmp_aureax_home / "id_ed25519.pem"
    victim.write_text("not a certificate")

    with pytest.raises(ValueError, match="single path segment"):
        tlsstore.regenerate(names=["edge-01", "../id_ed25519"])

    assert victim.exists()


def test_a_stray_file_in_the_store_is_not_reported_as_a_leaf(tmp_aureax_home) -> None:
    """`aureax tls status` hands every stem to leaf_paths, which raises for
    anything that is not one path segment. A Finder duplicate or a hand-copied
    file would turn a read-only status command into a traceback."""
    tlsstore.ensure_node_leaf(_target())
    (tlsstore.store_dir() / "orchestrator copy.pem").write_text("not a certificate")
    (tlsstore.store_dir() / "..hidden.key").write_text("not a key")

    report = tlsstore.status()

    assert [leaf["name"] for leaf in report["leaves"]] == ["edge-01"]


def test_a_stray_file_cannot_abort_a_ca_rotation_partway(tmp_aureax_home) -> None:
    """regenerate(ca=True) replaces the root and THEN reissues every leaf. A
    name the path builder refuses, reached inside that loop, would leave the
    fleet split across two CAs — half the nodes verifying against a root that
    no longer signs anything."""
    tlsstore.ensure_node_leaf(_target())
    tlsstore.ensure_node_leaf(_target("edge-02", "10.77.0.3"))
    (tlsstore.store_dir() / "my ca.pem").write_text("not a certificate")

    tlsstore.regenerate(ca=True, permitted_ips=["10.0.0.0/8", "127.0.0.0/8", "::1/128"],
                        permitted_dns=["localhost"])

    ca = _cert(tlsstore.ca_cert_path())
    for node in ("edge-01", "edge-02"):
        cert_path, _ = tlsstore.leaf_paths(node)
        assert _cert(cert_path).issuer == ca.subject, f"{node} was left on the old CA"


def test_regenerate_on_an_empty_store_mints_nothing(tmp_aureax_home) -> None:
    """It promises to replace what is there. A CA created here would be a root
    no node has ever been given, so every request would still fail — with a
    verification error pointing at the node rather than at the empty store."""
    tlsstore.regenerate()

    assert not tlsstore.ca_cert_path().exists()


def test_regenerate_refuses_the_ca_as_a_leaf_name(tmp_aureax_home) -> None:
    tlsstore.ensure_ca()
    before = tlsstore.ca_cert_path().read_bytes()

    with pytest.raises(ValueError, match="reserved for the fleet CA"):
        tlsstore.regenerate(names=["ca"])

    assert tlsstore.ca_cert_path().read_bytes() == before


def test_regenerate_ca_takes_the_new_constraints(tmp_aureax_home) -> None:
    """The one place a root's constraints change — a fleet that outgrew the old
    CA has no other way back to a working store."""
    tlsstore.ensure_ca(permitted_ips=["10.0.0.0/8"], permitted_dns=["localhost"])

    tlsstore.regenerate(ca=True, permitted_ips=["10.0.0.0/8", "192.0.2.7/32"],
                        permitted_dns=["localhost", "mac.local"])

    assert tlsstore.ca_name_constraints() == (
        ["10.0.0.0/8", "192.0.2.7/32"], ["localhost", "mac.local"]
    )


def test_regenerate_warns_about_a_leaf_the_new_ca_cannot_certify(
    tmp_aureax_home, caplog
) -> None:
    """Reissued anyway — this is the repair command and aborting halfway would
    split the fleet across two CAs — but nothing will accept the result, so the
    line has to say so and name what to run next."""
    cert_path, _ = tlsstore.ensure_node_leaf(_target("edge-01", "8.8.8.8"))

    with caplog.at_level(logging.WARNING, logger="aureax.config.tlsstore"):
        tlsstore.regenerate(ca=True, permitted_ips=["10.0.0.0/8", "127.0.0.0/8", "::1/128"],
                            permitted_dns=["localhost"])

    assert any("IP:8.8.8.8" in r.getMessage() for r in caplog.records)
    assert cert_path.exists()


def test_regenerate_named_leaf_only(tmp_aureax_home) -> None:
    tlsstore.ensure_orchestrator_leaf("10.77.0.2")
    node_cert, _ = tlsstore.ensure_node_leaf(_target())
    orch_cert, _ = tlsstore.leaf_paths("orchestrator", allow_reserved=True)
    orch_serial = _cert(orch_cert).serial_number
    node_serial = _cert(node_cert).serial_number

    tlsstore.regenerate(names=["edge-01"])

    assert _cert(node_cert).serial_number != node_serial
    assert _cert(orch_cert).serial_number == orch_serial
