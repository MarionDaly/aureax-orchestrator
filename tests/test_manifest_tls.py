"""The TLS half of the shipped images and manifests.

TLS is mandatory fleet-wide, and three of the pieces that make it work live
outside Python entirely — where nothing else reads them:

* **The probe scheme.** llama-server serves HTTPS on :8080 now. The kubelet's
  httpGet default is HTTP, and a plaintext GET against a TLS listener fails
  every single time, so a probe left on the default kills a container that is
  answering perfectly — after the startupProbe's full 3015s budget, with
  nothing in the events naming the scheme.

* **The OpenSSL build.** llama.cpp's ``LLAMA_OPENSSL`` defaults ON but degrades
  to a CMake ``message(WARNING)`` when ``find_package(OpenSSL)`` fails
  (``vendor/cpp-httplib/CMakeLists.txt`` at b11429), so a builder image without
  ``libssl-dev`` still exits 0 and produces a llama-server that dies with
  "the server is built without SSL support" on the first head that starts. The
  Dockerfiles therefore assert on the artefact with ``ldd``; this module
  asserts the assertion is still there, in both of them, along with the three
  ingredients it checks for.

* **The option's spelling.** There is no ``LLAMA_SERVER_SSL`` at b11429. CMake
  reports a misspelt ``-D`` as an unused cache variable and succeeds, so that
  typo is invisible until the fleet crash-loops.

The images cannot be built here (no Docker on the orchestrator's test host), so
these are text assertions over the build recipe. They are the only automated
check that stands between a silently plaintext build and a deploy.
"""

from __future__ import annotations

import re

import pytest
from ruamel.yaml import YAML

from .conftest import PROJECT_ROOT

K8S_DIR = PROJECT_ROOT / "k8s"
DOCKERFILES = [
    PROJECT_ROOT / "docker" / "Dockerfile.client",
    PROJECT_ROOT / "docker" / "Dockerfile.client.cuda",
]

# --- k8s manifests: every httpGet probe speaks HTTPS --------------------------


def _containers(node) -> list[dict]:
    """Every container dict anywhere in a manifest document.

    Recursive rather than reaching into ``spec.template.spec.containers``: this
    repo ships DaemonSets today but the rule is about probes, not about a
    workload kind, and an initContainer or a manifest nested one level deeper
    would otherwise be silently exempt from the scheme check.
    """
    found: list[dict] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("containers", "initContainers") and isinstance(value, list):
                found += [c for c in value if isinstance(c, dict)]
            else:
                found += _containers(value)
    elif isinstance(node, list):
        for item in node:
            found += _containers(item)
    return found


def _http_probes() -> list[tuple[str, str, str, dict]]:
    """(manifest, container, probe name, httpGet block) for every HTTP probe.

    Only ``httpGet`` probes are collected. The shard worker's startupProbe (an
    ``exec`` of ``client.rpc_probe``) connects to rpc-server's :50052, which
    speaks a bespoke binary protocol and no HTTP at all — a scheme there would
    be meaningless, and :50052 is the one port that stays plaintext.
    """
    probes: list[tuple[str, str, str, dict]] = []
    for manifest in sorted(K8S_DIR.glob("*.yaml")):
        for doc in YAML().load_all(manifest.read_text()):
            for container in _containers(doc):
                name = str(container.get("name", "<unnamed>"))
                for key, probe in container.items():
                    if not key.endswith("Probe") or not isinstance(probe, dict):
                        continue
                    http_get = probe.get("httpGet")
                    if isinstance(http_get, dict):
                        probes.append((manifest.name, name, key, http_get))
    return probes


def test_every_httpget_probe_in_the_manifests_declares_https() -> None:
    """A probe left on the kubelet's HTTP default kills a healthy TLS server."""
    for manifest, container, probe_name, http_get in _http_probes():
        assert http_get.get("scheme") == "HTTPS", (
            f"{manifest}: container {container!r} has {probe_name}.httpGet without "
            f"scheme: HTTPS (got {http_get.get('scheme')!r}). llama-server serves TLS "
            "on this port, so the kubelet's plaintext default fails every check and "
            "the container is killed even though it is answering."
        )


def test_the_scheme_check_has_something_to_check() -> None:
    """Guards the test above against passing vacuously.

    Delete the probes — or rename the key so the walk stops finding them — and
    a for-loop over an empty list is green. This is the assertion that says the
    manifests still carry HTTP probes at all.
    """
    probes = _http_probes()
    assert probes, (
        "no httpGet probe found in any k8s/*.yaml — either the probes were removed "
        "(model load is then unguarded; see tests/test_probe_budget.py) or _containers() "
        "no longer finds them, which would make the scheme check pass on nothing"
    )


def test_the_shard_workers_probe_is_a_plain_connect_to_the_bound_address() -> None:
    """The exclusion above is scope, not an oversight.

    rpc-server on :50052 serves no HTTP and has no TLS at any layer, so its
    probe must never be an ``httpGet`` (it would fail forever). It is an
    ``exec`` of ``client.rpc_probe`` rather than a ``tcpSocket``: a tcpSocket
    probe can only dial the pod IP — the node's LAN address — while a worker
    whose group has ``wireguard.enabled`` binds its tunnel address, so the
    kubelet's dial would be refused until the budget expired. The module dials
    what ``resolve_rpc_host`` says rpc-server binds, which on a LAN-bound
    worker is the very address the tcpSocket probe used to dial
    (tests/test_rpc_probe.py).
    """
    manifest = YAML().load((K8S_DIR / "daemonset-shard.yaml").read_text())
    containers = _containers(manifest)
    assert containers, "daemonset-shard.yaml has no containers"
    for container in containers:
        if container.get("name") == "rpc-firewall":
            continue  # Init containers complete before probes are meaningful.
        probe = container.get("startupProbe")
        assert probe is not None, f"{container.get('name')!r} has no startupProbe"
        assert "httpGet" not in probe, "rpc-server speaks no HTTP; an httpGet probe never succeeds"
        assert list(probe["exec"]["command"]) == ["/usr/local/bin/python", "-m", "client.rpc_probe"], (
            "the shard worker's startupProbe must dial the address rpc-server binds "
            "(client.rpc_probe), not the pod IP"
        )


# --- Dockerfiles: the OpenSSL build, asserted on the artefact -----------------


def _instructions(text: str) -> list[str]:
    """Logical Dockerfile instructions, line continuations joined.

    An `ldd` assertion split over four physical lines is one instruction, and
    "is the check in the same RUN as its failure branch" is only answerable
    once the continuations are folded together.
    """
    joined = re.sub(r"\\\n", " ", text)
    return [line.strip() for line in joined.splitlines() if line.strip()]


def _builder_stage(text: str) -> str:
    """Everything up to the runtime stage's FROM.

    The assertion has to run in the BUILDER, where build/bin/llama-server
    exists. The same command in the runtime stage would test a path that is not
    there and fail the build for the wrong reason.
    """
    stages = re.split(r"(?mi)^FROM ", text)
    assert len(stages) >= 3, "expected a multi-stage Dockerfile (builder + runtime)"
    return stages[1]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_the_builder_installs_the_openssl_headers(dockerfile) -> None:
    """Without libssl-dev the build still succeeds and HTTPS silently vanishes."""
    builder = _builder_stage(dockerfile.read_text())
    apt = [i for i in _instructions(builder) if i.startswith("RUN apt-get")]
    assert apt, f"{dockerfile.name}: builder stage installs no apt packages"
    assert any("libssl-dev" in i for i in apt), (
        f"{dockerfile.name}: the builder stage does not install libssl-dev, so "
        "find_package(OpenSSL) fails, cpp-httplib emits a WARNING, the build passes, "
        "and every head crash-loops with 'the server is built without SSL support'"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_the_cmake_invocation_asks_for_openssl_by_its_real_name(dockerfile) -> None:
    text = dockerfile.read_text()
    cmake = [i for i in _instructions(text) if i.startswith("RUN cmake")]
    assert cmake, f"{dockerfile.name}: no cmake instruction"
    assert any("-DLLAMA_OPENSSL=ON" in i for i in cmake), (
        f"{dockerfile.name}: cmake is not passed -DLLAMA_OPENSSL=ON. The option "
        "defaults ON today, but a future flip upstream would reach the fleet as a "
        "crash loop rather than as a line in this diff."
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_no_dockerfile_uses_the_option_name_that_does_not_exist(dockerfile) -> None:
    """``LLAMA_SERVER_SSL`` is not a thing at b11429.

    CMake treats an unknown ``-D`` as an unused cache variable and succeeds, so
    this typo produces a green build and a plaintext image — the exact failure
    the ldd assertion exists to catch, arriving one step earlier.
    """
    text = dockerfile.read_text()
    message = (
        f"{dockerfile.name} passes LLAMA_SERVER_SSL; the option is LLAMA_OPENSSL "
        "(llama.cpp top-level CMakeLists.txt at b11429) and a misspelt -D is silently "
        "ignored"
    )
    # The flag form, not the bare name: the comment above the cmake invocation
    # says the name does not exist, and that sentence is the reason a reader
    # does not reintroduce it.
    assert "-DLLAMA_SERVER_SSL" not in text, message
    assert not [i for i in _instructions(text) if i.startswith("RUN cmake") and "SERVER_SSL" in i], (
        message
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_the_build_is_asserted_against_the_linked_binary(dockerfile) -> None:
    """The ldd assertion, its two libraries, and a non-zero exit.

    Checking build log text would not do: `cmake --build` exits 0 whether or
    not the WARNING was printed, so the only reliable evidence is what the
    produced binary links against.
    """
    text = dockerfile.read_text()
    builder = _builder_stage(text)
    instructions = _instructions(builder)
    ldd = [i for i in instructions if "ldd build/bin/llama-server" in i]
    assert ldd, (
        f"{dockerfile.name}: no `ldd build/bin/llama-server` assertion in the builder "
        "stage. Without it a build that lost OpenSSL ships, and the failure surfaces "
        "minutes later on another machine as a crash-looping head."
    )
    assertion = ldd[0]
    for library in ("libssl", "libcrypto"):
        assert library in assertion, (
            f"{dockerfile.name}: the ldd assertion does not check for {library} — "
            "llama-server needs both to serve HTTPS"
        )
    assert "exit 1" in assertion, (
        f"{dockerfile.name}: the ldd assertion does not exit non-zero, so a build "
        "without OpenSSL still produces an image"
    )

    build_index = max(i for i, line in enumerate(instructions) if line.startswith("RUN cmake"))
    assert instructions.index(assertion) > build_index, (
        f"{dockerfile.name}: the ldd assertion runs before the build that produces "
        "build/bin/llama-server"
    )


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_the_runtime_stage_declares_the_openssl_runtime(dockerfile) -> None:
    """libssl3 is declared, not inherited.

    Both bases happen to carry OpenSSL transitively today (CPython's ``_ssl``).
    llama-server links it directly now, so an image that only gets it as
    somebody else's dependency loses it the day that dependency changes — and
    the binary then fails to exec, on a node.
    """
    text = dockerfile.read_text()
    runtime = re.split(r"(?mi)^FROM ", text)[-1]
    apt = [i for i in _instructions(runtime) if i.startswith("RUN apt-get")]
    assert apt, f"{dockerfile.name}: runtime stage installs no apt packages"
    assert any("libssl3" in i for i in apt), (
        f"{dockerfile.name}: the runtime stage does not install libssl3"
    )


def _llama_pins(dockerfile) -> set[str]:
    """Every ``ARG LLAMA_CPP_VERSION=`` value in one file, deduplicated."""
    return set(re.findall(r"(?m)^ARG LLAMA_CPP_VERSION=(\S+)\s*$", dockerfile.read_text()))


def _llama_commit_pins(dockerfile) -> set[str]:
    """Every ``ARG LLAMA_CPP_COMMIT=`` value in one file, deduplicated."""
    return set(re.findall(r"(?m)^ARG LLAMA_CPP_COMMIT=(\S+)\s*$", dockerfile.read_text()))


def test_both_dockerfiles_pin_the_same_llama_cpp_build() -> None:
    """A pytest twin of scripts/test-all.sh's shell check (:400-415).

    The RPC wire format is version-locked and a mismatch HANGS at the handshake
    rather than reporting an error, so a group whose members were built from
    different llama.cpp versions simply never answers. The shell check only runs
    inside ./scripts/test-all.sh; this one runs in every pytest invocation, and
    it matters more now that both files were edited for OpenSSL — the pins sit
    a few lines from the changes.
    """
    pins = {path.name: _llama_pins(path) for path in DOCKERFILES}
    for name, values in pins.items():
        assert len(values) == 1, (
            f"{name} declares {sorted(values) or 'no'} LLAMA_CPP_VERSION pin(s); every "
            "stage in one file has to agree on exactly one"
        )
    (client,), (cuda,) = sorted(pins["Dockerfile.client"]), sorted(pins["Dockerfile.client.cuda"])
    assert client == cuda, (
        f"docker/Dockerfile.client pins {client} but docker/Dockerfile.client.cuda pins "
        f"{cuda} — mixed llama.cpp versions in one sharded group hang at the RPC "
        "handshake with no error anywhere"
    )


def test_both_dockerfiles_pin_the_same_llama_cpp_commit() -> None:
    """The tag is mutable; the commit it must resolve to is the actual pin.

    Each Dockerfile compares ``git rev-parse HEAD`` against its own
    ``LLAMA_CPP_COMMIT`` and fails the build when the two disagree, so a moved
    upstream tag is caught on the node that builds next. That check is only
    worth its lines while both files expect the same commit — otherwise the two
    images pass their own gate and still hang at the RPC handshake.
    """
    pins = {path.name: _llama_commit_pins(path) for path in DOCKERFILES}
    for name, values in pins.items():
        assert len(values) == 1, (
            f"{name} declares {sorted(values) or 'no'} LLAMA_CPP_COMMIT pin(s); without "
            "exactly one, the tag is the only pin and a re-pushed tag reaches the fleet"
        )
    (client,), (cuda,) = sorted(pins["Dockerfile.client"]), sorted(pins["Dockerfile.client.cuda"])
    assert client == cuda, (
        f"docker/Dockerfile.client pins commit {client} but docker/Dockerfile.client.cuda "
        f"pins {cuda} — the two images would build different llama.cpp trees while each "
        "passed its own tag check"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", client), (
        f"{client!r} is not a full 40-character commit sha; an abbreviated or empty pin "
        "never equals `git rev-parse HEAD` and fails every build"
    )


def test_the_pinned_build_clears_the_security_floor() -> None:
    """The pin must be a ``bNNNN`` tag at or above MIN_LLAMA_BUILD.

    ``group doctor`` reads each member's build number with parse_llama_build,
    which understands only the ``bNNNN`` form — a pin spelt any other way (a
    release name, a bare sha) reads as "unknown" on every node. And a floor
    raised without moving the pin would make every correctly built node warn.
    """
    from orchestrator.config.schema import MIN_LLAMA_BUILD, parse_llama_build

    for path in DOCKERFILES:
        for pin in _llama_pins(path):
            build = parse_llama_build(pin)
            assert build is not None, (
                f"{path.name}: LLAMA_CPP_VERSION={pin} is not a bNNNN tag, so group "
                "doctor cannot read the build number back off a node"
            )
            assert build >= MIN_LLAMA_BUILD, (
                f"{path.name} pins b{build}, below the security floor b{MIN_LLAMA_BUILD} "
                "(orchestrator/config/schema.py)"
            )
