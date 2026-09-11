"""Shared pytest fixtures for orchestrator smoke tests."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The tracked, scrubbed fleet definition. Same shape as a real one (nine
# targets, two groups, one planner, the same per-node `config_file` names) but
# with documentation-range addresses and no credentials, so it is present in a
# fresh clone and identical for everyone.
EXAMPLE_NETWORK_CONFIG = PROJECT_ROOT / "configs" / "aureax-network.example.yaml"

# The operator's own copy. Gitignored because it carries real addresses and SSH
# credentials, so it is ABSENT from a fresh clone and its contents vary per
# operator. Only opt-in tests may read it, and they skip when it is missing.
REAL_NETWORK_CONFIG = PROJECT_ROOT / "configs" / "aureax-network.yaml"


@pytest.fixture
def network_config_path() -> Path:
    """Path to the example network config — used for happy-path load tests.

    Deliberately the tracked ``configs/aureax-network.example.yaml`` rather
    than ``configs/aureax-network.yaml``: the example is checked in and the
    same for every checkout, so a test reading it is deterministic and still
    runs on a fresh clone. The real file is operator-owned, gitignored and may
    be absent entirely — a test that genuinely needs it takes
    :data:`REAL_NETWORK_CONFIG` directly and skips when it does not exist.
    """
    return EXAMPLE_NETWORK_CONFIG


@pytest.fixture
def tmp_aureax_home(tmp_path, monkeypatch) -> Path:
    """Redirect ``$HOME`` to a tmp dir so tests don't read/write the user's ~/.aureax."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


# --- Shared network-config scaffolding --------------------------------------
# Several test modules need a minimal-but-valid NetworkConfig. Building it here
# keeps them from drifting into slightly different versions of "a valid
# config" — which matters because the schema's validators are themselves under
# test, so a locally-invented fixture that happens to dodge one of them makes a
# test pass for the wrong reason.
#
# test_lifecycle_groups.py deliberately keeps its own `_cfg`: it pins specific
# node ids and hosts that its assertions read back, so it is a fixture rather
# than a generic builder.

NETWORK_BASE: dict = {
    "network": {"name": "t", "version": "0.1.0"},
    "model_registry": {"models": []},
}


def make_network_config(
    targets: list[dict],
    groups: list[dict] | None = None,
    models: list[dict] | None = None,
):
    """Validated NetworkConfig from bare target/group dicts."""
    from orchestrator.config.schema import NetworkConfig

    payload = {**NETWORK_BASE, "ssh": {"targets": targets}}
    if groups is not None:
        payload["node_groups"] = groups
    if models is not None:
        payload["model_registry"] = {"models": models}
    return NetworkConfig.model_validate(payload)


def make_targets(*node_ids: str, disabled: tuple[str, ...] = ()) -> list[dict]:
    """Targets at 10.77.0.1, .2, ... each with its own ``<node_id>.yaml``."""
    return [
        {
            "node_id": n,
            "host": f"10.77.0.{i + 1}",
            "config_file": f"{n}.yaml",
            "enabled": n not in disabled,
        }
        for i, n in enumerate(node_ids)
    ]


def write_network_yaml(
    configs_dir: Path,
    targets: list[dict],
    groups: list[dict] | None = None,
    ssh_extra: dict | None = None,
    filename: str = "aureax-network.yaml",
) -> Path:
    """Write a minimal-but-valid network YAML into ``configs_dir``; return its path.

    The counterpart to :func:`make_network_config` for the tests that need a
    FILE — anything driving the CLI, the generator or the dashboard, all of
    which take a path and re-load it. ``targets`` are raw dicts so a caller can
    carry any per-target key (``password``, ``tags``, ``user``, ``enabled``);
    ``ssh_extra`` carries top-level ``ssh`` keys such as
    ``bootstrap_public_key``.
    """
    from ruamel.yaml import YAML

    configs_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {**NETWORK_BASE, "ssh": {**(ssh_extra or {}), "targets": targets}}
    if groups is not None:
        payload["node_groups"] = groups
    path = configs_dir / filename
    with open(path, "w") as f:
        YAML().dump(payload, f)
    return path


# --- Control-plane collaborator stand-ins ------------------------------------
# Several test modules stand in for the same production surfaces, so these live
# here rather than being re-declared (and quietly diverging) per file: a
# signature change in SystemLogger or orchestrator.cluster.kubectl has exactly
# one place to be chased.


@pytest.fixture
def fake_clock(monkeypatch):
    """A clock that only advances when the code under test sleeps.

    Every readiness wait in the deploy and apply paths is a ``time.time()``
    deadline around a ``time.sleep()``, so faking both turns a multi-minute
    timeout into an assertion that runs instantly and deterministically.
    """
    now = {"t": 1_000.0}
    monkeypatch.setattr(time, "time", lambda: now["t"])
    monkeypatch.setattr(time, "sleep", lambda s: now.update(t=now["t"] + s))
    return now


class RecordingLog:
    """Stands in for ``SystemLogger``; records ``(level, text, source)`` triples.

    ``emit`` accepts ``source`` both positionally and by keyword because
    callers use it both ways, and swallows any other kwarg so a new one cannot
    turn into a TypeError that only production sees. That signature lives here
    once: a keyword added to a ``SystemLogger.emit`` call must reach every
    recorder at the same time, not raise in the modules whose own copy was
    written without ``**_kw``.

    ``sink`` appends into a caller-owned list instead, for tests that read the
    lines back after the recorder has gone out of scope. Subclasses choose what
    to keep by overriding ``_record``.
    """

    def __init__(self, sink: list | None = None) -> None:
        self.lines: list = [] if sink is None else sink

    def emit(self, level: str, text: str, source: str = "lifecycle", **_kw) -> None:
        self._record(level, text, source)

    def _record(self, level: str, text: str, source: str) -> None:
        self.lines.append((level, text, source))

    def texts(self) -> list[str]:
        return [entry[1] for entry in self.lines]


class FakeLog(RecordingLog):
    """``RecordingLog`` that keeps only the ``(level, text)`` pair.

    The control-plane tests assert on pairs; ``source`` is fixed for a whole
    phase there, so recording it would add a column every assertion ignores.
    """

    def _record(self, level: str, text: str, _source: str) -> None:
        self.lines.append((level, text))


class FakeKubectl:
    """Stands in for ``orchestrator.cluster.kubectl``.

    Serves fixed node/pod listings and records every label write as
    ``(name, key, value)`` so a test can assert both which nodes were labelled
    and with what.
    """

    def __init__(self, nodes: list | None = None, pods: list | None = None) -> None:
        self._nodes = nodes or []
        self._pods = pods or []
        self.labelled: list[tuple[str, str, str]] = []

    def get_nodes(self) -> list:
        return self._nodes

    def get_pods(self, _namespace: str) -> list:
        return self._pods

    def label_node(self, name: str, key: str, value: str) -> None:
        self.labelled.append((name, key, value))

    def patch(self, *_a, **_k) -> str:
        return ""

    @property
    def labelled_names(self) -> list[str]:
        return [name for name, _key, _value in self.labelled]


# --- Edge-supervisor scaffolding ---------------------------------------------


@pytest.fixture
def bare_client(tmp_path, monkeypatch):
    """Factory for an ``AureaxClient`` built without reading a config file.

    ``__init__`` reads /etc/aureax/aureax-edge.yaml and opens a log store, so
    every supervisor test constructs the object through ``__new__`` instead.
    Doing that inline meant each site encoded its own guess at which of
    ``__init__``'s attributes the code under test touches — and a guess that is
    one attribute short fails as an AttributeError in a test rather than as the
    behaviour under test. This sets the FULL attribute set ``__init__`` creates
    (client/entrypoint.py) and applies ``**overrides`` on top.

    It also redirects ``process_ownership._record_path`` into ``tmp_path``, so
    a test that spawns or reaps a child writes its ownership record there
    rather than into the node's real ``/opt/aureax/run``.
    """
    from client import process_ownership
    from client.entrypoint import AureaxClient
    monkeypatch.setattr(process_ownership, "_record_path", lambda node, binary: tmp_path / f"{node}-{Path(binary).name}.json")


    def make(**overrides):
        client = AureaxClient.__new__(AureaxClient)
        client.config = {}
        client.node_id = "test-node"
        client._child_proc = None
        client._child_name = "llama-server"
        client._child_log_file = None
        client._child_log_thread = None
        client._shutdown = False
        client._telemetry_thread = None
        client.telemetry_storage = None
        client._log_store = None
        client._log_reporter = None
        client._child_capture = None
        client._log_thread = None
        for name, value in overrides.items():
            setattr(client, name, value)
        return client

    return make


@pytest.fixture(autouse=True)
def _reset_lifecycle_caches():
    """Clear the process-global caches around every test.

    They persist between tests, so a test that populates one changes the
    behaviour of whichever test happens to run next — an ordering dependency
    that only shows up when the suite is reordered. The dashboard's kubectl
    TTL cache is here for the same reason: it holds a monkeypatched answer for
    three seconds after the patch is gone.

    The config caches are cleared too. They key on (path, mtime), which makes
    them safe in production but not under pytest: tmp_path directories are
    reused across tests and a rewritten file can land on the same timestamp, so
    a later test reads an earlier one's parse.
    """
    from orchestrator import lifecycle
    from orchestrator.config import cache as config_cache
    from orchestrator.dashboard import api as dashboard_api

    def clear() -> None:
        lifecycle._node_status_cache.update({"ts": 0.0, "path": None, "data": []})
        lifecycle._hostname_cache.clear()
        lifecycle._role_cache.clear()
        lifecycle._NODE_TO_TARGET_CACHE.clear()
        # "nodes", "pods", "readyz" and the SQLite-backed "nodes-metrics:<db>"
        # all live here — the watchers share this cache rather than keeping one
        # of their own, so this clear covers them too.
        dashboard_api._kubectl_cache.clear()
        config_cache._RAW_CONFIG_CACHE.clear()
        config_cache._TEMPLATE_CACHE.clear()
        config_cache._NETWORK_CONFIG_CACHE.clear()

    clear()
    yield
    clear()
