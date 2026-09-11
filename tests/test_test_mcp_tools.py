"""scripts/test-mcp-tools.py reaches a node the way the fleet requires.

The helper runs against a live fleet from scripts/test-all.sh section 4, which
parses its ``PASS|``/``FAIL|`` lines and greps its captured output for
``aureax_sk_``. Three properties follow from that and are pinned here:

* it dials **https** and verifies against the fleet CA — a helper that fell
  back to plaintext, or that turned verification off, would report a fleet
  whose TLS is broken as passing;
* the bearer key comes from the environment and **never** from argv, because
  argv is readable by any local process for the life of the run (the same
  reason test-all.sh hands llama's key to curl through ``-K <(printf ...)``);
* every failure line it can emit names the URL or the path, never the key.

The script is loaded by path: it is not importable as a module, and that is
deliberate — it imports nothing from ``orchestrator`` or ``client`` so the only
thing it can fail on is the node.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import socket
import ssl
import sys
from pathlib import Path

import httpx
import pytest

from orchestrator.config import tlsstore
from tests.conftest import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "test-mcp-tools.py"

FAKE_KEY = "aureax_sk_" + "q" * 43


@pytest.fixture(scope="module")
def tool_script():
    """The helper, loaded from its hyphenated path."""
    spec = importlib.util.spec_from_file_location("aureax_test_mcp_tools", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _isolated_env(tool_script, tmp_aureax_home, monkeypatch) -> Path:
    """A redirected $HOME with a real fleet CA, and no ambient key.

    All three variables the helper reads exist on the machine this suite is
    developed on, so without clearing them a precedence test would pass because
    of the developer's environment.
    """
    for name in (tool_script.ENV_API_KEY, tool_script.ENV_KEY_FILE, tool_script.ENV_CA):
        monkeypatch.delenv(name, raising=False)
    return tlsstore.ensure_ca()


def _closed_port() -> int:
    """A port nothing is listening on, so connect() is refused immediately."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(tool_script, port: int, api_key: str | None = None) -> str:
    """Run the helper against a closed port and return everything it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        failures = asyncio.run(tool_script.run_tests("127.0.0.1", port, api_key))
    assert failures == 1, out.getvalue()
    return out.getvalue()


# --- TLS ---------------------------------------------------------------------


def test_it_dials_https(tool_script) -> None:
    """The connection-level FAIL line carries the URL it tried."""
    output = _run(tool_script, _closed_port())
    assert "https://127.0.0.1:" in output
    assert "http://" not in output


def test_the_ca_comes_from_the_environment_then_the_store(tool_script, monkeypatch, tmp_path) -> None:
    """$AUREAX_CA overrides the default under ~/.aureax/tls."""
    assert tool_script.ca_path() == Path.home() / ".aureax" / "tls" / "ca.pem"
    monkeypatch.setenv(tool_script.ENV_CA, str(tmp_path / "other-ca.pem"))
    assert tool_script.ca_path() == tmp_path / "other-ca.pem"


def test_the_context_verifies_against_that_bundle(tool_script, _isolated_env) -> None:
    """A real context built from a file — never a disabled check."""
    ctx = tool_script.ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    assert ctx.get_ca_certs(), "the bundle loaded but holds no certificates"


def test_a_missing_ca_is_one_parseable_fail_line(tool_script, monkeypatch, tmp_path) -> None:
    """count_results counts lines it can parse; a traceback is none of them."""
    monkeypatch.setenv(tool_script.ENV_CA, str(tmp_path / "absent.pem"))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        failures = asyncio.run(tool_script.run_tests("127.0.0.1", _closed_port(), None))
    lines = [line for line in out.getvalue().splitlines() if line.startswith(("PASS|", "FAIL|"))]
    assert failures == 1
    assert len(lines) == 1
    assert lines[0].startswith("FAIL|mcp_tls_ca|")
    assert "absent.pem" in lines[0]


# --- the key ------------------------------------------------------------------


def test_the_key_comes_from_the_environment(tool_script, monkeypatch, tmp_path) -> None:
    """AUREAX_MCP_API_KEY wins; AUREAX_MCP_KEY_FILE is the file spelling."""
    key_file = tmp_path / "node.mcp.key"
    key_file.write_text(f"{FAKE_KEY}\n")

    assert tool_script.api_key_from_env() is None

    monkeypatch.setenv(tool_script.ENV_KEY_FILE, str(key_file))
    assert tool_script.api_key_from_env() == FAKE_KEY

    monkeypatch.setenv(tool_script.ENV_API_KEY, "aureax_sk_direct")
    assert tool_script.api_key_from_env() == "aureax_sk_direct"


def test_the_key_is_never_taken_from_argv(tool_script, monkeypatch) -> None:
    """A third argv word is ignored, not read as a key.

    argv is world-readable for the life of the run. A caller that tries the
    obvious thing must get a 401 it can see rather than a key in `ps`.
    """
    seen: dict[str, object] = {}

    async def fake_run_tests(host, port, api_key=None):
        seen.update(host=host, port=port, api_key=api_key)
        return 0

    monkeypatch.setattr(tool_script, "run_tests", fake_run_tests)
    monkeypatch.setattr(sys, "argv", ["test-mcp-tools.py", "10.77.0.1", "30900", FAKE_KEY])
    with pytest.raises(SystemExit) as exit_info:
        tool_script.main()
    assert exit_info.value.code == 0
    assert seen == {"host": "10.77.0.1", "port": 30900, "api_key": None}


def test_the_bearer_header_is_set_on_the_client(tool_script, monkeypatch) -> None:
    """streamable_http_client takes no headers, so the client carries them.

    Also pins that ``verify`` is the CA context rather than a bool: passing
    ``True`` here would verify a private CA against the system trust store and
    fail every node.
    """
    captured: list[dict] = []
    real_client = httpx.AsyncClient

    class RecordingClient(real_client):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", RecordingClient)
    _run(tool_script, _closed_port(), FAKE_KEY)

    assert captured, "the helper built no http client"
    kwargs = captured[0]
    assert kwargs["headers"] == {"Authorization": f"Bearer {FAKE_KEY}"}
    assert isinstance(kwargs["verify"], ssl.SSLContext)

    captured.clear()
    _run(tool_script, _closed_port(), None)
    assert captured[0]["headers"] is None


def test_no_failure_line_carries_the_key(tool_script) -> None:
    """A 401 already surfaces as an HTTP status; the key adds nothing but risk."""
    output = _run(tool_script, _closed_port(), FAKE_KEY)
    assert FAKE_KEY not in output
    assert "aureax_sk_" not in output


def test_an_unreadable_key_file_is_one_parseable_fail_line(tool_script, monkeypatch, tmp_path) -> None:
    """Reported, not a traceback — and the path, never the contents."""
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    monkeypatch.setenv(tool_script.ENV_KEY_FILE, str(directory))
    monkeypatch.setattr(sys, "argv", ["test-mcp-tools.py", "10.77.0.1", "30900"])

    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exit_info:
        tool_script.main()
    assert exit_info.value.code == 1
    assert out.getvalue().startswith("FAIL|mcp_key_file|")
    assert str(directory) in out.getvalue()
