#!/usr/bin/env python3
"""MCP tool integration tests — called by test-all.sh.

Connects to an Aureax edge node's MCP server via streamable-http and
thoroughly tests every tool, validating response structure, data types,
and semantic correctness.

Usage:
    uv run python scripts/test-mcp-tools.py <host> <port>

Environment (never argv — see main):
    AUREAX_MCP_API_KEY   the node's MCP bearer key, or
    AUREAX_MCP_KEY_FILE  a file holding it (~/.aureax/keys/<node-id>.mcp.key)
    AUREAX_CA            the CA bundle that signed the node's certificate
                         (default ~/.aureax/tls/ca.pem)

Output format (one line per test):
    PASS|tool_name|details
    FAIL|tool_name|error message

Imports nothing from `orchestrator` or `client` on purpose: this runs as a
plain script against a live fleet, and the only thing it should be able to
fail on is the node.
"""

import asyncio
import json
import os
import ssl
import sys
from pathlib import Path
from typing import Any

# Both spellings the suite may export. A key NEVER arrives as an argv word:
# argv is readable by any local process for the life of the run, which is why
# test-all.sh passes llama's key to curl through `-K <(printf ...)` rather than
# `-H` (see that file's section 7 preamble) and why the supervisor hands
# llama-server its key in the environment.
ENV_API_KEY = "AUREAX_MCP_API_KEY"
ENV_KEY_FILE = "AUREAX_MCP_KEY_FILE"
ENV_CA = "AUREAX_CA"


def result_line(status: str, name: str, detail: str) -> None:
    """Print a pipe-delimited result line parseable by test-all.sh."""
    print(f"{status}|{name}|{detail}")


def describe_exception(exc: BaseException) -> str:
    """Render an exception — or an ExceptionGroup's leaves — as `Type: message`.

    The MCP SDK runs its transport inside a task group, so a refused
    connection, an unpublished NodePort and an HTTP 404 on /mcp all surface as
    ``ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)`` — a
    message that names no cause and makes the three indistinguishable. The
    causes live only in ``.exceptions``, and when the connection never opened
    the detail string built from this is the ONLY thing the operator gets, so
    it has to carry them.
    """
    if isinstance(exc, BaseExceptionGroup):
        return "; ".join(describe_exception(sub) for sub in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def tool_json(r: Any) -> Any:
    """Decode a tool result's single JSON item, or raise the server's own message.

    ``ClientSession.call_tool`` returns a CallToolResult and does NOT raise
    when the tool fails — the server puts the exception text in that one
    content item and sets ``isError``. That text is not JSON, so it must never
    reach ``json.loads``: the decoder would report every genuine server-side
    failure to test-all.sh as "Expecting value: line 1 column 1 (char 0)", and
    an empty content list as "list index out of range". Raising here puts the
    server's own message into the caller's ``except`` block, which is what
    reaches the suite.
    """
    if getattr(r, "isError", False) or not r.content:
        detail = getattr(r.content[0], "text", repr(r.content[0])) if r.content else "empty tool result"
        raise RuntimeError(str(detail)[:300])
    return json.loads(r.content[0].text)


async def sensor_checks(session) -> int:
    """Run get_sensor_status, then read_sensor against everything it declared.

    Returns the number of failures.

    The set of sensor types and the unit each must report are read from the
    server's own inventory rather than restated on this side. A literal set
    here would be a second copy of the node's sensor implementation: adding a
    sensor to the node would make this suite report a correct server as
    "validation failed" until the copy was updated to match.
    """
    failures = 0
    expected_units: dict[str, Any] = {}

    # --- get_sensor_status: the inventory the two read_sensor checks use ---
    try:
        r = await session.call_tool("get_sensor_status", {})
        data = tool_json(r)
        sensors = data.get("sensors") or {}
        offline = [k for k, v in sensors.items() if v.get("status") != "online"]
        unitless = [k for k, v in sensors.items() if not v.get("unit")]
        if not sensors:
            result_line("FAIL", "get_sensor_status", f"no sensors reported, keys={list(data.keys())}")
            failures += 1
        elif offline or unitless:
            result_line("FAIL", "get_sensor_status", f"not online: {offline}; no unit declared: {unitless}")
            failures += 1
        else:
            expected_units = {k: v["unit"] for k, v in sensors.items()}
            result_line("PASS", "get_sensor_status", f"{len(sensors)} sensors, all online")
    except Exception as e:
        result_line("FAIL", "get_sensor_status", str(e))
        failures += 1

    if not expected_units:
        # Both read_sensor checks below expect what the inventory declared, so
        # neither can run without it. One FAIL, named for what did not run.
        result_line("FAIL", "read_sensor",
                    "not run: get_sensor_status returned no usable sensor inventory")
        return failures + 1

    # --- read_sensor: each declared type, reporting its declared unit ---
    for sensor_type, unit in sorted(expected_units.items()):
        try:
            r = await session.call_tool("read_sensor", {"sensor_type": sensor_type})
            data = tool_json(r)
            checks = [
                "value" in data,
                "unit" in data,
                "sensor_type" in data,
                "timestamp" in data,
                isinstance(data.get("value"), (int, float)),
                data.get("unit") == unit,
                data.get("sensor_type") == sensor_type,
            ]
            if all(checks):
                result_line("PASS", f"read_sensor({sensor_type})", f"{data['value']} {data['unit']}")
            else:
                result_line("FAIL", f"read_sensor({sensor_type})", f"validation failed: {data}")
                failures += 1
        except Exception as e:
            result_line("FAIL", f"read_sensor({sensor_type})", str(e))
            failures += 1

    # --- read_sensor(all): exactly the declared types, no more, no fewer ---
    try:
        r = await session.call_tool("read_sensor", {"sensor_type": "all"})
        data = tool_json(r)
        if not isinstance(data, list):
            result_line("FAIL", "read_sensor(all)", f"expected a list, got {type(data).__name__}")
            failures += 1
        else:
            types_returned = sorted(str(d.get("sensor_type")) for d in data)
            types_expected = sorted(expected_units)
            if types_returned == types_expected:
                result_line("PASS", "read_sensor(all)", f"{len(data)} sensors, all types present")
            else:
                result_line("FAIL", "read_sensor(all)", f"expected {types_expected}, got {types_returned}")
                failures += 1
    except Exception as e:
        result_line("FAIL", "read_sensor(all)", str(e))
        failures += 1

    return failures


def ca_path() -> Path:
    """The CA bundle to verify the node against: $AUREAX_CA, else the store default."""
    return Path(os.environ.get(ENV_CA) or Path.home() / ".aureax" / "tls" / "ca.pem")


def ssl_context() -> ssl.SSLContext:
    """The context every request verifies against.

    Built from a file, never disabled: the whole point of the fleet CA is that
    a node presenting some other certificate is a failure, and a helper that
    turned verification off would report that fleet as passing.
    """
    return ssl.create_default_context(cafile=str(ca_path()))


def api_key_from_env() -> str | None:
    """The node's MCP key from AUREAX_MCP_API_KEY, else AUREAX_MCP_KEY_FILE.

    Returns None when neither is set, which is correct for a node whose
    template carries no ``api_key`` line — the request then goes out with no
    Authorization header and the node answers it.
    """
    direct = (os.environ.get(ENV_API_KEY) or "").strip()
    if direct:
        return direct
    key_file = os.environ.get(ENV_KEY_FILE)
    if key_file:
        return Path(key_file).read_text().strip() or None
    return None


async def run_tests(host: str, port: int, api_key: str | None = None) -> int:
    """Run all MCP tool tests against a single node.

    Returns the number of failures.

    `api_key` is a parameter rather than an env read in here so the caller
    owns the one place a key is sourced (main), and so nothing on this path
    can put it in a message.
    """
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    url = f"https://{host}:{port}/mcp"
    failures = 0

    try:
        verify = ssl_context()
    except OSError as e:
        # One FAIL line, same contract as the connection failure below: none of
        # the per-tool checks can run. Named for the CA rather than the node,
        # because the fix is on this Mac (`aureax generate`), not on the node.
        result_line("FAIL", "mcp_tls_ca",
                    f"cannot verify TLS against {ca_path()}: {e}")
        return 1

    try:
        # Every client, transport and session is entered as a context manager,
        # so a failure anywhere — the transport failing to open, `initialize()`
        # raising, anything in the body — still closes them on the way to the
        # handler below. An unclosed transport left to the garbage collector
        # writes a warning into the same stream test-all.sh parses for PASS|
        # and FAIL| lines.
        #
        # Entered in one parenthesized group: the managers open in order, so
        # `streamable_http_client` still receives the client bound above it.
        #
        # `connect` is 5s, matching client/mcp/pool.py and the harness: an
        # unready MCP pod behind the NodePort DROPs the SYN rather than
        # refusing it, so the connect leg always runs to its full budget, and
        # test-all.sh gates on only the FIRST MCP node before running this
        # helper against every one of them. `read` is a silence budget for one
        # read, not a total, and stays generous enough to cover prefill on a
        # cold sharded head.
        #
        # The bearer header goes on the client rather than per request:
        # `streamable_http_client` takes no headers of its own, so the
        # pre-built client is the SDK's supported path for both the header and
        # the CA context.
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0, read=600.0),
                follow_redirects=True,
                verify=verify,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            ) as http_client,
            streamable_http_client(
                url, http_client=http_client, terminate_on_close=False
            ) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()

            # --- get_health: verify structure and llama_server is healthy ---
            try:
                r = await session.call_tool("get_health", {})
                data = tool_json(r)
                checks = [
                    "node_id" in data,
                    "llama_server" in data,
                    "timestamp" in data,
                    "uptime_seconds" in data,
                    isinstance(data.get("uptime_seconds"), int),
                    data.get("llama_server", {}).get("healthy") is True,
                ]
                if all(checks):
                    result_line("PASS", "get_health", f"node={data['node_id']}, healthy=true, uptime={data['uptime_seconds']}s")
                else:
                    result_line("FAIL", "get_health", f"validation failed: healthy={data.get('llama_server',{}).get('healthy')}, keys={list(data.keys())}")
                    failures += 1
            except Exception as e:
                result_line("FAIL", "get_health", str(e))
                failures += 1

            # --- get_metrics: verify structure ---
            try:
                r = await session.call_tool("get_metrics", {"since_minutes": 5})
                data = tool_json(r)
                checks = [
                    "metrics" in data,
                    "node_id" in data,
                    "timestamp" in data,
                    "period_minutes" in data,
                ]
                if all(checks):
                    result_line("PASS", "get_metrics", f"period={data['period_minutes']}min")
                else:
                    result_line("FAIL", "get_metrics", f"missing keys, got: {list(data.keys())}")
                    failures += 1
            except Exception as e:
                result_line("FAIL", "get_metrics", str(e))
                failures += 1

            # --- the sensor tools, all derived from the server's inventory ---
            failures += await sensor_checks(session)

            # --- report_anomaly: verify it logs and returns an ID ---
            try:
                r = await session.call_tool("report_anomaly", {
                    "sensor_type": "temperature",
                    "severity": "low",
                    "description": "test anomaly from test-mcp-tools.py",
                    "value": 99.9,
                })
                data = tool_json(r)
                checks = [
                    data.get("status") == "logged",
                    isinstance(data.get("anomaly_id"), int),
                    data.get("anomaly", {}).get("severity") == "low",
                    data.get("anomaly", {}).get("sensor_type") == "temperature",
                    data.get("anomaly", {}).get("value") == 99.9,
                ]
                if all(checks):
                    result_line("PASS", "report_anomaly", f"id={data['anomaly_id']}, severity=low")
                else:
                    result_line("FAIL", "report_anomaly", f"validation failed: {data}")
                    failures += 1
            except Exception as e:
                result_line("FAIL", "report_anomaly", str(e))
                failures += 1

            # --- run_inference: the end-to-end LLM test ---
            # Validates that the full pipeline works: MCP → tool handler →
            # asyncio.to_thread → LlamaClient → llama-server → model → response.
            # Checks response structure, token counts, and timing.
            try:
                r = await session.call_tool("run_inference", {
                    "prompt": "What is 2+2? Answer with just the number.",
                    "max_tokens": 64,
                    "temperature": 0.1,
                })
                data = tool_json(r)

                if "error" in data:
                    result_line("FAIL", "run_inference", f"tool returned error: {data['error']}")
                    failures += 1
                else:
                    checks = [
                        "content" in data,
                        isinstance(data.get("content"), str),
                        len(data.get("content", "")) > 0,
                        isinstance(data.get("tokens_predicted"), int),
                        data.get("tokens_predicted", 0) > 0,
                        isinstance(data.get("tokens_evaluated"), int),
                        data.get("tokens_evaluated", 0) > 0,
                        isinstance(data.get("generation_time_ms"), (int, float)),
                        data.get("generation_time_ms", 0) > 0,
                    ]
                    if all(checks):
                        preview = data["content"][:80].replace("\n", " ").strip()
                        tok = data["tokens_predicted"]
                        ms = data["generation_time_ms"]
                        tok_s = (tok / ms * 1000) if ms > 0 else 0
                        result_line("PASS", "run_inference", f"{tok} tokens, {ms:.0f}ms ({tok_s:.1f} tok/s) — {preview}")
                    else:
                        result_line("FAIL", "run_inference", f"response validation failed: content_len={len(data.get('content',''))}, tokens={data.get('tokens_predicted')}, time={data.get('generation_time_ms')}")
                        failures += 1
            except Exception as e:
                result_line("FAIL", "run_inference", str(e))
                failures += 1

    except Exception as e:
        # Connection-level failure: none of the per-tool checks got to run, so
        # this is the only result line this process emits. ONE FAIL line is the
        # whole contract — test-all.sh's count_results counts the lines it
        # parses, not any total this file declares, and `failures` is only ever
        # compared against zero for the exit status below.
        #
        # Flattened through describe_exception, because the SDK's own failure
        # shape names no cause at all — see that function.
        #
        # The key is not in this line and must never be: the URL and the
        # exception text are all that is rendered, and a 401 already arrives as
        # an HTTP status through describe_exception. test-all.sh greps its own
        # captured output for `aureax_sk_`.
        result_line("FAIL", "mcp_connect",
                    f"Could not connect to {url}: {describe_exception(e)[:300]}")
        failures += 1

    return failures


def main() -> None:
    """Run MCP tool tests against a node specified by host and port args.

    The key comes from the environment, never from argv — see ENV_API_KEY. Any
    extra argv word is ignored rather than read as a key, so a caller that
    tries the obvious thing gets a 401 it can see instead of a key in `ps`.
    """
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <host> <port>", file=sys.stderr)
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    try:
        api_key = api_key_from_env()
    except OSError as e:
        # One parseable FAIL line rather than a traceback: an unreadable key
        # file would otherwise make every tool check vanish with no result for
        # count_results to count. The path, never the contents.
        result_line("FAIL", "mcp_key_file",
                    f"cannot read {ENV_KEY_FILE}={os.environ.get(ENV_KEY_FILE)}: {e}")
        sys.exit(1)
    failures = asyncio.run(run_tests(host, port, api_key))
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
