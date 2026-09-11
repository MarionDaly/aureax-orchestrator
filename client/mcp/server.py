"""MCP server for the Aureax edge client.

Exposes tools for health, inference, and simulated IoT sensors.

SECURITY: this port is bearer-authenticated whenever the node's config carries
``mcp.server.api_key``, which every currently deployed edge and head template
sets — the frozen rollback pair (``configs/aureax-shard-head.yaml``,
``configs/aureax-shard-worker.yaml``) deliberately does not, so a fleet rolled
back onto it serves this port ungated behind the LAN firewall. See
:func:`build_app` and ``client/mcp/auth.py``. It is reachable twice, as
``hostNetwork`` :9000 and as NodePort :30900, and both doors lead into the one
uvicorn process, so the single ASGI wrapper covers both. That gate is what
closes ``run_inference``: the tool forwards to llama-server with the node's
llama key attached, so an open MCP port was a way to reach key-protected
inference through the proxy.

``/health`` is deliberately exempt. The deploy and verify scripts poll it with
an unauthenticated curl before any key is in hand on a freshly imaged node
(``scripts/lib/common.sh::mcp_health_wait``), so requiring a key there would
turn every provisioning run into a manual step. It answers a status dict and
nothing else.

Transport is HTTPS with the node's own leaf certificate, verified against the
fleet CA — the supervisor passes ``ssl_certfile``/``ssl_keyfile`` to uvicorn
(``client/entrypoint.py::_start_mcp_server``) and every in-pod client dials
with ``client/tls.py::ssl_context``. There is no plaintext mode.

``rpc-server`` :50052 is now the one remaining unauthenticated door on a node;
it deserializes arbitrary tensor graphs and stays firewall-only on a trusted
LAN. The clamps below (``MAX_INFERENCE_TOKENS``, ``_MAX_METRICS_WINDOW_MINUTES``,
the anomaly truncation) still bound what a caller can cost this pod — an
*authenticated* one now, which is a smaller set of people but not a trusted
one: one key is shared by every client of the node, so a clamp is still the
only thing between a fat-fingered agent loop and the pod's memory.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import platform
import ssl
import time
from collections import deque
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse

from client.inference.llama_client import LlamaClient
from client.mcp.auth import BearerAuthMiddleware
from client.sensors.simulator import SensorSimulator
from client.telemetry.storage import TelemetryStorage

logger = logging.getLogger("aureax.mcp.server")


# MCP server instance — created at module level with defaults.
# Host/port are set via init() before run() is called, because FastMCP
# reads host/port from its constructor (self.settings), not from run().
mcp = FastMCP(
    "aureax-edge",
    instructions="Aureax edge agent — IoT sensor monitoring and local LLM inference",
    host="0.0.0.0",
    port=9000,
)


@mcp.custom_route("/health", methods=["GET"])
async def _health_endpoint(_request: StarletteRequest) -> JSONResponse:
    """Plain-HTTP health check at /health, alongside /mcp on the MCP port.

    Consumed by the deploy/verify scripts (setup-all.sh, test-all.sh,
    update-all.sh curl https://<node>:30900/health) because the /mcp path only
    speaks the MCP protocol and does not answer a plain GET. No K8s probe
    reaches this port: all three edge probes target llama-server on :8080.

    Exempt from the bearer gate (client/mcp/auth.py), because those scripts run
    on a freshly imaged node before any key has been pushed to it.

    Registered with FastMCP's public ``custom_route`` decorator, which adds it
    to the Starlette route table ``streamable_http_app()`` builds.
    """
    return JSONResponse({"status": "ok", "service": "aureax-mcp"})


# These get initialized by the entrypoint with actual config values
_llama_client: LlamaClient | None = None
_sensor_sim: SensorSimulator | None = None
_telemetry_storage: TelemetryStorage | None = None
_node_id: str = "unknown"
# The node's MCP bearer key, from mcp.server.api_key in its resolved config.
# Read only by build_app(): the tools below never see it, so no tool response
# and no log line can echo it.
_mcp_api_key: str | None = None
# Monotonic, so the reported uptime cannot jump or go negative: edge nodes
# commonly have no RTC and step the clock once time sync converges after boot,
# which happens after this pod starts, and a wall-clock origin would move with
# the correction.
#
# This is time since MCP-MODULE IMPORT, not pod or node uptime: entrypoint
# imports this module only after the RPC-worker wait and the model load, which
# is up to 2700s on a sharded head. The key stays "uptime_seconds" because
# scripts/test-mcp-tools.py's get_health check asserts its type and prints it.
_start_time = time.monotonic()
# Bounded on purpose. The stored dicts are never read back — report_anomaly
# echoes the record it just built, not an entry from here — and the only
# consumers are the len() counts in get_health/get_metrics. The deque exists
# so serving that counter cannot grow memory for the life of the pod. maxlen
# bounds the ENTRY COUNT only; report_anomaly truncates each entry's caller
# strings, because 200 unbounded descriptions is not a memory bound.
_anomaly_log: deque[dict] = deque(maxlen=200)
# Anomaly IDs come from this counter so they stay unique after the deque
# saturates; the deque itself only serves the recent-window echo and counts.
_anomaly_seq = itertools.count(1)

# Caller-supplied bounds. Every tool below is reachable by any holder of the
# node's one MCP key, which every client of the node shares (see the module
# docstring), so each value that sizes work or memory is clamped at the tool
# boundary rather than trusted.
MAX_INFERENCE_TOKENS = 4096
# A fixed ceiling on the work one caller can ask a node for, not a claim about
# what is stored: rows CAN outlive it (telemetry.local.retention_days is
# configurable, and TelemetryStorage.rotate keeps unpushed rows for three times
# that), so a wider window is silently shortened to this. Both ends are clamped
# because the value is unvalidated caller input: 10**15 overflows timedelta,
# and a negative one puts the cutoff in the future, reporting "no data" instead
# of a bad request.
_MAX_METRICS_WINDOW_MINUTES = 7 * 24 * 60
# Per-entry caps for report_anomaly's free-text arguments. All three are
# caller-controlled strings, so all three are capped: severity is documented as
# one of low/medium/high/critical, but nothing on the wire enforces that, and it
# is truncated rather than rejected so an unexpected value still logs.
_MAX_ANOMALY_DESCRIPTION = 512
_MAX_ANOMALY_SENSOR_TYPE = 64
_MAX_ANOMALY_SEVERITY = 32
# run_inference holds a default-executor thread (min(32, cpu+4)) for up to
# LLAMA_HTTP_TIMEOUT_SECONDS. get_health and get_metrics dispatch into that
# same shared executor, so unbounded inference concurrency stops them
# answering. This bounds the executor footprint; llama-server's own slot queue
# still decides ordering beyond it.
_INFERENCE_SLOTS = asyncio.Semaphore(4)


def init(
    node_id: str,
    llama_base_url: str = "https://127.0.0.1:8080",
    *,
    mcp_host: str = "0.0.0.0",
    mcp_port: int = 9000,
    telemetry_storage: TelemetryStorage | None = None,
    llama_api_key: str | None = None,
    mcp_api_key: str | None = None,
    verify: ssl.SSLContext | None = None,
) -> None:
    """Initialize the MCP server with runtime dependencies.

    Must be called before mcp.run(). Sets the llama.cpp client, sensor
    simulator, and overrides FastMCP's bind address.

    Only the PORT comes from config. ``mcp_host`` is passed the literal
    0.0.0.0 by the entrypoint: the pod runs hostNetwork, so this binds the
    node's own addresses and is a second door beside the :30900 NodePort —
    the SECURITY block in k8s/daemonset-edge.yaml carries the rationale and
    the trust model it assumes.

    ``llama_api_key``, ``mcp_api_key`` and ``verify`` are appended at the END
    of the signature on purpose, and everything after ``llama_base_url`` is
    keyword-only: the entrypoint passes the first two arguments positionally,
    so inserting a parameter mid-signature would otherwise silently reassign
    meaning under any third positional.

    ``mcp_api_key`` is stored, not used here — :func:`build_app` is what turns
    it into a gate. ``verify`` is the fleet-CA SSLContext the LlamaClient dials
    this node's own llama-server with; there is no plaintext fallback.
    """
    global _llama_client, _sensor_sim, _telemetry_storage, _node_id, _mcp_api_key
    _llama_client = LlamaClient(llama_base_url, api_key=llama_api_key, verify=verify)
    _mcp_api_key = mcp_api_key or None
    _sensor_sim = SensorSimulator(node_id)
    _telemetry_storage = telemetry_storage
    _node_id = node_id
    # Override host/port from config (FastMCP reads these from self.settings)
    mcp.settings.host = mcp_host
    mcp.settings.port = mcp_port


def build_app():
    """The ASGI app uvicorn serves: the MCP app, gated when a key is configured.

    Called once per process by ``client/entrypoint.py::_start_mcp_server``,
    after :func:`init`. Serving ``mcp.streamable_http_app()`` directly is what
    the supervisor must NOT do — the wrapper is the only enforcement point,
    because FastMCP has no auth of its own and both :9000 and :30900 reach this
    same app.

    The log line is the deployable evidence: ``kubectl logs <pod> | grep
    'bearer auth'`` is how an operator confirms which state a node came up in,
    and a node whose template lost its ``api_key:`` line otherwise looks
    identical to one that has it. Neither line ever carries the key.
    """
    app = mcp.streamable_http_app()
    if not _mcp_api_key:
        logger.warning(
            "MCP bearer auth DISABLED — this port is open to anyone who can reach it"
        )
        return app
    logger.info(
        "MCP bearer auth ENABLED — /mcp requires Authorization: Bearer; /health stays open"
    )
    return BearerAuthMiddleware(app, _mcp_api_key)


def _assert_initialized() -> None:
    """Raise RuntimeError if init() hasn't wired the runtime dependencies yet.

    Asserts BOTH runtime deps together, because init() wires them in one call:
    a tool reaching this guard at all means startup was misordered, so failing
    loudly beats returning a JSON error string.

    get_health and get_metrics deliberately skip it and report degraded state
    instead (``if _llama_client:`` / ``if _telemetry_storage:``) — they are the
    tools an operator hits while startup is still in progress, which is exactly
    when they must still answer.
    """
    if _llama_client is None or _sensor_sim is None:
        raise RuntimeError(
            "MCP server tools invoked before init() — call client.mcp.server.init(...) first."
        )


@mcp.tool()
async def get_health() -> str:
    """Get node health status including llama-server and system metrics.

    Async with the probe in a thread for the same reason as run_inference: a
    sync tool runs inline on the event loop, and a llama-server that accepts
    the socket but never answers (mid-load) would freeze the whole MCP server
    for the probe's timeout.
    """
    llama_healthy = False
    llama_status = "unknown"
    if _llama_client:
        try:
            health = await asyncio.to_thread(_llama_client.health)
            llama_healthy = health.get("status") == "ok"
            llama_status = health.get("status", "unknown")
        except Exception as e:
            logger.exception("llama-server health probe failed")
            llama_status = f"error: {e}"

    uptime_seconds = int(time.monotonic() - _start_time)

    result = {
        "node_id": _node_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime_seconds": uptime_seconds,
        "llama_server": {
            "healthy": llama_healthy,
            "status": llama_status,
        },
        "system": {
            "platform": platform.machine(),
            "python": platform.python_version(),
        },
        # Size of the capped recent window, NOT a running total — _anomaly_log
        # saturates at 200. The monotonic total is _anomaly_seq.
        "anomalies_detected": len(_anomaly_log),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
async def get_metrics(since_minutes: int = 60) -> str:
    """Get collected telemetry metrics from this node."""
    since_minutes = max(1, min(int(since_minutes), _MAX_METRICS_WINDOW_MINUTES))
    if _telemetry_storage:
        # Aggregated in SQL (latest value per metric) rather than
        # materializing the whole window — since_minutes comes straight from
        # the MCP caller (clamped above), and the sync sqlite read runs off the
        # event loop.
        latest, sample_count = await asyncio.to_thread(
            _telemetry_storage.latest_per_metric, since_minutes
        )

        result = {
            "node_id": _node_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "period_minutes": since_minutes,
            "metrics": latest,
            "sample_count": sample_count,
        }
    else:
        # Fallback: live snapshot when telemetry is not initialized
        result = {
            "node_id": _node_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "period_minutes": since_minutes,
            "metrics": {
                "uptime_seconds": int(time.monotonic() - _start_time),
                # Same capped-window count as get_health, same key name.
                "anomalies_detected": len(_anomaly_log),
            },
            # Present in both branches on purpose: one tool, one key set.
            "sample_count": 0,
        }

    return json.dumps(result, indent=2)


@mcp.tool()
async def run_inference(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Send a prompt to the local LLM and get a response.

    The prompt goes out as a single user MESSAGE on /v1/chat/completions, not
    as a raw prompt: the raw path skips the model's chat template, which Gemma
    4 does not survive (it answered ", ok, ok, ok…" and leaked `<channel|>`
    tokens to a prompt the chat endpoint answered with "ok"). See
    client/inference/llama_client.py for the full case.

    Uses asyncio.to_thread() to run the synchronous llama.cpp HTTP call
    off the event loop. Without this, the blocking call would freeze the
    entire MCP server (including session heartbeats), causing the client's
    session to time out with "Session terminated".

    max_tokens is clamped and concurrency is capped: one MCP key is shared by
    every client of the node (see the module docstring) and each call pins a
    shared default-executor thread for up to the llama HTTP timeout.

    A reasoning model that spends the whole budget thinking comes back as an
    {"error": ...} result, not as empty content — the caller asked for an
    answer and needs to be told to raise max_tokens.
    """
    _assert_initialized()
    max_tokens = max(1, min(int(max_tokens), MAX_INFERENCE_TOKENS))
    try:
        # Run the sync HTTP call in a thread so the async event loop stays free
        # for MCP protocol messages (heartbeats, session management).
        async with _INFERENCE_SLOTS:
            result = await asyncio.to_thread(
                _llama_client.chat,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        content = result.get("content", "")
        budget_spent_reasoning = (
            not content
            and result.get("finish_reason") == "length"
            and bool(result.get("has_reasoning"))
        )
        if budget_spent_reasoning:
            # Same {"error": ...} shape as the except branch below, because the
            # caller needs a reason it can act on: an empty string reads as
            # "the model had nothing to say", and the live failure was the
            # opposite — the model said plenty, all of it into
            # `reasoning_content` until max_tokens ran out. LlamaClient already
            # asks the chat template to disable thinking; this catches a model
            # or template that ignores the request.
            return json.dumps({
                "error": (
                    f"model spent the whole max_tokens budget ({max_tokens}) "
                    "reasoning; raise max_tokens"
                )
            })
        return json.dumps({
            "content": content,
            "tokens_predicted": result.get("tokens_predicted", 0),
            "tokens_evaluated": result.get("tokens_evaluated", 0),
            "generation_time_ms": result.get("timings", {}).get("predicted_ms", 0),
        }, indent=2)
    except Exception as e:
        logger.exception("run_inference failed")
        return json.dumps({"error": str(e)})


@mcp.tool()
def read_sensor(sensor_type: str) -> str:
    """Read current values from a simulated IoT sensor.

    sensor_type: one of temperature, humidity, pressure, light, battery, all
    """
    _assert_initialized()
    data = _sensor_sim.read(sensor_type)
    return json.dumps(data, indent=2)


@mcp.tool()
def get_sensor_status() -> str:
    """Get operational status of all simulated sensors."""
    _assert_initialized()
    status = _sensor_sim.get_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def report_anomaly(sensor_type: str, severity: str, description: str, value: float = 0.0) -> str:
    """Report a detected anomaly for orchestrator review.

    severity: one of low, medium, high, critical
    """
    # Truncate on ingest: _anomaly_log's maxlen caps how MANY entries stay
    # resident, not how big each one is, and this tool takes free text from any
    # holder of the node's shared MCP key and keeps it for the life of the pod.
    anomaly = {
        "node_id": _node_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "sensor_type": sensor_type[:_MAX_ANOMALY_SENSOR_TYPE],
        "severity": severity[:_MAX_ANOMALY_SEVERITY],
        "description": description[:_MAX_ANOMALY_DESCRIPTION],
        "value": value,
    }
    _anomaly_log.append(anomaly)

    return json.dumps({
        "status": "logged",
        # A monotonic counter, NOT len(_anomaly_log): the deque is capped at
        # 200, so its length stops being unique exactly when anomalies are
        # frequent enough to matter.
        "anomaly_id": next(_anomaly_seq),
        "anomaly": anomaly,
    }, indent=2)
