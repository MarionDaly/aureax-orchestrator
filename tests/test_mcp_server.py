"""Edge MCP tool behaviour, plus the inference/sensor helpers it sits on.

The MCP port is bearer-gated whenever the node's config carries an
``mcp.server.api_key`` (client/mcp/server.py::build_app), but the clamps still
bound what one caller can cost this pod: one key is shared by every client of
the node, so an authenticated caller is a smaller set of people, not a trusted
one. These tests pin those bounds, the gate's wiring, and the response shapes
the verify scripts and the agent loop read back.

Tools are plain functions: FastMCP's @mcp.tool() registers and returns ``fn``
unchanged, so each one is called directly here — no transport, no network.
"""

from __future__ import annotations

import json

import httpx
import pytest

from client.inference.chat_client import strip_thinking
from client.inference.llama_client import LlamaClient
from client.mcp import auth as mcp_auth
from client.mcp import server as srv
from client.sensors.simulator import SensorSimulator


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload
        # chat() parses resp.text (strict=False, for control characters in
        # thinking output), health() reads resp.json() — both are served here
        # from the one payload so a test never has to know which it hits.
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"raise_for_status reached for {self.status_code}")

    def json(self) -> dict:
        return self._payload


# ---- llama-server health ----------------------------------------------------

def test_health_reports_loading_instead_of_raising_on_503(monkeypatch) -> None:
    """503 is llama-server's mid-load state, not a failure.

    raise_for_status turned it into an HTTPStatusError that get_health reported
    as "error: Server error '503 ...'" — indistinguishable from an unreachable
    server, for the minutes a sharded head spends streaming tensors.
    """
    client = LlamaClient("https://127.0.0.1:8080")
    monkeypatch.setattr(
        client._client, "get", lambda *_a, **_kw: _FakeResponse(503, {"error": "loading model"})
    )
    assert client.health() == {"status": "loading"}

    monkeypatch.setattr(client._client, "get", lambda *_a, **_kw: _FakeResponse(200, {"status": "ok"}))
    assert client.health() == {"status": "ok"}


async def test_get_health_surfaces_the_loading_state(monkeypatch) -> None:
    class _LoadingLlama:
        def health(self) -> dict:
            return {"status": "loading"}

    monkeypatch.setattr(srv, "_llama_client", _LoadingLlama())
    data = json.loads(await srv.get_health())
    assert data["llama_server"] == {"healthy": False, "status": "loading"}


# ---- get_metrics ------------------------------------------------------------

async def test_get_metrics_clamps_the_window(monkeypatch) -> None:
    """since_minutes comes straight from the MCP caller.

    10**15 raised OverflowError out of timedelta and a negative value put the
    cutoff in the future, reporting "no data" instead of "bad request".
    """
    seen: list[int] = []

    class _Storage:
        def latest_per_metric(self, since_minutes: int):
            seen.append(since_minutes)
            return {}, 0

    monkeypatch.setattr(srv, "_telemetry_storage", _Storage())

    assert json.loads(await srv.get_metrics(since_minutes=10**15))["period_minutes"] == 7 * 24 * 60
    assert json.loads(await srv.get_metrics(since_minutes=-5))["period_minutes"] == 1
    assert json.loads(await srv.get_metrics(since_minutes=30))["period_minutes"] == 30
    assert seen == [7 * 24 * 60, 1, 30]


async def test_get_metrics_returns_one_key_set(monkeypatch) -> None:
    """Both branches must answer with identical keys.

    A consumer reading data["sample_count"] KeyErrored exactly when telemetry
    was not wired — which is the whole point of the fallback existing.
    """
    class _Storage:
        def latest_per_metric(self, _since_minutes: int):
            return {"cpu_percent": 42.0}, 3

    monkeypatch.setattr(srv, "_telemetry_storage", _Storage())
    wired = json.loads(await srv.get_metrics())

    monkeypatch.setattr(srv, "_telemetry_storage", None)
    fallback = json.loads(await srv.get_metrics())

    assert wired.keys() == fallback.keys()
    assert fallback["sample_count"] == 0


# ---- LlamaClient.chat -------------------------------------------------------


def _chat_client(monkeypatch, payload: dict, *, api_key: str | None = None):
    """A LlamaClient whose POST is captured instead of dialled.

    Returns (client, sent) where ``sent`` collects (url, json body) tuples.
    """
    client = LlamaClient("https://127.0.0.1:8080", api_key=api_key)
    sent: list[tuple] = []

    def _post(url, json=None, **_kw):
        sent.append((url, json))
        return _FakeResponse(200, payload)

    monkeypatch.setattr(client._client, "post", _post)
    return client, sent


def _chat_body(content, *, reasoning: str | None = None, **extra) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    body = {"choices": [{"message": message, "finish_reason": "stop"}]}
    body.update(extra)
    return body


def test_chat_posts_a_chat_completion_not_a_raw_prompt(monkeypatch) -> None:
    """A raw /completion prompt skips the model's chat template.

    Observed with a Gemma 4 31B model on one test bench: "Reply with the single word: ok" came
    back from /completion as ", ok, ok, ok…" with `<channel|>` tokens leaking
    through, and as "ok" from /v1/chat/completions. Qwen tolerating the raw
    path was luck, not a contract.
    """
    client, sent = _chat_client(monkeypatch, _chat_body("ok"))

    client.chat([{"role": "user", "content": "hi"}], max_tokens=64, temperature=0.1)

    url, body = sent[0]
    assert url == "/v1/chat/completions"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.1
    # Explicit, because both callers parse one JSON body and an SSE stream
    # would not be readable by json.loads at all.
    assert body["stream"] is False
    assert "prompt" not in body and "n_predict" not in body
    client.close()


def test_chat_asks_the_chat_template_to_disable_thinking(monkeypatch) -> None:
    """A tool call is not a chat: the budget must buy an answer.

    Measured on both reasoning models in the fleet (the sharded Qwen3.5-35B
    head and a Gemma 4 GPU node on one test bench): run_inference with max_tokens=64
    returned EMPTY content, the whole budget spent in `reasoning_content` with
    finish_reason "length". This request field fixed both; `reasoning_budget:
    0` did not. Every call carries it — there is no caller-facing switch.
    """
    client, sent = _chat_client(monkeypatch, _chat_body("ok"))

    client.chat([{"role": "user", "content": "hi"}], max_tokens=64)

    assert sent[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    client.close()


def test_chat_keeps_the_bearer_header_on_the_client(monkeypatch) -> None:
    """/v1/chat/completions is NOT one of the auth-exempt paths.

    The key rides on the httpx client rather than a per-call kwargs dict, so
    switching endpoints must not have left it behind on /completion.
    """
    client, _sent = _chat_client(monkeypatch, _chat_body("ok"), api_key="aureax_sk_AAAA")

    assert client._client.headers["Authorization"] == "Bearer aureax_sk_AAAA"
    client.close()


def test_chat_returns_content_and_never_reasoning_content(monkeypatch) -> None:
    """With --jinja a reasoning model splits its thinking into a second field.

    Returning that hands the caller the model's scratchpad as the answer.
    """
    client, _sent = _chat_client(
        monkeypatch, _chat_body("the answer", reasoning="scratchpad the user must not see")
    )

    result = client.chat([{"role": "user", "content": "q"}])

    assert result["content"] == "the answer"
    assert "scratchpad" not in json.dumps(result)
    client.close()


def test_chat_still_strips_an_inlined_think_block(monkeypatch) -> None:
    """Belt-and-braces: some templates leave <think> inside `content`."""
    client, _sent = _chat_client(monkeypatch, _chat_body("<think>plan</think>final"))

    assert client.chat([{"role": "user", "content": "q"}])["content"] == "final"
    client.close()


def test_chat_reads_a_null_content_as_an_empty_answer(monkeypatch) -> None:
    """A reasoning model that spends its whole budget thinking returns
    content: null with finish_reason "length" — an empty answer, not a crash,
    and json.dumps of the tool result must not carry a bare None."""
    client, _sent = _chat_client(monkeypatch, _chat_body(None))

    assert client.chat([{"role": "user", "content": "q"}])["content"] == ""
    client.close()


def test_chat_maps_usage_onto_the_token_counts_the_verify_script_asserts(monkeypatch) -> None:
    """scripts/test-mcp-tools.py fails a node whose run_inference reports zero
    tokens or zero time, so the OpenAI field names have to be translated."""
    client, _sent = _chat_client(
        monkeypatch,
        _chat_body(
            "hi",
            usage={"completion_tokens": 7, "prompt_tokens": 11},
            timings={"predicted_ms": 1234.5},
        ),
    )

    result = client.chat([{"role": "user", "content": "q"}])

    assert result["tokens_predicted"] == 7
    assert result["tokens_evaluated"] == 11
    assert result["timings"]["predicted_ms"] == 1234.5
    client.close()


def test_chat_falls_back_to_the_measured_round_trip_without_timings(monkeypatch) -> None:
    """`timings` is a llama.cpp extension, not part of the OpenAI shape.

    A build that stops sending it must not make every node look broken to the
    verify script's non-zero check.
    """
    client, _sent = _chat_client(monkeypatch, _chat_body("hi"))

    result = client.chat([{"role": "user", "content": "q"}])

    assert result["timings"]["predicted_ms"] > 0
    client.close()


def test_chat_surfaces_an_http_error(monkeypatch) -> None:
    """raise_for_status still fires, so run_inference reports it as an error
    result instead of returning an empty completion."""
    client = LlamaClient("https://127.0.0.1:8080")

    def _post(*_a, **_kw):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(client._client, "post", _post)
    with pytest.raises(httpx.HTTPError):
        client.chat([{"role": "user", "content": "q"}])
    client.close()


# ---- run_inference ----------------------------------------------------------

async def test_run_inference_clamps_max_tokens(monkeypatch) -> None:
    """Unbounded max_tokens pins a shared default-executor thread for 600s."""
    seen: list[int] = []

    class _Llama:
        def chat(self, messages: list[dict], *, max_tokens: int, temperature: float) -> dict:
            seen.append(max_tokens)
            return {"content": "hi"}

    monkeypatch.setattr(srv, "_llama_client", _Llama())
    monkeypatch.setattr(srv, "_sensor_sim", SensorSimulator("edge-01"))

    await srv.run_inference("p", max_tokens=10**9)
    await srv.run_inference("p", max_tokens=0)
    await srv.run_inference("p", max_tokens=64)

    assert seen == [srv.MAX_INFERENCE_TOKENS, 1, 64]


async def test_run_inference_sends_the_prompt_as_a_single_user_message(monkeypatch) -> None:
    """The tool takes a bare prompt string but must not send one: the raw
    endpoint bypasses the chat template, which is what broke Gemma 4."""
    client, sent = _chat_client(
        monkeypatch,
        _chat_body(
            "ok",
            reasoning="thinking the caller must not receive",
            usage={"completion_tokens": 2, "prompt_tokens": 5},
        ),
    )
    monkeypatch.setattr(srv, "_llama_client", client)
    monkeypatch.setattr(srv, "_sensor_sim", SensorSimulator("edge-01"))

    payload = json.loads(await srv.run_inference("Reply with the single word: ok", max_tokens=64))

    url, body = sent[0]
    assert url == "/v1/chat/completions"
    assert body["messages"] == [
        {"role": "user", "content": "Reply with the single word: ok"}
    ]
    assert payload["content"] == "ok"
    assert "thinking" not in json.dumps(payload)
    # The key set the verify script and the harness read back, unchanged from
    # the /completion era.
    assert payload["tokens_predicted"] == 2
    assert payload["tokens_evaluated"] == 5
    assert payload["generation_time_ms"] > 0
    client.close()


async def test_run_inference_reports_a_budget_spent_on_reasoning_as_an_error(
    monkeypatch,
) -> None:
    """Empty content plus finish_reason "length" plus reasoning is not an
    answer, and returning "" tells the caller the model had nothing to say.

    The live shape this reproduces: max_tokens=64 against a reasoning model
    put the entire budget into `reasoning_content`. LlamaClient now disables
    thinking on every call, so reaching this branch means the template ignored
    the request — the caller still has to be told to raise max_tokens.
    """
    body = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "thinking that used the whole budget",
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 64, "prompt_tokens": 5},
    }
    client, _sent = _chat_client(monkeypatch, body)
    monkeypatch.setattr(srv, "_llama_client", client)
    monkeypatch.setattr(srv, "_sensor_sim", SensorSimulator("edge-01"))

    payload = json.loads(await srv.run_inference("q", max_tokens=64))

    assert "raise max_tokens" in payload["error"]
    assert "64" in payload["error"]
    # The same {"error": ...} shape the transport failure below returns, and
    # never the model's scratchpad.
    assert set(payload) == {"error"}
    assert "thinking that used the whole budget" not in json.dumps(payload)
    client.close()


async def test_run_inference_catches_the_inlined_think_form_of_the_same_failure(
    monkeypatch,
) -> None:
    """Same phenomenon, other shape: a template that leaves the block inside
    `content` instead of splitting it into `reasoning_content`.

    strip_thinking reduces an unterminated <think> to "", so without this the
    caller would get an empty answer for a budget that was entirely spent.
    """
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "<think>reasoning that never ends"},
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 64, "prompt_tokens": 5},
    }
    client, _sent = _chat_client(monkeypatch, body)
    monkeypatch.setattr(srv, "_llama_client", client)
    monkeypatch.setattr(srv, "_sensor_sim", SensorSimulator("edge-01"))

    payload = json.loads(await srv.run_inference("q", max_tokens=64))

    assert "raise max_tokens" in payload["error"]
    assert "reasoning that never ends" not in json.dumps(payload)
    client.close()


async def test_run_inference_still_returns_an_empty_answer_when_nothing_was_thought(
    monkeypatch,
) -> None:
    """Only the reasoning case becomes an error. A model that simply stops
    with nothing to say still returns the four-key result the verify script
    and the harness read back."""
    body = {
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 0, "prompt_tokens": 5},
    }
    client, _sent = _chat_client(monkeypatch, body)
    monkeypatch.setattr(srv, "_llama_client", client)
    monkeypatch.setattr(srv, "_sensor_sim", SensorSimulator("edge-01"))

    payload = json.loads(await srv.run_inference("q", max_tokens=64))

    assert "error" not in payload
    assert payload["content"] == ""
    client.close()


async def test_run_inference_reports_a_transport_failure_as_an_error_result(monkeypatch) -> None:
    """An exception must come back as {"error": ...}, not escape the tool —
    the harness renders that as an ordinary result."""
    client = LlamaClient("https://127.0.0.1:8080")

    def _post(*_a, **_kw):
        raise httpx.HTTPError("connection refused")

    monkeypatch.setattr(client._client, "post", _post)
    monkeypatch.setattr(srv, "_llama_client", client)
    monkeypatch.setattr(srv, "_sensor_sim", SensorSimulator("edge-01"))

    payload = json.loads(await srv.run_inference("p"))

    assert "connection refused" in payload["error"]
    client.close()


# ---- report_anomaly ---------------------------------------------------------

def test_report_anomaly_truncates_caller_text() -> None:
    """maxlen bounds the entry COUNT; each entry's free text is bounded here."""
    payload = json.loads(
        srv.report_anomaly("t" * 5000, "s" * 10_000, "d" * 100_000, value=1.0)
    )
    stored = payload["anomaly"]
    assert len(stored["description"]) == srv._MAX_ANOMALY_DESCRIPTION
    assert len(stored["sensor_type"]) == srv._MAX_ANOMALY_SENSOR_TYPE
    # severity is documented as low/medium/high/critical, but nothing on the
    # wire enforces it, so it is capped like the other two caller strings.
    assert len(stored["severity"]) == srv._MAX_ANOMALY_SEVERITY
    assert srv._anomaly_log[-1]["description"] == stored["description"]
    assert srv._anomaly_log[-1]["severity"] == stored["severity"]


# ---- uptime -----------------------------------------------------------------

async def test_uptime_is_monotonic_not_wall_clock(monkeypatch) -> None:
    """Edge nodes step the clock once time sync converges, after the pod starts.

    A wall-clock origin made the reported uptime jump by the size of the
    correction, or go negative on a backwards step.
    """
    # A monotonic origin is seconds since boot; a wall-clock one is ~1.7e9, so
    # subtracting this from time.time() could not land in the window below.
    monkeypatch.setattr(srv, "_start_time", srv.time.monotonic() - 42)

    monkeypatch.setattr(srv, "_llama_client", None)
    health = json.loads(await srv.get_health())
    assert 42 <= health["uptime_seconds"] <= 43

    monkeypatch.setattr(srv, "_telemetry_storage", None)
    metrics = json.loads(await srv.get_metrics())
    assert 42 <= metrics["metrics"]["uptime_seconds"] <= 43


async def test_anomaly_count_uses_one_key_in_both_tools(monkeypatch) -> None:
    """One saturating counter, one name — the vocabulary the configs use."""
    monkeypatch.setattr(srv, "_llama_client", None)
    monkeypatch.setattr(srv, "_telemetry_storage", None)
    assert "anomalies_detected" in json.loads(await srv.get_health())
    assert "anomalies_detected" in json.loads(await srv.get_metrics())["metrics"]


# ---- sensor simulator -------------------------------------------------------

def test_unknown_sensor_type_raises() -> None:
    """A returned {"error": ...} dict came back as isError=False.

    MCPPool then parsed it into the same shape it uses for "No session for
    node" and transport failures, and the agent loop fed a bad argument back to
    the model as a normal tool result.
    """
    sim = SensorSimulator("edge-01")
    with pytest.raises(ValueError, match="Unknown sensor type"):
        sim.read("humidty")


def test_battery_agrees_between_read_and_status(monkeypatch) -> None:
    """get_sensor_status read the cached field, which only read_battery moved.

    A node whose battery sensor was never read reported 100.0 for the life of
    the pod while read_sensor("battery") returned the drained value.
    """
    clock = [1_700_000_000.0]
    monkeypatch.setattr("client.sensors.simulator.time.time", lambda: clock[0])
    sim = SensorSimulator("edge-01")

    clock[0] += 10 * 3600  # ten hours, drain rate is 0.5-0.8 %/h
    status_level = sim.get_status()["sensors"]["battery"]["level"]
    assert status_level < 100.0
    assert sim.read_battery()["value"] == status_level


def test_baselines_are_drawn_once_per_instance(monkeypatch) -> None:
    """Two reads from ONE instance must not jump by whole degrees.

    Comparing two fresh instances cannot catch this — both replay the same RNG
    sequence whether the baseline is drawn in __init__ or inside
    read_temperature. Pin the clock and the noise draw and only the once-drawn
    baseline keeps the two reads equal.
    """
    monkeypatch.setattr("client.sensors.simulator.time.time", lambda: 1_700_000_000.0)
    sim = SensorSimulator("edge-01")
    sim._rng.gauss = lambda _mu, _sigma: 0.0

    assert sim.read_temperature()["value"] == sim.read_temperature()["value"]


# ---- strip_thinking ---------------------------------------------------------

def test_strip_thinking_drops_an_unterminated_block() -> None:
    """chat() caps generation and never inspects finish_reason.

    A thinking model cut off mid-reasoning returns "<think>..." with no closing
    tag; engine.py stored that verbatim and replayed it on every later turn.
    """
    assert strip_thinking("<think>reasoning that never ends") == ""
    assert strip_thinking("<think>a</think>answer<think>cut off") == "answer"
    # Closed blocks still behave.
    assert strip_thinking("<think>plan</think>final") == "final"


# ---- the bearer gate --------------------------------------------------------

@pytest.fixture
def restore_mcp_key():
    """Put ``_mcp_api_key`` back however the test left it.

    It is module state that init() writes, so a test that sets it would
    otherwise decide whether the NEXT test's build_app() wraps — an ordering
    dependency that only shows up when the suite is reordered.
    """
    previous = srv._mcp_api_key
    yield
    srv._mcp_api_key = previous


def test_build_app_wraps_the_app_when_a_key_is_configured(restore_mcp_key, caplog) -> None:
    """Serving mcp.streamable_http_app() directly is how this port loses its auth.

    The gate is an ASGI wrapper around the whole app — there is no auth inside
    FastMCP — so what build_app returns is the entire enforcement point for
    both doors into this process (hostNetwork :9000 and NodePort :30900).
    """
    srv._mcp_api_key = "aureax_sk_" + "k" * 32

    with caplog.at_level("INFO", logger="aureax.mcp.server"):
        app = srv.build_app()

    assert isinstance(app, mcp_auth.BearerAuthMiddleware)
    lines = [r.message for r in caplog.records if "bearer auth" in r.message]
    assert len(lines) == 1, f"expected exactly one state line, got {lines}"
    assert lines[0] == (
        "MCP bearer auth ENABLED — /mcp requires Authorization: Bearer; /health stays open"
    )
    assert "aureax_sk_" not in lines[0], "the key must never reach a log line"


def test_build_app_returns_the_bare_app_and_warns_when_no_key_is_set(
    restore_mcp_key, caplog
) -> None:
    """A node whose template lost its api_key line looks identical otherwise.

    `kubectl logs <pod> | grep 'bearer auth'` is the only way to tell the two
    apart from the Mac, so the disabled state has to say so at WARNING.
    """
    srv._mcp_api_key = None

    with caplog.at_level("INFO", logger="aureax.mcp.server"):
        app = srv.build_app()

    assert not isinstance(app, mcp_auth.BearerAuthMiddleware)
    records = [r for r in caplog.records if "bearer auth" in r.message]
    assert len(records) == 1
    assert records[0].levelname == "WARNING"
    assert records[0].message == (
        "MCP bearer auth DISABLED — this port is open to anyone who can reach it"
    )


def test_the_configured_key_is_the_one_the_gate_compares(restore_mcp_key) -> None:
    """init() stores the key; build_app() is what turns it into a gate.

    Pinned because the two halves live in different functions: an init that
    stored the key under a second name would leave every node unauthenticated
    with both halves looking correct in isolation.
    """
    key = "aureax_sk_" + "z" * 32
    srv._mcp_api_key = key

    app = srv.build_app()

    assert app._expected == key.encode()
    assert app._exempt == frozenset({"/health"})


@pytest.fixture
def restore_init_state():
    """Put back every module global :func:`init` writes, and close its client.

    Broader than ``restore_mcp_key`` because init() wires five globals and
    FastMCP's bind settings in one call — a test that runs the real thing would
    otherwise leave a live LlamaClient and this node's id behind for whatever
    runs next.
    """
    saved = (
        srv._llama_client, srv._sensor_sim, srv._telemetry_storage,
        srv._node_id, srv._mcp_api_key, srv.mcp.settings.host, srv.mcp.settings.port,
    )
    yield
    if srv._llama_client is not None and srv._llama_client is not saved[0]:
        srv._llama_client.close()
    (
        srv._llama_client, srv._sensor_sim, srv._telemetry_storage,
        srv._node_id, srv._mcp_api_key,
    ) = saved[:5]
    srv.mcp.settings.host, srv.mcp.settings.port = saved[5], saved[6]


@pytest.mark.parametrize("configured", [True, False])
def test_init_hands_its_key_to_the_gate_build_app_reads(restore_init_state, configured) -> None:
    """The real init() -> the real build_app(), with nothing stubbed between.

    Everything else about this seam is tested from one side or the other: the
    tests above assign ``srv._mcp_api_key`` directly, and the supervisor's tests
    monkeypatch both init and build_app, so the ONE line that carries init()'s
    ``mcp_api_key`` into the global build_app() gates on is crossed by nothing.
    Drop it and every node's MCP port serves unauthenticated with a green suite:
    pods Ready, /health answering, `aureax status` clean.
    """
    key = "aureax_sk_" + "q" * 32

    srv.init(
        "edge-01",
        "https://127.0.0.1:8080",
        mcp_api_key=key if configured else None,
    )
    app = srv.build_app()

    if configured:
        assert isinstance(app, mcp_auth.BearerAuthMiddleware)
        assert app._expected == key.encode()
    else:
        assert not isinstance(app, mcp_auth.BearerAuthMiddleware)
