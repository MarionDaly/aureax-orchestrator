"""HTTP client for communicating with llama-server.

Speaks the chat-completions endpoint, NOT the raw /completion one. A raw
prompt bypasses the model's chat template, and a model whose template carries
its own control tokens then never sees them: on one test bench's Gemma 4 31B node, a raw
/completion of "Reply with the single word: ok" came back as ", ok, ok, ok,
ok…" and leaked `<channel|>` tokens, while the same text as a user message on
/v1/chat/completions answered "ok". Qwen tolerated the raw path; that was luck,
not a contract. llama-server is started with --jinja (client/config/parser.py),
so the GGUF's own template applies.
"""

from __future__ import annotations

import ssl
import time

import httpx

from client.inference.chat_client import (
    CHAT_COMPLETIONS_PATH,
    build_chat_payload,
    parse_chat_response,
    strip_thinking,
)

# 600s covers ~2400 tokens at ~4 tok/s on an edge node. The MCP protocol and
# harness CLI handle their own timeouts upstream — this client should never
# be the bottleneck.
LLAMA_HTTP_TIMEOUT_SECONDS = 600.0


class LlamaClient:
    """Client for the llama.cpp HTTP server running on localhost."""

    def __init__(
        self,
        base_url: str = "https://127.0.0.1:8080",
        api_key: str | None = None,
        verify: ssl.SSLContext | None = None,
    ) -> None:
        """Create an HTTP client pointed at the given llama-server URL.

        The bearer header is set on the client, not per request: every path is
        then covered by construction (harmless on the auth-exempt /health), and
        the key never sits in a per-call kwargs dict that could surface in a
        traceback.

        ``verify`` is the fleet CA context (``client.tls.ssl_context``). The
        default URL is https even though it is loopback: llama-server is
        started with ``--ssl-cert-file``, so it speaks TLS on :8080 and nothing
        else, and the certificate carries ``IP:127.0.0.1`` as a SAN precisely
        so this in-pod dial verifies. Leaving ``verify`` None checks against
        the public trust store, which a fleet-CA certificate fails — a loud
        error rather than an unverified connection.
        """
        self.base_url = base_url
        self.api_key = api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(
            base_url=base_url,
            timeout=LLAMA_HTTP_TIMEOUT_SECONDS,
            headers=headers,
            verify=verify,
        )

    def health(self) -> dict:
        """Check llama-server health. Returns status dict.

        Its own 5s timeout, overriding the client's completion budget: a probe
        that has not answered in 5s has told the caller everything, and this
        runs on a shared executor thread that a wedged mid-load server would
        otherwise pin for the full budget.

        503 is a STATE, not a failure, so it is returned as
        ``{"status": "loading"}`` and never raised: llama-server answers /health
        with 503 while the model loads — minutes on a sharded head — and
        raise_for_status would report that as an error indistinguishable from an
        unreachable server.
        """
        resp = self._client.get("/health", timeout=5.0)
        if resp.status_code == 503:
            return {"status": "loading"}
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the underlying httpx client, releasing its connection pool.

        Symmetric with ChatLlamaClient.close(), so neither caller has to reach
        into the private client to release a pool.
        """
        self._client.close()

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """Send a chat completion and return the assistant's answer.

        Returns the flat shape run_inference re-emits — ``content``,
        ``tokens_predicted``, ``tokens_evaluated``, ``timings`` — rather than
        llama-server's response, so the OpenAI field names stay in this one
        place and the MCP tool's output keys are unchanged from the /completion
        era (scripts/test-mcp-tools.py asserts all four on a live node).
        Two further keys, ``finish_reason`` and ``has_reasoning``, exist only
        so run_inference can tell an empty answer apart from a budget burned
        on hidden thinking; they are not re-emitted to the MCP caller.
        """
        started = time.monotonic()
        resp = self._client.post(
            CHAT_COMPLETIONS_PATH,
            json=build_chat_payload(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # Thinking OFF, on every call: a tool call is not a chat, so
                # the caller's token budget has to buy an ANSWER rather than a
                # scratchpad nobody receives. Measured on the fleet against
                # both reasoning models — the sharded Qwen3.5-35B head and
                # Gemma 4 on one test bench's GPU node — a run_inference call with
                # max_tokens=64 came back with EMPTY content: the whole budget
                # went to `reasoning_content` and finish_reason was "length".
                # This one request field fixed BOTH (content "ok", 2 tokens,
                # no reasoning); the request field `reasoning_budget: 0` did
                # NOT. The dashboard's ChatLlamaClient passes nothing here on
                # purpose, so interactive chat keeps its thinking.
                chat_template_kwargs={"enable_thinking": False},
            ),
        )
        resp.raise_for_status()
        body = parse_chat_response(resp.text)

        choices = body.get("choices") or [{}]
        message = choices[0].get("message") or {}
        # `content`, NEVER `reasoning_content`: with --jinja a reasoning model
        # returns its thinking in that separate field, and handing the caller
        # the scratchpad instead of the answer is the failure this endpoint was
        # adopted to avoid. content is null when the model spent its whole
        # budget thinking (finish_reason "length") — an empty answer, not a
        # crash. strip_thinking stays as belt-and-braces for templates that
        # inline <think> into content instead of splitting it out.
        raw_content = message.get("content") or ""
        content = strip_thinking(raw_content)

        usage = body.get("usage") or {}
        timings = body.get("timings") or {}
        # llama.cpp attaches `timings` to OAI responses, but that is a build
        # detail and not part of the OpenAI shape, so fall back to the measured
        # round trip: the live verify script requires a non-zero number here,
        # and a missing field would report the node as broken rather than slow.
        predicted_ms = timings.get("predicted_ms")
        if not predicted_ms:
            predicted_ms = (time.monotonic() - started) * 1000.0

        return {
            "content": content,
            "tokens_predicted": int(usage.get("completion_tokens") or 0),
            "tokens_evaluated": int(usage.get("prompt_tokens") or 0),
            "timings": {"predicted_ms": predicted_ms},
            # Why the answer is what it is, for run_inference's empty-content
            # check. A BOOLEAN, never the reasoning text: carrying the
            # scratchpad out of this method is the failure the field split
            # above exists to prevent. True for BOTH shapes of the same
            # phenomenon — a separate `reasoning_content`, and a template that
            # inlined <think> into content, which strip_thinking has just
            # reduced to nothing.
            "finish_reason": str(choices[0].get("finish_reason") or ""),
            "has_reasoning": bool(message.get("reasoning_content")) or (
                bool(raw_content) and not content
            ),
        }
