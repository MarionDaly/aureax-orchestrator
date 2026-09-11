"""Chat completion client for llama-server's OpenAI-compatible API.

Uses /v1/chat/completions with tool-calling support. The dashboard's
ChatService builds one ChatLlamaClient per chat target
(orchestrator/dashboard/chat_service.py::_llm_for) and client/engine.py's
PlannerEngine consumes it.

The edge node's LlamaClient (client/inference/llama_client.py) speaks the same
endpoint for its own run_inference path and shares the request/response
shaping below, so there is ONE chat-completions body in the codebase rather
than one per client that can drift apart.
"""

from __future__ import annotations

import contextlib
import json
import re
import ssl

import httpx

# Sized against chat()'s own 4096 max_tokens default at ~4 tok/s, the slow end
# for an edge node: 4096 / 4 ≈ 1024s, so a full-length generation finishes
# inside this budget instead of hitting a ReadTimeout that costs the caller a
# completed-but-undelivered turn. Lowering max_tokens is not the alternative —
# a reasoning model cut short returns empty content with finish_reason
# "length". Raise both together, or neither.
#
# Deliberately NOT shared with llama_client's constant of the same name, which
# budgets the edge's own /v1/chat/completions calls and is what
# mcp.pool.MCP_CALL_TIMEOUT_SECONDS is sized above; keeping this local also
# avoids a cross-module import for one constant.
LLAMA_HTTP_TIMEOUT_SECONDS = 1200.0

# The one chat endpoint both clients dial. Named once so a path change is one
# edit, not a grep across two modules.
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def build_chat_payload(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Build the /v1/chat/completions request body.

    Shared by ChatLlamaClient (dashboard, tool-calling) and LlamaClient (the
    edge node's own run_inference), because the two were one copy-paste away
    from disagreeing about a field that only one of them would ever exercise.

    ``stream`` is stated explicitly even though false is llama.cpp's default:
    both callers parse ONE JSON body, and a build that ever flipped that
    default would hand them an SSE stream that json.loads cannot read.

    ``tool_choice`` rides with ``tools`` and only with it — llama.cpp rejects
    ``tools`` outright unless llama-server was started with --jinja, and an
    empty list is not worth that risk.

    ``chat_template_kwargs`` is emitted only when a caller passes something
    truthy, for the same reason: the dict is handed to the GGUF's own Jinja
    template, so an empty one buys nothing and only widens what a template has
    to tolerate. The edge's LlamaClient passes
    ``{"enable_thinking": False}`` on every call (the live evidence is in
    client/inference/llama_client.py); the dashboard's ChatLlamaClient passes
    nothing, deliberately, so interactive chat keeps its thinking.
    """
    payload: dict = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    return payload


def parse_chat_response(text: str) -> dict:
    """Parse a chat-completions body with strict=False.

    Shared for the reason it exists at all: reasoning models emit raw control
    characters inside thinking output, which strict JSON rejects, and a caller
    that reached for plain ``resp.json()`` would fail on exactly the models
    this fleet runs.
    """
    return json.loads(text, strict=False)


class ChatLlamaClient:
    """Chat completion client with tool-calling support."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        verify: ssl.SSLContext | None = None,
    ) -> None:
        """Point at a llama-server running with --jinja.

        ``base_url`` must be a full URL (``https://host:port``) — every
        llama-server in the fleet listens with a leaf signed by the fleet CA.
        ``verify`` is the ``SSLContext`` that trusts that CA
        (``tlsstore.ca_ssl_context()``); leaving it None falls back to the
        system trust store, which no fleet certificate chains to.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.verify = verify
        self._client = self._build_client()
        self._apply_auth_header()

    def _build_client(self) -> httpx.Client:
        """The ONE place an httpx client for this target is constructed.

        set_base_url rebuilds the transport, and when that rebuild had its own
        inline ``httpx.Client(...)`` it was one edit away from silently dropping
        whatever the constructor had configured — a host change would leave the
        client verifying against the system trust store, where nothing the
        fleet CA signed validates, and every turn would fail with a certificate
        error naming the node rather than the missing CA.
        """
        return httpx.Client(
            base_url=self.base_url,
            timeout=LLAMA_HTTP_TIMEOUT_SECONDS,
            # httpx treats None as "no explicit context" and would let httpcore
            # build a certifi-backed default, so name the default verifying
            # behaviour explicitly instead of passing the None through.
            verify=True if self.verify is None else self.verify,
        )

    def _apply_auth_header(self) -> None:
        if self.api_key:
            self._client.headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            self._client.headers.pop("Authorization", None)

    def set_api_key(self, api_key: str | None) -> None:
        """Update the Authorization header on the live httpx client.

        MUTATES rather than replaces, for the same reason _pool_for refreshes
        an MCPPool in place (orchestrator/dashboard/chat_service.py): cached
        PlannerEngines hold their own reference to this object, so swapping the
        ChatService dict entry would leave every already-built engine sending
        the old key.
        """
        self.api_key = api_key
        self._apply_auth_header()

    def set_base_url(self, base_url: str) -> None:
        """Repoint this client at a new address, rebuilding its httpx client.

        MUTATES rather than replaces, for the same reason set_api_key does:
        cached PlannerEngines hold their own reference to this object, so
        swapping the ChatService dict entry would leave every already-built
        engine dialling the old address. httpx freezes base_url into the client
        at construction and offers no setter, so the transport is rebuilt here.

        The previous transport is closed rather than left to the garbage
        collector: a host edit is rare, but leaking a whole connection pool per
        edit is the failure ChatService caches these to avoid. Any request
        still in flight on it is aimed at an address this target no longer has.

        The rebuild goes through _build_client and re-applies the auth header,
        so the CA context and the bearer key both survive a repoint — the
        transport is new, everything configured on it is not.
        """
        if base_url == self.base_url:
            return
        old = self._client
        self.base_url = base_url
        self._client = self._build_client()
        self._apply_auth_header()
        with contextlib.suppress(Exception):
            old.close()

    def close(self) -> None:
        """Close the underlying httpx client, releasing its connection pool.

        The dashboard's ChatService owns one of these per chat target and calls
        this on shutdown; without it the pools live until the process exits.
        """
        self._client.close()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat completion request with optional tool schemas.

        Returns the raw response dict from /v1/chat/completions — the caller
        (client/engine.py) reads tool_calls off the message itself, so nothing
        is normalized away here.
        """
        payload = build_chat_payload(
            messages, temperature=temperature, max_tokens=max_tokens, tools=tools
        )
        resp = self._client.post(CHAT_COMPLETIONS_PATH, json=payload)
        resp.raise_for_status()
        return parse_chat_response(resp.text)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output.

    Two passes. The second drops an UNTERMINATED block: chat() caps generation
    at max_tokens and never inspects finish_reason, so a thinking model cut off
    mid-reasoning returns "<think>..." with no closing tag. After the pair
    substitution any remaining "<think>" is by definition unclosed, and it has
    to go: engine.py stores what this returns as the assistant message and
    replays it to the model on every later turn.
    """
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()
