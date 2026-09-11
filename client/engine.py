"""Agentic tool-calling loop.

Sends user messages to an LLM with remote tool schemas attached. When the
LLM returns tool_calls, dispatches them to the correct remote node via the
MCP pool and feeds results back. Repeats until the LLM returns a final text
response.

Its only caller today is the orchestrator's dashboard chat
(``orchestrator/dashboard/chat_service.py``), which runs one engine per chat
session. It lives under ``client/`` because it is written against the edge
client's own LLM and MCP-pool interfaces, and because a node is intended to
run its own loop eventually — but the planner pod does NOT use it yet:
``client/entrypoint.py::_run_planner_host`` only holds the MCP sessions open.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from client.inference.chat_client import ChatLlamaClient, strip_thinking
from client.mcp.pool import MCPPool

logger = logging.getLogger("aureax.engine")

MAX_TOOL_ROUNDS = 10

# Approximate character budget for the history sent to the model. MAX_TOOL_ROUNDS
# bounds one turn; this bounds the SESSION, which is the axis that grows without
# limit — an unbounded history overflows the served context, and because the
# same oversized history reloads on the next turn, every later request in that
# session fails the same way.
#
# This is the CEILING, and the fallback when the caller does not know what the
# target serves. At ~4 chars/token it is roughly 50k tokens, which fits the
# 131072 a sharded head serves. Targets are served at very different context
# sizes, and a caller holding the target's model.server.context_size passes
# history_budget_chars(context_size) instead: against a small context a fixed
# 200k budget never trims, so the session bricks in exactly the way described
# above.
MAX_HISTORY_CHARS = 200_000

# Rough tokens-to-characters ratio for English prose plus JSON tool payloads.
CHARS_PER_TOKEN = 4

# Share of the served context the replayed history may occupy. The remainder
# carries the tool schemas and the reply (chat() asks for up to 4096 tokens),
# neither of which _fits_context can see and neither of which is optional.
HISTORY_CONTEXT_FRACTION = 0.4

# Floor for a derived budget: below this every turn would drop the user's own
# prompt, which _fits_context would do silently.
MIN_HISTORY_CHARS = 2048


def history_budget_chars(context_size: int | None) -> int:
    """History budget in characters for a target served at ``context_size``.

    Falls back to MAX_HISTORY_CHARS when the size is unknown or nonsensical,
    and never exceeds it: a bigger context does not make replaying a bigger
    session on every turn a good idea.
    """
    if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size <= 0:
        return MAX_HISTORY_CHARS
    derived = int(context_size * HISTORY_CONTEXT_FRACTION) * CHARS_PER_TOKEN
    return min(MAX_HISTORY_CHARS, max(MIN_HISTORY_CHARS, derived))


def _msg_chars(message: dict) -> int:
    """What one stored message costs in the request this replays it into.

    ``tool_calls`` counts alongside ``content``: an assistant turn that calls a
    tool carries a stripped or empty ``content`` and its whole JSON arguments
    payload under ``tool_calls``, so measuring content alone prices the largest
    messages in a tool-heavy session at zero — the overflow the budget exists
    to prevent.
    """
    return (
        len(str(message.get("content") or ""))
        + len(json.dumps(message.get("tool_calls") or []))
    )


def _fits_context(messages: list[dict], max_chars: int = MAX_HISTORY_CHARS) -> list[dict]:
    """The system message plus as much recent history as fits `max_chars`.

    A COPY: `self.messages` stays complete because the dashboard persists it by
    INDEX (chat_service.ask records len(engine.messages) before the turn and
    _drain_engine slices from there), so trimming the real list would misalign
    every row it writes.

    Cuts only at a `role == "user"` boundary, so an assistant message carrying
    tool_calls is never separated from its matching `role: tool` replies —
    llama-server rejects that pairing outright. Budgeted in characters rather
    than messages because one tool result is a whole JSON payload and can
    outweigh many turns.
    """
    if not messages:
        return []
    system, rest = messages[:1], messages[1:]
    budget = max_chars - len(str(system[0].get("content") or ""))

    # The newest user turn and everything after it is kept unconditionally,
    # even if it alone blows the budget: sending a question the model may
    # reject beats silently dropping the question being answered.
    last_user = next(
        (i for i in range(len(rest) - 1, -1, -1) if rest[i].get("role") == "user"), 0
    )
    keep_from = last_user
    used = sum(_msg_chars(m) for m in rest[last_user:])

    # Then walk backwards, extending only as far as a whole user turn fits.
    for i in range(last_user - 1, -1, -1):
        used += _msg_chars(rest[i])
        if used > budget:
            break
        if rest[i].get("role") == "user":
            keep_from = i
    return system + rest[keep_from:]


class PlannerEngine:
    """Agentic loop: LLM reasons, calls tools, synthesizes answers."""

    def __init__(
        self,
        llm: ChatLlamaClient,
        mcp_pool: MCPPool,
        system_prompt: str,
        max_history_chars: int = MAX_HISTORY_CHARS,
    ) -> None:
        """Initialize with LLM client, MCP pool, system prompt and history budget.

        ``max_history_chars`` belongs to the ENGINE, not the module, because it
        has to match what this engine's target serves — see
        history_budget_chars. The default is the ceiling, for a caller that
        does not know.
        """
        self.llm = llm
        self.mcp_pool = mcp_pool
        self.max_history_chars = max_history_chars
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    async def arun(self, user_message: str) -> AsyncIterator[dict]:
        """Run the tool-calling loop, yielding events as it advances.

        Event shapes:
          {"type": "tool_call",   "call_id", "node_id", "tool", "args"}
          {"type": "tool_result", "call_id", "result"}
          {"type": "final",       "content"}

        The LLM call is offloaded to a thread so the event loop stays
        responsive — important for the dashboard's SSE path.
        """
        self.messages.append({"role": "user", "content": user_message})
        tools = self.mcp_pool.get_openai_tools()

        for _ in range(MAX_TOOL_ROUNDS):
            response = await asyncio.to_thread(
                self.llm.chat,
                messages=_fits_context(self.messages, self.max_history_chars),
                tools=tools if tools else None,
            )
            message = response["choices"][0]["message"]

            if message.get("tool_calls"):
                # Stripped on THIS branch too, and as a copy so the response
                # dict _dispatch_tool_calls reads stays untouched. A thinking
                # block before a tool call is stored, rendered and replayed
                # exactly like one before a final answer.
                stored = dict(message)
                if isinstance(stored.get("content"), str):
                    stored["content"] = strip_thinking(stored["content"])
                self.messages.append(stored)
                async for event in self._dispatch_tool_calls(message["tool_calls"]):
                    yield event
                continue

            # Strip BEFORE storing: consumers render from self.messages (the
            # dashboard persists engine.messages verbatim and discards the
            # `final` payload), so thinking left in the stored copy reaches
            # the user — and gets replayed to the model — on every later turn.
            # The invariant holds on both branches.
            content = strip_thinking(message.get("content") or "")
            self.messages.append({"role": "assistant", "content": content})
            yield {"type": "final", "content": content}
            return

        # Stored as a real assistant turn, not only yielded: the persisted
        # history otherwise ends on dangling tool results, and the user sees
        # the spinner end with no reply at all.
        capped = "(Reached maximum tool rounds without a final response)"
        self.messages.append({"role": "assistant", "content": capped})
        yield {"type": "final", "content": capped}

    async def _dispatch_tool_calls(self, tool_calls: list[dict]) -> AsyncIterator[dict]:
        """Run one round's tool calls, append the tool messages, and stream events.

        Concurrent ACROSS nodes, serial WITHIN one, so a round costs the
        slowest node rather than the sum of every call. Each call is an HTTP
        round-trip to a different edge node, and ``run_inference`` runs a whole
        generation there. Same-node calls stay in sequence because the edge runs
        ``parallel_slots: 1``: fanning out on one node only queues behind itself
        while burning the pool's 300s read timeout.
        """
        planned = []
        for tc in tool_calls:
            func = tc["function"]
            node_id, tool_name = self.mcp_pool.parse_tool_call(func["name"])
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            planned.append((tc["id"], node_id, tool_name, args))

        # Every tool_call event first, so the dashboard shows the whole round
        # starting rather than one call at a time.
        for call_id, node_id, tool_name, args in planned:
            yield {
                "type": "tool_call",
                "call_id": call_id,
                "node_id": node_id,
                "tool": tool_name,
                "args": args,
            }
            logger.info("Calling %s on %s with %s", tool_name, node_id, args)

        by_node: dict[str, list[int]] = {}
        for i, (_cid, node_id, _tool, _args) in enumerate(planned):
            by_node.setdefault(node_id, []).append(i)

        results: dict[int, dict] = {}

        async def run_node(indexes: list[int]) -> None:
            for i in indexes:
                _cid, node_id, tool_name, args = planned[i]
                results[i] = await self.mcp_pool.call_tool(node_id, tool_name, args)

        # return_exceptions like _run_planner_host's connect_pass: call_tool
        # catches Exception, but anyio can raise BaseExceptionGroup, which
        # would otherwise cancel the sibling calls.
        outcomes = await asyncio.gather(
            *(run_node(idx) for idx in by_node.values()), return_exceptions=True
        )
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                logger.warning("Tool dispatch task failed: %s", outcome)

        # Appended and yielded in the ORIGINAL order: chat_service._drain_engine
        # persists engine.messages by index, so each tool message has to stay
        # aligned with the assistant message's tool_calls.
        for i, (call_id, _node_id, _tool, _args) in enumerate(planned):
            result = results.get(i, {"error": "tool call did not complete"})
            self.messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                # Compact, not indent=2: this text is replayed to the model on
                # every later round and turn, and the dashboard pretty-prints it
                # at render time anyway (static/sections/chat.js::pretty).
                "content": json.dumps(result),
            })
            yield {"type": "tool_result", "call_id": call_id, "result": result}
