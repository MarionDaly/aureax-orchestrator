"""Ingest JSONL log entries pushed by edge nodes.

Entries land in the SAME log the dashboard already renders, tagged
``node:<id>`` in the ``source`` column. That is deliberate: LogStore's own
docstring describes it as a shape shared by the orchestrator and the edge
client, so a worker crash shows up in the existing UI without new endpoints or
new storage.

Routed through ``SystemLogger.emit`` rather than ``LogStore.append``. Both
persist, but only ``emit`` also fans out to the SSE subscribers that feed the
dashboard's live log console: appending straight to the store would put shipped
lines on disk where nothing shows them until the next dashboard restart
re-seeds the buffer. The store is still accepted as a fallback for callers that
have no logger (and for tests).
"""

from __future__ import annotations

import logging
from datetime import datetime

from orchestrator.telemetry.receiver import parse_records

logger = logging.getLogger("aureax.logs")

_ALLOWED_LEVELS = {"debug", "info", "warn", "error", "success", "hint"}

_LOG_FIELDS = ("node_id", "level", "text")

# One shipped line can be a whole GGML backtrace frame; keep them bounded so a
# runaway child cannot bloat the orchestrator's database one row at a time.
_MAX_TEXT = 4000

# node_id and the edge's own source are bounded separately: they are composed
# INTO the text and into the indexed source column, so leaving them uncapped
# put the row size back under a poster's control. /api/v1/logs is
# unauthenticated on 0.0.0.0.
_MAX_SOURCE = 120


def parse_jsonl(body: str) -> list[dict]:
    """Parse a JSONL body into log records, dropping anything malformed."""
    return parse_records(body, _LOG_FIELDS, what="log", log=logger)


def _iso_or_none(ts) -> str | None:
    """Keep a shipped timestamp only when it is a parseable ISO string.

    The column is compared as text by retention and by the SSE ordering, so a
    number sorts wrong — and so does an arbitrary string, which a type check
    alone would let straight into a column every other writer fills with ISO
    values. Anything unparseable takes the same fallback path a number does.
    """
    if not isinstance(ts, str):
        return None
    try:
        datetime.fromisoformat(ts)
    except ValueError:
        return None
    return ts


def ingest(body: str, store, system_log=None) -> tuple[int, int]:
    """Record parsed entries on the orchestrator. Returns (stored, received).

    ``system_log`` is preferred when present: it persists to the same store and
    additionally broadcasts to live SSE subscribers.

    Both numbers come back because the caller compares them to decide whether
    the shipper may advance its cursor, and one parse answers both: deriving
    the second from a fresh ``parse_jsonl`` would JSON-parse every push twice.
    """
    records = parse_jsonl(body)
    batch: list[tuple[str, str, str, str | None]] = []
    for r in records:
        level = str(r.get("level", "info")).lower()
        if level not in _ALLOWED_LEVELS:
            level = "info"
        node_id = str(r["node_id"])[:_MAX_SOURCE]
        # The edge's own source (e.g. "llama-server") is folded into the text
        # rather than the source column, so the column stays a stable per-node
        # key the dashboard can filter on. Composed BEFORE the cut, so
        # _MAX_TEXT bounds the whole stored row: prepending afterwards would
        # let a huge "source" with a one-byte text store a huge row.
        origin = str(r.get("source", ""))[:_MAX_SOURCE].strip()
        raw = str(r["text"])
        text = (f"[{origin}] {raw}" if origin and origin != node_id else raw)[:_MAX_TEXT]
        # The shipper sends the time of the ORIGINAL event, and it is kept:
        # dropping it would stamp a whole batch — including a crash's run-up,
        # shipped after the fact — with one ingest time, which is not when the
        # node said any of it.
        batch.append((level, text, f"node:{node_id}", _iso_or_none(r.get("ts"))))

    # One transaction for the whole shipped batch — per-record writes cost an
    # fsync each, and a shipped backlog can be hundreds of rows.
    #
    # `stored` reflects what was actually written, not what was received: the
    # edge shipper advances its cursor when this endpoint answers 200, so
    # reporting a failed write as success drops those entries permanently.
    try:
        stored = (
            system_log.emit_many(batch)
            if system_log is not None
            else len(store.append_many(batch))
        )
    except Exception as e:
        logger.warning("Failed to store %d log entr(ies): %s", len(batch), e)
        stored = 0
    if stored < len(batch):
        logger.warning("Stored only %d of %d shipped log entr(ies)", stored, len(batch))
    return stored, len(batch)
