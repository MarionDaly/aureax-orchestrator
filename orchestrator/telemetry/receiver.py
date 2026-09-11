"""HTTP endpoint for receiving JSONL telemetry from edge nodes.

Also owns :func:`parse_records`, the line loop ``orchestrator/logs/receiver.py``
parses its shipped batches with: the two endpoints differ only in which fields
they require and how they validate them.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable

from orchestrator.telemetry.storage import MetricsStorage

logger = logging.getLogger("aureax.telemetry")

_METRIC_FIELDS = ("node_id", "metric_name", "metric_value", "timestamp")

# Bounds the strings that reach indexed columns. /api/v1/telemetry is
# unauthenticated on 0.0.0.0, so anything on the LAN can post a row.
_MAX_FIELD = 200


def parse_records(
    body: str,
    required: tuple[str, ...],
    *,
    what: str,
    log: logging.Logger,
    validate: Callable[[dict], str | None] | None = None,
) -> list[dict]:
    """Parse a JSONL (newline-delimited JSON) body into records.

    Every rejection is a skip-and-warn, never a raise: both shippers advance
    only past a batch this endpoint answered 200 to, so one poison line failing
    the request would make the same batch arrive again every push interval,
    forever.

    ``validate`` returns None for an acceptable record, or a short reason to
    log. It is per-endpoint — a metric value has constraints a log line does
    not.
    """
    records = []
    for line in body.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Skipping malformed %s line: %s", what, line[:100])
            continue
        # A line that decodes to a bare scalar ("5", "null") is valid JSON but
        # not a record, and `"node_id" in 5` raises TypeError — which would
        # escape this function and 500 the whole shipped batch.
        if not isinstance(record, dict):
            log.warning("Skipping non-object %s line: %s", what, line[:100])
            continue
        if not all(k in record for k in required):
            log.warning("Skipping %s record with missing fields: %s", what, line[:100])
            continue
        problem = validate(record) if validate is not None else None
        if problem:
            log.warning("Skipping %s record (%s): %s", what, problem, line[:100])
            continue
        records.append(record)
    return records


def _metric_problem(record: dict) -> str | None:
    """Reject a metric row that storage or the dashboard cannot carry.

    Presence alone is not enough. A null value trips ``NOT NULL constraint
    failed`` and a dict/list raises sqlite3.ProgrammingError, both from inside
    executemany — escaping ingest as a 500, which the reporter reads as
    "retry", so the node re-sends the same poison batch every 60s forever. A
    non-finite float survives the insert instead and 500s /api/nodes for the
    whole fleet, where Starlette renders with allow_nan=False.
    """
    value = record["metric_value"]
    # bool is an int subclass, and True would land in the REAL column as 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "metric_value is not a number"
    if not math.isfinite(value):
        return "metric_value is not finite"
    for field in ("node_id", "metric_name", "timestamp"):
        text = record[field]
        if not isinstance(text, str) or not text or len(text) > _MAX_FIELD:
            return f"{field} is not a bounded non-empty string"
    return None


def parse_jsonl(body: str) -> list[dict]:
    """Parse a JSONL body into metric records, dropping anything unusable."""
    return parse_records(
        body, _METRIC_FIELDS, what="metric", log=logger, validate=_metric_problem
    )


def ingest(body: str, storage: MetricsStorage) -> tuple[int, list[dict]]:
    """Parse a JSONL body and insert it. Returns (count ingested, records).

    The records come back because the caller needs them too: the dashboard
    logs a per-node rollup of every push, and building it from the body meant
    a second full json.loads pass over lines this function had just parsed.
    """
    records = parse_jsonl(body)
    return (storage.insert(records) if records else 0), records
