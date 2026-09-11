"""JSONL reporter — pushes buffered metrics to the orchestrator.

Also owns :func:`post_jsonl`, the POST-with-retry loop ``client/log/reporter.py``
ships its batches through. Both edge reporters send newline-delimited JSON to
the same dashboard over the same transport, so the retry and backoff policy
lives here once instead of in two copies that drift.

Both pushes now ride TLS and verify the orchestrator against the fleet CA
(``verify``, threaded from the supervisor), so the metrics and log lines a node
ships cannot be read or altered in flight and cannot be delivered to something
impersonating the dashboard. Neither push carries a credential: ``/api/v1/telemetry``
and ``/api/v1/logs`` are unauthenticated on the orchestrator by design, so
anything on the LAN can still post under any node_id. TLS authenticates the
SERVER, not the client — an ingest token is a separate, deferred decision.
"""

from __future__ import annotations

import json
import logging
import ssl
import time

import httpx

from client.telemetry.storage import TelemetryStorage

logger = logging.getLogger("aureax.reporter")


def post_jsonl(
    endpoint: str,
    payload: str,
    attempts: int,
    *,
    backoff_base: float,
    backoff_max: float,
    log: logging.Logger,
    what: str,
    attempt_level: int = logging.WARNING,
    verify: ssl.SSLContext | None = None,
) -> bool:
    """POST one JSONL batch with exponential backoff. True only on HTTP 200.

    Any non-200 is a failure, which is what lets both callers hold their
    progress marker (``mark_pushed`` here, the cursor in the log reporter) and
    retry the same batch rather than losing it.

    ``httpx.HTTPError`` is caught as a whole family on purpose: retrying the
    transport — a reset mid-body, a bad response frame — is exactly this loop's
    job, and a narrower catch would abort a batch neither caller has marked
    shipped. Both callers' loops are separately guarded (the suppress around
    the push in client/entrypoint.py's _telemetry_loop and _log_loop), so an
    escape here costs one push rather than the thread.

    ``attempt_level`` exists because these per-attempt records have different
    fates: the log shipper's are filtered out of the local store, while the
    telemetry reporter's would be stored, shipped, and fail again.

    ``verify`` is the fleet CA context. A certificate that does not verify
    raises ``httpx.ConnectError`` — an ``httpx.HTTPError`` — so it retries and
    then fails the batch like an unreachable dashboard: the rows stay on the
    node instead of being handed to whatever answered.
    """
    for attempt in range(attempts):
        try:
            resp = httpx.post(
                endpoint,
                content=payload,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=30,
                verify=verify,
            )
            if resp.status_code == 200:
                return True
            log.log(
                attempt_level, "Push of %s failed (HTTP %d), attempt %d/%d",
                what, resp.status_code, attempt + 1, attempts,
            )
        except httpx.HTTPError as e:
            log.log(
                attempt_level, "Push of %s failed (%s), attempt %d/%d",
                what, e, attempt + 1, attempts,
            )

        # Backoff — but not after the last attempt: sleeping there delays only
        # the caller's next cycle, by up to backoff_max, for a retry that will
        # never happen.
        if attempt < attempts - 1:
            time.sleep(min(backoff_base * (2 ** attempt), backoff_max))

    log.error("Failed to push %s after %d attempts", what, attempts)
    return False


class TelemetryReporter:
    """Pushes buffered metrics to the orchestrator endpoint as JSONL."""

    def __init__(
        self,
        storage: TelemetryStorage,
        endpoint: str,
        batch_size: int = 50,
        max_attempts: int = 3,
        backoff_base: int = 30,
        backoff_max: int = 300,
        verify: ssl.SSLContext | None = None,
    ) -> None:
        """Configure reporter with storage backend and push endpoint.

        ``verify`` is appended last so the positional callers in
        ``client/entrypoint.py`` keep working; it is the fleet CA context every
        push is made under.
        """
        self.storage = storage
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.verify = verify

    def push(self, max_batches: int = 1, attempts: int | None = None) -> bool:
        """Push unpushed metrics to the orchestrator. Returns True on success.

        Sends up to ``max_batches`` batches, stopping as soon as the buffer is
        empty or a send fails. More than one matters after an outage: a single
        batch per call clears (batch_size - production rate) rows per push
        interval — 50/min against the ~11/min ``collect_all`` produces — so a
        day offline takes ~7 hours to catch up, with the dashboard showing
        stale values throughout.

        ``attempts`` overrides the configured retry count for one call. The
        telemetry loop passes 1 — the rows are already durably buffered, so the
        in-loop backoff buys no durability and only stretches the sampling
        interval during the very incident an operator would reconstruct later.
        """
        for _ in range(max(1, max_batches)):
            unpushed = self.storage.get_unpushed(self.batch_size)
            if not unpushed:
                return True

            # Build JSONL payload
            lines = []
            ids = []
            for m in unpushed:
                record = {
                    "node_id": m["node_id"],
                    "metric_name": m["metric_name"],
                    "metric_value": m["metric_value"],
                    "timestamp": m["timestamp"],
                }
                lines.append(json.dumps(record))
                ids.append(m["id"])

            # Per-attempt failures at DEBUG, so a failed push stores one row and
            # not four: _init_log_store mirrors this logger's WARNING/ERROR
            # records into the shipped log store, and while the dashboard is
            # down those rows are the bulk of what floods it on recovery.
            ok = post_jsonl(
                self.endpoint,
                "\n".join(lines),
                attempts or self.max_attempts,
                backoff_base=self.backoff_base,
                backoff_max=self.backoff_max,
                log=logger,
                what="metrics",
                attempt_level=logging.DEBUG,
                verify=self.verify,
            )
            if not ok:
                return False
            self.storage.mark_pushed(ids)
            logger.info("Pushed %d metrics to %s", len(ids), self.endpoint)
        return True
