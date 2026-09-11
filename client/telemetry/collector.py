"""Metrics collector — samples system and inference metrics."""

from __future__ import annotations

import logging
import math
import os
import ssl
from datetime import UTC, datetime

import httpx

# ---------------------------------------------------------------------------
# Cross-sample state.
#
# Both of these exist because the interesting numbers are RATES, and every
# source below exposes a monotonic counter instead. Holding the previous
# reading is what turns "total jiffies since boot" and "tokens predicted since
# the server started" into "what happened during the last sampling interval".
#
# Without it, 100*(1 - idle/total) read straight from /proc/stat is average
# CPU SINCE BOOT: on a long-lived node that converges to a constant and stops
# responding to load altogether, reporting single-digit utilisation straight
# through a saturating inference run and making the metric useless for
# diagnosing throughput.
# ---------------------------------------------------------------------------
_prev_cpu_jiffies: tuple[int, int] | None = None
_prev_llama_counters: dict[str, float] | None = None

# One-shot latch for the /metrics 401 error below — the collector runs on a
# ~60s loop and the message is shipped to the orchestrator, so emitting it per
# sample would flood the log store with copies of the same fact.
_metrics_auth_warned = False


def _metric(node_id: str, name: str, value: float, timestamp: str) -> dict:
    return {
        "node_id": node_id,
        "metric_name": name,
        "metric_value": value,
        "timestamp": timestamp,
    }


def collect_system_metrics(node_id: str) -> list[dict]:
    """Collect system metrics. Works on Linux with /proc."""
    timestamp = datetime.now(UTC).isoformat()
    metrics = []

    # CPU usage over the SAMPLING INTERVAL, from consecutive /proc/stat reads.
    #
    # The first call after start seeds the baseline and emits nothing — one
    # missing sample is better than one misleading sample, and every call after
    # it reports real interval utilisation.
    global _prev_cpu_jiffies
    try:
        with open("/proc/stat") as f:
            cpu_line = f.readline().split()
        idle = int(cpu_line[4])
        total = sum(int(x) for x in cpu_line[1:])

        prev = _prev_cpu_jiffies
        _prev_cpu_jiffies = (idle, total)
        if prev is not None:
            d_idle = idle - prev[0]
            d_total = total - prev[1]
            # d_total <= 0 means the counters wrapped or the file lied; skip
            # rather than publishing a negative or a divide-by-zero.
            if d_total > 0:
                cpu_pct = round(100 * (1 - d_idle / d_total), 1)
                metrics.append(_metric(node_id, "cpu_percent", cpu_pct, timestamp))
    except (OSError, IndexError, ValueError):
        pass

    # Load average, 1 minute only.
    #
    # It answers what cpu_percent cannot: a node pinned at 100% looks identical
    # whether it is keeping up or running a queue eight deep, and the run queue
    # is the difference. The 5m and 15m fields are the same signal smoothed, so
    # they would cost two more rows a minute on every node for nothing a 30-day
    # history does not already show.
    try:
        with open("/proc/loadavg") as f:
            load_1m = float(f.read().split()[0])
        metrics.append(_metric(node_id, "load_1m", load_1m, timestamp))
    except (OSError, IndexError, ValueError):
        pass

    # Host uptime, deliberately: procfs uptime is not namespaced, so this read
    # inside the container returns the machine's. K8s pod age already covers
    # the container, and only the host figure tells a crash-looping pod apart
    # from a node that rebooted underneath it.
    try:
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        metrics.append(_metric(node_id, "uptime_seconds", round(uptime), timestamp))
    except (OSError, IndexError, ValueError):
        pass

    # Memory from /proc/meminfo
    #
    # Both fields are read without a fallback value: defaulting MemAvailable to
    # 0 publishes a hard 100.0 every sample forever, and defaulting MemTotal to
    # 1 publishes ~-8e8. A missing metric is recoverable; a fabricated one is
    # indistinguishable from a real reading. MemAvailable predates every kernel
    # this runs on, so absence means the file was not /proc/meminfo.
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                meminfo[parts[0].rstrip(":")] = int(parts[1])
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable")
        if total > 0 and available is not None:
            mem_pct = round(100 * (1 - available / total), 1)
            metrics.append(_metric(node_id, "memory_percent", mem_pct, timestamp))

        # Swap, off the dict already in hand — a second read of the same file
        # would buy nothing. Reported only where swap exists: these nodes run
        # without any, and SwapTotal 0 there would publish "0% swap used" every
        # minute, which reads as measured headroom rather than as no swap at
        # all. Once swap does exist, a model that no longer fits is the thing
        # this metric catches, since paging shows up as collapsed tok/s long
        # before anything fails.
        swap_total = meminfo.get("SwapTotal", 0)
        swap_free = meminfo.get("SwapFree")
        if swap_total > 0 and swap_free is not None:
            swap_pct = round(100 * (1 - swap_free / swap_total), 1)
            metrics.append(_metric(node_id, "swap_used_percent", swap_pct, timestamp))
    except (OSError, IndexError, ValueError):
        pass

    # Disk usage
    #
    # Nothing is published when the filesystem reports zero blocks, by the same
    # rule as the memory block above: a pseudo-filesystem where the hostPath
    # should be would otherwise publish disk_usage_percent = 0.0, which the
    # dashboard renders as a healthy empty disk.
    try:
        st = os.statvfs("/opt/aureax")
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        if total > 0:
            disk_pct = round(100 * (1 - free / total), 1)
            metrics.append(_metric(node_id, "disk_usage_percent", disk_pct, timestamp))
    except OSError:
        pass

    # CPU temperature (Linux sysfs)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp_mc = int(f.read().strip())
        metrics.append(_metric(node_id, "temperature_c", round(temp_mc / 1000, 1), timestamp))
    except (OSError, ValueError):
        pass

    # Clocks. Both travel alongside throughput because decode on CPU-class
    # nodes is bounded by memory bandwidth: where the DRAM controller is its
    # own DVFS domain its clock swings several-fold during a run, over the
    # exact resource that sets the token rate. A tok/s number is not
    # interpretable without knowing what clock produced it.
    #
    # cpu4 first, cpu0 second: on a big.LITTLE SoC the big cluster is the one
    # llama.cpp is pinned to and the little cores idle at a different
    # frequency, so sampling cpu0 there reports a clock no inference ran at.
    # Falling back to cpu0 covers uniform CPUs, and a machine with no cpufreq
    # at all emits nothing.
    for cpu in ("cpu4", "cpu0"):
        try:
            with open(f"/sys/devices/system/cpu/{cpu}/cpufreq/scaling_cur_freq") as f:
                khz = int(f.read().strip())
            metrics.append(_metric(node_id, "cpu_freq_mhz", round(khz / 1000), timestamp))
            break
        except (OSError, ValueError):
            continue

    # The DRAM controller's devfreq node, present on some SoCs and absent on
    # others (and on every x86 machine); nothing is emitted where it is
    # missing.
    try:
        with open("/sys/class/devfreq/dmc/cur_freq") as f:
            hz = int(f.read().strip())
        metrics.append(_metric(node_id, "dram_freq_mhz", round(hz / 1_000_000), timestamp))
    except (OSError, ValueError):
        pass

    return metrics


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse llama-server's Prometheus exposition into {name: value}.

    Deliberately minimal: llama-server emits no labels on these series, so a
    two-field split is sufficient and avoids taking a parser dependency.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        # +Inf is a legal exposition value for an undefined gauge, and it
        # survives every hop after this one: stored REAL, shipped through
        # json.dumps (allow_nan defaults True), then rendered by Starlette's
        # JSONResponse with allow_nan=False — where it raises and 500s
        # /api/nodes for the WHOLE fleet, from one sample on one node.
        if not math.isfinite(value):
            continue
        out[parts[0]] = value
    return out


def collect_inference_metrics(
    node_id: str,
    llama_url: str = "https://127.0.0.1:8080",
    api_key: str | None = None,
    verify: ssl.SSLContext | None = None,
) -> list[dict]:
    """Collect throughput metrics from llama-server's /metrics endpoint.

    Requires ``--metrics`` on llama-server (``model.server.metrics`` in the
    edge config, on by default). Without it the endpoint 404s and this returns
    only liveness, which is the same graceful-degradation path as an unreachable
    server.

    /metrics rather than /health: llama-server's /health response carries no
    throughput fields at all, so polling it records nothing usable and leaves
    ``tokens_per_second`` — declared in every config's
    ``telemetry.local.collect.inference`` list — produced by nothing.

    Rates are computed as counter deltas between consecutive samples, which
    yields the true decode rate independent of the sampling interval: dividing
    tokens by wall-clock would instead average generation together with the idle
    gaps between requests and report a number several times too low.

    ``verify`` is the fleet CA context (``client.tls.ssl_context``). The dial is
    loopback but still https — llama-server runs with ``--ssl-cert-file`` and
    speaks nothing else — and the node's certificate carries ``IP:127.0.0.1`` as
    a SAN so this verifies. A certificate that does not verify raises
    ``httpx.ConnectError``, which the handler below treats exactly like an
    unreachable server: no metrics, no crash, and the supervisor's TLS
    preflight (``client/tls.py::require_tls_files``, exit 78) is what makes a
    node with missing material loud instead.
    """
    global _prev_llama_counters, _metrics_auth_warned
    timestamp = datetime.now(UTC).isoformat()
    metrics: list[dict] = []

    # /metrics is NOT on llama-server's auth-exempt list, unlike /health — with
    # an API key set, this is the one collector call that must authenticate.
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        resp = httpx.get(f"{llama_url}/metrics", timeout=5, headers=headers, verify=verify)
    except httpx.HTTPError:
        return metrics
    if resp.status_code in (401, 403):
        # Without this, a key mismatch (rotated-but-not-restarted, stale env)
        # produces a fleet that looks perfectly healthy with a permanently flat
        # throughput graph — the exact silent-failure class this function's
        # own docstring records. Logged ONCE, not per 60s sample, and under
        # the "aureax." logger hierarchy so the shipping handler attached to
        # the "aureax" root forwards it to the orchestrator.
        if not _metrics_auth_warned:
            _metrics_auth_warned = True
            logging.getLogger("aureax.telemetry").error(
                "llama-server /metrics returned %d — this node's "
                "model.server.api_key does not match the key llama-server was "
                "started with. Throughput telemetry is dead until a config "
                "push + pod restart (aureax apply --node <id>) fixes it.",
                resp.status_code,
            )
        return metrics
    if resp.status_code != 200:
        return metrics

    cur = _parse_prometheus(resp.text)

    # Levels: meaningful on their own, no delta needed.
    #
    # kv_cache_tokens is the ratio's numerator, and only it says how far a
    # session is from the configured context — 40% of 131072 and 40% of 4096
    # are the same ratio and different situations.
    #
    # tokens_predicted_total is a counter published as a level ON PURPOSE: the
    # rate is derived from it below, and the running total is what separates a
    # server that generated nothing this interval from one that has generated
    # nothing since it started.
    for series, name in (
        ("llamacpp:requests_processing", "requests_processing"),
        ("llamacpp:requests_deferred", "requests_deferred"),
        ("llamacpp:kv_cache_usage_ratio", "kv_cache_usage_ratio"),
        ("llamacpp:kv_cache_tokens", "kv_cache_tokens"),
        ("llamacpp:tokens_predicted_total", "tokens_predicted_total"),
    ):
        if series in cur:
            metrics.append(_metric(node_id, name, cur[series], timestamp))

    prev = _prev_llama_counters
    _prev_llama_counters = cur

    if prev is not None:
        for tok_series, sec_series, name in (
            (
                "llamacpp:tokens_predicted_total",
                "llamacpp:tokens_predicted_seconds_total",
                "tokens_per_second",
            ),
            (
                "llamacpp:prompt_tokens_total",
                "llamacpp:prompt_seconds_total",
                "prompt_tokens_per_second",
            ),
        ):
            # The pair must be in BOTH readings. Treating a series missing
            # from `prev` as 0.0 makes the delta equal the lifetime totals, so
            # a single truncated scrape publishes the server's since-start
            # average as current throughput — the exact thing the docstring
            # above says this function must never emit.
            if not all(s in cur and s in prev for s in (tok_series, sec_series)):
                continue
            d_tok = cur[tok_series] - prev[tok_series]
            d_sec = cur[sec_series] - prev[sec_series]
            # A negative delta means llama-server restarted and its counters
            # went back to zero; skip this interval rather than emitting a
            # nonsense rate. The next sample recovers on its own.
            if d_tok > 0 and d_sec > 0:
                metrics.append(_metric(node_id, name, round(d_tok / d_sec, 3), timestamp))

    return metrics


def collect_all(
    node_id: str,
    llama_url: str | None = "https://127.0.0.1:8080",
    api_key: str | None = None,
    verify: ssl.SSLContext | None = None,
) -> list[dict]:
    """Collect all metrics.

    The metric set is fixed here in code — the ``telemetry.local.collect``
    lists in the configs are RESERVED and select nothing (see LoggingConfig's
    sibling note in orchestrator/config/schema.py). Editing a config's list
    changes nothing; this module is where a new metric is added.

    ``llama_url=None`` skips the inference sample entirely — a shard worker
    runs rpc-server and no HTTP server, so there is nothing to poll. System
    metrics still flow, which is what keeps every member of a group visible
    in the dashboard (``/api/nodes`` is derived from the telemetry DB).
    """
    metrics = collect_system_metrics(node_id)
    if llama_url:
        metrics.extend(
            collect_inference_metrics(node_id, llama_url, api_key=api_key, verify=verify)
        )
    return metrics
