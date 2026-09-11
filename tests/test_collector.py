"""Telemetry collection behaviours that decide whether a number is trustworthy.

The collector's whole job here is turning monotonic counters into rates. Every
case below is one where getting that wrong publishes a plausible-looking number
that is silently meaningless — which is exactly what the previous version did
for the entire life of the fleet.
"""

from __future__ import annotations

import errno
import io
import logging
import os
import ssl

import httpx
import pytest

from client.telemetry import collector


@pytest.fixture(autouse=True)
def _reset_collector_state():
    """Module-level deltas and latches leak between tests otherwise.

    ``_metrics_auth_warned`` is a one-shot: the first test to trip the 401
    branch would otherwise silence it for whichever test runs next.
    """
    collector._prev_cpu_jiffies = None
    collector._prev_llama_counters = None
    collector._metrics_auth_warned = False
    yield
    collector._prev_cpu_jiffies = None
    collector._prev_llama_counters = None
    collector._metrics_auth_warned = False


class _Resp:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


def _serve(monkeypatch, *payloads) -> list[dict]:
    """Feed successive /metrics bodies to consecutive collector calls.

    Returns the list of ``{"url", "headers", "verify"}`` records the fake
    appends per call. The scrape carries both a credential and a TLS context,
    so the fake records the whole call rather than one argument of it.
    """
    seq = list(payloads)
    calls: list[dict] = []

    def fake_get(url, timeout=5, headers=None, verify=None):
        calls.append({"url": url, "headers": headers, "verify": verify})
        return seq.pop(0)

    monkeypatch.setattr(collector.httpx, "get", fake_get)
    return calls


def _by_name(metrics) -> dict:
    return {m["metric_name"]: m["metric_value"] for m in metrics}


# --- Prometheus parsing ------------------------------------------------------

def test_parse_skips_comments_blanks_and_malformed_lines() -> None:
    text = (
        "# HELP llamacpp:tokens_predicted_total Number of tokens.\n"
        "# TYPE llamacpp:tokens_predicted_total counter\n"
        "llamacpp:tokens_predicted_total 42\n"
        "\n"
        "llamacpp:broken\n"
        "llamacpp:not_a_number abc\n"
        "llamacpp:kv_cache_usage_ratio 0.25\n"
    )
    parsed = collector._parse_prometheus(text)
    assert parsed == {
        "llamacpp:tokens_predicted_total": 42.0,
        "llamacpp:kv_cache_usage_ratio": 0.25,
    }


def test_non_finite_values_never_enter_the_metric_set() -> None:
    """+Inf is a legal exposition value for an undefined gauge.

    It survives storage and json.dumps, then reaches Starlette's JSONResponse,
    which renders with allow_nan=False and raises — so one bad sample from one
    node 500s /api/nodes for the entire fleet, and the result is TTL-cached.
    """
    parsed = collector._parse_prometheus(
        "llamacpp:kv_cache_usage_ratio +Inf\n"
        "llamacpp:requests_deferred nan\n"
        "llamacpp:requests_processing 2\n"
    )
    assert parsed == {"llamacpp:requests_processing": 2.0}


# --- rate derivation ---------------------------------------------------------

_T0 = (
    "llamacpp:tokens_predicted_total 100\n"
    "llamacpp:tokens_predicted_seconds_total 100\n"
    "llamacpp:prompt_tokens_total 50\n"
    "llamacpp:prompt_seconds_total 5\n"
)
_T1 = (
    "llamacpp:tokens_predicted_total 200\n"
    "llamacpp:tokens_predicted_seconds_total 150\n"
    "llamacpp:prompt_tokens_total 90\n"
    "llamacpp:prompt_seconds_total 9\n"
)


def test_first_sample_publishes_no_rate(monkeypatch) -> None:
    """With one reading there is no interval, so there is no rate to report.

    Emitting total/total here would publish the server's lifetime average and
    call it current throughput.
    """
    _serve(monkeypatch, _Resp(_T0))
    out = _by_name(collector.collect_inference_metrics("n1"))
    assert "tokens_per_second" not in out
    assert "prompt_tokens_per_second" not in out


def test_rate_is_the_delta_not_the_lifetime_average(monkeypatch) -> None:
    """100 tokens over 50 seconds of decode is 2 tok/s, not 200/150 = 1.33."""
    _serve(monkeypatch, _Resp(_T0), _Resp(_T1))
    collector.collect_inference_metrics("n1")
    out = _by_name(collector.collect_inference_metrics("n1"))
    assert out["tokens_per_second"] == 2.0
    assert out["prompt_tokens_per_second"] == 10.0


def test_counter_reset_is_skipped_rather_than_reported_negative(monkeypatch) -> None:
    """llama-server restarting zeroes its counters.

    Subtracting across that boundary yields a negative delta; publishing it
    would put a nonsense point in the middle of a benchmark series.
    """
    restarted = (
        "llamacpp:tokens_predicted_total 5\n"
        "llamacpp:tokens_predicted_seconds_total 2\n"
    )
    _serve(monkeypatch, _Resp(_T1), _Resp(restarted))
    collector.collect_inference_metrics("n1")
    out = _by_name(collector.collect_inference_metrics("n1"))
    assert "tokens_per_second" not in out


def test_idle_interval_emits_no_rate(monkeypatch) -> None:
    """No tokens generated between samples means no throughput to report.

    Reporting 0 tok/s would drag the average down and misrepresent a server
    that was simply not asked to do anything.
    """
    _serve(monkeypatch, _Resp(_T0), _Resp(_T0))
    collector.collect_inference_metrics("n1")
    out = _by_name(collector.collect_inference_metrics("n1"))
    assert "tokens_per_second" not in out


def test_a_series_missing_from_the_previous_scrape_publishes_no_rate(monkeypatch) -> None:
    """A counter absent from `prev` is not a zero.

    Subtracting 0.0 makes the delta equal the LIFETIME totals, so one
    truncated body — or a build that renames a series — publishes the server's
    since-start average as current throughput. That is the same defect
    test_first_sample_publishes_no_rate forbids, one series at a time.
    """
    _serve(monkeypatch, _Resp("llamacpp:requests_processing 1\n"), _Resp(_T1))
    collector.collect_inference_metrics("n1")
    out = _by_name(collector.collect_inference_metrics("n1"))
    assert "tokens_per_second" not in out
    assert "prompt_tokens_per_second" not in out


def test_gauges_are_reported_on_the_very_first_sample(monkeypatch) -> None:
    """Unlike counters, a gauge is meaningful immediately."""
    _serve(monkeypatch, _Resp(_T0 + "llamacpp:requests_processing 3\n"))
    out = _by_name(collector.collect_inference_metrics("n1"))
    assert out["requests_processing"] == 3.0


def test_the_kv_and_token_totals_are_published_as_levels(monkeypatch) -> None:
    """tokens_predicted_total is a counter shipped as a level on purpose.

    The rate derived from the same series is reported separately, so the total
    is what remains to distinguish a server that generated nothing during this
    interval from one that has generated nothing since it started. And
    kv_cache_tokens is the ratio's numerator: 40% of 131072 and 40% of 4096 are
    the same ratio and different situations.
    """
    kv = "llamacpp:kv_cache_tokens 4096\n"
    _serve(monkeypatch, _Resp(_T0 + kv), _Resp(_T1 + kv))

    first = _by_name(collector.collect_inference_metrics("n1"))
    assert first["kv_cache_tokens"] == 4096.0
    assert first["tokens_predicted_total"] == 100.0

    second = _by_name(collector.collect_inference_metrics("n1"))
    assert second["tokens_predicted_total"] == 200.0, "published the delta, not the total"
    assert second["tokens_per_second"] == 2.0


# --- degradation -------------------------------------------------------------

def test_metrics_endpoint_disabled_degrades_quietly(monkeypatch) -> None:
    """--metrics is off by default in llama.cpp, so a 404 is a real scenario.

    It must not raise: the same collect_all() call also gathers system metrics,
    and losing those too would blank the node out of the dashboard entirely.
    """
    _serve(monkeypatch, _Resp("Not Found", status_code=404))
    assert collector.collect_inference_metrics("n1") == []


def test_unreachable_server_degrades_quietly(monkeypatch) -> None:
    def boom(url, timeout=5, headers=None, verify=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(collector.httpx, "get", boom)
    assert collector.collect_inference_metrics("n1") == []


def test_shard_workers_skip_inference_collection_entirely(monkeypatch) -> None:
    """A worker runs rpc-server and serves no HTTP, so there is nothing to poll."""

    def boom(url, timeout=5, headers=None, verify=None):
        raise AssertionError("must not poll on a shard worker")

    monkeypatch.setattr(collector.httpx, "get", boom)
    collector.collect_all("worker-1", llama_url=None)


# --- authentication ----------------------------------------------------------

def test_the_scrape_carries_the_node_key_as_a_bearer_header(monkeypatch) -> None:
    """/metrics is NOT on llama-server's auth-exempt list, unlike /health.

    Drop the header and every keyed node's throughput telemetry dies for good
    while the node goes on reporting healthy system metrics.
    """
    sent = _serve(monkeypatch, _Resp(_T0), _Resp(_T0))

    collector.collect_inference_metrics("n1", api_key="aureax_sk_x")
    assert sent[0]["headers"] == {"Authorization": "Bearer aureax_sk_x"}

    collector.collect_inference_metrics("n1")
    assert sent[1]["headers"] is None, "sent an Authorization header for an unkeyed node"


def test_collect_all_forwards_the_key_to_the_inference_scrape(monkeypatch) -> None:
    """The key reaches httpx only if collect_all passes it through."""
    sent = _serve(monkeypatch, _Resp(_T0))
    collector.collect_all("n1", llama_url="https://127.0.0.1:8080", api_key="aureax_sk_y")
    assert [c["headers"] for c in sent] == [{"Authorization": "Bearer aureax_sk_y"}]


def test_a_key_mismatch_is_reported_once_not_once_per_sample(monkeypatch, caplog) -> None:
    """A rotated-but-not-restarted key looks exactly like a healthy idle node.

    The 401 branch is the only thing that names it, and it ships to the
    orchestrator — so it must fire, and must not repeat on a ~60s loop.
    """
    caplog.set_level(logging.ERROR, logger="aureax.telemetry")
    _serve(monkeypatch, _Resp("Unauthorized", 401), _Resp("Unauthorized", 401))

    assert collector.collect_inference_metrics("n1", api_key="stale") == []
    assert collector.collect_inference_metrics("n1", api_key="stale") == []

    records = [r for r in caplog.records if r.name == "aureax.telemetry"]
    assert len(records) == 1, "warned per sample instead of once"
    message = records[0].getMessage()
    assert "aureax apply" in message
    assert "stale" not in message, "leaked the key into a shipped log line"


def test_a_403_takes_the_same_one_shot_path(monkeypatch, caplog) -> None:
    caplog.set_level(logging.ERROR, logger="aureax.telemetry")
    _serve(monkeypatch, _Resp("Forbidden", 403))
    assert collector.collect_inference_metrics("n1", api_key="stale") == []
    assert len([r for r in caplog.records if r.name == "aureax.telemetry"]) == 1


# --- transport -----------------------------------------------------------------

def test_the_scrape_is_https_by_default() -> None:
    """llama-server runs with --ssl-cert-file and speaks nothing but TLS.

    An http:// default would connect to a port that answers only TLS, so every
    unconfigured caller — and this default is what client/mcp/server.py and the
    supervisor fall back to — would lose throughput telemetry outright.
    """
    import inspect

    for fn in (collector.collect_inference_metrics, collector.collect_all):
        default = inspect.signature(fn).parameters["llama_url"].default
        assert default.startswith("https://"), f"{fn.__name__} defaults to plaintext"


def test_the_scrape_carries_the_fleet_ca_context(monkeypatch) -> None:
    """The node's certificate is signed by the fleet CA, not a public root.

    Without the context the loopback scrape fails verification and lands in the
    quiet ConnectError branch, which is indistinguishable from a llama-server
    that has not finished loading — a flat throughput graph and no error.
    """
    ctx = ssl.create_default_context()
    sent = _serve(monkeypatch, _Resp(_T0))

    collector.collect_inference_metrics("n1", verify=ctx)

    assert sent[0]["verify"] is ctx


def test_collect_all_forwards_the_context_to_the_inference_scrape(monkeypatch) -> None:
    """The supervisor holds one context and hands it to collect_all only.

    collect_all is the sole entry point the telemetry loop calls
    (client/entrypoint.py), so a context it drops never reaches httpx at all.
    """
    ctx = ssl.create_default_context()
    sent = _serve(monkeypatch, _Resp(_T0))

    collector.collect_all("n1", llama_url="https://127.0.0.1:8080", verify=ctx)

    assert [c["verify"] for c in sent] == [ctx]


def test_an_untrusted_certificate_costs_the_scrape_not_the_sample(monkeypatch) -> None:
    """A TLS failure is an httpx.ConnectError, i.e. an httpx.HTTPError.

    It therefore takes the same graceful path as an unreachable server: system
    metrics still flow, so the node stays visible in /api/nodes instead of
    disappearing from the dashboard the moment a certificate goes stale.
    """

    def untrusted(url, timeout=5, headers=None, verify=None):
        raise httpx.ConnectError("certificate verify failed: unable to get local issuer")

    monkeypatch.setattr(collector.httpx, "get", untrusted)

    assert collector.collect_inference_metrics("n1") == []


# --- CPU interval utilisation ------------------------------------------------

def _fake_files(monkeypatch, files: dict) -> None:
    """Serve the named /proc and /sys paths; every other path stays unreadable.

    A value may be an Exception to raise instead of contents. Every read in
    collect_system_metrics is individually guarded, so letting the unlisted
    paths fail is the same code path a non-Linux or partially-featured node
    takes.
    """

    def fake_open(path, *a, **kw):
        value = files.get(str(path))
        if value is None:
            raise FileNotFoundError(path)
        if isinstance(value, Exception):
            raise value
        return io.StringIO(value)

    monkeypatch.setattr("builtins.open", fake_open)


def _fake_proc_stat(monkeypatch, idle: int, total: int) -> None:
    """Serve one /proc/stat line and nothing else."""
    busy = total - idle
    # /proc/stat column order: user nice system idle iowait irq softirq ...
    _fake_files(monkeypatch, {"/proc/stat": f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\n"})


def test_cpu_percent_reports_the_interval_not_since_boot(monkeypatch) -> None:
    """A node 50% busy for the last interval must read 50%, not its lifetime mean.

    The regression this guards: a long-running node whose since-boot average has
    converged reports the same flat number under full load as at idle. The fleet
    showed 4-9% throughout a saturating run because of exactly that.
    """
    # Seed: heavily idle history (10% busy since boot).
    _fake_proc_stat(monkeypatch, idle=900, total=1000)
    assert _by_name(collector.collect_system_metrics("n1")).get("cpu_percent") is None

    # Next interval: 100 jiffies elapsed, 50 of them busy.
    _fake_proc_stat(monkeypatch, idle=950, total=1100)
    out = _by_name(collector.collect_system_metrics("n1"))
    assert out["cpu_percent"] == 50.0


def test_cpu_percent_ignores_a_non_advancing_clock(monkeypatch) -> None:
    """Two identical reads mean no elapsed jiffies — a divide-by-zero, not 0%."""
    _fake_proc_stat(monkeypatch, idle=900, total=1000)
    collector.collect_system_metrics("n1")
    _fake_proc_stat(monkeypatch, idle=900, total=1000)
    out = _by_name(collector.collect_system_metrics("n1"))
    assert "cpu_percent" not in out


# --- system reads degrade one metric at a time -------------------------------

_MEMINFO = "MemTotal: 16000000 kB\nMemFree: 1000000 kB\nMemAvailable: 4000000 kB\n"


def test_memory_percent_is_computed_from_the_file(monkeypatch) -> None:
    _fake_files(monkeypatch, {"/proc/meminfo": _MEMINFO})
    assert _by_name(collector.collect_system_metrics("n1"))["memory_percent"] == 75.0


def test_a_meminfo_without_memavailable_publishes_nothing(monkeypatch) -> None:
    """Defaulting MemAvailable to 0 published a hard 100.0 on every sample forever.

    The dashboard renders that as "mem 100%", which is indistinguishable from
    a node genuinely out of memory. A missing metric is recoverable; a
    fabricated one is not.
    """
    _fake_files(monkeypatch, {"/proc/meminfo": "MemTotal: 16000000 kB\nMemFree: 1000000 kB\n"})
    assert "memory_percent" not in _by_name(collector.collect_system_metrics("n1"))


def test_a_malformed_meminfo_line_costs_one_metric_not_the_sample(monkeypatch) -> None:
    """int(parts[1]) raises ValueError, which the old guard did not catch.

    It escaped collect_system_metrics before the disk, temperature and clock
    blocks ran, so one bad line lost every metric in the sample.
    """
    _fake_files(monkeypatch, {
        "/proc/meminfo": "MemTotal: not-a-number kB\n",
        "/sys/class/thermal/thermal_zone0/temp": "48000\n",
    })
    out = _by_name(collector.collect_system_metrics("n1"))
    assert "memory_percent" not in out
    assert out["temperature_c"] == 48.0


def test_a_sysfs_read_failing_with_eio_costs_one_metric_not_the_sample(monkeypatch) -> None:
    """sysfs fails with more than FileNotFoundError.

    A cpufreq policy whose CPU is offline returns ENODEV and some thermal
    drivers return EIO. Those used to escape the sample entirely: the
    supervisor logged "Telemetry sample failed" and discarded the cpu, memory
    and disk metrics already collected, so a node doing it every interval
    wrote no rows and vanished from /api/nodes.
    """
    _fake_files(monkeypatch, {
        "/proc/meminfo": _MEMINFO,
        "/sys/class/thermal/thermal_zone0/temp": OSError(errno.EIO, "I/O error"),
        "/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq":
            OSError(errno.ENODEV, "No such device"),
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq": "1800000\n",
    })
    out = _by_name(collector.collect_system_metrics("n1"))
    assert "temperature_c" not in out
    assert out["memory_percent"] == 75.0
    # And the cpu4 -> cpu0 fallback still runs on an OSError that is not ENOENT.
    assert out["cpu_freq_mhz"] == 1800


def test_a_filesystem_reporting_zero_blocks_publishes_no_disk_metric(monkeypatch) -> None:
    """Publishing 0 for an unmeasurable disk is a fabricated reading.

    statvfs reports f_blocks == 0 for a pseudo-filesystem, which is what sits
    at /opt/aureax when the hostPath failed to mount — precisely the case an
    operator needs to see. The dashboard renders 0.0 as a healthy empty disk.
    """
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(
        collector.os, "statvfs", lambda _p: os.statvfs_result((0, 0, 0, 0, 0, 0, 0, 0, 0, 255))
    )
    assert "disk_usage_percent" not in _by_name(collector.collect_system_metrics("n1"))


def test_disk_usage_is_published_when_the_filesystem_is_real(monkeypatch) -> None:
    """The companion to the zero-block case: a real reading still travels."""
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
    # f_bsize, f_frsize, f_blocks, f_bfree, f_bavail, f_files, ...
    monkeypatch.setattr(
        collector.os,
        "statvfs",
        lambda _p: os.statvfs_result((4096, 4096, 1000, 400, 250, 0, 0, 0, 0, 255)),
    )
    assert _by_name(collector.collect_system_metrics("n1"))["disk_usage_percent"] == 75.0


_MEMINFO_SWAP = _MEMINFO + "SwapTotal: 4000000 kB\nSwapFree: 3000000 kB\n"


def test_load_average_reports_the_run_queue_alongside_utilisation(monkeypatch) -> None:
    """cpu_percent saturates at 100 and stops moving; the run queue does not.

    A head keeping up and a head eight requests deep both read 100% busy, and
    the load figure is what separates them. 5m and 15m are the same signal
    smoothed, so they are deliberately absent — the 30-day history already is
    the smoothing, and each would cost a row a minute on every node.
    """
    _fake_files(monkeypatch, {"/proc/loadavg": "3.24 2.10 1.80 2/512 9\n"})
    out = _by_name(collector.collect_system_metrics("n1"))
    assert out["load_1m"] == 3.24
    assert "load_5m" not in out
    assert "load_15m" not in out


@pytest.mark.parametrize("contents", ["", "\n", "n/a 2.10 1.80 2/512 9\n"])
def test_an_unreadable_loadavg_publishes_nothing(monkeypatch, contents) -> None:
    """A truncated or non-numeric first field is not a load of zero.

    0.0 renders as a perfectly idle node, which is the one reading nobody
    looks at twice.
    """
    _fake_files(monkeypatch, {"/proc/loadavg": contents})
    assert "load_1m" not in _by_name(collector.collect_system_metrics("n1"))


def test_uptime_is_the_hosts_not_the_containers(monkeypatch) -> None:
    """procfs uptime is not namespaced, so an in-container read is the host's.

    That is what is wanted: K8s pod age already covers the container, and only
    the host figure tells a crash-looping pod apart from a node that rebooted
    underneath it. The second field is idle time summed over cores, not uptime.
    """
    _fake_files(monkeypatch, {"/proc/uptime": "350735.47 234388.90\n"})
    assert _by_name(collector.collect_system_metrics("n1"))["uptime_seconds"] == 350735


def test_swap_used_percent_comes_out_of_the_memory_read(monkeypatch) -> None:
    """One /proc/meminfo read serves both metrics; a second would buy nothing."""
    _fake_files(monkeypatch, {"/proc/meminfo": _MEMINFO_SWAP})
    out = _by_name(collector.collect_system_metrics("n1"))
    assert out["swap_used_percent"] == 25.0
    assert out["memory_percent"] == 75.0


def test_a_swapless_node_publishes_no_swap_metric(monkeypatch) -> None:
    """SwapTotal 0 is not 0% used.

    These nodes run without swap, and a 0.0 row every minute reads as measured
    headroom rather than as the absence of anything to measure.
    """
    _fake_files(monkeypatch, {"/proc/meminfo": _MEMINFO + "SwapTotal: 0 kB\nSwapFree: 0 kB\n"})
    assert "swap_used_percent" not in _by_name(collector.collect_system_metrics("n1"))


def test_swap_without_a_swapfree_field_publishes_nothing(monkeypatch) -> None:
    """The MemAvailable rule again: a missing free figure is not a zero.

    Treating it as one publishes a permanent 100% swap used, which is
    indistinguishable from a node that really is paging itself to death.
    """
    _fake_files(monkeypatch, {"/proc/meminfo": _MEMINFO + "SwapTotal: 4000000 kB\n"})
    out = _by_name(collector.collect_system_metrics("n1"))
    assert "swap_used_percent" not in out
    assert out["memory_percent"] == 75.0


def test_the_new_proc_reads_fail_one_metric_at_a_time(monkeypatch) -> None:
    """/proc is no more exempt from EIO than sysfs is.

    Folding these reads into the memory block's try would put two more paths
    behind one guard, so a single failing one would take the whole /proc/meminfo
    result down with it. swap_used_percent stays inside that guard on purpose:
    it is derived from the same meminfo dict, so it has no read of its own to
    fail separately.
    """
    _fake_files(monkeypatch, {
        "/proc/loadavg": OSError(errno.EIO, "I/O error"),
        "/proc/uptime": OSError(errno.EIO, "I/O error"),
        "/proc/meminfo": _MEMINFO_SWAP,
    })
    out = _by_name(collector.collect_system_metrics("n1"))
    assert "load_1m" not in out
    assert "uptime_seconds" not in out
    assert out["memory_percent"] == 75.0
    assert out["swap_used_percent"] == 25.0
