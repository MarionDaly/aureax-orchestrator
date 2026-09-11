"""Tests for the shared ``Broadcaster`` fan-out class."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.dashboard.broadcaster import Broadcaster


async def test_subscribe_emit_unsubscribe():
    bc = Broadcaster()
    queue = bc.subscribe()
    bc.emit({"id": 1})
    event = await asyncio.wait_for(queue.get(), timeout=0.5)
    assert event == {"id": 1}
    bc.unsubscribe(queue)


async def test_emit_with_no_subscribers_is_silent():
    bc = Broadcaster()
    bc.emit("anything")  # should not raise


async def test_multiple_subscribers_all_receive_event():
    bc = Broadcaster()
    q1 = bc.subscribe()
    q2 = bc.subscribe()
    bc.emit({"x": 1})
    e1 = await asyncio.wait_for(q1.get(), timeout=0.5)
    e2 = await asyncio.wait_for(q2.get(), timeout=0.5)
    assert e1 == {"x": 1}
    assert e2 == {"x": 1}


async def test_a_stalled_subscriber_loses_its_oldest_events_not_the_producer():
    """Queues are bounded, and the bound drops the OLDEST event.

    Nothing applies backpressure to a producer: a chat run emits a frame per
    message and per tool result whether or not the browser is reading, so an
    unbounded queue grew for as long as a half-open connection stayed open.
    """
    bc = Broadcaster(maxsize=2)
    q = bc.subscribe()

    for i in range(4):
        bc.emit(i)
    await asyncio.sleep(0)  # emit hands off via call_soon_threadsafe

    assert q.qsize() == 2, "queue grew past its bound"
    assert [q.get_nowait(), q.get_nowait()] == [2, 3], "dropped the newest instead"


async def test_unsubscribed_queue_no_longer_receives():
    bc = Broadcaster()
    q = bc.subscribe()
    bc.unsubscribe(q)
    bc.emit("ignored")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(q.get(), timeout=0.05)


async def test_emit_from_a_worker_thread_wakes_the_subscribers_loop():
    """The reason this class exists, and the one thing nothing tested.

    Every other test emits from inside the subscriber's own loop, where a bare
    ``queue.put_nowait`` would pass just as well. From a FOREIGN thread — which
    is where SystemLogger fans lifecycle logs out of — put_nowait leaves the
    loop parked in select() and the SSE consumer stalls until some later event
    happens to arrive in-loop. Hence ``loop.call_soon_threadsafe``.
    """
    import threading
    import time

    bc = Broadcaster()
    q = bc.subscribe()

    def producer():
        time.sleep(0.05)  # let this loop park in select() with nothing to run
        bc.emit({"from": "thread"})

    threading.Thread(target=producer, daemon=True).start()
    started = time.perf_counter()
    event = await asyncio.wait_for(q.get(), timeout=2.0)
    elapsed = time.perf_counter() - started

    assert event == {"from": "thread"}
    assert elapsed < 0.5, f"delivered only when the timeout woke the loop ({elapsed:.2f}s)"


async def test_each_stalled_subscriber_gets_its_own_full_queue_warning(caplog):
    """The warning is once per SUBSCRIBER, not once per process.

    A single flag lets the first stalled consumer in the process's life silence
    it for every later one, and there is exactly one Broadcaster behind
    SystemLogger for the whole dashboard.
    """
    import logging

    def full_warnings():
        return [r for r in caplog.records if "queue full" in r.message]

    with caplog.at_level(logging.WARNING, logger="aureax.broadcaster"):
        bc = Broadcaster(maxsize=1)
        first = bc.subscribe()
        for i in range(3):
            bc.emit(i)
        await asyncio.sleep(0)
        assert len(full_warnings()) == 1, "the first stalled subscriber warns once"

        # Cleared, so what the assertion below finds can only have come from the
        # SECOND subscriber: the first one's warning is already on the record and
        # would satisfy the count on its own.
        bc.unsubscribe(first)
        caplog.clear()

        bc.subscribe()  # a second subscriber, stalling the same way
        for i in range(3):
            bc.emit(i)
        await asyncio.sleep(0)

    assert len(full_warnings()) == 1, "the second stalled subscriber warns too"
