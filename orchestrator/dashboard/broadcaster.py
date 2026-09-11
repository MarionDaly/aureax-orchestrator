"""Thread-safe fan-out from any thread to per-event-loop asyncio queues.

Used by ``SystemLogger`` (lifecycle log fan-out) and ``ChatService`` (per-session
event fan-out) to broadcast events from worker threads to async SSE consumers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

logger = logging.getLogger("aureax.broadcaster")

# Per-subscriber queue bound. Producers never wait on a consumer — a chat run
# emits a frame per message and per tool result whether or not anyone is
# reading — so an unbounded queue would grow for as long as a half-open
# connection stays half-open. Generous enough that a browser doing normal work
# never reaches it.
MAX_QUEUE_SIZE = 1000


class Broadcaster:
    """Multi-subscriber fan-out queue.

    ``emit`` is safe to call from any thread; each subscriber's queue is
    populated via ``loop.call_soon_threadsafe`` so the event lands on the
    loop the subscriber lives on. Subscribers that have died (loop closed)
    are tolerated: ``RuntimeError`` from ``call_soon_threadsafe`` is
    swallowed silently.

    Queues are bounded (``MAX_QUEUE_SIZE``). A subscriber that stops draining
    loses its OLDEST events, not its newest: the newest frame is the one that
    describes the current state, and a consumer that far behind has already
    lost the thread. The drop is logged once per subscriber so a stalled
    consumer is visible without the log itself becoming the flood.
    """

    def __init__(self, maxsize: int = MAX_QUEUE_SIZE) -> None:
        self._subs: list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []
        self._maxsize = maxsize
        # Keyed by id(queue): the warning is per SUBSCRIBER, so one stalled
        # consumer earlier in the process's life cannot silence the warning for
        # every later one (there is a single Broadcaster behind SystemLogger).
        self._warned_full: set[int] = set()

    def emit(self, event: object) -> None:
        for queue, loop in list(self._subs):
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._deliver, queue, event)

    def _deliver(self, queue: asyncio.Queue, event: object) -> None:
        """Put one event on a subscriber's queue, dropping the oldest if full.

        Runs on the subscriber's own loop, which is the only place
        ``get_nowait`` is safe against a consumer awaiting ``get``.
        """
        if queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
            if id(queue) not in self._warned_full:
                self._warned_full.add(id(queue))
                logger.warning(
                    "subscriber queue full at %d events — dropping oldest",
                    self._maxsize,
                )
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subs.append((queue, loop))
        return queue

    @property
    def has_subscribers(self) -> bool:
        return bool(self._subs)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subs = [(q, lp) for (q, lp) in self._subs if q is not queue]
        # id() is only unique among LIVE objects, so a dropped id must go with
        # the queue it named — the next queue at that address is a different
        # subscriber and gets its own warning.
        self._warned_full.discard(id(queue))
