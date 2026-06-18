"""
utils/event_bus.py
──────────────────
Lightweight asyncio pub/sub bus for broadcasting SSE events to all connected
frontend clients.

Any backend component (daemon, GoalManager, direct API call) publishes events
here. Every open SSE connection is a subscriber and receives a copy of each
event.

Usage
─────
    # In a SSE endpoint:
    q = event_bus.subscribe()
    try:
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\\n\\n"
    finally:
        event_bus.unsubscribe(q)

    # From anywhere in the backend:
    await event_bus.publish({"event": "trigger_queued", "trigger_id": str(t.id)})
"""

import asyncio
import logging

logger = logging.getLogger("noesis.event_bus")


class EventBus:
    """
    Single-process pub/sub. Each subscriber gets its own asyncio.Queue so
    slow consumers don't block fast ones. Events are dropped for a subscriber
    only if their queue fills up (maxsize protection).
    """

    _QUEUE_MAXSIZE = 256  # drop events for a subscriber that falls this far behind

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    # ── Subscription management ────────────────────────────────────────────────

    def subscribe(self) -> asyncio.Queue:
        """
        Register a new subscriber. Call from SSE endpoint setup.
        Returns the Queue to await events from.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._subscribers.append(q)
        logger.debug("EventBus: subscriber added (%d total)", len(self._subscribers))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """
        Remove a subscriber. Call in the SSE endpoint's finally block.
        """
        try:
            self._subscribers.remove(q)
            logger.debug("EventBus: subscriber removed (%d remaining)", len(self._subscribers))
        except ValueError:
            pass  # already removed (double-close guard)

    # ── Publishing ─────────────────────────────────────────────────────────────

    async def publish(self, event: dict) -> None:
        """
        Broadcast an event to all active subscribers.
        Subscribers whose queue is full are silently skipped (they're too slow).
        """
        if not self._subscribers:
            return
        for q in list(self._subscribers):  # snapshot to avoid mutation during iteration
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("EventBus: subscriber queue full — event dropped for one client.")

    def publish_sync(self, event: dict) -> None:
        """
        Non-async variant for use in synchronous contexts.
        Schedules the publish on the running event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.publish(event))
            )
        except RuntimeError:
            pass  # no event loop running — silently drop


# ── Global singleton ───────────────────────────────────────────────────────────
# Import this from anywhere:  from utils.event_bus import event_bus
event_bus = EventBus()
