"""
perception/stages/intake.py
────────────────────────────
Stage 1 — Intake Buffer

Collects raw signals over a configurable time window before processing begins.
Prevents each individual message from triggering a pipeline run.

Flush triggers:
  1. Buffer reaches 3 messages               → immediate flush
  2. window_seconds (1 minute) elapses        → flush on timeout

Simple logic: process when we have enough messages OR enough time passes.
"""

from __future__ import annotations

import asyncio
from collections import deque

from core.events import UnifiedIngestEvent, SenderClass, PriorityLevel
from utils.log_writer import emit


class IntakeBuffer:
    """
    Async buffer that accumulates UnifiedIngestEvent objects and drains them as a batch.

    Usage
    ─────
        buffer = IntakeBuffer(window_seconds=60.0, max_buffer_size=3)

        # Producer side (Feeder Task):
        await buffer.feed(event)

        # Consumer side (Processing Task):
        batch = await buffer.drain()   # blocks until flush trigger
    """

    def __init__(self, window_seconds: float = 60.0, max_buffer_size: int = 3) -> None:
        self.window_seconds = window_seconds
        self.max_buffer_size = max_buffer_size
        self._buffer: deque[UnifiedIngestEvent] = deque()
        self._lock = asyncio.Lock()
        self.flush_trigger: asyncio.Event = asyncio.Event()
        self._first_arrival_time: float | None = None

    async def feed(self, event: UnifiedIngestEvent) -> None:
        """
        Add an event to the buffer.
        Triggers flush based on event priority or buffer size.
        """
        async with self._lock:
            if self._first_arrival_time is None:
                self._first_arrival_time = asyncio.get_event_loop().time()
            self._buffer.append(event)

            # Fast-lane: Operator, Agent (self-initiative), or explicitly High-priority
            if (
                event.sender_class in (SenderClass.OPERATOR, SenderClass.AGENT)
                or event.priority_level == PriorityLevel.HIGH
            ):
                emit("perception.flush_trigger", "perception", {"reason": "fast_lane"}, level="debug")
                self.flush_trigger.set()
            # Batch-lane: flush when buffer is full
            elif len(self._buffer) >= self.max_buffer_size:
                emit("perception.flush_trigger", "perception", {"reason": "buffer_size"}, level="debug")
                self.flush_trigger.set()

    async def drain(self) -> list[UnifiedIngestEvent]:
        """
        Block until the flush trigger fires or the window expires, then
        return all buffered events and reset state for the next window.

        IMPORTANT: the asyncio.Lock is released before the wait so that
        concurrent feed() calls are never blocked while this coroutine sleeps.
        """
        # Snapshot remaining window without holding the lock during the wait
        async with self._lock:
            if self._first_arrival_time is None:
                remaining_timeout = self.window_seconds
            else:
                elapsed = asyncio.get_event_loop().time() - self._first_arrival_time
                remaining_timeout = max(0.0, self.window_seconds - elapsed)

        # Wait OUTSIDE the lock — feed() can safely append during this wait
        try:
            await asyncio.wait_for(
                self.flush_trigger.wait(),
                timeout=remaining_timeout,
            )
        except asyncio.TimeoutError:
            pass  # Normal window expiry — drain whatever is buffered

        self.flush_trigger.clear()

        async with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()
            self._first_arrival_time = None

        return batch

    @property
    def size(self) -> int:
        return len(self._buffer)
