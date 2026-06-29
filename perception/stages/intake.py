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

from perception.schemas import Priority, RawSignal
from utils.log_writer import emit


class IntakeBuffer:
    """
    Async buffer that accumulates RawSignal objects and drains them as a batch.

    Usage
    ─────
        buffer = IntakeBuffer(window_seconds=60.0)

        # Producer side (any coroutine / Discord handler / webhook handler):
        await buffer.ingest(signal)

        # Consumer side (pipeline loop):
        batch = await buffer.drain()   # blocks until 3 messages or 60 seconds
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._buffer: deque[RawSignal] = deque()
        self._lock = asyncio.Lock()
        self.flush_trigger: asyncio.Event = asyncio.Event()
        self._first_arrival_time: float | None = None

    async def ingest(self, signal: RawSignal) -> None:
        """
        Add a signal to the buffer.
        Triggers flush when buffer reaches 3 messages.
        """
        async with self._lock:
            if self._first_arrival_time is None:
                self._first_arrival_time = asyncio.get_event_loop().time()
            self._buffer.append(signal)
            if len(self._buffer) >= 3:
                emit("perception.flush_trigger", "perception", {"reason": "buffer_size_3"}, level="debug")
                self.flush_trigger.set()

    async def drain(self) -> list[RawSignal]:
        """
        Wait for window_seconds (60s) OR the flush trigger (3 messages),
        then return all buffered signals and reset for the next window.
        """
        async with self._lock:
            if self._first_arrival_time is None:
                # No signals yet, wait full window
                remaining_timeout = self.window_seconds
            else:
                elapsed = asyncio.get_event_loop().time() - self._first_arrival_time
                remaining_timeout = max(0, self.window_seconds - elapsed)

        try:
            await asyncio.wait_for(
                self.flush_trigger.wait(),
                timeout=remaining_timeout,
            )
        except asyncio.TimeoutError:
            pass  # Normal window expiry

        self.flush_trigger.clear()

        async with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()
            self._first_arrival_time = None

        return batch

    @property
    def size(self) -> int:
        return len(self._buffer)
