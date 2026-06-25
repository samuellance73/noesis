"""
perception/stages/intake.py
────────────────────────────
Stage 1 — Intake Buffer

Collects raw signals over a configurable time window before processing begins.
Prevents each individual message from triggering a pipeline run.

Flush triggers (in priority order):
  1. High-priority signal arrives          → immediate flush (bypass window)
  2. Buffer reaches max_size               → immediate flush
  3. window_seconds elapses without flush  → flush on timeout

Buffer is *lossy* at max_size: when the queue is already full, the oldest
non-high-priority signal is dropped to make room for the incoming one.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque

from perception.schemas import Priority, RawSignal

logger = logging.getLogger("noesis.perception")


class IntakeBuffer:
    """
    Async buffer that accumulates RawSignal objects and drains them as a batch.

    Usage
    ─────
        buffer = IntakeBuffer(window_seconds=5.0, max_size=100)

        # Producer side (any coroutine / Discord handler / webhook handler):
        await buffer.ingest(signal)

        # Consumer side (pipeline loop):
        batch = await buffer.drain()   # blocks until window expires or flush
    """

    def __init__(self, window_seconds: float = 5.0, max_size: int = 100) -> None:
        self.window_seconds = window_seconds
        self.max_size = max_size
        self._buffer: deque[RawSignal] = deque()
        self._lock = asyncio.Lock()
        self.flush_trigger: asyncio.Event = asyncio.Event()

    async def ingest(self, signal: RawSignal) -> None:
        """
        Add a signal to the buffer.

        - HIGH priority signals set the flush trigger immediately.
        - If the buffer is full, the oldest normal-priority signal is dropped.
        """
        async with self._lock:
            if len(self._buffer) >= self.max_size:
                # Drop oldest non-HIGH signal to make room
                dropped = self._drop_oldest_normal()
                if dropped is None:
                    # All buffered signals are HIGH — drop the new one instead
                    logger.warning(
                        "IntakeBuffer full of HIGH-priority signals; dropping incoming signal id=%s",
                        signal.id,
                    )
                    return
                logger.warning(
                    "IntakeBuffer full; dropped oldest normal signal id=%s to fit id=%s",
                    dropped.id, signal.id,
                )

            self._buffer.append(signal)

        if signal.priority == Priority.HIGH:
            logger.debug("IntakeBuffer: HIGH-priority signal received — setting flush trigger.")
            self.flush_trigger.set()
        elif len(self._buffer) >= self.max_size:
            logger.debug("IntakeBuffer: buffer full — setting flush trigger.")
            self.flush_trigger.set()

    async def drain(self) -> list[RawSignal]:
        """
        Wait for window_seconds OR the flush trigger, then return all buffered
        signals and reset for the next window.
        """
        try:
            await asyncio.wait_for(
                self.flush_trigger.wait(),
                timeout=self.window_seconds,
            )
        except asyncio.TimeoutError:
            pass  # Normal window expiry — proceed with whatever's buffered

        self.flush_trigger.clear()

        async with self._lock:
            batch = list(self._buffer)
            self._buffer.clear()

        return batch

    def _drop_oldest_normal(self) -> RawSignal | None:
        """
        Remove and return the oldest non-HIGH signal from the buffer.
        Returns None if no such signal exists.
        """
        for i, sig in enumerate(self._buffer):
            if sig.priority != Priority.HIGH:
                del self._buffer[i]
                return sig
        return None

    @property
    def size(self) -> int:
        return len(self._buffer)
