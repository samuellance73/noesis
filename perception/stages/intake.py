"""
perception/stages/intake.py
────────────────────────────
Stage 1 — Intake Buffer

Collects raw signals over a configurable time window before processing begins.
Prevents each individual message from triggering a pipeline run.

Flush triggers (in priority order):
  1. High-priority signal arrives          → immediate flush (bypass window)
  2. Buffer reaches max_size               → immediate flush
  3. window_seconds elapses without flush  → flush on timeout, IF the batch
     passes the intake threshold (see below).

Intake threshold (resource conservation)
─────────────────────────────────────────
After a window timeout, the buffer is flushed ONLY if at least one of these
conditions is met:
  • len(buffer) >= 3   (enough signal volume to justify a pipeline run)
  • ANY signal in the buffer has Priority.HIGH

If neither condition is met, the signals are left in the buffer and an empty
list is returned.  They will be carried over to the next window and retested.

Buffer is *lossy* at max_size: when the queue is already full, the oldest
non-high-priority signal is dropped to make room for the incoming one.
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
                    emit("perception.warning", "perception", {"msg": f"IntakeBuffer full of HIGH-priority signals; dropping incoming signal id={signal.id}"}, level="warn")
                    return
                emit("perception.warning", "perception", {"msg": f"IntakeBuffer full; dropped oldest normal signal id={dropped.id} to fit id={signal.id}"}, level="warn")

            self._buffer.append(signal)

        if signal.priority == Priority.HIGH:
            emit("perception.flush_trigger", "perception", {"reason": "high_priority"}, level="debug")
            self.flush_trigger.set()
        elif len(self._buffer) >= self.max_size:
            emit("perception.flush_trigger", "perception", {"reason": "buffer_full"}, level="debug")
            self.flush_trigger.set()

    async def drain(self) -> list[RawSignal]:
        """
        Wait for window_seconds OR the flush trigger, then return all buffered
        signals and reset for the next window.

        Intake threshold (applied only on timeout, not on forced flushes):
        If the window expires and the buffer contains fewer than 3 signals with
        no Priority.HIGH signals, do nothing — return an empty list and leave
        the signals in the buffer for the next window.
        """
        timed_out = False
        try:
            await asyncio.wait_for(
                self.flush_trigger.wait(),
                timeout=self.window_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True  # Normal window expiry

        self.flush_trigger.clear()

        async with self._lock:
            # ── Intake threshold check (only on timeout, not on forced flush) ──
            if timed_out:
                has_high = any(s.priority == Priority.HIGH for s in self._buffer)
                if len(self._buffer) < 3 and not has_high:
                    emit(
                        "perception.intake_threshold_not_met",
                        "perception",
                        {
                            "buffered": len(self._buffer),
                            "msg": "Threshold not met (< 3 signals, no HIGH); holding signals.",
                        },
                        level="debug",
                    )
                    return []  # Leave signals in buffer; carry over to next window

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
