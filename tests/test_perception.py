"""
tests/test_perception.py
─────────────────────────
Unit tests for the Perception Layer pipeline stages (v2 API).

Covers:
  Stage 1 — IntakeBuffer  (windowing, fast-lane flush, batch-size flush)
  Stage 2 — AuthorityScorer (base scores, recency bonus, clamping)
  WorldModel facade       (absorb, flag_for_interrupt, drain)

Run with:
    uv run pytest tests/test_perception.py -v
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from core.events import PriorityLevel, UnifiedIngestEvent, SenderClass
from perception.schemas import (
    PerceptionEvent,
    PerceptionType,
    PerceptionWorldModel,
    ResponseJob,
    ScoredSignal,
)
from perception.stages.authority import AuthorityScorer
from perception.stages.intake import IntakeBuffer


# ── Helpers ────────────────────────────────────────────────────────────────────

def _event_now(
    text: str = "hello world test message",
    sender_class: SenderClass = SenderClass.EXTERNAL,
    priority: PriorityLevel = PriorityLevel.NORMAL,
    channel_id: str | None = None,
    seconds_ago: float = 0.0,
) -> UnifiedIngestEvent:
    """Build a UnifiedIngestEvent with a monotonic_timestamp from seconds_ago seconds in the past."""
    return UnifiedIngestEvent(
        source_channel="test_channel",
        sender_identifier="u1",
        sender_class=sender_class,
        raw_content=text,
        target_conversation_identifier=channel_id or "test_conversation",
        priority_level=priority,
        monotonic_timestamp=time.monotonic() - seconds_ago,
    )


def _scored_signal(
    text: str = "some sample text here",
    sender_class: SenderClass = SenderClass.EXTERNAL,
    frequency: int = 1,
    ptype: PerceptionType | None = None,
    authority: float = 0.0,
    seconds_ago: float = 5.0,
) -> ScoredSignal:
    sig = _event_now(text=text, sender_class=sender_class, seconds_ago=seconds_ago)
    return ScoredSignal(
        representative=sig,
        frequency=frequency,
        perception_type=ptype,
        authority_score=authority,
    )


def _perception_event(
    ptype: PerceptionType = PerceptionType.INFORMATION,
    urgency: float = 0.5,
    authority: float = 0.5,
    immediate: bool = False,
    affects: bool = True,
) -> PerceptionEvent:
    return PerceptionEvent(
        summary="test event",
        type=ptype,
        urgency=urgency,
        authority_score=authority,
        requires_immediate_response=immediate,
        affects_objectives=affects,
        frequency=1,
        source_ids=[uuid4()],
    )


# ── Stage 1: IntakeBuffer ──────────────────────────────────────────────────────

class TestIntakeBuffer:

    @pytest.mark.asyncio
    async def test_drain_returns_fed_signals(self):
        buf = IntakeBuffer(window_seconds=0.05, max_buffer_size=100)
        s1 = _event_now("first signal test words")
        s2 = _event_now("second signal test words")
        await buf.feed(s1)
        await buf.feed(s2)
        batch = await buf.drain()
        assert len(batch) == 2
        ids = {s.event_identifier for s in batch}
        assert s1.event_identifier in ids
        assert s2.event_identifier in ids

    @pytest.mark.asyncio
    async def test_operator_sender_triggers_immediate_flush(self):
        buf = IntakeBuffer(window_seconds=60.0, max_buffer_size=100)
        s = _event_now(sender_class=SenderClass.OPERATOR)
        await buf.feed(s)
        # Should return quickly despite 60s window (fast-lane)
        batch = await asyncio.wait_for(buf.drain(), timeout=2.0)
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_high_priority_triggers_immediate_flush(self):
        buf = IntakeBuffer(window_seconds=60.0, max_buffer_size=100)
        s = _event_now(priority=PriorityLevel.HIGH)
        await buf.feed(s)
        batch = await asyncio.wait_for(buf.drain(), timeout=2.0)
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_agent_sender_triggers_immediate_flush(self):
        buf = IntakeBuffer(window_seconds=60.0, max_buffer_size=100)
        s = _event_now(sender_class=SenderClass.AGENT)
        await buf.feed(s)
        batch = await asyncio.wait_for(buf.drain(), timeout=2.0)
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_buffer_clears_after_drain(self):
        buf = IntakeBuffer(window_seconds=0.05, max_buffer_size=100)
        await buf.feed(_event_now())
        await buf.drain()
        # Second drain should return empty after window expires
        batch = await buf.drain()
        assert batch == []

    @pytest.mark.asyncio
    async def test_max_buffer_flush_trigger(self):
        buf = IntakeBuffer(window_seconds=60.0, max_buffer_size=3)
        for _ in range(3):
            await buf.feed(_event_now())
        # Should flush immediately when max_buffer_size reached
        batch = await asyncio.wait_for(buf.drain(), timeout=2.0)
        assert len(batch) == 3

    @pytest.mark.asyncio
    async def test_size_property_reflects_buffer(self):
        buf = IntakeBuffer(window_seconds=60.0, max_buffer_size=100)
        assert buf.size == 0
        await buf.feed(_event_now())
        await buf.feed(_event_now())
        assert buf.size == 2

    @pytest.mark.asyncio
    async def test_external_user_does_not_fast_lane(self):
        """External users should not trigger fast-lane (no early flush)."""
        buf = IntakeBuffer(window_seconds=0.15, max_buffer_size=100)
        await buf.feed(_event_now(sender_class=SenderClass.EXTERNAL))
        # With a 150ms window, the external event should NOT trigger immediate flush —
        # we wait for the window to expire naturally.
        t0 = time.monotonic()
        batch = await buf.drain()
        elapsed = time.monotonic() - t0
        assert len(batch) == 1
        assert elapsed >= 0.10, f"Expected window to elapse, but drain returned in {elapsed:.3f}s"


# ── Stage 2: AuthorityScorer ──────────────────────────────────────────────────

class TestAuthorityScorer:

    def setup_method(self):
        self.scorer = AuthorityScorer(operator_ids=["op1"])

    def _score(
        self,
        sender_class: SenderClass,
        seconds_ago: float = 5.0,
    ) -> float:
        sig = _event_now("sample text here test", sender_class=sender_class, seconds_ago=seconds_ago)
        result = self.scorer.score(sig)
        return result.authority_score

    def test_operator_base_score(self):
        score = self._score(SenderClass.OPERATOR)
        assert score >= 1.0  # 1.0 base + recency bonus, clamped to 1.0

    def test_trusted_base_score(self):
        score = self._score(SenderClass.TRUSTED)
        assert 0.75 <= score <= 1.0

    def test_external_base_score(self):
        score = self._score(SenderClass.EXTERNAL, seconds_ago=60.0)  # no recency bonus
        assert 0.50 <= score <= 0.56  # base only (0.50) + tiny margin

    def test_agent_base_score(self):
        # AGENT uses _DEFAULT_BASE_SCORE (0.30) since it's not in _BASE_SCORES yet
        score = self._score(SenderClass.AGENT, seconds_ago=60.0)
        assert 0.0 <= score <= 0.56  # default base or external-equivalent

    def test_recency_bonus_recent_signal(self):
        score_recent = self._score(SenderClass.EXTERNAL, seconds_ago=0.0)
        score_old    = self._score(SenderClass.EXTERNAL, seconds_ago=60.0)
        assert score_recent > score_old, "Recent signals should score higher"

    def test_score_clamped_to_1(self):
        score = self._score(SenderClass.OPERATOR, seconds_ago=0.0)
        assert score <= 1.0

    def test_returns_scored_signal(self):
        sig = _event_now("hello world test text", sender_class=SenderClass.EXTERNAL)
        result = self.scorer.score(sig)
        assert isinstance(result, ScoredSignal)

    def test_score_batch(self):
        events = [_event_now(sender_class=SenderClass.EXTERNAL) for _ in range(5)]
        results = self.scorer.score_batch(events)
        assert len(results) == 5
        assert all(isinstance(r, ScoredSignal) for r in results)

    def test_works_with_scored_signal_input(self):
        """AuthorityScorer.score() now accepts UnifiedIngestEvent directly; score_batch also."""
        sig = _event_now("hello world", sender_class=SenderClass.TRUSTED)
        result = self.scorer.score(sig)
        assert isinstance(result, ScoredSignal)
        assert result.authority_score > 0


# ── WorldModel facade ──────────────────────────────────────────────────────────

class TestPerceptionWorldModel:

    @pytest.mark.asyncio
    async def test_absorb_and_drain(self):
        wm = PerceptionWorldModel()
        ev = _perception_event()
        await wm.absorb(ev)
        drained = wm.drain_perceptions()
        assert len(drained) == 1
        assert drained[0].id == ev.id

    @pytest.mark.asyncio
    async def test_flag_and_drain_interrupts(self):
        wm = PerceptionWorldModel()
        ev = _perception_event(ptype=PerceptionType.CORRECTION, authority=0.9)
        await wm.flag_for_interrupt(ev)
        flags = wm.drain_interrupts()
        assert len(flags) == 1
        assert flags[0].id == ev.id
        # Second drain should be empty
        assert wm.drain_interrupts() == []

    @pytest.mark.asyncio
    async def test_drain_empty_queue_returns_empty(self):
        wm = PerceptionWorldModel()
        assert wm.drain_perceptions() == []

    @pytest.mark.asyncio
    async def test_multiple_absorbs(self):
        wm = PerceptionWorldModel()
        for _ in range(5):
            await wm.absorb(_perception_event())
        drained = wm.drain_perceptions()
        assert len(drained) == 5

    @pytest.mark.asyncio
    async def test_add_and_drain_contexts(self):
        wm = PerceptionWorldModel()
        await wm.add_perception_context({"text": "test", "source": "discord", "summary": "test summary"})
        contexts = wm.drain_contexts()
        assert len(contexts) == 1
        assert contexts[0]["summary"] == "test summary"
        # Second drain should be empty
        assert wm.drain_contexts() == []
