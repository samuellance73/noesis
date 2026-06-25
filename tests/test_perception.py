"""
tests/test_perception.py
─────────────────────────
Unit tests for the Perception Layer pipeline stages.

Covers:
  Stage 1 — IntakeBuffer  (windowing, HIGH flush, lossy overflow)
  Stage 2 — Deduplicator  (exact match, fuzzy, channel+time proximity)
  Stage 3 — Classifier    (noise, correction, directive, query, information, feedback)
  Stage 4 — AuthorityScorer (base scores, frequency/recency bonuses, clamping)
  Stage 5 — Synthesizer   (single-signal fast path, LLM timeout fallback)
  Stage 6 — Router        (routing matrix, all PerceptionTypes)
  WorldModel facade       (absorb, flag_for_interrupt, drain)

Run with:
    uv run pytest tests/test_perception.py -v
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from perception.schemas import (
    DeduplicatedSignal,
    PerceptionEvent,
    PerceptionType,
    PerceptionWorldModel,
    Priority,
    RawSignal,
    RawSignalSource,
    ResponseJob,
    ScoredSignal,
    SourceType,
)
from perception.stages.authority import AuthorityScorer
from perception.stages.classifier import Classifier
from perception.stages.dedup import Deduplicator
from perception.stages.intake import IntakeBuffer
from perception.stages.router import Router
from perception.stages.synthesizer import Synthesizer


# ── Helpers ────────────────────────────────────────────────────────────────────

def _source(stype: SourceType = SourceType.USER, uid: str = "u1") -> RawSignalSource:
    return RawSignalSource(type=stype, identifier=uid, display_name=uid)


def _signal(
    text: str = "hello world test message",
    stype: SourceType = SourceType.USER,
    priority: Priority = Priority.NORMAL,
    channel_id: str | None = None,
    ts: datetime | None = None,
) -> RawSignal:
    return RawSignal(
        source=_source(stype),
        text=text,
        priority=priority,
        channel_id=channel_id,
        timestamp=ts or datetime.now(timezone.utc),
    )


def _deduped(
    text: str = "some sample text here",
    stype: SourceType = SourceType.USER,
    frequency: int = 1,
    ptype: PerceptionType | None = None,
) -> DeduplicatedSignal:
    sig = _signal(text=text, stype=stype)
    ds = DeduplicatedSignal(
        representative=sig,
        frequency=frequency,
        sources=[_source(stype)],
        raw_signals=[sig],
    )
    ds.perception_type = ptype
    return ds


def _scored(
    text: str = "some sample text here",
    ptype: PerceptionType = PerceptionType.INFORMATION,
    authority: float = 0.5,
    frequency: int = 1,
) -> ScoredSignal:
    sig = _signal(text=text)
    return ScoredSignal(
        representative=sig,
        frequency=frequency,
        sources=[_source()],
        perception_type=ptype,
        authority_score=authority,
    )


def _event(
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
    async def test_drain_returns_ingested_signals(self):
        buf = IntakeBuffer(window_seconds=0.05, max_size=100)
        s1 = _signal("first signal test words")
        s2 = _signal("second signal test words")
        await buf.ingest(s1)
        await buf.ingest(s2)
        batch = await buf.drain()
        assert len(batch) == 2
        ids = {s.id for s in batch}
        assert s1.id in ids and s2.id in ids

    @pytest.mark.asyncio
    async def test_high_priority_triggers_immediate_flush(self):
        buf = IntakeBuffer(window_seconds=60.0, max_size=100)
        s = _signal(priority=Priority.HIGH)
        await buf.ingest(s)
        # Should return quickly despite 60s window
        batch = await asyncio.wait_for(buf.drain(), timeout=2.0)
        assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_buffer_clears_after_drain(self):
        buf = IntakeBuffer(window_seconds=0.05, max_size=100)
        await buf.ingest(_signal())
        await buf.drain()
        # Second drain should return empty after window
        batch = await buf.drain()
        assert batch == []

    @pytest.mark.asyncio
    async def test_overflow_drops_oldest_normal(self):
        buf = IntakeBuffer(window_seconds=0.05, max_size=3)
        s1, s2, s3 = _signal("a b c d"), _signal("e f g h"), _signal("i j k l")
        await buf.ingest(s1)
        await buf.ingest(s2)
        await buf.ingest(s3)
        s_new = _signal("new new new new")
        await buf.ingest(s_new)  # s1 should be dropped
        batch = await buf.drain()
        assert len(batch) == 3
        assert s1.id not in {s.id for s in batch}
        assert s_new.id in {s.id for s in batch}

    @pytest.mark.asyncio
    async def test_max_buffer_flush_trigger(self):
        buf = IntakeBuffer(window_seconds=60.0, max_size=3)
        for _ in range(3):
            await buf.ingest(_signal())
        batch = await asyncio.wait_for(buf.drain(), timeout=2.0)
        assert len(batch) == 3


# ── Stage 2: Deduplicator ─────────────────────────────────────────────────────

class TestDeduplicator:

    def test_empty_input(self):
        d = Deduplicator()
        assert d.deduplicate([]) == []

    def test_single_signal_passes_through(self):
        d = Deduplicator()
        result = d.deduplicate([_signal("unique message here today")])
        assert len(result) == 1
        assert result[0].frequency == 1

    def test_exact_duplicates_merged(self):
        d = Deduplicator()
        text = "what is the meaning of life"
        sigs = [_signal(text), _signal(text), _signal(text)]
        result = d.deduplicate(sigs)
        assert len(result) == 1
        assert result[0].frequency == 3

    def test_fuzzy_duplicates_merged(self):
        d = Deduplicator(similarity_threshold=0.75)
        # Very similar — should cluster
        s1 = _signal("please update the deployment configuration for production")
        s2 = _signal("please update the deployment configuration for prod")
        result = d.deduplicate([s1, s2])
        assert len(result) == 1
        assert result[0].frequency == 2

    def test_distinct_signals_not_merged(self):
        d = Deduplicator()
        s1 = _signal("what is the weather today in tokyo")
        s2 = _signal("delete all files from the production database now")
        result = d.deduplicate([s1, s2])
        assert len(result) == 2

    def test_representative_is_most_recent(self):
        d = Deduplicator()
        text = "reboot the server right now please"
        old_ts = datetime.now(timezone.utc) - timedelta(minutes=5)
        new_ts = datetime.now(timezone.utc)
        old_sig = RawSignal(source=_source(), text=text, timestamp=old_ts)
        new_sig = RawSignal(source=_source(), text=text, timestamp=new_ts)
        result = d.deduplicate([old_sig, new_sig])
        assert len(result) == 1
        assert result[0].representative.id == new_sig.id

    def test_frequency_reflects_cluster_size(self):
        d = Deduplicator()
        text = "hello world check status"
        sigs = [_signal(text) for _ in range(5)]
        result = d.deduplicate(sigs)
        assert result[0].frequency == 5


# ── Stage 3: Classifier ───────────────────────────────────────────────────────

class TestClassifier:

    def setup_method(self):
        self.c = Classifier()

    def _classify(self, text: str, stype: SourceType = SourceType.USER) -> PerceptionType:
        ds = _deduped(text=text, stype=stype)
        return self.c.classify(ds)

    def test_noise_too_short(self):
        assert self._classify("ok") == PerceptionType.NOISE

    def test_noise_pure_emoji(self):
        assert self._classify("👍👍👍") == PerceptionType.NOISE

    def test_noise_bot_command(self):
        assert self._classify("!status running now") == PerceptionType.NOISE

    def test_correction_actually(self):
        result = self._classify("actually that answer you gave was completely wrong")
        assert result == PerceptionType.CORRECTION

    def test_correction_you_said(self):
        result = self._classify("you said the deployment was done but it clearly was not")
        assert result == PerceptionType.CORRECTION

    def test_directive_create(self):
        result = self._classify("create a new GitHub repository for the project")
        assert result == PerceptionType.DIRECTIVE

    def test_directive_please_do(self):
        result = self._classify("can you update the configuration file for us")
        assert result == PerceptionType.DIRECTIVE

    def test_query_question_mark(self):
        result = self._classify("what is the current deployment status?")
        assert result == PerceptionType.QUERY

    def test_query_wh_word(self):
        result = self._classify("what happened to the last deployment cycle")
        assert result == PerceptionType.QUERY

    def test_information_declarative(self):
        result = self._classify("the deployment finished at 14:30 UTC without issues")
        assert result == PerceptionType.INFORMATION

    def test_feedback_good_job(self):
        result = self._classify("good job on the last release that was great work")
        assert result == PerceptionType.FEEDBACK


# ── Stage 4: AuthorityScorer ──────────────────────────────────────────────────

class TestAuthorityScorer:

    def setup_method(self):
        self.scorer = AuthorityScorer(operator_ids=["op1"])

    def _score(
        self,
        stype: SourceType,
        frequency: int = 1,
        seconds_ago: float = 5.0,
    ) -> float:
        ts = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        sig = RawSignal(source=_source(stype), text="sample text here test", timestamp=ts)
        ds = DeduplicatedSignal(
            representative=sig,
            frequency=frequency,
            sources=[_source(stype)],
            raw_signals=[sig],
            perception_type=PerceptionType.INFORMATION,
        )
        scored = self.scorer.score(ds)
        return scored.authority_score

    def test_operator_base_score(self):
        score = self._score(SourceType.OPERATOR)
        assert score >= 1.0  # 1.0 base + bonuses clamped to 1.0

    def test_trusted_base_score(self):
        score = self._score(SourceType.TRUSTED)
        assert 0.75 <= score <= 1.0

    def test_user_base_score(self):
        score = self._score(SourceType.USER, seconds_ago=60)  # no recency bonus
        assert 0.50 <= score <= 0.71  # base + up to freq bonus

    def test_anonymous_base_score(self):
        score = self._score(SourceType.ANONYMOUS, seconds_ago=60)
        assert 0.20 <= score <= 0.41

    def test_agent_base_score(self):
        score = self._score(SourceType.AGENT, seconds_ago=60)
        assert 0.60 <= score <= 0.81

    def test_frequency_bonus_capped(self):
        # 100 signals should give max 0.20 bonus, not more
        score_high_freq = self._score(SourceType.USER, frequency=100, seconds_ago=60)
        score_low_freq = self._score(SourceType.USER, frequency=1, seconds_ago=60)
        assert score_high_freq > score_low_freq
        assert score_high_freq <= 0.5 + 0.20 + 0.05 + 0.01  # base + cap + max_recency + margin

    def test_score_clamped_to_1(self):
        score = self._score(SourceType.OPERATOR, frequency=100, seconds_ago=0)
        assert score <= 1.0

    def test_returns_scored_signal(self):
        ts = datetime.now(timezone.utc)
        sig = RawSignal(source=_source(SourceType.USER), text="hello world test text", timestamp=ts)
        ds = DeduplicatedSignal(
            representative=sig, frequency=1, sources=[_source()], raw_signals=[sig],
            perception_type=PerceptionType.DIRECTIVE,
        )
        result = self.scorer.score(ds)
        assert isinstance(result, ScoredSignal)
        assert result.perception_type == PerceptionType.DIRECTIVE

    def test_raises_without_type(self):
        ts = datetime.now(timezone.utc)
        sig = RawSignal(source=_source(SourceType.USER), text="hello world test text", timestamp=ts)
        ds = DeduplicatedSignal(
            representative=sig, frequency=1, sources=[_source()], raw_signals=[sig],
            perception_type=None,
        )
        with pytest.raises(ValueError, match="perception_type"):
            self.scorer.score(ds)


# ── Stage 5: Synthesizer ──────────────────────────────────────────────────────

class TestSynthesizer:

    def _make_synthesizer(self, llm_content: str | None = None) -> Synthesizer:
        """Build a Synthesizer with a mocked ModelRouter that returns fixed content."""
        from core.model_router import ModelResponse, ModelTier
        router = MagicMock()
        mock_resp = ModelResponse(
            content=llm_content or "[]",
            model_used="test-model",
            tier=ModelTier.NANO,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=1.0,
        )
        router.complete = AsyncMock(return_value=mock_resp)
        router.resolve_model = MagicMock(return_value="test-model")
        return Synthesizer(router=router, timeout=5.0, max_tokens=800)

    @pytest.mark.asyncio
    async def test_single_signal_skips_llm(self):
        from core.model_router import ModelResponse, ModelTier
        router = MagicMock()
        router.complete = AsyncMock()
        router.resolve_model = MagicMock(return_value="test-model")
        synth = Synthesizer(router=router, timeout=5.0, max_tokens=800)

        signals = [_scored()]
        events, _ = await synth.synthesize(signals)

        assert len(events) == 1
        router.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_trivial_event_preserves_type(self):
        synth = self._make_synthesizer()
        s = _scored(ptype=PerceptionType.DIRECTIVE, authority=0.9)
        events, _ = await synth.synthesize([s])
        assert events[0].type == PerceptionType.DIRECTIVE

    @pytest.mark.asyncio
    async def test_llm_response_parsed_correctly(self):
        good_json = """
        [
          {
            "summary": "User wants deployment status",
            "type": "query",
            "urgency": 0.6,
            "authority_score": 0.5,
            "requires_immediate_response": true,
            "affects_objectives": false,
            "frequency": 2,
            "source_ids": [],
            "response_context": "Check deployment status"
          }
        ]
        """
        synth = self._make_synthesizer(good_json)
        signals = [_scored(), _scored(text="second signal text here")]
        events, _ = await synth.synthesize(signals)
        assert len(events) == 1
        assert events[0].type == PerceptionType.QUERY
        assert events[0].requires_immediate_response is True

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_trivial(self):
        from core.model_router import ModelTier
        router = MagicMock()

        async def slow(*args, **kwargs):
            await asyncio.sleep(10)

        router.complete = slow
        router.resolve_model = MagicMock(return_value="test-model")
        synth = Synthesizer(router=router, timeout=0.1, max_tokens=800)

        signals = [_scored(), _scored(text="another different signal here")]
        events, _ = await synth.synthesize(signals)
        # Fallback: one event per signal
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back_to_trivial(self):
        synth = self._make_synthesizer("this is not json at all!!!")
        signals = [_scored(), _scored(text="another sample text signal")]
        events, _ = await synth.synthesize(signals)
        assert len(events) == 2  # fallback: one per signal

    @pytest.mark.asyncio
    async def test_empty_signals_returns_empty(self):
        synth = self._make_synthesizer()
        events, latency = await synth.synthesize([])
        assert events == []
        assert latency == 0.0


# ── Stage 6: Router ───────────────────────────────────────────────────────────

class TestRouter:

    def _make_router(self):
        pool = MagicMock()
        pool.enqueue = AsyncMock()
        wm = MagicMock()
        wm.absorb = AsyncMock()
        wm.flag_for_interrupt = AsyncMock()
        return Router(reactive_pool=pool, world_model=wm), pool, wm

    @pytest.mark.asyncio
    async def test_noise_is_dropped(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.NOISE)
        await router.route([ev])
        pool.enqueue.assert_not_called()
        wm.absorb.assert_not_called()

    @pytest.mark.asyncio
    async def test_directive_goes_to_reactive_and_world(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.DIRECTIVE, authority=0.5)
        await router.route([ev])
        pool.enqueue.assert_called_once()
        wm.absorb.assert_called_once()
        wm.flag_for_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_directive_high_authority_interrupts(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.DIRECTIVE, authority=0.9)
        await router.route([ev])
        pool.enqueue.assert_called_once()
        wm.absorb.assert_called_once()
        wm.flag_for_interrupt.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_reactive_only(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.QUERY)
        await router.route([ev])
        pool.enqueue.assert_called_once()
        wm.absorb.assert_not_called()
        wm.flag_for_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_information_world_only(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.INFORMATION)
        await router.route([ev])
        pool.enqueue.assert_not_called()
        wm.absorb.assert_called_once()

    @pytest.mark.asyncio
    async def test_correction_low_authority_no_interrupt(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.CORRECTION, authority=0.5)
        await router.route([ev])
        pool.enqueue.assert_not_called()
        wm.absorb.assert_called_once()
        wm.flag_for_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_correction_high_authority_interrupts(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.CORRECTION, authority=0.8)
        await router.route([ev])
        wm.absorb.assert_called_once()
        wm.flag_for_interrupt.assert_called_once()
        pool.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_feedback_world_only(self):
        router, pool, wm = self._make_router()
        ev = _event(ptype=PerceptionType.FEEDBACK)
        await router.route([ev])
        pool.enqueue.assert_not_called()
        wm.absorb.assert_called_once()
        wm.flag_for_interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_events_routed_independently(self):
        router, pool, wm = self._make_router()
        events = [
            _event(ptype=PerceptionType.DIRECTIVE, authority=0.5),
            _event(ptype=PerceptionType.NOISE),
            _event(ptype=PerceptionType.QUERY),
            _event(ptype=PerceptionType.INFORMATION),
        ]
        await router.route(events)
        assert pool.enqueue.call_count == 2   # directive + query
        assert wm.absorb.call_count == 2      # directive + information


# ── WorldModel facade ──────────────────────────────────────────────────────────

class TestPerceptionWorldModel:

    @pytest.mark.asyncio
    async def test_absorb_and_drain(self):
        wm = PerceptionWorldModel()
        ev = _event()
        await wm.absorb(ev)
        drained = wm.drain_perceptions()
        assert len(drained) == 1
        assert drained[0].id == ev.id

    @pytest.mark.asyncio
    async def test_flag_and_drain_interrupts(self):
        wm = PerceptionWorldModel()
        ev = _event(ptype=PerceptionType.CORRECTION, authority=0.9)
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
            await wm.absorb(_event())
        drained = wm.drain_perceptions()
        assert len(drained) == 5
