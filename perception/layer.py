"""
perception/layer.py
────────────────────
PerceptionLayer — top-level orchestrator that wires all six pipeline stages
into a continuously running async loop.

Architecture
────────────
  External signal producers call `await layer.ingest(raw_signal)`.
  The layer runs a background loop (started via `await layer.start()`):

    while True:
        batch = await intake_buffer.drain()      # blocks until window or flush
        if not batch:
            continue
        try:
            deduped  = deduplicator.deduplicate(batch)
            # classify + score each deduplicated signal
            for ds in deduped:
                ds.perception_type = classifier.classify(ds)
            deduped = [s for s in deduped if s.perception_type != NOISE]
            scored   = authority_scorer.score_batch(classified)
            events, latency = await synthesizer.synthesize(scored, world_model_summary)
            await router.route(events)
        except Exception:
            log + continue

Error isolation
───────────────
Each stage is wrapped in try/except.  A crashed stage passes its input through
unmodified rather than halting the pipeline (§7).  The pipeline loop itself
never blocks new signal intake regardless of downstream failures.

Logging (§9)
────────────
Perception events are logged to logs/perception.log via the
`noesis.perception` logger, matching the pattern of other log files.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from perception.config import PerceptionConfig
from perception.reactive_pool import ReactivePool
from perception.schemas import (
    DeduplicatedSignal,
    PerceptionEvent,
    PerceptionType,
    PerceptionWorldModel,
    RawSignal,
)
from perception.stages.authority import AuthorityScorer
from perception.stages.classifier import Classifier
from perception.stages.dedup import Deduplicator
from perception.stages.intake import IntakeBuffer
from perception.stages.router import Router
from perception.stages.synthesizer import Synthesizer

logger = logging.getLogger("noesis.perception")


class PerceptionLayer:
    """
    The full perception pipeline as a single manageable component.

    Usage (in app/lifespan.py)
    ──────────────────────────
        config = PerceptionConfig()
        layer  = PerceptionLayer(
            config=config,
            llm_service=app.state.upstream_service,
            world_model=my_world_model,       # PerceptionWorldModel or custom
            executor_factory=my_factory,      # optional ReactivePool executor
        )
        await layer.start()     # launch intake loop + reactive pool
        ...
        await layer.stop()      # graceful shutdown

    Producing signals from any source
    ──────────────────────────────────
        signal = RawSignal(source=..., text="...", priority=Priority.NORMAL)
        await layer.ingest(signal)
    """

    def __init__(
        self,
        config: PerceptionConfig,
        llm_service: Any,                          # UpstreamService (duck-typed)
        world_model: PerceptionWorldModel | None = None,
        executor_factory=None,
    ) -> None:
        self.config = config

        # ── Stage 1: Intake buffer ─────────────────────────────────────────────
        self._intake = IntakeBuffer(
            window_seconds=config.intake_window_seconds,
            max_size=config.intake_max_buffer_size,
        )

        # ── Stage 2: Deduplicator ──────────────────────────────────────────────
        self._dedup = Deduplicator(
            similarity_threshold=config.similarity_threshold,
        )

        # ── Stage 3: Classifier ────────────────────────────────────────────────
        self._classifier = Classifier()

        # ── Stage 4: Authority scorer ──────────────────────────────────────────
        self._authority = AuthorityScorer(
            operator_ids=config.operator_ids,
        )

        # ── Stage 5: Synthesizer ───────────────────────────────────────────────
        self._synthesizer = Synthesizer(
            llm_service=llm_service,
            model=config.synthesizer_model,
            timeout=config.synthesizer_timeout_seconds,
            max_tokens=config.synthesizer_max_tokens,
        )

        # ── ReactivePool ───────────────────────────────────────────────────────
        self._reactive_pool = ReactivePool(
            max_workers=config.reactive_pool_size,
            executor_timeout=config.reactive_executor_timeout_seconds,
            executor_factory=executor_factory,
        )

        # ── WorldModel facade ──────────────────────────────────────────────────
        self._world_model: PerceptionWorldModel = world_model or PerceptionWorldModel()

        # ── Stage 6: Router ────────────────────────────────────────────────────
        self._router = Router(
            reactive_pool=self._reactive_pool,
            world_model=self._world_model,
        )

        self._loop_task: asyncio.Task | None = None
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ingest(self, signal: RawSignal) -> None:
        """
        Accept a raw signal from any external source (Discord, CLI, API, cron).
        Thread-safe — may be called from any coroutine.
        """
        await self._intake.ingest(signal)
        logger.debug(
            "PerceptionLayer.ingest: signal id=%s  source=%s  priority=%s  text=%r",
            signal.id, signal.source.identifier, signal.priority.value,
            signal.text[:80],
        )

    async def start(self) -> None:
        """Launch the intake loop and reactive pool as background tasks."""
        if self._running:
            logger.warning("PerceptionLayer.start() called while already running.")
            return

        self._running = True
        await self._reactive_pool.start()
        self._loop_task = asyncio.create_task(
            self._pipeline_loop(), name="perception-pipeline-loop"
        )
        logger.info(
            "PerceptionLayer started.  window=%.1fs  max_buf=%d  pool=%d  model=%s",
            self.config.intake_window_seconds,
            self.config.intake_max_buffer_size,
            self.config.reactive_pool_size,
            self.config.synthesizer_model,
        )

    async def stop(self) -> None:
        """Gracefully shut down the pipeline loop and reactive pool."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        await self._reactive_pool.stop()
        logger.info("PerceptionLayer stopped.")

    @property
    def world_model(self) -> PerceptionWorldModel:
        """Access the WorldModel facade for the GoalManager to drain."""
        return self._world_model

    # ── Pipeline loop ──────────────────────────────────────────────────────────

    async def _pipeline_loop(self) -> None:
        """
        Main async loop.  Drains the intake buffer and runs each batch through
        the full pipeline.  Runs forever until `stop()` cancels this task.
        """
        logger.debug("PerceptionLayer: pipeline loop started.")
        while self._running:
            try:
                batch = await self._intake.drain()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("PerceptionLayer: intake drain error: %s", exc, exc_info=True)
                await asyncio.sleep(1.0)
                continue

            if not batch:
                continue

            await self._run_pipeline(batch)

        logger.debug("PerceptionLayer: pipeline loop exited.")

    async def _run_pipeline(self, batch: list[RawSignal]) -> None:
        """Process one batch through all six stages with per-stage error isolation."""
        raw_count = len(batch)
        logger.info(
            "PerceptionLayer: batch received  raw_count=%d  window=%.1fs",
            raw_count, self.config.intake_window_seconds,
        )

        # ── Stage 2: Deduplication ─────────────────────────────────────────────
        try:
            deduped: list[DeduplicatedSignal] = self._dedup.deduplicate(batch)
        except Exception as exc:
            logger.error("PerceptionLayer: Deduplicator crashed — passing signals through unmodified. error=%s", exc)
            # Fallback: treat each raw signal as its own deduplicated signal
            deduped = [
                DeduplicatedSignal(
                    representative=s,
                    frequency=1,
                    sources=[s.source],
                    raw_signals=[s],
                )
                for s in batch
            ]

        logger.info(
            "PerceptionLayer: post-dedup count=%d (collapsed %d signals)",
            len(deduped), raw_count - len(deduped),
        )

        # ── Stage 3: Classification ────────────────────────────────────────────
        classified: list[DeduplicatedSignal] = []
        for ds in deduped:
            try:
                ds.perception_type = self._classifier.classify(ds)
            except Exception as exc:
                logger.error(
                    "PerceptionLayer: Classifier crashed for signal id=%s — defaulting to INFORMATION. error=%s",
                    ds.representative.id, exc,
                )
                ds.perception_type = PerceptionType.INFORMATION
            classified.append(ds)

        # Drop noise before scoring and synthesis
        non_noise = [ds for ds in classified if ds.perception_type != PerceptionType.NOISE]
        noise_count = len(classified) - len(non_noise)
        if noise_count:
            logger.info("PerceptionLayer: dropped %d NOISE signals.", noise_count)

        if not non_noise:
            logger.info("PerceptionLayer: entire batch classified as noise — skipping synthesis.")
            return

        # ── Stage 4: Authority scoring ─────────────────────────────────────────
        try:
            scored = self._authority.score_batch(non_noise)
        except Exception as exc:
            logger.error(
                "PerceptionLayer: AuthorityScorer crashed — passing signals with default score 0.5. error=%s", exc
            )
            from perception.schemas import ScoredSignal
            scored = [
                ScoredSignal(
                    representative=ds.representative,
                    frequency=ds.frequency,
                    sources=ds.sources,
                    perception_type=ds.perception_type or PerceptionType.INFORMATION,
                    authority_score=0.5,
                )
                for ds in non_noise
            ]

        # ── Stage 5: Synthesis (single LLM call) ───────────────────────────────
        world_summary = self._get_world_summary()
        try:
            events, latency = await self._synthesizer.synthesize(scored, world_summary)
        except Exception as exc:
            logger.error(
                "PerceptionLayer: Synthesizer crashed — falling back to trivial events. error=%s", exc
            )
            events = [self._synthesizer._trivial_event(s) for s in scored]
            latency = 0.0

        logger.info(
            "PerceptionLayer: synthesizer produced %d event(s) in %.2fs",
            len(events), latency,
        )

        # Log each PerceptionEvent for perception.log
        for ev in events:
            logger.info(
                "PerceptionLayer: event id=%s  type=%s  urgency=%.2f  authority=%.2f  "
                "frequency=%d  immediate=%s  affects_objectives=%s  summary=%r",
                ev.id, ev.type.value, ev.urgency, ev.authority_score,
                ev.frequency, ev.requires_immediate_response, ev.affects_objectives,
                ev.summary[:120],
            )

        # ── Stage 6: Routing ───────────────────────────────────────────────────
        try:
            await self._router.route(events)
        except Exception as exc:
            logger.error("PerceptionLayer: Router crashed. error=%s", exc, exc_info=True)

    def _get_world_summary(self) -> str:
        """
        Return a brief world model summary for the Synthesizer prompt.
        In the current architecture the perception layer does not hold the full
        GoalState, so this returns a stub unless the caller provides a richer
        world_model implementation with a `summarize()` method.
        """
        if hasattr(self._world_model, "summarize"):
            try:
                return self._world_model.summarize()  # type: ignore[attr-defined]
            except Exception:
                pass
        pending = self._world_model.pending_perceptions.qsize()
        return f"Pending perceptions in queue: {pending}"
