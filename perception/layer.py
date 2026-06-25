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

from perception.config import PerceptionConfig
from perception.reactive_pool import ReactivePool
from perception.schemas import (
    DeduplicatedSignal,
    PerceptionWorldModel,
    RawSignal,
    ScoredSignal,
)
from perception.stages.authority import AuthorityScorer
from perception.stages.dedup import Deduplicator
from perception.stages.intake import IntakeBuffer
from perception.stages.router import Router
from core.model_router import ModelRouter, ModelRequest, ModelTier
from utils.log_writer import emit
from utils.json_parser import _clean_llm_json


class PerceptionLayer:
    """
    The full perception pipeline as a single manageable component.

    Usage (in app/lifespan.py)
    ──────────────────────────
        config = PerceptionConfig()
        layer  = PerceptionLayer(
            config=config,
            router=app.state.model_router,
            world_model=my_world_model,       # PerceptionWorldModel or custom
            executor_factory=my_factory,      # callable(ResponseJob) -> Awaitable
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
        router: ModelRouter,
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

        # ── Stage 3: Authority scorer ──────────────────────────────────────────
        self._authority = AuthorityScorer(
            operator_ids=config.operator_ids,
        )

        # ── ModelRouter for LLM bundle processing ───────────────────────────────
        self._router_llm = router

        # ── ReactivePool ───────────────────────────────────────────────────────
        self._reactive_pool = ReactivePool(
            max_workers=config.reactive_pool_size,
            executor_timeout=config.reactive_executor_timeout_seconds,
            executor_factory=executor_factory,
        )

        # ── WorldModel facade ──────────────────────────────────────────────────
        self._world_model: PerceptionWorldModel = world_model or PerceptionWorldModel()

        # ── Stage 5: Router ────────────────────────────────────────────────────
        self._router = Router(
            world_model=self._world_model,
            reactive_pool=self._reactive_pool,
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
        emit(
            event="perception.ingested",
            layer="perception",
            level="debug",
            data={
                "signal_id": signal.id,
                "source": signal.source.identifier,
                "priority": signal.priority.value,
                "text": signal.text[:80],
            }
        )

    async def start(self) -> None:
        """Launch the intake loop and reactive pool as background tasks."""
        if self._running:
            emit("perception.warning", "perception", {"msg": "PerceptionLayer.start() called while already running."}, level="warn")
            return

        self._running = True
        await self._reactive_pool.start()
        self._loop_task = asyncio.create_task(
            self._pipeline_loop(), name="perception-pipeline-loop"
        )
        emit(
            event="perception.started",
            layer="perception",
            data={
                "window": self.config.intake_window_seconds,
                "max_buf": self.config.intake_max_buffer_size,
                "pool": self.config.reactive_pool_size,
            }
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
        emit("perception.stopped", "perception", {})

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
        emit("perception.loop_started", "perception", {}, level="debug")
        while self._running:
            try:
                batch = await self._intake.drain()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                emit("perception.error", "perception", {"msg": f"intake drain error: {exc}"}, level="error")
                await asyncio.sleep(1.0)
                continue

            if not batch:
                continue

            await self._run_pipeline(batch)

        emit("perception.loop_exited", "perception", {}, level="debug")

    async def _run_pipeline(self, batch: list[RawSignal]) -> None:
        """Process one batch through the 3-stage pipeline with per-stage error isolation."""
        raw_count = len(batch)
        emit(
            event="perception.batch_received",
            layer="perception",
            data={
                "raw_count": raw_count,
                "window": self.config.intake_window_seconds,
            }
        )

        # ── Stage 2: Deduplication ─────────────────────────────────────────────
        try:
            deduped: list[DeduplicatedSignal] = self._dedup.deduplicate(batch)
        except Exception as exc:
            emit("perception.error", "perception", {"msg": f"Deduplicator crashed: {exc}"}, level="error")
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

        emit(
            event="perception.deduped",
            layer="perception",
            data={
                "count": len(deduped),
                "collapsed": raw_count - len(deduped),
            }
        )

        # ── Stage 3: Authority scoring ─────────────────────────────────────────
        try:
            scored = self._authority.score_batch(deduped)
        except Exception as exc:
            emit("perception.error", "perception", {"msg": f"AuthorityScorer crashed: {exc}"}, level="error")
            scored = [
                ScoredSignal(
                    representative=ds.representative,
                    frequency=ds.frequency,
                    sources=ds.sources,
                    perception_type=None,
                    authority_score=0.5,
                )
                for ds in deduped
            ]

        # ── Stage 4: LLM bundle processing ───────────────────────────────────────
        try:
            decisions = await self._process_bundle(scored)
        except Exception as exc:
            emit("perception.warning", "perception", {"msg": f"LLM bundle call failed: {exc}"}, level="warn")
            decisions = [
                {"index": i, "priority": "medium", "action": "queue",
                 "summary": s.representative.text[:100], "reason": "fallback"}
                for i, s in enumerate(scored)
            ]

        emit(
            event="perception.classified",
            layer="perception",
            data={"decision_count": len(decisions)}
        )

        # ── Stage 5: Routing ───────────────────────────────────────────────────
        try:
            await self._router.route(scored, decisions)
        except Exception as exc:
            emit("perception.error", "perception", {"msg": f"Router crashed: {exc}"}, level="error")

    async def _process_bundle(self, signals: list[ScoredSignal]) -> list[dict]:
        """
        Process a bundle of signals through a single LLM call.
        Returns a list of decision dicts with keys: index, priority, action, summary, reason.
        """
        import json
        import time

        # Build the signal block for the prompt
        signal_block_lines = []
        for i, signal in enumerate(signals):
            timestamp = int(signal.representative.timestamp.timestamp())
            source = signal.representative.source.identifier
            channel = signal.representative.channel_id or "unknown"
            authority = signal.authority_score
            text = signal.representative.text

            signal_block_lines.append(
                f"[{i}] source={source} channel={channel} authority={authority:.2f} time={timestamp}\n"
                f'    "{text}"'
            )

        signal_block = "\n\n".join(signal_block_lines)

        system_prompt = """\
You are the perception layer of an autonomous AI agent system called Noesis.
You have received a batch of signals from multiple sources since the last
processing cycle. Analyze all signals together and return a structured JSON
decision for each one.

For each signal, decide:
- priority: "high", "medium", or "low"
- action: "interrupt", "queue", or "drop"
  - interrupt = wake GoalManager immediately
  - queue = add to next GoalManager cycle context
  - drop = ignore entirely
- summary: one sentence describing what this signal means in context of the
  others (can reference other signals in the batch)
- reason: one sentence explaining your action decision

Use authority_score as a strong signal. A score below 0.3 should rarely
result in "interrupt". Obvious noise (very short, punctuation only, bot
commands starting with !) should be "drop".

Return ONLY a JSON array. One object per signal, in the same order received.
Schema per object:
{
  "index": <int>,
  "priority": "high" | "medium" | "low",
  "action": "interrupt" | "queue" | "drop",
  "summary": "<string>",
  "reason": "<string>"
}
"""

        user_prompt = f"""\
SIGNALS:
{signal_block}
"""

        request = ModelRequest(
            tier=ModelTier.NANO,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            component="PerceptionLayer._process_bundle",
        )

        response = await self._router_llm.complete(request)
        raw_content = response.content

        if not raw_content:
            emit("perception.warning", "perception", {"msg": "LLM returned empty content."}, level="warn")
            raise ValueError("Empty LLM response")

        # Parse the JSON response
        try:
            cleaned = _clean_llm_json(raw_content)
            data = json.loads(cleaned)

            if not isinstance(data, list):
                data = [data]

            # Validate basic structure
            for item in data:
                if not isinstance(item, dict):
                    raise ValueError("Decision item is not a dict")
                if "index" not in item or "action" not in item:
                    raise ValueError("Decision item missing required fields")

            return data

        except (json.JSONDecodeError, ValueError) as parse_err:
            emit(
                event="perception.warning",
                layer="perception",
                level="warn",
                data={
                    "msg": f"JSON parse failed ({parse_err})",
                    "raw_response": raw_content[:500],
                }
            )
            raise
