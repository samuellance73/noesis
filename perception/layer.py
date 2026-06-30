"""
perception/layer.py
────────────────────
PerceptionLayer — top-level orchestrator wiring all pipeline stages.

Architecture (v2)
─────────────────
  while True:
      batch = await intake_buffer.drain()          # blocks until 3 messages or 60s
      if not batch: continue
      scored   = authority_scorer.score_batch(batch)   # Stage 2
      decision = await TriageDispatcher.evaluate_batch(scored, router)  # Stage 3
      await BatchActionDispatcher.dispatch(scored, decision, world_model, router)
          ├─ fast_path_actions      → asyncio.gather (concurrent tool calls + Discord reply)
          └─ slow_path_escalations  → asyncio.create_task(GoalManager.run_stream())

One STANDARD-tier LLM call handles the full batch: classify, triage, and generate
responses for all signals simultaneously.  Complex tasks are run as parallel
asyncio tasks via GoalManager — no polling, no TriggerStore.

Error isolation: each stage is wrapped in try/except; a crashed stage never
halts the pipeline loop.
"""


from __future__ import annotations

import asyncio
import json
import os

from perception.config import PerceptionConfig
from perception.reactive_pool import ReactivePool
from perception.dispatcher import BatchActionDispatcher
from utils.prompts import load_prompt
from perception.schemas import (
    PerceptionWorldModel,
    ScoredSignal,
)
from perception.stages.authority import AuthorityScorer
from perception.stages.intake import IntakeBuffer
from datetime import datetime
from core.model_router import ModelRouter, ModelRequest, ModelTier
from triggers.triage import BatchTriageDecision, FastPathAction, SlowPathEscalation, TriageDispatcher
from utils.log_writer import emit
from utils.json_parser import _clean_llm_json

from core.events import UnifiedIngestEvent
from core.queues import ingest_queue


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
        intake_buffer: IntakeBuffer,  # New: IntakeBuffer instance passed in
        world_model: PerceptionWorldModel | None = None,
        executor_factory=None,
    ) -> None:
        self.config = config
        self._intake = intake_buffer  # Assign the passed-in IntakeBuffer

        # ── Stage 2: Authority scorer ──────────────────────────────────────────
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

    # ── Public API ─────────────────────────────────────────────────────────────

    async def ingest(self, event: UnifiedIngestEvent) -> None:
        """
        Accept a UnifiedIngestEvent from any external source (Discord, CLI, API, cron).
        Thread-safe — may be called from any coroutine.
        """
        await ingest_queue.put(event)
        emit(
            event="perception.ingested",
            layer="perception",
            level="debug",
            data={
                "event_identifier": event.event_identifier,
                "source_channel": event.source_channel,
                "priority_level": event.priority_level.value,
                "raw_content": event.raw_content[:80],
            }
        )

    async def start(self) -> None:
        """Placeholder for compatibility. ReactivePool started by daemon."""
        emit("perception.started", "perception", {"msg": "PerceptionLayer initialized, awaiting daemon start."})

    async def stop(self) -> None:
        """Placeholder for compatibility. ReactivePool stopped by daemon."""
        emit("perception.stopped", "perception", {"msg": "PerceptionLayer stopped, assuming daemon handles ReactivePool shutdown."})

    @property
    def world_model(self) -> PerceptionWorldModel:
        """Access the WorldModel facade for the GoalManager to drain."""
        return self._world_model

    async def process_batch(self, batch: list[UnifiedIngestEvent]) -> None:
        """Process one batch through the pipeline with per-stage error isolation."""
        raw_count = len(batch)
        emit(
            event="perception.batch_received",
            layer="perception",
            data={
                "raw_count": raw_count,
                "window": self.config.intake_window_seconds,
            }
        )

        # ── Stage 2: Authority scoring ─────────────────────────────────────────
        try:
            scored = self._authority.score_batch(batch) # AuthorityScorer now expects UnifiedIngestEvent
        except Exception as exc:
            emit("perception.error", "perception", {"msg": f"AuthorityScorer crashed: {exc}"}, level="error")
            scored = [
                ScoredSignal(
                    representative=event,
                    frequency=1,
                    perception_type=None,
                    authority_score=0.5,
                )
                for event in batch
            ]

        # ── Stage 3+4: Batch Triage → Concurrent Execution ────────────────────
        try:
            decision = await TriageDispatcher.evaluate_batch(scored, self._router_llm)
        except Exception as exc:
            emit("perception.warning", "perception",
                 {"msg": f"Batch triage failed, queuing all signals as context: {exc}"}, level="warn")
            for sig in scored:
                # Triage failed: preserve content in the WorldModel for the next planning cycle
                await self._world_model.add_perception_context({
                    "text":      sig.representative.raw_content,
                    "source":    sig.representative.sender_identifier,
                    "channel":   sig.representative.target_conversation_identifier,
                    "authority": sig.authority_score,
                })
            return

        # Hand off execution to the dispatcher
        await BatchActionDispatcher.dispatch(scored, decision, self._world_model, self._router_llm)

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
            timestamp = int(signal.representative.monotonic_timestamp)
            source = signal.representative.sender_identifier
            channel = signal.representative.target_conversation_identifier or "unknown"
            authority = signal.authority_score
            text = signal.representative.raw_content

            signal_block_lines.append(
                f"[{i}] source={source} channel={channel} authority={authority:.2f} time={timestamp}\n"
                f'    "{text}"'
            )

        signal_block = "\n\n".join(signal_block_lines)

        system_prompt = load_prompt("triage_batch_system.txt")

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
