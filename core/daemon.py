"""
core/daemon.py
──────────────
The Unified Daemon (Component 3) — the single infinite background loop.

Architecture
────────────
  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐
  │   Feeder Task    │    │ Processing Task   │    │ Self-Initiative Task    │
  │                  │    │                   │    │                         │
  │ ingest_queue     │    │ IntakeBuffer      │    │ Polls for system idle;  │
  │  ──► feed()      │    │  ──► process_batch│    │ submits autonomous goals│
  └──────────────────┘    └──────────────────┘    └─────────────────────────┘

The Feeder Task and Processing Task run forever in parallel.  The Self-Initiative
Task wakes every CHECK_INTERVAL_SECONDS and, if the system is idle and the
engine's built-in cooldown has elapsed, fires SelfInitiativeEngine.run().

Accessing _reactive_pool through the PerceptionLayer's private attribute is
intentional: the daemon owns the pool lifecycle (start/stop) while the
PerceptionLayer owns the pool reference.
"""

from __future__ import annotations

import asyncio
import os

from core.events import UnifiedIngestEvent
from core.queues import ingest_queue
from perception.config import PerceptionConfig
from perception.layer import PerceptionLayer
from perception.schemas import PerceptionWorldModel
from perception.stages.intake import IntakeBuffer
from core.model_router import ModelRouter
from agents.self_initiative import SelfInitiativeEngine
from utils.log_writer import emit

# How often the self-initiative task wakes to check system idleness (seconds).
_SI_CHECK_INTERVAL: float = float(os.getenv("SELF_INITIATIVE_CHECK_INTERVAL", "10"))


async def _feeder_task(intake_buffer: IntakeBuffer) -> None:
    """
    Feeder Task — continuously drains ingest_queue into the IntakeBuffer.
    Never raises; errors are logged and the loop continues.
    """
    while True:
        try:
            event: UnifiedIngestEvent = await ingest_queue.get()
            await intake_buffer.feed(event)
            ingest_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            emit("daemon.error", "daemon",
                 {"msg": f"Feeder task error: {exc}"}, level="error")


async def _processing_task(
    perception_layer: PerceptionLayer,
    intake_buffer: IntakeBuffer,
) -> None:
    """
    Processing Task — waits on IntakeBuffer.drain(), then runs the full
    perception pipeline on the flushed batch.
    """
    while True:
        try:
            batch = await intake_buffer.drain()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            emit("daemon.error", "daemon",
                 {"msg": f"IntakeBuffer drain error: {exc}"}, level="error")
            await asyncio.sleep(1.0)  # brief back-off to avoid tight error loops
            continue

        if not batch:
            continue

        try:
            await perception_layer.process_batch(batch)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            emit("daemon.error", "daemon",
                 {"msg": f"process_batch error: {exc}"}, level="error")


async def _self_initiative_task(
    router: ModelRouter,
    intake_buffer: IntakeBuffer,
) -> None:
    """
    Self-Initiative Task (Component 7).

    Wakes every _SI_CHECK_INTERVAL seconds.  When both the ingest_queue and
    IntakeBuffer are empty (system is genuinely idle) it delegates to
    SelfInitiativeEngine.run(), which enforces its own cooldown timer so this
    task never needs to sleep an extra 60 seconds.
    """
    engine = SelfInitiativeEngine()
    while True:
        try:
            await asyncio.sleep(_SI_CHECK_INTERVAL)

            # System idle check
            if ingest_queue.empty() and intake_buffer.size == 0:
                submitted = await engine.run(router, intake_buffer)
                if submitted:
                    emit("daemon.self_initiative_triggered", "daemon", {})
        except asyncio.CancelledError:
            break
        except Exception as exc:
            emit("daemon.error", "daemon",
                 {"msg": f"Self-initiative task error: {exc}"}, level="error")


async def start_daemon(
    config: PerceptionConfig,
    router: ModelRouter,
    world_model: PerceptionWorldModel | None = None,
    executor_factory=None,
) -> None:
    """
    Unified daemon entry-point.  Called once from app/lifespan.py as an
    asyncio.create_task().

    Starts three concurrent tasks:
      1. Feeder Task         — ingest_queue → IntakeBuffer
      2. Processing Task     — IntakeBuffer.drain() → PerceptionLayer.process_batch()
      3. Self-Initiative Task — idle detection → SelfInitiativeEngine
    """
    emit("daemon.started", "daemon", {"msg": "Unified daemon starting…"})

    # Build the shared IntakeBuffer and wire up the PerceptionLayer
    intake_buffer = IntakeBuffer(
        window_seconds=config.intake_window_seconds,
        max_buffer_size=config.intake_max_buffer_size,
    )
    perception_layer = PerceptionLayer(
        config=config,
        router=router,
        intake_buffer=intake_buffer,
        world_model=world_model,
        executor_factory=executor_factory,
    )

    # Start the ReactivePool background workers
    await perception_layer._reactive_pool.start()

    feeder     = asyncio.create_task(_feeder_task(intake_buffer),            name="daemon-feeder")
    processing = asyncio.create_task(
        _processing_task(perception_layer, intake_buffer),                   name="daemon-processing",
    )
    si         = asyncio.create_task(_self_initiative_task(router, intake_buffer), name="daemon-self-initiative")

    try:
        await asyncio.gather(feeder, processing, si)
    except asyncio.CancelledError:
        emit("daemon.cancelled", "daemon", {"msg": "Unified daemon cancelled."})
    finally:
        feeder.cancel()
        processing.cancel()
        si.cancel()
        await asyncio.gather(feeder, processing, si, return_exceptions=True)
        await perception_layer._reactive_pool.stop()
        emit("daemon.stopped", "daemon", {"msg": "Unified daemon stopped."})
