"""
utils/daemon.py
───────────────
The background daemon loop — the brainless scheduler behind the unified
trigger architecture.

Responsibilities (and ONLY these):
  1. Wake up periodically (poll interval)
  2. Immediately wake when a human trigger arrives (fast-lane)
  3. Drain all pending triggers from TriggerStore
  4. For each trigger: launch GoalManager.run_stream() as a background task
  5. Publish all GoalManager events to the EventBus so SSE clients receive them
  6. Update trigger status (processing → done/failed)

The daemon has ZERO intelligence. It does not decide what to do — that is
GoalManager's job. It only moves triggers from the queue into GoalManager.

Architecture
────────────
  TriggerStore (pending) ──► Daemon ──► GoalManager ──► EventBus ──► SSE clients
                                   ↑
                         human_ready.Event (fast-lane, no sleep needed)
"""

import asyncio
import logging
from uuid import UUID

from agents.trigger_store import Trigger, trigger_store
from agents.goal_manager import GoalManager
from integrations.llm.service import UpstreamService
from utils.event_bus import event_bus

logger = logging.getLogger("noesis.daemon")

# Daemon runs the GoalManager with a tighter cycle cap than interactive use.
# A triggered task is expected to be concrete enough that 1–3 cycles suffice.
_DAEMON_MAX_CYCLES = 3


async def _run_trigger(trigger: Trigger, service: UpstreamService) -> None:
    """
    Run one trigger through a GoalManager, publishing all events to the bus.
    Updates trigger status when done.
    """
    logger.info(
        "[Daemon] Starting trigger id=%s source=%s: %r",
        trigger.id, trigger.source, trigger.description[:80],
    )

    # Announce to SSE clients that this trigger is now being processed
    await event_bus.publish({
        "event":       "trigger_started",
        "trigger_id":  str(trigger.id),
        "source":      trigger.source,
        "description": trigger.description,
    })

    manager = GoalManager(
        llm_service=service,
        model=trigger.model,
        max_cycles=_DAEMON_MAX_CYCLES,
    )

    try:
        async for event in manager.run_stream(trigger.description):
            # Tag every event with the trigger that produced it so the frontend
            # can correlate events to the right trigger card.
            event["trigger_id"] = str(trigger.id)
            event["trigger_source"] = trigger.source
            await event_bus.publish(event)

        trigger_store.mark_done(trigger.id)
        logger.info("[Daemon] Trigger %s completed.", trigger.id)

    except Exception as e:
        error_msg = str(e)
        trigger_store.mark_failed(trigger.id, error=error_msg)
        logger.error("[Daemon] Trigger %s failed: %s", trigger.id, error_msg, exc_info=True)
        await event_bus.publish({
            "event":      "trigger_failed",
            "trigger_id": str(trigger.id),
            "message":    error_msg,
        })


async def _process_batch(service: UpstreamService) -> None:
    """Drain pending triggers and run them all in parallel."""
    pending = trigger_store.get_pending()
    if not pending:
        return
    logger.info("[Daemon] Processing batch of %d trigger(s).", len(pending))
    await asyncio.gather(*[_run_trigger(t, service) for t in pending])


async def start_daemon(
    service: UpstreamService,
    interval_seconds: int = 60,
) -> None:
    """
    Main daemon loop. Runs forever as a background asyncio task.

    Fast-lane: human triggers fire immediately (no sleep) because the
    TriggerStore sets `human_ready` when a human trigger is submitted.

    Poll interval: all other trigger sources are picked up on the next
    regular tick.

    To start this, call:
        asyncio.create_task(start_daemon(service=app.state.upstream_service))
    """
    logger.info(
        "[Daemon] Starting. Poll interval=%ds. Human triggers fire immediately.",
        interval_seconds,
    )

    # Small boot delay to let FastAPI fully initialize before first run
    await asyncio.sleep(2)

    while True:
        try:
            # Wait for EITHER the poll interval OR a human trigger, whichever
            # comes first. This is the fast-lane mechanism.
            try:
                await asyncio.wait_for(
                    trigger_store.human_ready.wait(),
                    timeout=interval_seconds,
                )
                # Clear the flag so the next human trigger can re-arm it
                trigger_store.human_ready.clear()
                logger.debug("[Daemon] Woke on human trigger fast-lane.")
            except asyncio.TimeoutError:
                logger.debug("[Daemon] Woke on poll interval.")

            await _process_batch(service)

        except asyncio.CancelledError:
            logger.info("[Daemon] Cancelled — shutting down cleanly.")
            raise  # re-raise so the task terminates properly
        except Exception as e:
            # Log but never crash the daemon — keep polling
            logger.error("[Daemon] Unexpected error in main loop: %s", e, exc_info=True)
            await asyncio.sleep(5)  # brief back-off before retry
