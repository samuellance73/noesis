"""
triggers/daemon.py
──────────────────
The background daemon loop — the brainless scheduler behind the unified
trigger architecture.

Responsibilities (and ONLY these):
  1. Wake up periodically (poll interval)
  2. Immediately wake when a human trigger arrives (fast-lane)
  3. Drain all pending triggers from TriggerStore
  4. Route each trigger through the TriageDispatcher, then to the right agent
  5. Publish all agent events to the EventBus so SSE clients receive them
  6. Update trigger status (processing → done/failed)
  7. Invoke SelfInitiativeEngine when the store is idle

Routing strategy
────────────────
  Every trigger → TriageDispatcher.evaluate()  →  decision

  Fast-Path (decision.can_solve_immediately == True):
      Execute 0 or 1 tool call(s) in-process, publish final_answer, done.
      GoalManager is NOT invoked.

  Slow-Path (decision.can_solve_immediately == False):
      Mutate trigger.description with decision.escalation_instructions,
      then route to GoalManager (human/perception) or AgentExecutor (all else).

Architecture
────────────
  TriggerStore (pending) ──► Daemon ──► TriageDispatcher
                                         ├──► Fast-Path: in-process tool exec ──► EventBus ──► SSE
                                         └──► Slow-Path: GoalManager / AgentExecutor ──► EventBus
                                    ↑
                         human_ready.Event (fast-lane, no sleep needed)

  Idle store → SelfInitiativeEngine → submit(source="agent") → TriageDispatcher
"""

import asyncio
import os

from triggers.store import Trigger, trigger_store
from triggers.triage import TriageDispatcher
from agents.executor import AgentExecutor
from agents.goal_manager import GoalManager
from agents.self_initiative import SelfInitiativeEngine
from core.model_router import ModelRouter
from utils.event_bus import event_bus
from utils.log_writer import emit

# GoalManager cycle cap for daemon-triggered human runs.
_DAEMON_MAX_CYCLES = 5

# Minimum wall-clock seconds per cycle — prevents the agent from hammering
# the API in rapid succession. Set to None to disable pacing.
_DAEMON_CYCLE_INTERVAL: float | None = 60.0

# AgentExecutor iteration cap for non-human (neutral/cron/webhook) triggers.
_DAEMON_MAX_ITERATIONS = 5

# Singleton self-initiative engine (shared across all daemon ticks).
_self_initiative = SelfInitiativeEngine()


async def _run_trigger(trigger: Trigger, router: ModelRouter) -> None:
    """
    Route one trigger through the TriageDispatcher, then to the appropriate
    agent or Fast-Path executor. Stream events to the bus and update trigger
    status when done.
    """
    from datetime import datetime
    import uuid
    from pathlib import Path
    import shutil

    # Generate run_id and create run directory immediately for all trigger types
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"
    runs_root = Path("logs") / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the 10 most recent runs, sorted by time order
    try:
        all_runs = sorted([d for d in runs_root.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
        for old_run in all_runs[10:]:
            shutil.rmtree(old_run, ignore_errors=True)
    except Exception:
        pass

    emit(
        event="daemon.trigger_dispatched",
        layer="daemon",
        data={
            "trigger_id": str(trigger.id),
            "source": trigger.source,
            "description": trigger.description[:80],
        },
        run_id=run_id,
    )

    # Announce to SSE clients that this trigger is now being processed
    await event_bus.publish({
        "event":       "trigger_started",
        "trigger_id":  str(trigger.id),
        "source":      trigger.source,
        "description": trigger.description,
    })

    try:
        # ── Triage: Fast-Path vs Slow-Path ─────────────────────────────────
        decision = await TriageDispatcher.evaluate(trigger, router)

        if decision.can_solve_immediately:
            await _run_fast_path(trigger, decision, run_id)
        else:
            # Mutate description with the refined escalation plan before routing
            if decision.escalation_instructions:
                trigger.description = decision.escalation_instructions

            if trigger.source in ("human", "perception", "agent"):
                await _run_as_goal_manager(trigger, router, run_id)
            else:
                await _run_as_executor(trigger, router, run_id)
        # ───────────────────────────────────────────────────────────────

        trigger_store.mark_done(trigger.id)
        emit("daemon.trigger_completed", "daemon", {"trigger_id": str(trigger.id)}, run_id=run_id)
        await _update_discord_reaction(trigger, "✅")

    except Exception as e:
        error_msg = str(e)
        trigger_store.mark_failed(trigger.id, error=error_msg)
        emit("daemon.trigger_failed", "daemon", {"trigger_id": str(trigger.id), "error": error_msg}, level="error", run_id=run_id)
        await event_bus.publish({
            "event":      "trigger_failed",
            "trigger_id": str(trigger.id),
            "message":    error_msg,
        })
        await _update_discord_reaction(trigger, "❌")



async def _run_fast_path(trigger: Trigger, decision, run_id: str) -> None:
    """
    Fast-Path: execute 0 or 1 tool call(s) in-process, then publish the
    concatenated result as a final_answer event.  GoalManager is bypassed.
    """
    from agents.tools import tools_registry  # local import keeps circular deps clean

    # Emit strategic.loop_started for dashboard consistency
    emit("strategic.loop_started", "strategic", {"goal": trigger.description}, run_id=run_id)

    output_parts: list[str] = []

    for tool_call in decision.immediate_tool_calls:
        emit(
            "daemon.fast_path_tool",
            "daemon",
            {
                "trigger_id": str(trigger.id),
                "tool": tool_call.tool_name,
                "input": str(tool_call.tool_input)[:120],
            },
            run_id=run_id,
        )
        # Log tactical events for dashboard/log analyzers
        emit("tactical.tool_call", "tactical", {"tool": tool_call.tool_name, "input": str(tool_call.tool_input)}, run_id=run_id)
        result = await tools_registry.execute(tool_call.tool_name, tool_call.tool_input)
        emit("tactical.tool_result", "tactical", {"tool": tool_call.tool_name, "result": str(result)[:80]}, run_id=run_id)
        output_parts.append(result)

    if decision.final_answer:
        output_parts.append(decision.final_answer)

    final_text = "\n".join(output_parts).strip() or "(no output)"

    await event_bus.publish({
        "event":          "final_answer",
        "trigger_id":     str(trigger.id),
        "trigger_source": trigger.source,
        "trigger_metadata": trigger.metadata or {},
        "answer":         final_text,
        "task_goal":      trigger.description[:80],
        "fast_path":      True,
    })
    # Log tactical.final_answer for dashboard/log analyzers
    emit("tactical.final_answer", "tactical", {"answer": final_text}, run_id=run_id)
    emit(
        "daemon.fast_path_complete",
        "daemon",
        {"trigger_id": str(trigger.id), "answer_len": len(final_text)},
        run_id=run_id,
    )


async def _run_as_goal_manager(trigger: Trigger, router: ModelRouter, run_id: str) -> None:
    """
    Human-operator path: full GoalManager loop with sub-task decomposition,
    multi-cycle reasoning, and human-readable run logs under logs/runs/.
    """
    manager = GoalManager(
        router=router,
        max_cycles=_DAEMON_MAX_CYCLES,
        cycle_interval_seconds=_DAEMON_CYCLE_INTERVAL,
    )
    async for event in manager.run_stream(trigger.description, run_id=run_id):
        event["trigger_id"]       = str(trigger.id)
        event["trigger_source"]   = trigger.source
        event["trigger_metadata"] = trigger.metadata or {}
        await event_bus.publish(event)


async def _run_as_executor(trigger: Trigger, router: ModelRouter, run_id: str) -> None:
    """
    Non-human path: lightweight single-turn AgentExecutor — no multi-cycle
    overhead, no run logs written to disk.
    """
    from agents.schemas import AgentState

    # Emit strategic.loop_started for dashboard consistency
    emit("strategic.loop_started", "strategic", {"goal": trigger.description}, run_id=run_id)

    executor = AgentExecutor(
        router=router,
        task_label=f"trigger-{str(trigger.id)[:8]}",
    )
    executor.state = AgentState(max_iterations=_DAEMON_MAX_ITERATIONS)
    async for event in executor.run_generator(trigger.description):
        event["trigger_id"]       = str(trigger.id)
        event["trigger_source"]   = trigger.source
        event["trigger_metadata"] = trigger.metadata or {}
        await event_bus.publish(event)

        # Log tactical events for dashboard/log analyzers
        ev = event.get("event")
        if ev == "thought":
            emit("tactical.thought", "tactical", {"thought": event.get("thought", ""), "iteration": event.get("step_index", 0) + 1}, run_id=run_id)
        elif ev == "tool_start":
            emit("tactical.tool_call", "tactical", {"tool": event.get("tool_name", ""), "input": str(event.get("tool_input", ""))}, run_id=run_id)
        elif ev == "tool_observation":
            emit("tactical.tool_result", "tactical", {"tool": event.get("tool_name", ""), "result": str(event.get("observation", ""))[:80]}, run_id=run_id)
        elif ev == "final_answer":
            emit("tactical.final_answer", "tactical", {"answer": str(event.get("answer", ""))}, run_id=run_id)
        elif ev == "error":
            emit("tactical.error", "tactical", {"msg": event.get("message", "")}, level="error", run_id=run_id)


async def _update_discord_reaction(trigger: Trigger, emoji: str) -> None:
    """Helper to update a message reaction on Discord when a trigger finishes."""
    # Allow both direct discord executor tasks and perception manager tasks to clear reactions
    if trigger.source not in ("discord", "perception") or not trigger.metadata:
        return
    channel_id = trigger.metadata.get("channel_id")
    message_id = trigger.metadata.get("message_id")
    if not channel_id or not message_id:
        return

    from utils.callbacks import ServiceRegistry
    try:
        await ServiceRegistry.call("update_discord_reaction", int(channel_id), int(message_id), emoji)
    except Exception as e:
        emit("daemon.warning", "daemon", {"msg": f"Failed to update Discord reaction callback: {e}"}, level="warn")



async def _process_batch(router: ModelRouter) -> None:
    """Drain pending triggers and run them all in parallel.

    If the store is empty after draining, offer the SelfInitiativeEngine a
    chance to submit a new autonomous goal for the next tick.
    """
    pending = trigger_store.get_pending()
    if not pending:
        # Store is idle — give the self-initiative engine a chance to act
        if os.getenv("SELF_INITIATIVE_ENABLED", "true").lower() == "true":
            submitted = await _self_initiative.maybe_submit(router)
            if submitted:
                emit("daemon.self_initiative_triggered", "daemon", {})
        return
    emit("daemon.batch_processing", "daemon", {"count": len(pending)})
    await asyncio.gather(*[_run_trigger(t, router) for t in pending])



async def start_daemon(
    router: ModelRouter,
    interval_seconds: int = 60,
) -> None:
    """
    Main daemon loop. Runs forever as a background asyncio task.

    Fast-lane: human triggers fire immediately (no sleep) because the
    TriggerStore sets `human_ready` when a human trigger is submitted.

    Poll interval: all other trigger sources are picked up on the next
    regular tick.

    To start this, call:
        asyncio.create_task(start_daemon(router=app.state.model_router))
    """
    emit(
        event="daemon.started",
        layer="daemon",
        data={"interval_seconds": interval_seconds}
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
                emit("daemon.woke", "daemon", {"reason": "human_fast_lane"}, level="debug")
            except asyncio.TimeoutError:
                emit("daemon.woke", "daemon", {"reason": "poll_interval"}, level="debug")

            await _process_batch(router)

        except asyncio.CancelledError:
            emit("daemon.cancelled", "daemon", {})
            raise  # re-raise so the task terminates properly
        except Exception as e:
            # Log but never crash the daemon — keep polling
            emit("daemon.error", "daemon", {"msg": f"Unexpected error in main loop: {e}"}, level="error")
            await asyncio.sleep(5)  # brief back-off before retry
