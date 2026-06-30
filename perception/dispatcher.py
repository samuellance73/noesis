"""
perception/dispatcher.py
────────────────────────
Batch action dispatcher — executes triage decisions from the PerceptionLayer.

Fast-path actions   → concurrent asyncio.gather (tool call + Discord reply)
Slow-path escalations → asyncio.create_task(GoalManager.run_stream(...))
Unlisted signals    → WorldModel passive context (for the next GoalManager cycle)

Lazy run-directory creation (per spec Component 6)
───────────────────────────────────────────────────
A unique run_id is assigned to every dispatched task but the physical
logs/runs/{run_id}/ directory is NOT created here.  The GoalManager (and its
EpisodicMemoryAdapter) will create the directory only when it actually writes
a summary.json.  Fast-path tasks never touch the filesystem at all.

This prevents the 10-run pruning limit from discarding valuable strategic
planning folders in favour of empty fast-path placeholders.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from perception.schemas import ScoredSignal, PerceptionWorldModel
from triggers.triage import BatchTriageDecision, FastPathAction, SlowPathEscalation
from utils.log_writer import emit
from agents.tools import tools_registry
from utils.callbacks import ServiceRegistry
from agents.goal_manager import GoalManager
from core.model_router import ModelRouter


class BatchActionDispatcher:
    """
    Stateless dispatcher for executing batch triage decisions.
    All methods are static — no instance state.
    """

    @staticmethod
    async def dispatch(
        signals: list[ScoredSignal],
        decision: BatchTriageDecision,
        world_model: PerceptionWorldModel,
        router: ModelRouter,
    ) -> None:
        """
        Execute a BatchTriageDecision:
          • fast_path_actions      → asyncio.gather (concurrent tool calls + Discord reply)
          • slow_path_escalations  → asyncio.create_task(GoalManager) — non-blocking
          • unlisted signals       → WorldModel passive context queue
        """
        handled: set[int] = set()

        # ── Fast-path: run all actions concurrently ────────────────────────────
        if decision.fast_path_actions:
            for act in decision.fast_path_actions:
                handled.add(act.signal_index)

            results = await asyncio.gather(
                *[BatchActionDispatcher._run_fast_action(act, signals) for act in decision.fast_path_actions],
                return_exceptions=True,
            )
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    emit("perception.warning", "perception",
                         {"msg": f"Fast-path action {i} raised: {r}"}, level="warn")
            emit("perception.fast_path_batch_done", "perception",
                 {"count": len(decision.fast_path_actions)})

        # ── Slow-path: spawn GoalManager tasks ────────────────────────────────
        for esc in decision.slow_path_escalations:
            idx = esc.signal_index
            if idx < 0 or idx >= len(signals):
                emit("perception.warning", "perception",
                     {"msg": f"SlowPathEscalation index {idx} out of range ({len(signals)} signals)"}, level="warn")
                continue
            handled.add(idx)
            sig = signals[idx]

            # Assign a unique run_id.  DO NOT create the directory here —
            # GoalManager's EpisodicMemoryAdapter creates it lazily on first write.
            run_id = str(uuid.uuid4())

            goal_manager = GoalManager(router=router)
            asyncio.create_task(
                BatchActionDispatcher._run_goal_manager(goal_manager, esc.refined_goal, run_id),
                name=f"goal-manager-{run_id[:8]}",
            )

            emit("perception.slow_path_escalated", "perception",
                 {"index": idx, "goal": esc.refined_goal[:80], "run_id": run_id})

        # ── Remainder: passive WorldModel context ──────────────────────────────
        for i, sig in enumerate(signals):
            if i not in handled:
                await world_model.add_perception_context({
                    "text":      sig.representative.raw_content,
                    "source":    sig.representative.sender_identifier,
                    "channel":   sig.representative.target_conversation_identifier,
                    "authority": sig.authority_score,
                })

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    async def _run_fast_action(action: FastPathAction, signals: list[ScoredSignal]) -> None:
        """Execute one fast-path action: optional tool call + Discord reply."""
        idx = action.signal_index
        if idx < 0 or idx >= len(signals):
            emit("perception.warning", "perception",
                 {"msg": f"FastPathAction index {idx} out of range"}, level="warn")
            return
        sig = signals[idx]
        meta = sig.representative.metadata or {}
        channel_id = meta.get("channel_id") or sig.representative.target_conversation_identifier

        tool_output = ""
        if action.tool_name:
            emit("perception.fast_path_tool", "perception",
                 {"index": idx, "tool": action.tool_name,
                  "input": str(action.tool_input or "")[:80]})
            tool_output = await tools_registry.execute(
                action.tool_name, action.tool_input or ""
            )

        reply = action.final_answer
        if tool_output:
            reply = f"{reply}\n\n**Output:**\n{tool_output}" if reply else tool_output

        if channel_id and reply:
            try:
                await ServiceRegistry.call(
                    "send_discord_message",
                    json.dumps({"channel_id": int(channel_id), "message": reply}),
                )
            except Exception as exc:
                emit("perception.warning", "perception",
                     {"msg": f"Fast-path Discord send failed: {exc}"}, level="warn")

        emit("perception.fast_path_done", "perception",
             {"index": idx, "channel": channel_id, "reply_len": len(reply or "")})

    @staticmethod
    async def _run_goal_manager(manager: GoalManager, goal: str, run_id: str) -> None:
        """
        Drive a GoalManager to completion and absorb all streamed events.
        Errors are caught and logged; they must not crash the task silently.
        """
        try:
            async for event in manager.run_stream(goal, run_id=run_id):
                # Publish to the event bus so SSE clients / Discord bot see progress
                from utils.event_bus import event_bus
                await event_bus.publish(event)
        except Exception as exc:
            emit("perception.error", "perception",
                 {"msg": f"GoalManager task failed: {exc}", "run_id": run_id}, level="error")
