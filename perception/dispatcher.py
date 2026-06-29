"""
perception/dispatcher.py
────────────────────────
Batch action dispatcher for executing fast-path and slow-path actions.
"""

import asyncio
import json
import os

from perception.schemas import ScoredSignal
from triggers.triage import BatchTriageDecision, FastPathAction, SlowPathEscalation
from utils.log_writer import emit
from agents.tools import tools_registry
from utils.callbacks import ServiceRegistry


class BatchActionDispatcher:
    """
    Stateless dispatcher for executing batch triage decisions.
    Handles fast-path tool execution and slow-path trigger submission.
    """

    @staticmethod
    async def dispatch(
        signals: list[ScoredSignal],
        decision: BatchTriageDecision,
        world_model,
    ) -> None:
        """
        Act on a BatchTriageDecision:
          • fast_path_actions      → asyncio.gather (concurrent tool calls + Discord reply)
          • slow_path_escalations  → trigger_store.submit(source="perception")
          • unlisted signals       → WorldModel context (passive queue)
        """
        from triggers.store import trigger_store

        model = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")
        handled: set[int] = set()

        # ── Fast-path: run all actions concurrently ────────────────────────────
        async def _run_action(action: FastPathAction) -> None:
            idx = action.signal_index
            if idx < 0 or idx >= len(signals):
                emit("perception.warning", "perception",
                     {"msg": f"FastPathAction index {idx} out of range"}, level="warn")
                return
            sig = signals[idx]
            meta = sig.representative.metadata or {}
            channel_id = meta.get("channel_id") or sig.representative.channel_id

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

        if decision.fast_path_actions:
            for act in decision.fast_path_actions:
                handled.add(act.signal_index)
            results = await asyncio.gather(
                *[_run_action(act) for act in decision.fast_path_actions],
                return_exceptions=True,
            )
            # Log any exceptions that surfaced from individual actions
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    emit("perception.warning", "perception",
                         {"msg": f"Fast-path action {i} raised: {r}"}, level="warn")
            emit("perception.fast_path_batch_done", "perception",
                 {"count": len(decision.fast_path_actions)})

        # ── Slow-path: submit to trigger_store for GoalManager ─────────────────
        for esc in decision.slow_path_escalations:
            idx = esc.signal_index
            if idx < 0 or idx >= len(signals):
                continue
            handled.add(idx)
            sig = signals[idx]
            meta = sig.representative.metadata or {}
            channel_id = meta.get("channel_id") or sig.representative.channel_id
            message_id = meta.get("message_id")

            trigger_store.submit(
                source="perception",
                description=esc.refined_goal,
                model=model,
                metadata={
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "original_signal": sig.representative.text[:200],
                    "authority": sig.authority_score,
                },
            )
            emit("perception.slow_path_escalated", "perception",
                 {"index": idx, "goal": esc.refined_goal[:80]})

        # ── Remainder: passive WorldModel context ──────────────────────────────
        for i, sig in enumerate(signals):
            if i not in handled:
                await world_model.add_perception_context({
                    "text": sig.representative.text,
                    "source": sig.representative.source.identifier,
                    "channel": sig.representative.channel_id,
                    "authority": sig.authority_score,
                })
