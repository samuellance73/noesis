"""
perception/stages/router.py
────────────────────────────
Stage 5 — Router

Receives LLM decisions for scored signals and routes each signal to the
appropriate downstream consumer based on the "action" field from the LLM.

Routing logic
─────────────
  action="interrupt" → Submit to trigger_store and wake GoalManager immediately
  action="queue"     → Add to WorldModel context for next GoalManager cycle
  action="drop"      → Ignore entirely

Notes
─────
  • The WorldModel interface is injected at construction time as a duck-typed
    async callable. This keeps the Router decoupled from concrete GoalManager
    implementations.
  • Routing decisions are logged at INFO level for the perception.log file..
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from perception.schemas import ScoredSignal
from utils.log_writer import emit


@runtime_checkable
class WorldModelProtocol(Protocol):
    async def add_perception_context(self, context: dict) -> None: ...


class Router:
    """
    Routes signals based on LLM decisions to the WorldModel and/or trigger_store.

    Parameters
    ──────────
    world_model   : WorldModelProtocol   — receives context the GoalManager
                    should be aware of on its next cycle.
    """

    def __init__(
        self,
        world_model: WorldModelProtocol,
        reactive_pool=None,
    ) -> None:
        self.world_model = world_model
        self._reactive_pool = reactive_pool  # Kept for compatibility but not used in new pipeline

    async def route(self, signals: list[ScoredSignal], decisions: list[dict]) -> None:
        """Route all signals based on LLM decisions."""
        for i, decision in enumerate(decisions):
            if i >= len(signals):
                emit("perception.warning", "perception", {"msg": f"Router: decision index {i} out of range for signals list"}, level="warn")
                continue
            
            signal = signals[i]
            await self._route_one(signal, decision)

    async def _route_one(self, signal: ScoredSignal, decision: dict) -> None:
        action = decision.get("action", "queue")
        priority = decision.get("priority", "medium")
        summary = decision.get("summary", signal.representative.text[:100])
        reason = decision.get("reason", "")

        # ── DROP: ignore entirely ───────────────────────────────────────────────
        if action == "drop":
            emit(
                event="perception.routed",
                layer="perception",
                data={
                    "action": "drop",
                    "signal_id": signal.id,
                    "authority": signal.authority_score,
                    "reason": reason,
                }
            )
            return

        # ── INTERRUPT: wake GoalManager immediately ───────────────────────────────
        if action == "interrupt":
            from triggers.store import trigger_store
            model = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")
            
            sig_meta = signal.representative.metadata or {}
            # Use the full text (which includes context) rather than just the current message
            description = signal.representative.text if signal.representative.text else summary

            trigger_store.submit(
                description=description,
                source="perception",
                model=model,
                metadata={
                    "original_signal": signal.representative.text,
                    "channel_id": sig_meta.get("channel_id") or signal.representative.channel_id,
                    "message_id": sig_meta.get("message_id"),
                    "channel": signal.representative.channel_id,
                    "authority": signal.authority_score,
                    "priority": priority,
                    "llm_summary": summary,
                }
            )
            trigger_store.human_ready.set()
            
            emit(
                event="perception.routed",
                layer="perception",
                data={
                    "action": "interrupt",
                    "signal_id": signal.id,
                    "authority": signal.authority_score,
                    "priority": priority,
                    "summary": summary[:120],
                }
            )
            return

        # ── QUEUE: add to WorldModel context for next GoalManager cycle ─────────
        if action == "queue":
            await self.world_model.add_perception_context({
                "text": signal.representative.text,
                "source": signal.representative.source.identifier,
                "channel": signal.representative.channel_id,
                "summary": summary,
                "priority": priority,
                "authority": signal.authority_score,
            })
            
            emit(
                event="perception.routed",
                layer="perception",
                data={
                    "action": "queue",
                    "signal_id": signal.id,
                    "authority": signal.authority_score,
                    "priority": priority,
                    "summary": summary[:120],
                }
            )
            return

        # ── Unknown action: default to queue ───────────────────────────────────────
        emit(
            event="perception.warning",
            layer="perception",
            level="warn",
            data={
                "msg": f"Router: Unknown action '{action}' for signal_id={signal.id}, defaulting to queue"
            }
        )
        await self.world_model.add_perception_context({
            "text": signal.representative.text,
            "source": signal.representative.source.identifier,
            "channel": signal.representative.channel_id,
            "summary": summary,
            "priority": priority,
            "authority": signal.authority_score,
        })
