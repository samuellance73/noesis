"""
triggers/triage.py
──────────────────
TriageDispatcher — stateless middleware router.

Two evaluation modes
────────────────────
1. evaluate(trigger, router)          — single trigger (daemon path)
   Returns TriageDecision.

2. evaluate_batch(signals, router)    — batch path (perception pipeline)
   Accepts the full list of ScoredSignals, fires ONE STANDARD-tier LLM call,
   and returns a BatchTriageDecision mapping each signal index to:
     • FastPathAction      — execute ≤1 tool call concurrently, reply to channel
     • SlowPathEscalation  — create a trigger_store entry for the GoalManager

Batch execution (in perception/layer.py)
─────────────────────────────────────────
  fast_path_actions   → asyncio.gather (parallel tool execution)
  slow_path_escalations → trigger_store.submit(source="perception")
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field

from agents.schemas import ToolCall
from agents.tools import tools_registry
from core.model_router import ModelRequest, ModelRouter, ModelTier
from triggers.store import Trigger
from utils.log_writer import emit


# ── Single-trigger schema ─────────────────────────────────────────────────────

class TriageDecision(BaseModel):
    """Routing decision for one daemon trigger."""
    can_solve_immediately: bool = Field(
        ...,
        description=(
            "True ONLY if the task needs zero tool calls OR exactly one tool call."
        ),
    )
    rationale: str = Field(..., description="One sentence explaining the routing choice.")
    immediate_tool_calls: list[ToolCall] = Field(default_factory=list)
    final_answer: Optional[str] = Field(None)
    escalation_instructions: Optional[str] = Field(None)


# ── Batch schemas ─────────────────────────────────────────────────────────────

class FastPathAction(BaseModel):
    """
    A signal resolved immediately (≤1 tool call).
    All FastPathActions are executed concurrently via asyncio.gather.
    """
    signal_index: int = Field(..., description="0-based index into the batch list.")
    tool_name: Optional[str] = Field(None, description="Tool to call, or null.")
    tool_input: Optional[str] = Field(None, description="Input for the tool.")
    final_answer: str = Field(..., description="Response to send back to the user/channel.")


class SlowPathEscalation(BaseModel):
    """A signal that needs multi-step GoalManager planning."""
    signal_index: int = Field(..., description="0-based index into the batch list.")
    refined_goal: str = Field(
        ...,
        description="Numbered, step-by-step breakdown for the GoalManager.",
    )


class BatchTriageDecision(BaseModel):
    """
    One LLM call partitions an entire signal batch into fast-path actions
    (concurrent execution) and slow-path escalations (GoalManager triggers).
    Signals omitted from both lists are silently dropped.
    """
    rationale: str = Field(..., description="Strategic overview of the batch routing.")
    fast_path_actions: list[FastPathAction] = Field(default_factory=list)
    slow_path_escalations: list[SlowPathEscalation] = Field(default_factory=list)


# ── Prompt builders ───────────────────────────────────────────────────────────

def _tools_block() -> str:
    lines = [
        f"  • {name}: {getattr(func, 'description', 'No description.')}"
        for name, func in tools_registry.tools.items()
    ]
    return "\n".join(lines) or "  (no tools registered)"


def _build_single_system_prompt() -> str:
    return f"""You are the TriageDispatcher, a stateless routing middleware for an autonomous AI agent.
Evaluate the incoming task and return ONE JSON routing decision.

AVAILABLE TOOLS:
{_tools_block()}

FAST-PATH (can_solve_immediately=true): zero or exactly one tool call needed.
  → Populate immediate_tool_calls (0 or 1 items) and final_answer.

SLOW-PATH (can_solve_immediately=false): >1 tool calls, multi-step, or uncertain scope.
  → Populate escalation_instructions (numbered steps). Leave immediate_tool_calls=[].

OUTPUT — ONLY this JSON, no markdown:
{{
  "can_solve_immediately": true|false,
  "rationale": "<one sentence>",
  "immediate_tool_calls": [{{"tool_name":"<name>","tool_input":"<input>"}}],
  "final_answer": "<string or null>",
  "escalation_instructions": "<string or null>"
}}"""


def _build_batch_system_prompt() -> str:
    return f"""You are the TriageDispatcher, batch routing middleware for an autonomous AI agent.
Analyze ALL signals as a coherent group and return ONE BatchTriageDecision JSON.

AVAILABLE TOOLS:
{_tools_block()}

FAST-PATH → fast_path_actions: signal needs 0 or 1 tool call.
  If signals are contextually related, merge into a single action with one shared tool call.

SLOW-PATH → slow_path_escalations: >1 tool calls or multi-step reasoning needed.
  Provide refined_goal with numbered steps for the GoalManager.

DROP: omit noisy signals (very short text, punctuation only) from both lists.

OUTPUT — ONLY this JSON, no markdown:
{{
  "rationale": "<batch overview>",
  "fast_path_actions": [
    {{"signal_index":<int>,"tool_name":"<name|null>","tool_input":"<input|null>","final_answer":"<reply>"}}
  ],
  "slow_path_escalations": [
    {{"signal_index":<int>,"refined_goal":"<numbered steps>"}}
  ]
}}"""


# ── TriageDispatcher ──────────────────────────────────────────────────────────

class TriageDispatcher:
    """
    Stateless middleware — two static methods, no instance state.

    evaluate()       → single daemon trigger
    evaluate_batch() → perception pipeline signal batch
    """

    @staticmethod
    async def evaluate(trigger: Trigger, router: ModelRouter) -> TriageDecision:
        """Evaluate one daemon trigger. Always returns a valid TriageDecision."""
        emit("triage.evaluating", "triage",
             {"trigger_id": str(trigger.id), "source": trigger.source,
              "description": trigger.description[:120]}, level="debug")

        request = ModelRequest(
            tier=ModelTier.STANDARD,
            messages=[{"role": "user", "content":
                       f"Task to triage:\n\n{trigger.description}\n\nReturn your JSON routing decision now."}],
            system=_build_single_system_prompt(),
            component="triage_dispatcher.single",
        )
        try:
            response = await router.complete(request)
            decision = TriageDispatcher._parse(response.content, TriageDecision)
        except Exception as exc:
            emit("triage.error", "triage",
                 {"trigger_id": str(trigger.id), "error": str(exc)}, level="error")
            decision = TriageDecision(
                can_solve_immediately=False,
                rationale=f"Triage failed ({exc!s}); defaulting to Slow-Path.",
                escalation_instructions=trigger.description,
            )

        emit("triage.decision", "triage",
             {"trigger_id": str(trigger.id), "fast_path": decision.can_solve_immediately,
              "rationale": decision.rationale})
        return decision

    @staticmethod
    async def evaluate_batch(
        signals: list,   # list[ScoredSignal] — late import avoids circular dep
        router: ModelRouter,
    ) -> BatchTriageDecision:
        """
        Evaluate a full batch of ScoredSignals in one STANDARD-tier LLM call.
        Falls back to slow-pathing all signals on LLM/parse failure.
        """
        if not signals:
            return BatchTriageDecision(rationale="Empty batch.")

        lines = []
        for i, sig in enumerate(signals):
            rep = sig.representative
            channel = rep.target_conversation_identifier or "unknown"
            lines.append(
                f"[{i}] source={rep.sender_identifier} channel={channel} "
                f"authority={sig.authority_score:.2f}\n"
                f'    "{rep.raw_content[:400]}"'
            )

        user_msg = (
            f"SIGNAL BATCH ({len(signals)} signals):\n\n"
            + "\n\n".join(lines)
            + "\n\nReturn your BatchTriageDecision JSON now."
        )

        emit("triage.batch_evaluating", "triage", {"signal_count": len(signals)})

        request = ModelRequest(
            tier=ModelTier.STANDARD,
            messages=[{"role": "user", "content": user_msg}],
            system=_build_batch_system_prompt(),
            component="triage_dispatcher.batch",
        )
        try:
            response = await router.complete(request)
            decision = TriageDispatcher._parse(response.content, BatchTriageDecision)
        except Exception as exc:
            emit("triage.batch_error", "triage", {"error": str(exc)}, level="error")
            decision = BatchTriageDecision(
                rationale=f"Batch triage failed ({exc!s}); escalating all signals.",
                slow_path_escalations=[
                    SlowPathEscalation(
                        signal_index=i,
                        refined_goal=sig.representative.raw_content,
                    )
                    for i, sig in enumerate(signals)
                ],
            )

        emit("triage.batch_decision", "triage",
             {"fast": len(decision.fast_path_actions),
              "slow": len(decision.slow_path_escalations),
              "rationale": decision.rationale[:120]})
        return decision

    @staticmethod
    def _parse(raw: str, model_cls):
        """Extract and parse the outermost JSON object from an LLM response."""
        cleaned = raw.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)```", cleaned)
        if fence:
            cleaned = fence.group(1).strip()
        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if not obj_match:
            raise ValueError(f"No JSON object in response: {raw[:200]!r}")
        return model_cls(**json.loads(obj_match.group()))
