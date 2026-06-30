"""
agents/self_initiative.py
──────────────────────────
SelfInitiativeEngine — Proactive Agent Goals (Component 7).

Principle: Autonomous Self-Determination.

When both the ingest_queue and the IntakeBuffer are empty (system verified idle),
the core daemon invokes SelfInitiativeEngine.run().  The engine reviews recent
agent.jsonl log history to generate a new, meaningful self-determined goal and
submits it to the ingest_queue as a UnifiedIngestEvent with source_channel="agent".

The self-generated event is treated like any other ingest event — it flows through
the IntakeBuffer → PerceptionLayer → TriageDispatcher → Dispatcher pipeline on the
next processing tick.

Design constraints
──────────────────
• Lightweight: one STANDARD-tier LLM call, no heavy scanning.
• Idempotent: the engine enforces its own cooldown timer so the daemon can call
  run() on every idle check without risk of submission spam.
• Non-blocking: the goal is submitted to ingest_queue and the caller returns
  immediately — execution happens asynchronously on the next daemon tick.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from core.model_router import ModelRequest, ModelRouter, ModelTier
from core.events import UnifiedIngestEvent, SenderClass, PriorityLevel
from core.queues import ingest_queue
from perception.stages.intake import IntakeBuffer
from utils.log_writer import emit

# Minimum seconds between self-initiative submissions to prevent spam.
_COOLDOWN_SECONDS: float = float(os.getenv("SELF_INITIATIVE_COOLDOWN", "300"))

# Maximum number of recent log lines to inject as context.
_MAX_LOG_LINES: int = 60

# Default model string — reuses the same env var as the Discord interface.
_DEFAULT_MODEL: str = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")

# Log file path — same as used by log_writer.
_LOG_PATH: Path = Path("logs/agent.jsonl")


class SelfInitiativeEngine:
    """
    Generates autonomous goals when the agent has nothing to do.

    Usage (inside triggers/daemon.py):
        engine = SelfInitiativeEngine()
        await engine.maybe_submit(router)
    """

    def __init__(self) -> None:
        self._last_submission: float = 0.0
        self._lock = asyncio.Lock()

    async def run(self, router: ModelRouter, intake_buffer: IntakeBuffer) -> bool:
        """
        If the cooldown has elapsed and the system is idle, generate and
        submit a new self-initiative goal.

        Returns True if a new goal was submitted, False otherwise.
        """
        async with self._lock:
            if time.monotonic() - self._last_submission < _COOLDOWN_SECONDS:
                return False

            # Double-check the system is truly idle before spending tokens
            if not ingest_queue.empty() or intake_buffer.size > 0:
                return False

            goal_description = await self._generate_goal(router)
            if not goal_description:
                return False

            # Create a UnifiedIngestEvent for the self-initiated goal
            event = UnifiedIngestEvent(
                source_channel="agent",
                sender_identifier="self_initiative_engine",
                sender_class=SenderClass.AGENT,
                raw_content=goal_description,
                target_conversation_identifier="self_initiative", # Placeholder
                priority_level=PriorityLevel.NORMAL,
                metadata={
                    "reason": "self_initiated_goal",
                },
            )
            await ingest_queue.put(event)
            self._last_submission = time.monotonic()
            emit(
                "self_initiative.submitted",
                "self_initiative",
                {"goal": goal_description[:120]},
            )
            return True

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _generate_goal(self, router: ModelRouter) -> str | None:
        """
        Ask the STANDARD-tier LLM to review recent activity and propose
        a concrete, actionable goal.
        """
        context = self._read_recent_logs()
        system = (
            "You are an autonomous AI agent reviewing your own recent activity. "
            "Based on the context below, propose ONE concrete, actionable goal "
            "you should pursue right now to be most useful. "
            "Prioritise: completing unfinished work, monitoring external systems, "
            "self-improvement, or proactive research. "
            "Reply with ONLY the goal statement — no preamble, no numbering."
        )
        user_msg = (
            f"Recent agent activity (last {_MAX_LOG_LINES} log entries):\n\n"
            f"{context}\n\n"
            "What should I do next?"
        )

        request = ModelRequest(
            tier=ModelTier.STANDARD,
            messages=[{"role": "user", "content": user_msg}],
            system=system,
            component="self_initiative",
        )

        try:
            response = await router.complete(request)
            goal = response.content.strip()
            return goal if goal else None
        except Exception as exc:
            emit(
                "self_initiative.error",
                "self_initiative",
                {"error": str(exc)},
                level="error",
            )
            return None

    @staticmethod
    def _read_recent_logs() -> str:
        """
        Return the last _MAX_LOG_LINES lines from the agent log as plain text.
        Falls back gracefully if the log is missing or unreadable.
        """
        try:
            log_path = _LOG_PATH
            if not log_path.exists():
                return "(no log file found)"
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            recent = lines[-_MAX_LOG_LINES:]
            return "\n".join(recent) if recent else "(log is empty)"
        except Exception as exc:
            return f"(could not read log: {exc})"
