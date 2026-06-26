"""
agents/self_initiative.py
──────────────────────────
SelfInitiativeEngine — Proactive Agent Goals (Component 5).

Principle: Autonomous Self-Determination.

When the daemon's trigger_store is completely empty, it may invoke the
SelfInitiativeEngine.  The engine reviews past logs and workspace context
to generate a new, meaningful goal and submits it to the trigger_store
with source="agent".

The self-generated trigger is then treated like any other trigger — it
flows through the TriageDispatcher and is routed to the appropriate path
(Fast-Path or Slow-Path) on the next daemon tick.

Design constraints
──────────────────
• Lightweight: one STANDARD-tier LLM call, no heavy scanning.
• Idempotent: the engine throttles itself via a cooldown to avoid
  spamming the trigger_store when the daemon idles for a long time.
• Non-blocking: the result is submitted to the store and the caller
  returns immediately — execution happens on the next daemon tick.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from core.model_router import ModelRequest, ModelRouter, ModelTier
from triggers.store import trigger_store
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

    async def maybe_submit(self, router: ModelRouter) -> bool:
        """
        If the cooldown has elapsed and trigger_store is empty, generate and
        submit a new self-initiative goal.

        Returns True if a new goal was submitted, False otherwise.
        """
        async with self._lock:
            if time.monotonic() - self._last_submission < _COOLDOWN_SECONDS:
                return False

            # Double-check the store is truly empty before spending tokens
            if trigger_store.get_pending():
                return False

            goal = await self._generate_goal(router)
            if not goal:
                return False

            trigger_store.submit(
                source="agent",
                description=goal,
                model=_DEFAULT_MODEL,
            )
            self._last_submission = time.monotonic()
            emit(
                "self_initiative.submitted",
                "self_initiative",
                {"goal": goal[:120]},
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
