"""
perception/schemas.py
─────────────────────
All Pydantic data models used by the perception layer.

Signal lifecycle
────────────────
  RawSignal
    → (AuthorityScorer) →
  ScoredSignal                (final pre-synthesis view)
    → (Synthesizer) →
  PerceptionEvent             (one per semantically distinct concern)
    → (Router) →
  ResponseJob                 (injected into ReactivePool)
  or
  WorldModel.pending_perceptions (picked up by GoalManager)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from core.events import UnifiedIngestEvent, SenderClass, PriorityLevel


# ─── Perception type ──────────────────────────────────────────────────────────

class PerceptionType(str, Enum):
    DIRECTIVE   = "directive"    # "do X" — wants action taken
    QUERY       = "query"        # "what is X" — wants information
    INFORMATION = "information"  # "FYI, X happened" — context injection
    CORRECTION  = "correction"   # "you got X wrong" — feedback on prior output
    FEEDBACK    = "feedback"     # "good job / bad job" — evaluative
    NOISE       = "noise"        # irrelevant, drop


# ─── Scored signal (pre-synthesis view) ───────────────────────────────────────

class ScoredSignal(BaseModel):
    """
    A signal after authority scoring, containing the original unified ingest event.
    This is the input type for the LLM bundle processing.
    """
    id: UUID = Field(default_factory=uuid4)
    representative: UnifiedIngestEvent # Now holds the UnifiedIngestEvent
    frequency: int = 1
    perception_type: PerceptionType | None = None
    authority_score: float


# ─── Perception event (output of synthesis) ───────────────────────────────────

class PerceptionEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    summary: str                          # one sentence describing the concern
    type: PerceptionType
    urgency: float                        # 0.0 to 1.0
    authority_score: float                # 0.0 to 1.0 (weighted avg of contributors)
    requires_immediate_response: bool
    affects_objectives: bool              # does this potentially change active goals?
    frequency: int                        # how many raw signals contributed
    source_ids: list[UUID]
    response_context: str | None = None   # only if requires_immediate_response
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Response job (for ReactivePool) ──────────────────────────────────────────

class ResponseJob(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    event: PerceptionEvent
    priority: float                       # inherits urgency from event
    status: Literal["queued", "running", "done", "failed"] = "queued"
    assigned_worker: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─── WorldModel perception interface ──────────────────────────────────────────

class PerceptionWorldModel:
    """
    Lightweight async queue facade used by the Router to feed the GoalManager.

    The full GoalManager keeps its own GoalState.world_model (a Pydantic object).
    This class is intentionally a *separate* component — the perception layer
    has no dependency on GoalManager internals.  Wire them together in
    app/lifespan.py by passing GoalManager.absorb / flag_for_interrupt as
    callbacks, or simply attach a PerceptionWorldModel instance and let the
    GoalManager drain it via drain_perceptions().
    """

    def __init__(self) -> None:
        self.pending_perceptions: asyncio.Queue[PerceptionEvent] = asyncio.Queue()
        self.interrupt_flags: list[PerceptionEvent] = []
        self.perception_contexts: list[dict] = []

    async def absorb(self, event: PerceptionEvent) -> None:
        await self.pending_perceptions.put(event)

    async def flag_for_interrupt(self, event: PerceptionEvent) -> None:
        self.interrupt_flags.append(event)

    async def add_perception_context(self, context: dict) -> None:
        """Add a perception context dict for the next GoalManager cycle."""
        self.perception_contexts.append(context)

    def drain_perceptions(self) -> list[PerceptionEvent]:
        events: list[PerceptionEvent] = []
        while not self.pending_perceptions.empty():
            events.append(self.pending_perceptions.get_nowait())
        return events

    def drain_interrupts(self) -> list[PerceptionEvent]:
        flags = list(self.interrupt_flags)
        self.interrupt_flags.clear()
        return flags

    def drain_contexts(self) -> list[dict]:
        contexts = list(self.perception_contexts)
        self.perception_contexts.clear()
        return contexts
