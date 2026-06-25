"""
perception/schemas.py
─────────────────────
All Pydantic data models used by the perception layer.

Signal lifecycle
────────────────
  RawSignal
    → (Deduplicator) →
  DeduplicatedSignal          (adds frequency, merged sources)
    → (Classifier) →
  DeduplicatedSignal.perception_type set
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


# ─── Source / priority primitives ─────────────────────────────────────────────

class SourceType(str, Enum):
    OPERATOR  = "operator"    # config-defined, highest trust
    TRUSTED   = "trusted"     # whitelisted users / internal services
    USER      = "user"        # default authenticated user
    ANONYMOUS = "anonymous"   # webhook / unknown origin
    AGENT     = "agent"       # signals from other agents in the system


class Priority(str, Enum):
    HIGH   = "high"    # bypasses intake window
    NORMAL = "normal"


# ─── Raw signal ───────────────────────────────────────────────────────────────

class RawSignalSource(BaseModel):
    type: SourceType
    identifier: str                   # user id, api key hash, agent id, etc.
    display_name: str | None = None


class RawSignal(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: RawSignalSource
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: Priority = Priority.NORMAL
    channel_id: str | None = None
    thread_id: str | None = None
    metadata: dict = Field(default_factory=dict)


# ─── Post-deduplication ───────────────────────────────────────────────────────

class DeduplicatedSignal(BaseModel):
    representative: RawSignal        # most-recent signal in the cluster
    frequency: int = 1               # how many raw signals were collapsed
    sources: list[RawSignalSource]   # all contributing sources
    raw_signals: list[RawSignal] = Field(default_factory=list)
    # Set by Classifier and AuthorityScorer in subsequent stages:
    perception_type: "PerceptionType | None" = None
    authority_score: float | None = None


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
    A deduplicated signal after classification and authority scoring.
    This is the input type for the Synthesizer stage.
    """
    id: UUID = Field(default_factory=uuid4)
    representative: RawSignal
    frequency: int
    sources: list[RawSignalSource]
    perception_type: PerceptionType
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

    async def absorb(self, event: PerceptionEvent) -> None:
        await self.pending_perceptions.put(event)

    async def flag_for_interrupt(self, event: PerceptionEvent) -> None:
        self.interrupt_flags.append(event)

    def drain_perceptions(self) -> list[PerceptionEvent]:
        events: list[PerceptionEvent] = []
        while not self.pending_perceptions.empty():
            events.append(self.pending_perceptions.get_nowait())
        return events

    def drain_interrupts(self) -> list[PerceptionEvent]:
        flags = list(self.interrupt_flags)
        self.interrupt_flags.clear()
        return flags
