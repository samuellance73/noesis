"""
perception/stages/router.py
────────────────────────────
Stage 6 — Router

Receives the list of PerceptionEvent objects from the Synthesizer and routes
each one to the appropriate downstream consumer according to the routing matrix
defined in the spec (§3.6).

Routing matrix
──────────────
  Type        authority   immediate response   to WorldModel   interrupt GM?
  ─────────   ─────────   ──────────────────   ─────────────   ─────────────
  DIRECTIVE   any         yes                  yes             if >= 0.75
  QUERY       any         yes                  no              no
  INFORMATION any         no                   yes             no
  CORRECTION  < 0.75      no                   yes             no
  CORRECTION  >= 0.75     no                   yes             yes
  FEEDBACK    any         no                   yes             no
  NOISE       any         no                   no              no

Notes
─────
  • The ReactivePool and WorldModel interfaces are injected at construction time
    as duck-typed async callables.  This keeps the Router decoupled from
    concrete GoalManager or pool implementations.
  • Routing decisions are logged at INFO level for the perception.log file.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from perception.schemas import PerceptionEvent, PerceptionType, ResponseJob

logger = logging.getLogger("noesis.perception")

# Authority threshold above which corrections trigger a GoalManager interrupt
_INTERRUPT_THRESHOLD = 0.75


@runtime_checkable
class ReactivePoolProtocol(Protocol):
    async def enqueue(self, job: ResponseJob) -> None: ...


@runtime_checkable
class WorldModelProtocol(Protocol):
    async def absorb(self, event: PerceptionEvent) -> None: ...
    async def flag_for_interrupt(self, event: PerceptionEvent) -> None: ...


class Router:
    """
    Routes PerceptionEvents to the ReactivePool and/or WorldModel.

    Parameters
    ──────────
    reactive_pool : ReactivePoolProtocol — receives ResponseJob objects for
                    events that require an immediate response.
    world_model   : WorldModelProtocol   — receives events the GoalManager
                    should be aware of on its next cycle.
    """

    def __init__(
        self,
        reactive_pool: ReactivePoolProtocol,
        world_model: WorldModelProtocol,
    ) -> None:
        self.reactive_pool = reactive_pool
        self.world_model = world_model

    async def route(self, events: list[PerceptionEvent]) -> None:
        """Route all events in this perception batch."""
        for event in events:
            await self._route_one(event)

    async def _route_one(self, event: PerceptionEvent) -> None:
        etype = event.type

        # ── NOISE: drop immediately ────────────────────────────────────────────
        if etype == PerceptionType.NOISE:
            logger.info(
                "Router: DROP  event_id=%s  type=noise  authority=%.2f",
                event.id, event.authority_score,
            )
            return

        # ── Compute routing flags from matrix ──────────────────────────────────
        send_to_reactive = event.requires_immediate_response
        send_to_world = False
        interrupt_gm = False

        if etype == PerceptionType.DIRECTIVE:
            send_to_reactive = True
            send_to_world = True
            interrupt_gm = event.authority_score >= _INTERRUPT_THRESHOLD

        elif etype == PerceptionType.QUERY:
            send_to_reactive = True
            send_to_world = False
            interrupt_gm = False

        elif etype == PerceptionType.INFORMATION:
            send_to_reactive = False
            send_to_world = True
            interrupt_gm = False

        elif etype == PerceptionType.CORRECTION:
            send_to_reactive = False
            send_to_world = True
            interrupt_gm = event.authority_score >= _INTERRUPT_THRESHOLD

        elif etype == PerceptionType.FEEDBACK:
            send_to_reactive = False
            send_to_world = True
            interrupt_gm = False

        # ── Dispatch ───────────────────────────────────────────────────────────
        if send_to_reactive:
            job = ResponseJob(event=event, priority=event.urgency)
            await self.reactive_pool.enqueue(job)
            logger.info(
                "Router: REACTIVE  event_id=%s  type=%s  urgency=%.2f  job_id=%s",
                event.id, etype.value, event.urgency, job.id,
            )

        if send_to_world:
            await self.world_model.absorb(event)
            logger.info(
                "Router: WORLD_MODEL  event_id=%s  type=%s  authority=%.2f",
                event.id, etype.value, event.authority_score,
            )

        if interrupt_gm:
            await self.world_model.flag_for_interrupt(event)
            logger.info(
                "Router: INTERRUPT_FLAG  event_id=%s  type=%s  authority=%.2f",
                event.id, etype.value, event.authority_score,
            )
