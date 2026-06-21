"""
triggers/store.py
─────────────────
In-memory trigger queue — the single source of work for the daemon.

Every source of work (human, cron, Discord, webhook, agent-generated) writes
a Trigger here. The daemon drains it periodically and fires an AgentExecutor run
for each pending trigger.

Design notes
────────────
- Pure in-memory: no SQLite, no disk I/O. Triggers are lost on server restart.
  This is acceptable for the current stage; upgrade to SQLite by replacing the
  internal dict with an aiosqlite connection.
- Thread-safe for asyncio: all mutations happen inside the event loop; no locks
  needed as long as you don't call these from threads.
- Human fast-lane: human-sourced triggers are also placed in a separate
  asyncio.Event so the daemon can fire them immediately instead of waiting
  for the next poll interval.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger("noesis.trigger_store")

TriggerSource = Literal["human", "executor", "cron", "discord", "webhook", "agent"]
TriggerStatus = Literal["pending", "processing", "done", "failed"]


class Trigger(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: TriggerSource
    description: str
    model: str
    status: TriggerStatus = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class TriggerStore:
    """
    In-memory store for all triggers.

    Usage
    ─────
        trigger = trigger_store.submit(source="human", description="...", model="...")
        pending  = trigger_store.get_pending()   # drains all pending atomically
        trigger_store.mark_done(trigger.id)
    """

    def __init__(self) -> None:
        self._triggers: dict[UUID, Trigger] = {}
        # Signals the daemon that at least one human trigger is ready — allows
        # immediate dispatch without waiting for the poll interval.
        self.human_ready: asyncio.Event = asyncio.Event()

    # ── Write ──────────────────────────────────────────────────────────────────

    def submit(
        self,
        source: TriggerSource,
        description: str,
        model: str,
        metadata: Optional[dict] = None,
        bunch_key: Optional[str] = None,
    ) -> Trigger:
        """Create and store a new pending trigger. Returns the created Trigger."""
        metadata = metadata or {}
        if bunch_key:
            # Look for an existing pending trigger to bunch with
            for t in self._triggers.values():
                if t.status == "pending" and t.source == source and t.metadata.get("bunch_key") == bunch_key:
                    t.description += f"\n[Follow-up]: {description}"
                    # Update metadata (e.g. latest message id)
                    t.metadata.update(metadata)
                    logger.info("TriggerStore: bunched with existing trigger id=%s", t.id)
                    return t
            metadata["bunch_key"] = bunch_key

        t = Trigger(source=source, description=description, model=model, metadata=metadata)
        self._triggers[t.id] = t
        logger.info(
            "TriggerStore: submitted trigger id=%s source=%s description=%r",
            t.id, t.source, t.description[:80],
        )
        if source in ("human", "executor"):
            self.human_ready.set()  # wake daemon immediately for operator triggers
        return t


    # ── Read / drain ───────────────────────────────────────────────────────────

    def get_pending(self) -> list[Trigger]:
        """
        Atomically drain all pending triggers and mark them as 'processing'.
        Returns a snapshot list — safe to iterate while the store accepts new ones.
        """
        pending = [t for t in self._triggers.values() if t.status == "pending"]
        for t in pending:
            t.status = "processing"
        if pending:
            logger.info("TriggerStore: drained %d pending trigger(s).", len(pending))
        return pending

    def get(self, trigger_id: UUID) -> Optional[Trigger]:
        return self._triggers.get(trigger_id)

    def all(self) -> list[Trigger]:
        return list(self._triggers.values())

    # ── Status updates ─────────────────────────────────────────────────────────

    def mark_done(self, trigger_id: UUID) -> None:
        if t := self._triggers.get(trigger_id):
            t.status = "done"
            logger.info("TriggerStore: trigger %s → done", trigger_id)

    def mark_failed(self, trigger_id: UUID, error: str = "") -> None:
        if t := self._triggers.get(trigger_id):
            t.status = "failed"
            t.error = error
            logger.warning("TriggerStore: trigger %s → failed: %s", trigger_id, error)


# ── Global singleton ───────────────────────────────────────────────────────────
# Import this from anywhere:  from triggers.store import trigger_store
trigger_store = TriggerStore()
