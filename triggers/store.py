"""
triggers/store.py
─────────────────
Legacy in-memory TriggerStore — kept for backward compatibility only.

NOTE: This component is RETIRED in Noesis v2.  The new architecture uses
asyncio.create_task() directly (core/daemon.py) instead of polling a store.
Do NOT add new dependencies on TriggerStore.

The Trigger model and trigger_store singleton are preserved here so that
any references in the old triggers/daemon.py (dead code, not called in
production) continue to import without errors.
"""
from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class Trigger(BaseModel):
    """
    Single trigger record (legacy — used only by triggers/daemon.py).
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(..., description="Origin of the trigger (e.g., 'human', 'cron', 'agent').")
    description: str = Field(..., description="The main task or goal of the trigger.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = Field("pending")
    error: Optional[str] = Field(None)


class TriggerStore:
    """
    In-memory trigger store (legacy).  All mutating methods are synchronous —
    the asyncio.Lock is retained only for the human_ready Event which is
    used in the legacy poll loop.

    In the v2 architecture this is never instantiated at runtime.
    """
    _instance: Optional["TriggerStore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self.triggers: Dict[str, Trigger] = {}
            self.human_ready = asyncio.Event()
            self._initialized = True

    def submit_sync(self, trigger: Trigger) -> None:
        """Synchronous submit (no lock needed for in-process dict)."""
        self.triggers[trigger.id] = trigger
        if trigger.source == "human":
            self.human_ready.set()

    async def submit(self, trigger: Trigger) -> None:
        """Async compat shim — delegates to submit_sync."""
        self.submit_sync(trigger)

    def get_pending(self) -> List[Trigger]:
        """Return pending triggers sorted by creation time."""
        return sorted(
            [t for t in self.triggers.values() if t.status == "pending"],
            key=lambda t: t.created_at,
        )

    def mark_done(self, trigger_id: str) -> None:
        if trigger_id in self.triggers:
            self.triggers[trigger_id].status = "done"

    def mark_failed(self, trigger_id: str, error: str) -> None:
        if trigger_id in self.triggers:
            self.triggers[trigger_id].status = "failed"
            self.triggers[trigger_id].error = error

    def clear(self) -> None:
        self.triggers.clear()
        self.human_ready.clear()


trigger_store = TriggerStore()
