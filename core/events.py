
from __future__ import annotations

import time
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SenderClass(str, Enum):
    OPERATOR = "operator"
    TRUSTED = "trusted"
    EXTERNAL = "external"
    AGENT    = "agent"       # Self-initiated goals from SelfInitiativeEngine


class PriorityLevel(str, Enum):
    HIGH   = "high"
    NORMAL = "normal"


class UnifiedIngestEvent(BaseModel):
    event_identifier: UUID = Field(default_factory=uuid4)
    source_channel: str
    sender_identifier: str
    sender_class: SenderClass
    raw_content: str
    target_conversation_identifier: str
    priority_level: PriorityLevel
    monotonic_timestamp: float = Field(default_factory=time.monotonic)
    metadata: dict[str, Any] = Field(default_factory=dict)

