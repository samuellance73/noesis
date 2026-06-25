"""
perception/__init__.py
──────────────────────
Public surface of the Perception Layer.

Import the PerceptionLayer class directly:

    from perception import PerceptionLayer, PerceptionConfig

Then wire it into app/lifespan.py:

    layer = PerceptionLayer(config, llm_service)
    await layer.start()               # launches the intake→route loop
    await layer.ingest(raw_signal)    # add a signal from any source
    await layer.stop()                # graceful shutdown
"""

from .layer import PerceptionLayer
from .config import PerceptionConfig
from .schemas import (
    RawSignal,
    RawSignalSource,
    SourceType,
    Priority,
    PerceptionType,
    PerceptionEvent,
    ResponseJob,
)

__all__ = [
    "PerceptionLayer",
    "PerceptionConfig",
    "RawSignal",
    "RawSignalSource",
    "SourceType",
    "Priority",
    "PerceptionType",
    "PerceptionEvent",
    "ResponseJob",
]
