import asyncio
from core.events import UnifiedIngestEvent

ingest_queue: asyncio.Queue[UnifiedIngestEvent] = asyncio.Queue()
