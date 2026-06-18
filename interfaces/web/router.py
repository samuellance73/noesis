import asyncio
import json
import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from integrations.llm.schemas import ChatPayload
from integrations.llm.service import UpstreamService, handle_upstream_errors
from agents.executor import AgentExecutor
from agents.goal_manager import GoalManager
from agents.trigger_store import trigger_store
from utils.event_bus import event_bus

router = APIRouter()


# ── System status (daemon health + queue snapshot) ─────────────────────────

@router.get("/api/status")
async def system_status(request: Request):
    """
    Single-call health check. Shows:
      - Daemon task state (running / done / cancelled / error)
      - Trigger queue counts by status
      - Last 10 triggers with details
      - EventBus subscriber count (= open SSE connections)
    """
    daemon_task = getattr(request.app.state, "daemon_task", None)
    if daemon_task is None:
        daemon_status = "not_started"
    elif daemon_task.done():
        exc = daemon_task.exception() if not daemon_task.cancelled() else None
        daemon_status = f"failed: {exc}" if exc else ("cancelled" if daemon_task.cancelled() else "done")
    else:
        daemon_status = "running"

    all_triggers = trigger_store.all()
    counts = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
    for t in all_triggers:
        counts[t.status] = counts.get(t.status, 0) + 1

    recent = sorted(all_triggers, key=lambda t: t.created_at, reverse=True)[:10]

    return {
        "daemon": daemon_status,
        "event_bus_subscribers": len(event_bus._subscribers),
        "trigger_counts": counts,
        "recent_triggers": [
            {
                "id":          str(t.id)[:8] + "…",  # short ID for readability
                "source":      t.source,
                "description": t.description[:60] + ("…" if len(t.description) > 60 else ""),
                "status":      t.status,
                "created_at":  t.created_at.strftime("%H:%M:%S"),
                "error":       t.error,
            }
            for t in recent
        ],
    }



class AgentRequest(BaseModel):
    model: str
    user_input: str
    stream: bool = False


class GoalRequest(BaseModel):
    model: str
    goal: str


class TriggerRequest(BaseModel):
    model: str
    description: str


# Dependency: return the shared UpstreamService built once at startup
def get_upstream_service(request: Request) -> UpstreamService:
    return request.app.state.upstream_service


@router.get("/api/models")
async def get_models(service: UpstreamService = Depends(get_upstream_service)):
    async with handle_upstream_errors():
        return await service.fetch_models()


@router.post("/api/chat")
async def chat_completion(
    payload: ChatPayload,
    service: UpstreamService = Depends(get_upstream_service)
):
    upstream_payload = payload.model_dump(exclude_none=True)

    if payload.stream:
        return StreamingResponse(
            service.stream_chat_completion(upstream_payload),
            media_type="text/event-stream"
        )
    else:
        async with handle_upstream_errors():
            return await service.get_chat_completion(upstream_payload)


# ── Single-turn agent (quick, non-autonomous) ──────────────────────────────

@router.post("/api/agent/run")
async def run_agent(
    payload: AgentRequest,
    service: UpstreamService = Depends(get_upstream_service)
):
    """Single-turn agent execution. Fast path for simple one-shot requests."""
    executor = AgentExecutor(llm_service=service, model=payload.model)

    if payload.stream:
        async def event_generator():
            # Synthetic plan_ready keeps any legacy frontend JS happy
            yield f"data: {json.dumps({'event': 'plan_ready', 'milestones': [{'goal': 'Execute user request'}]})}\n\n"
            async for event in executor.run_generator(payload.user_input):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async with handle_upstream_errors():
        result = await executor.run(payload.user_input)
        return {"result": result}


# ── Autonomous goal loop (direct, synchronous) ──────────────────────────────

@router.post("/api/agent/goal")
async def run_goal(
    payload: GoalRequest,
    service: UpstreamService = Depends(get_upstream_service)
):
    """
    Autonomous goal-directed loop. Streams SSE events until the goal is
    complete or the client closes the connection.

    To stop: close the SSE connection (the generator will clean up) or
    call POST /api/agent/goal/stop with the same session (future work).
    """
    manager = GoalManager(llm_service=service, model=payload.model)

    async def event_generator():
        try:
            async for event in manager.run_stream(payload.goal):
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            manager.request_stop()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Unified trigger architecture ────────────────────────────────────────────

@router.post("/api/triggers")
async def submit_trigger(payload: TriggerRequest):
    """
    Submit work to the trigger queue. Returns immediately — the daemon picks
    it up and runs GoalManager in the background.

    Human triggers bypass the poll interval (fast-lane) and fire as soon as
    the current batch (if any) finishes.

    Response: { "trigger_id": "<uuid>", "status": "pending" }
    """
    trigger = trigger_store.submit(
        source="human",
        description=payload.description,
        model=payload.model,
    )
    return {"trigger_id": str(trigger.id), "status": trigger.status}


@router.get("/api/triggers")
async def list_triggers():
    """Return all triggers with their current status (for debugging / dashboards)."""
    return {
        "triggers": [
            {
                "id":          str(t.id),
                "source":      t.source,
                "description": t.description,
                "status":      t.status,
                "created_at":  t.created_at.isoformat(),
                "error":       t.error,
            }
            for t in trigger_store.all()
        ]
    }


@router.get("/api/triggers/stream")
async def trigger_stream():
    """
    SSE endpoint — broadcasts all GoalManager events from all running triggers
    to every connected client.

    Connect once; receive events from every trigger the daemon processes,
    tagged with `trigger_id` and `trigger_source` for filtering.
    """
    q = event_bus.subscribe()

    async def event_generator():
        # Send a heartbeat immediately so the browser knows the connection is live
        yield "data: {\"event\": \"connected\"}\n\n"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive ping so proxies don't close the connection
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
