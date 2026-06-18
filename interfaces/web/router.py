import asyncio
import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from integrations.llm.schemas import ChatPayload
from integrations.llm.service import UpstreamService, handle_upstream_errors
from agents.executor import AgentExecutor
from agents.goal_manager import GoalManager

router = APIRouter()


class AgentRequest(BaseModel):
    model: str
    user_input: str
    stream: bool = False


class GoalRequest(BaseModel):
    model: str
    goal: str


# A dependency provider to extract the shared connection client from application state
def get_upstream_service(request: Request) -> UpstreamService:
    client = request.app.state.upstream_client
    return UpstreamService(client)


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


# ── Autonomous goal loop ────────────────────────────────────────────────────

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
