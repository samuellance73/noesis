import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .schemas import ChatPayload
from .service import UpstreamService, handle_upstream_errors
from agents.orchestrator import AgentOrchestrator

router = APIRouter()


class AgentRequest(BaseModel):
    model: str
    user_input: str
    stream: bool = False


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
    # Convert Pydantic schemas safely into python dicts, removing unset fields
    upstream_payload = payload.model_dump(exclude_none=True)

    if payload.stream:
        return StreamingResponse(
            service.stream_chat_completion(upstream_payload),
            media_type="text/event-stream"
        )
    else:
        async with handle_upstream_errors():
            return await service.get_chat_completion(upstream_payload)


@router.post("/api/agent/run")
async def run_agent(
    payload: AgentRequest,
    service: UpstreamService = Depends(get_upstream_service)
):
    orchestrator = AgentOrchestrator(llm_service=service, model=payload.model)

    if payload.stream:
        async def event_generator():
            async for event in orchestrator.run_stream(payload.user_input):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async with handle_upstream_errors():
        return await orchestrator.run(payload.user_input)
