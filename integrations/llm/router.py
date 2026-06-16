import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .schemas import ChatPayload
from .service import UpstreamService
from agents.executor import AgentExecutor

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
        return await service.get_chat_completion(upstream_payload)

@router.post("/api/agent/run")
async def run_agent(
    payload: AgentRequest,
    service: UpstreamService = Depends(get_upstream_service)
):
    executor = AgentExecutor(llm_service=service, model=payload.model)

    if payload.stream:
        async def event_generator():
            async for step_update in executor.run_generator(payload.user_input):
                # Format as standard Server-Sent Event (SSE)
                yield f"data: {json.dumps(step_update)}\n\n"
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    result = await executor.run(payload.user_input)
    return {
        "result": result,
        "steps": executor.state.steps
    }

