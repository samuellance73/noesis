from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from .schemas import ChatPayload
from .service import UpstreamService

router = APIRouter()

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
