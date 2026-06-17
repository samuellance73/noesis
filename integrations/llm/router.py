import json
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .schemas import ChatPayload
from .service import UpstreamService
from agents.executor import AgentExecutor
from agents.planner import plan

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
    if payload.stream:
        async def event_generator():
            try:
                # 1. Plan first
                yield f"data: {json.dumps({'event': 'planning_start'})}\n\n"
                milestones = await plan(payload.user_input, service)
                yield f"data: {json.dumps({'event': 'plan_ready', 'milestones': milestones})}\n\n"
                
                # 2. Run each milestone through the executor
                results = []
                for idx, milestone in enumerate(milestones):
                    yield f"data: {json.dumps({'event': 'step_start', 'step_index': idx, 'step_goal': milestone['goal']})}\n\n"
                    
                    executor = AgentExecutor(llm_service=service, model=payload.model)
                    final_result = None
                    async for step_update in executor.run_generator(milestone["goal"]):
                        step_update["step_index"] = idx
                        if step_update["event"] == "final_answer":
                            final_result = step_update["answer"]
                        yield f"data: {json.dumps(step_update)}\n\n"
                    
                    results.append({"milestone": milestone["goal"], "result": final_result})
                
                # 3. Return everything
                yield f"data: {json.dumps({'event': 'done', 'milestones': milestones, 'results': results})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Fallback to non-streaming
    milestones = await plan(payload.user_input, service)
    results = []
    for milestone in milestones:
        executor = AgentExecutor(llm_service=service, model=payload.model)
        result = await executor.run(milestone["goal"])
        results.append({"milestone": milestone["goal"], "result": result})
    return {"milestones": milestones, "results": results}

