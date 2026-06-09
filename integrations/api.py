import os
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv()

router = APIRouter()

UPSTREAM_API_URL = os.getenv("UPSTREAM_API_URL", "https://alisaajer-newrepo18.hf.space/v1")
API_KEY = os.getenv("API_KEY", "")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):       
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = True

@router.get("/api/models")
async def get_models():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{UPSTREAM_API_URL}/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=15.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Upstream error: {exc.response.text}"
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {str(exc)}"
            )

@router.post("/api/chat")
async def chat_completion(payload: ChatPayload):
    # Prepare the payload for upstream
    upstream_payload = {
        "model": payload.model,
        "messages": [msg.model_dump() for msg in payload.messages],
        "temperature": payload.temperature,
        "stream": payload.stream
    }
    if payload.max_tokens is not None:
        upstream_payload["max_tokens"] = payload.max_tokens

    async def stream_generator():
        async with httpx.AsyncClient() as client:
            try:
                # Request a stream from upstream
                async with client.stream(
                    "POST",
                    f"{UPSTREAM_API_URL}/chat/completions",
                    headers=headers,
                    json=upstream_payload,
                    timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        yield f"data: {{\"error\": \"Upstream error {response.status_code}: {error_body.decode('utf-8', errors='ignore')}\"}}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if line:
                            yield f"{line}\n"
            except Exception as e:
                yield f"data: {{\"error\": \"Streaming exception: {str(e)}\"}}\n\n"

    if payload.stream:
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{UPSTREAM_API_URL}/chat/completions",
                    headers=headers,
                    json=upstream_payload,
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status_code=exc.response.status_code,
                    detail=f"Upstream error: {exc.response.text}"
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal Server Error: {str(exc)}"
                )
