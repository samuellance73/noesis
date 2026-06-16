import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import httpx2
from fastapi import HTTPException

logger = logging.getLogger(__name__)

@asynccontextmanager
async def handle_upstream_errors():
    """Centralized context manager for uniform upstream exception handling."""
    try:
        yield
    except httpx2.HTTPStatusError as exc:
        logger.error(f"Upstream returned HTTP error {exc.response.status_code}: {exc.response.text}")
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream API Error: {exc.response.text}"
        )
    except httpx2.RequestError as exc:
        logger.error(f"Network error while connecting to upstream: {exc}")
        raise HTTPException(
            status_code=503,
            detail="The upstream service is temporarily unreachable."
        )
    except Exception as exc:
        logger.error(f"Unexpected internal server error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred while processing your request."
        )

class UpstreamService:
    def __init__(self, client: httpx2.AsyncClient):
        # We inject a shared, pooled client rather than creating one per request
        self.client = client

    async def fetch_models(self) -> dict:
        async with handle_upstream_errors():
            response = await self.client.get("models")
            response.raise_for_status()
            return response.json()

    async def get_chat_completion(self, payload: dict) -> dict:
        async with handle_upstream_errors():
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
            return response.json()

    async def stream_chat_completion(self, payload: dict) -> AsyncGenerator[str, None]:
        try:
            async with self.client.stream(
                "POST", 
                "chat/completions", 
                json=payload, 
                timeout=60.0
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    yield f"data: {{\"error\": \"Upstream error {response.status_code}: {error_body.decode('utf-8', errors='ignore')}\"}}\n\n"
                    return

                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"
        except Exception as exc:
            logger.error(f"Streaming failed: {exc}", exc_info=True)
            yield f"data: {{\"error\": \"Streaming interruption occurred: {str(exc)}\"}}\n\n"
