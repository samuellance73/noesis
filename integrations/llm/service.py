import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import httpx2
from fastapi import HTTPException
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient errors worth retrying."""
    if isinstance(exc, httpx2.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    if isinstance(exc, (httpx2.TimeoutException, httpx2.NetworkError)):
        return True
    return False


_retry_policy = dict(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

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

    @retry(**_retry_policy)
    async def fetch_models(self) -> dict:
        """Raw httpx exceptions bubble out so @retry can act on them."""
        response = await self.client.get("models")
        response.raise_for_status()
        return response.json()

    @retry(**_retry_policy)
    async def get_chat_completion(self, payload: dict) -> dict:
        """Raw httpx exceptions bubble out so @retry can act on them."""
        response = await self.client.post("chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def stream_chat_completion(self, payload: dict) -> AsyncGenerator[str, None]:
        """Stream chat completions with pre-connect retries.

        Retry logic only fires before the stream opens — retrying mid-stream
        would cause duplicate tokens, so once we start yielding we let any
        error surface naturally.
        """
        @retry(**{**_retry_policy, "stop": stop_after_attempt(3)})
        async def _open_stream():
            """Raises on non-200 so tenacity can retry the connection."""
            response = await self.client.send(
                self.client.build_request("POST", "chat/completions", json=payload),
                stream=True,
            )
            if response.status_code != 200:
                error_body = await response.aread()
                exc = httpx2.HTTPStatusError(
                    f"Upstream error {response.status_code}",
                    request=response.request,
                    response=response,
                )
                raise exc
            return response

        try:
            response = await _open_stream()
            async with response:
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"
        except RetryError as exc:
            logger.error(f"Stream connect failed after retries: {exc}")
            yield f"data: {{\"error\": \"Upstream unavailable after retries.\"}}\n\n"
        except Exception as exc:
            logger.error(f"Streaming failed: {exc}", exc_info=True)
            yield f"data: {{\"error\": \"Streaming interruption occurred: {str(exc)}\"}}\n\n"
