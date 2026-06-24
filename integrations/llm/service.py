import logging
import time
from typing import AsyncGenerator

import httpx2
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from utils.llm_log_formatter import format_chat_block, format_stream_block

logger     = logging.getLogger(__name__)
llm_logger = logging.getLogger("noesis.llm")


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


class UpstreamService:
    def __init__(self, client: httpx2.AsyncClient):
        # Inject a shared, pooled client rather than creating one per request.
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
        t0 = time.perf_counter()

        response = await self.client.post("chat/completions", json=payload)
        response.raise_for_status()
        data    = response.json()
        elapsed = time.perf_counter() - t0

        llm_logger.info(format_chat_block(payload, data, elapsed))
        return data

    async def stream_chat_completion(self, payload: dict) -> AsyncGenerator[str, None]:
        """Stream chat completions with pre-connect retries.

        Retry logic only fires before the stream opens — retrying mid-stream
        would cause duplicate tokens, so once we start yielding we let any
        error surface naturally.
        """
        t0 = time.perf_counter()

        @retry(**{**_retry_policy, "stop": stop_after_attempt(3)})
        async def _open_stream():
            """Raises on non-200 so tenacity can retry the connection."""
            response = await self.client.send(
                self.client.build_request("POST", "chat/completions", json=payload),
                stream=True,
            )
            if response.status_code != 200:
                await response.aread()
                raise httpx2.HTTPStatusError(
                    f"Upstream error {response.status_code}",
                    request=response.request,
                    response=response,
                )
            return response

        try:
            response = await _open_stream()
            async with response:
                raw_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"
                        raw_lines.append(line)

                elapsed = time.perf_counter() - t0
                llm_logger.info(format_stream_block(payload, raw_lines, elapsed))

        except RetryError as exc:
            logger.error("Stream connect failed after retries: %s", exc)
            yield 'data: {"error": "Upstream unavailable after retries."}\n\n'
        except Exception as exc:
            logger.error("Streaming failed: %s", exc, exc_info=True)
            yield f'data: {{"error": "Streaming interruption occurred: {exc}"}}\n\n'
