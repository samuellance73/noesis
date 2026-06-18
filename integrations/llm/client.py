import httpx2
from .config import settings


def get_client(timeout: float = 30.0) -> httpx2.AsyncClient:
    """
    Creates and returns an httpx2.AsyncClient configured for the upstream LLM API.
    This encapsulates the API key and base URL setup in one place.
    """
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }

    base_url = settings.upstream_api_url
    if not base_url.endswith("/"):
        base_url += "/"

    return httpx2.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=timeout,
    )
