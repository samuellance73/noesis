"""
agents/tools/web_search.py
─────────────────────────
Web search tool implementation.
"""

import httpx2
from integrations.llm.config import settings
from .registry import tools_registry


@tools_registry.register(
    "web_search",
    description="Perform a web search using Tavily API. Useful for finding current information on the internet.",
)
async def web_search(query: str) -> str:
    tavily_api_key = settings.tavily_api_key
    if not tavily_api_key:
        return "Error: TAVILY_API_KEY is not configured."

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5,
    }

    async with httpx2.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=10.0)
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if not results:
        return "No results found."

    formatted = [
        f"Title: {r.get('title', 'No Title')}\nURL: {r.get('url', '')}\nContent: {r.get('content', 'No Content')}\n"
        for r in results
    ]
    return "\n".join(formatted)
