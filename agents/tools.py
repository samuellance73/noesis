import logging
import httpx2
from typing import Dict, Any, Callable
from integrations.llm.config import settings
from utils.tracer import traced_tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str, description: str = ""):
        def decorator(func: Callable):
            func.description = description
            self.tools[name] = func
            return func
        return decorator

    async def execute(self, name: str, arg: Any) -> str:
        if name not in self.tools:
            logger.warning("Unknown tool requested: %r  available=%s", name, list(self.tools.keys()))
            return f"Error: Tool '{name}' is not available."
        try:
            func = self.tools[name]
            import inspect
            if inspect.iscoroutinefunction(func):
                result = await func(arg)
            else:
                result = func(arg)
            return str(result)
        except Exception as e:
            logger.error("Error executing tool %r: %s", name, e, exc_info=True)
            return f"Error executing tool: {str(e)}"


tools_registry = ToolRegistry()


@tools_registry.register(
    "web_search",
    description="Perform a web search using Tavily API. Useful for finding current information on the internet.",
)
@traced_tool("web_search", input_arg="query")
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
