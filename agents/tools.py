import logging
import httpx2
from typing import Dict, Any, Callable
from integrations.llm.config import settings

logger = logging.getLogger(__name__)

# Simple tools registry
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
            return f"Error: Tool '{name}' is not available."
        try:
            # Handle both sync and async tools
            func = self.tools[name]
            import inspect
            if inspect.iscoroutinefunction(func):
                result = await func(arg)
            else:
                result = func(arg)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error executing tool: {str(e)}"

tools_registry = ToolRegistry()

@tools_registry.register("web_search", description="Perform a web search using Tavily API. Useful for finding current information on the internet.")
async def web_search(query: str) -> str:
    if not settings.tavily_api_key:
        return "Error: tavily_api_key is not configured."
    
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
        "max_results": 5
    }
    
    try:
        async with httpx2.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("results", [])
            if not results:
                return "No results found."
                
            formatted_results = []
            for r in results:
                title = r.get("title", "No Title")
                url_str = r.get("url", "")
                content = r.get("content", "No Content")
                formatted_results.append(f"Title: {title}\nURL: {url_str}\nContent: {content}\n")
                
            return "\n".join(formatted_results)
    except Exception as e:
        logger.error(f"Tavily search error: {e}")
        return f"Error executing search: {str(e)}"
