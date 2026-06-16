import logging
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

# Simple tools registry
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(func: Callable):
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

# Example tool definition
@tools_registry.register("web_search")
async def web_search(query: str) -> str:
    # Integration logic with search service (e.g., Tavily or DuckDuckGo)
    return f"Search result placeholder for: {query}"
