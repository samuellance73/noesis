"""
agents/tools/registry.py
────────────────────────
Tool registry implementation.
"""

import inspect
from typing import Dict, Any, Callable
from utils.log_writer import emit


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
            emit("tactical.warning", "tactical", {"msg": f"Unknown tool requested: {name!r}  available={list(self.tools.keys())}"}, level="warn")
            return f"Error: Tool '{name}' is not available."
        try:
            func = self.tools[name]
            if inspect.iscoroutinefunction(func):
                result = await func(arg)
            else:
                result = func(arg)
            return str(result)
        except Exception as e:
            emit("tactical.error", "tactical", {"msg": f"Error executing tool {name!r}: {e}"}, level="error")
            return f"Error executing tool: {str(e)}"


tools_registry = ToolRegistry()


def build_specialized_registry(executor_type_str: str) -> ToolRegistry:
    """
    Creates a new ToolRegistry containing only the tools relevant to the specified executor profile.
    This limits tool availability for specialized agents (e.g. Synthesis agents have no tools).
    """
    from ..schemas import ExecutorType
    try:
        executor_type = ExecutorType(executor_type_str)
    except ValueError:
        executor_type = ExecutorType.FULL

    registry = ToolRegistry()
    if executor_type == ExecutorType.RESEARCH:
        if "web_search" in tools_registry.tools:
            registry.tools["web_search"] = tools_registry.tools["web_search"]
    elif executor_type == ExecutorType.CODE:
        for name in ["python_execute", "run_command"]:
            if name in tools_registry.tools:
                registry.tools[name] = tools_registry.tools[name]
    elif executor_type == ExecutorType.SYNTHESIS:
        pass
    else:
        registry.tools = dict(tools_registry.tools)
    return registry
