"""
agents/tools
────────────
Tool registry and specialized tool implementations.

Exposes the tool functions directly and loads the modules so they register
themselves with the global tools_registry.
"""

from .registry import ToolRegistry, tools_registry, build_specialized_registry

# Import and expose tool functions directly (this also triggers registration decorators)
from .discord import send_discord_message
from .python_execute import python_execute
from .run_command import run_command
from .web_search import web_search

__all__ = [
    "ToolRegistry",
    "tools_registry",
    "build_specialized_registry",
    "send_discord_message",
    "python_execute",
    "run_command",
    "web_search",
]
