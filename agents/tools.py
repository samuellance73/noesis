import logging
import sys
import asyncio
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


@tools_registry.register(
    "python_execute",
    description="Execute arbitrary Python 3 code in a separate process. Write code that prints output to stdout. The return value is the standard output of the process. Standard libraries are available, and you can also import installed packages.",
)
@traced_tool("python_execute", input_arg="code")
async def python_execute(code: str) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=code.encode("utf-8")),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            stdout, stderr = await process.communicate()
            return f"Error: Execution timed out after 10.0 seconds.\nStdout: {stdout.decode('utf-8')}\nStderr: {stderr.decode('utf-8')}"

        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")

        if process.returncode != 0:
            return f"Error: Process exited with code {process.returncode}\nStdout:\n{stdout_str}\nStderr:\n{stderr_str}"

        output = []
        if stdout_str:
            output.append(stdout_str)
        if stderr_str:
            output.append(f"Stderr:\n{stderr_str}")

        result = "\n".join(output)
        if not result:
            result = "Success (no output)"
        return result
    except Exception as e:
        return f"Error: Failed to execute python code: {str(e)}"


@tools_registry.register(
    "run_command",
    description="Run a shell command in the terminal and return its output (stdout and stderr).",
)
@traced_tool("run_command", input_arg="command")
async def run_command(command: str) -> str:
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            stdout, stderr = await process.communicate()
            return f"Error: Command timed out after 15.0 seconds.\nStdout: {stdout.decode('utf-8', errors='replace')}\nStderr: {stderr.decode('utf-8', errors='replace')}"

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")

        if process.returncode != 0:
            return f"Error: Command exited with code {process.returncode}\nStdout:\n{stdout_str}\nStderr:\n{stderr_str}"

        output = []
        if stdout_str:
            output.append(stdout_str)
        if stderr_str:
            output.append(f"Stderr:\n{stderr_str}")

        result = "\n".join(output)
        if not result:
            result = "Success (no output)"
        return result
    except Exception as e:
        return f"Error: Failed to execute command: {str(e)}"


