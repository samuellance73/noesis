"""
agents/tools/run_command.py
───────────────────────────
Shell command execution tool implementation.
"""

import asyncio
from .registry import tools_registry
from utils.log_writer import emit


@tools_registry.register(
    "run_command",
    description="Run a shell command in the terminal and return its output (stdout and stderr).",
)
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

        if stdout_str:
            emit("tactical.tool_output", "tactical", {"tool": "run_command", "stream": "stdout", "content": stdout_str.rstrip()}, level="debug")
        if stderr_str:
            emit("tactical.tool_output", "tactical", {"tool": "run_command", "stream": "stderr", "content": stderr_str.rstrip()}, level="warn")

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
