"""
agents/tools/python_execute.py
──────────────────────────────
Python code execution tool implementation.
"""

import sys
import asyncio
from .registry import tools_registry
from utils.log_writer import emit


@tools_registry.register(
    "python_execute",
    description=(
        "Execute arbitrary Python 3 code in a separate process and capture the stdout output. "
        "Standard libraries and pre-installed packages are fully available.\n\n"
        
        "INSTALLED LIBRARIES TO USE:\n"
        "- 'PyGithub' (import as: from github import Github) is installed. Use this for all GitHub-related tasks.\n"
        "- 'requests' and 'httpx' are installed for any raw API requests.\n\n"
        
        "AVAILABLE SECRET KEYS (ENVIRONMENT VARIABLES):\n"
        "You can read these secret keys securely inside your Python code using 'os.environ.get(...)'. "
        "Do NOT hardcode keys in your written code; always fetch them dynamically:\n"
        "- os.environ.get('GITHUB_TOKEN') is pre-loaded with your GitHub access token.\n"
        "- os.environ.get('TELEGRAM_BOT_TOKEN') is pre-loaded with your Telegram bot key.\n"
        "- os.environ.get('DISCORD_BOT_TOKEN') is a SELFBOT / USER-ACCOUNT token (not a bot application token). "
        "When making raw Discord API calls with this token, use 'Authorization: <token>' (NO 'Bot' prefix). "
        "Example header: {'Authorization': token, 'Content-Type': 'application/json'}. "
        "Do NOT use 'Authorization: Bot <token>' — that will return 401 Unauthorized."
    ),
)
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

        if stdout_str:
            emit("tactical.tool_output", "tactical", {"tool": "python_execute", "stream": "stdout", "content": stdout_str.rstrip()}, level="debug")
        if stderr_str:
            emit("tactical.tool_output", "tactical", {"tool": "python_execute", "stream": "stderr", "content": stderr_str.rstrip()}, level="warn")

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
