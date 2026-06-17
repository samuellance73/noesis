import pytest
import asyncio
import sys
import os

# Add the project root to sys.path so we can import agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.tools import run_command, tools_registry

@pytest.mark.asyncio
async def test_run_command_success():
    cmd = "echo 'hello from command line'"
    result = await run_command(cmd)
    assert result.strip() == "hello from command line"

@pytest.mark.asyncio
async def test_run_command_error():
    cmd = "nonexistent_command_12345"
    result = await run_command(cmd)
    assert "Error:" in result

@pytest.mark.asyncio
async def test_run_command_registry():
    assert "run_command" in tools_registry.tools
    func = tools_registry.tools["run_command"]
    res = await func("echo 123")
    assert res.strip() == "123"
