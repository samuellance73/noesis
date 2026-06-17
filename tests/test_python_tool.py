import pytest
import asyncio
import sys
import os

# Add the project root to sys.path so we can import agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.tools import python_execute, tools_registry

@pytest.mark.asyncio
async def test_python_execute_success():
    code = "print('hello from tool')"
    result = await python_execute(code)
    assert result.strip() == "hello from tool"

@pytest.mark.asyncio
async def test_python_execute_error():
    code = "raise ValueError('some error')"
    result = await python_execute(code)
    assert "ValueError: some error" in result

@pytest.mark.asyncio
async def test_python_execute_registry():
    assert "python_execute" in tools_registry.tools
    func = tools_registry.tools["python_execute"]
    res = await func("print(1 + 2)")
    assert res.strip() == "3"
