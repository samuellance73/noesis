import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.model_router import ModelRouter, ModelResponse, ModelTier
from agents.executor import AgentExecutor
from agents.tools import tools_registry


def _mock_router(responses: list[str]) -> ModelRouter:
    """Build a mock ModelRouter whose .complete() returns ModelResponses from content strings."""
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(side_effect=[
        ModelResponse(
            content=c,
            model_used="test-model",
            tier=ModelTier.STANDARD,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=1.0,
        )
        for c in responses
    ])
    router.resolve_model = MagicMock(return_value="test-model")
    router.config = MagicMock()
    router.config.tiers = {ModelTier.STANDARD: MagicMock(context_budget=8000)}
    return router


@pytest.mark.asyncio
async def test_agent_executor_final_answer_directly():
    content = json.dumps({
        "thought": "I can answer this directly.",
        "tool_calls": [],
        "final_answer": "The capital of France is Paris.",
    })
    router = _mock_router([content])

    executor = AgentExecutor(router=router)
    result = await executor.run("What is the capital of France?")

    assert result == "The capital of France is Paris."
    router.complete.assert_called_once()


@pytest.mark.asyncio
async def test_agent_executor_with_tool_call():
    content_1 = json.dumps({
        "thought": "I need to search for the current weather in Paris.",
        "tool_calls": [{"tool_name": "web_search", "tool_input": "Paris weather"}],
        "final_answer": None,
    })
    content_2 = json.dumps({
        "thought": "I have the search results. I can formulate the final answer.",
        "tool_calls": [],
        "final_answer": "The weather in Paris is sunny and 22 degrees.",
    })
    router = _mock_router([content_1, content_2])

    original_tool = tools_registry.tools["web_search"]
    mock_tool = AsyncMock(return_value="Search result placeholder for: Paris weather")
    tools_registry.tools["web_search"] = mock_tool

    try:
        executor = AgentExecutor(router=router)
        result = await executor.run("What is the weather in Paris?")

        assert result == "The weather in Paris is sunny and 22 degrees."
        assert router.complete.call_count == 2

        # Verify the second call's messages contain the observation
        second_call_request: "ModelRequest" = router.complete.call_args_list[1][0][0]
        messages = second_call_request.messages
        observation_msg = next(
            (m for m in messages if m.get("role") == "user" and "OBSERVATIONS:" in m.get("content", "")),
            None,
        )
        assert observation_msg is not None
        assert "Search result placeholder for: Paris weather" in observation_msg["content"]
    finally:
        tools_registry.tools["web_search"] = original_tool


def test_parse_agent_step_resilient():
    # 1. Clean JSON
    raw = '{"thought": "think", "tool_calls": [], "final_answer": "ans"}'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"
    assert step.final_answer == "ans"

    # 2. With think tags
    raw = '<think>some internal thought</think>\n{"thought": "think", "tool_calls": [], "final_answer": "ans"}'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"

    # 3. With markdown block
    raw = '```json\n{"thought": "think", "tool_calls": [], "final_answer": "ans"}\n```'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"

    # 4. With trailing commas
    raw = '{"thought": "think", "tool_calls": [], "final_answer": "ans",}'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"

    # 5. Single quotes repair
    raw = "{'thought': 'think', 'tool_calls': [], 'final_answer': 'ans'}"
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"
    assert step.final_answer == "ans"

    # 6. With nested markdown code fences inside final_answer
    raw = '```json\n{"thought": "think", "tool_calls": [], "final_answer": "Check this code:\\n```bash\\ncurl -X DELETE\\n```"}\n```'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"
    assert step.final_answer == "Check this code:\n```bash\ncurl -X DELETE\n```"
