import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from integrations.llm.service import UpstreamService
from agents.executor import AgentExecutor
from agents.tools import tools_registry

@pytest.mark.asyncio
async def test_agent_executor_final_answer_directly():
    # Mock LLM service
    llm_service = MagicMock(spec=UpstreamService)
    
    # LLM returns final answer on first step
    mock_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "I can answer this directly.",
                        "tool_calls": [],
                        "final_answer": "The capital of France is Paris."
                    })
                }
            }
        ]
    }
    llm_service.get_chat_completion = AsyncMock(return_value=mock_response)

    executor = AgentExecutor(llm_service=llm_service, model="test-model")
    result = await executor.run("What is the capital of France?")
    
    assert result == "The capital of France is Paris."
    llm_service.get_chat_completion.assert_called_once()


@pytest.mark.asyncio
async def test_agent_executor_with_tool_call():
    # Mock LLM service
    llm_service = MagicMock(spec=UpstreamService)
    
    # First call: agent decides to search
    mock_response_1 = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "I need to search for the current weather in Paris.",
                        "tool_calls": [{
                            "tool_name": "web_search",
                            "tool_input": "Paris weather"
                        }],
                        "final_answer": None
                    })
                }
            }
        ]
    }
    
    # Second call: agent uses the search observation to answer
    mock_response_2 = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "I have the search results. I can formulate the final answer.",
                        "tool_calls": [],
                        "final_answer": "The weather in Paris is sunny and 22 degrees."
                    })
                }
            }
        ]
    }
    
    llm_service.get_chat_completion = AsyncMock()
    llm_service.get_chat_completion.side_effect = [mock_response_1, mock_response_2]

    # Mock the tool implementation in the registry
    original_tool = tools_registry.tools["web_search"]
    mock_tool = AsyncMock(return_value="Search result placeholder for: Paris weather")
    tools_registry.tools["web_search"] = mock_tool

    try:
        executor = AgentExecutor(llm_service=llm_service, model="test-model")
        result = await executor.run("What is the weather in Paris?")
        
        assert result == "The weather in Paris is sunny and 22 degrees."
        assert llm_service.get_chat_completion.call_count == 2
        
        # Verify the history includes the observation
        called_payloads = [call[0][0] for call in llm_service.get_chat_completion.call_args_list]
        
        # The second payload messages should contain the observation from web_search
        messages = called_payloads[1]["messages"]
        # messages[0]: system prompt
        # messages[1]: user prompt
        # messages[2]: assistant thought/tool_call
        # messages[3]: observation
        assert "OBSERVATIONS:" in messages[3]["content"]
        assert "Search result placeholder for: Paris weather" in messages[3]["content"]
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
