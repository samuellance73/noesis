import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from main import app
from integrations.llm.service import UpstreamService
from agents.executor import AgentExecutor
from agents.tools import tools_registry

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

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
                        "tool_call": None,
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
                        "tool_call": {
                            "tool_name": "web_search",
                            "tool_input": "Paris weather"
                        },
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
                        "tool_call": None,
                        "final_answer": "The weather in Paris is sunny and 22 degrees."
                    })
                }
            }
        ]
    }
    
    llm_service.get_chat_completion = AsyncMock()
    llm_service.get_chat_completion.side_effect = [mock_response_1, mock_response_2]

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
    assert "Observation from 'web_search'" in messages[3]["content"]
    assert "Search result placeholder for: Paris weather" in messages[3]["content"]


def test_agent_endpoint_success(client):
    # Mock upstream client's post call, since router resolves service and calls get_chat_completion
    # which uses httpx2.AsyncClient
    with patch("httpx2.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "thought": "Ready.",
                            "tool_call": None,
                            "final_answer": "Agent response content"
                        })
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        payload = {
            "model": "test-agent-model",
            "user_input": "Hello agent"
        }
        
        response = client.post("/api/agent/run", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "result": "Agent response content",
            "steps": [
                {
                    "step": {
                        "thought": "Ready.",
                        "tool_call": None,
                        "final_answer": "Agent response content"
                    },
                    "observation": None
                }
            ]
        }
        mock_post.assert_called_once()
