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
        assert "Observation from 'web_search'" in messages[3]["content"]
        assert "Search result placeholder for: Paris weather" in messages[3]["content"]
    finally:
        tools_registry.tools["web_search"] = original_tool


def test_agent_endpoint_success(client):
    # Mock upstream client's post call, since router resolves service and calls get_chat_completion
    # which uses httpx2.AsyncClient
    with patch("httpx2.AsyncClient.post") as mock_post:
        # First call (planning stage)
        mock_response_1 = MagicMock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps([
                            {"id": 1, "goal": "Execute the task", "depends_on": []}
                        ])
                    }
                }
            ]
        }
        mock_response_1.raise_for_status = MagicMock()

        # Second call (execution stage)
        mock_response_2 = MagicMock()
        mock_response_2.status_code = 200
        mock_response_2.json.value = {
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
        mock_response_2.json.return_value = {
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
        mock_response_2.raise_for_status = MagicMock()

        mock_post.side_effect = [mock_response_1, mock_response_2]
        
        payload = {
            "model": "test-agent-model",
            "user_input": "Hello agent",
            "stream": False
        }
        
        response = client.post("/api/agent/run", json=payload)
        assert response.status_code == 200
        assert response.json() == {
            "milestones": [{"id": 1, "goal": "Execute the task", "depends_on": []}],
            "results": [{"milestone": "Execute the task", "result": "Agent response content"}]
        }
        assert mock_post.call_count == 2


def test_parse_agent_step_resilient():
    # 1. Clean JSON
    raw = '{"thought": "think", "tool_call": null, "final_answer": "ans"}'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"
    assert step.final_answer == "ans"

    # 2. With think tags
    raw = '<think>some internal thought</think>\n{"thought": "think", "tool_call": null, "final_answer": "ans"}'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"

    # 3. With markdown block
    raw = '```json\n{"thought": "think", "tool_call": null, "final_answer": "ans"}\n```'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"

    # 4. With trailing commas
    raw = '{"thought": "think", "tool_call": null, "final_answer": "ans",}'
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"

    # 5. Single quotes repair
    raw = "{'thought': 'think', 'tool_call': null, 'final_answer': 'ans'}"
    step = AgentExecutor._parse_agent_step(raw)
    assert step.thought == "think"
    assert step.final_answer == "ans"


@pytest.mark.asyncio
async def test_orchestrator_fail_fast_stream():
    from agents.orchestrator import AgentOrchestrator
    
    llm_service = MagicMock(spec=UpstreamService)
    orchestrator = AgentOrchestrator(llm_service=llm_service, model="test-model")

    # Mock planning to return two milestones
    milestones = [
        {"id": 1, "goal": "Milestone 1", "depends_on": []},
        {"id": 2, "goal": "Milestone 2", "depends_on": [1]}
    ]

    with patch("agents.orchestrator.plan", AsyncMock(return_value=milestones)):
        # Mock executor generator to yield error for milestone 1
        async def mock_run_generator_fail(goal):
            yield {"event": "iteration_start", "iteration": 1}
            yield {"event": "error", "message": "Simulated tool error"}

        with patch("agents.executor.AgentExecutor.run_generator", side_effect=mock_run_generator_fail):
            events = []
            async for event in orchestrator.run_stream("do something"):
                events.append(event)
            
            # Verify we aborted and did not run Milestone 2
            # Event sequence should contain planning_start, plan_ready, step_start (for 0), iteration_start, error (from executor), error (from orchestrator done fail fast)
            # Make sure Milestone 2's step_start (step_index=1) is NOT in events
            step_starts = [e for e in events if e.get("event") == "step_start"]
            assert len(step_starts) == 1
            assert step_starts[0]["step_index"] == 0
            
            # Should end with orchestrator abort error event
            assert events[-1] == {"event": "error", "message": "Execution aborted: Milestone 1 failed."}


@pytest.mark.asyncio
async def test_orchestrator_fail_fast_non_stream():
    from agents.orchestrator import AgentOrchestrator
    
    llm_service = MagicMock(spec=UpstreamService)
    orchestrator = AgentOrchestrator(llm_service=llm_service, model="test-model")

    # Mock planning to return two milestones
    milestones = [
        {"id": 1, "goal": "Milestone 1", "depends_on": []},
        {"id": 2, "goal": "Milestone 2", "depends_on": [1]}
    ]

    with patch("agents.orchestrator.plan", AsyncMock(return_value=milestones)):
        # Mock executor run to return error/failed text on milestone 1
        with patch("agents.executor.AgentExecutor.run", AsyncMock(return_value="Error: limit reached")):
            result = await orchestrator.run("do something")
            
            # Verify results indicate abortion and only contains the error result
            assert len(result["results"]) == 1
            assert result["results"][0]["milestone"] == "Milestone 1"
            assert "Aborted: Dependency failed" in result["results"][0]["result"]

