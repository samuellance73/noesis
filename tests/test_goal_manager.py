import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from integrations.llm.service import UpstreamService
from agents.goal_manager import GoalManager
from agents.schemas import GoalState, ManagerDecision, SubTask, WorldModel, WorldModelPatch, ExecutorType, Objective
from agents.memory.episodic_store import EpisodicStore, EpisodicEntry
from agents.memory.episodic_writer import EpisodicWriter
from agents.tools import build_specialized_registry

def test_build_specialized_registry():
    # Test research
    reg_research = build_specialized_registry("research")
    assert "web_search" in reg_research.tools
    assert len(reg_research.tools) == 1

    # Test code
    reg_code = build_specialized_registry("code")
    assert "python_execute" in reg_code.tools
    assert "run_command" in reg_code.tools
    assert len(reg_code.tools) == 2

    # Test synthesis
    reg_synth = build_specialized_registry("synthesis")
    assert len(reg_synth.tools) == 0

    # Test full / default fallback
    reg_full = build_specialized_registry("full")
    assert len(reg_full.tools) > 2


@pytest.mark.asyncio
async def test_goal_manager_with_memory_and_critic():
    # 1. Mock upstream LLM service
    llm_service = MagicMock(spec=UpstreamService)

    # Manager decision for cycle 1
    # It thoughts, identifies gaps, updates domains/beliefs, spawns one subtask with executor specialization
    decision_1 = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "I need to query information about gravity.",
                        "tasks_to_spawn": [
                            {
                                "goal": "Find the definition of gravity",
                                "context": "None",
                                "executor_type": "research"
                            }
                        ],
                        "progress_update": "Spawning gravity query.",
                        "world_model_patch": {
                            "gaps_closed": [],
                            "gaps_added": ["What is gravity?"],
                            "domain_updates": {"physics": "Exploring basic laws of motion."},
                            "belief_updates": {"Gravity exists": 0.99}
                        },
                        "updated_objectives": [
                            {
                                "id": "obj-1",
                                "description": "Obtain physics context",
                                "status": "active",
                                "spawned_cycle": 1
                            }
                        ],
                        "updated_open_questions": [],
                        "is_goal_complete": False,
                        "final_answer": None
                    })
                }
            }
        ]
    }

    # Critic rating for the subtask result (0.95 score)
    critic_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "score": 0.95,
                        "reason": "Accurate definition of gravity."
                    })
                }
            }
        ]
    }

    # Manager decision for cycle 2 (finishing the run)
    decision_2 = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "I have the gravity definition. Goal complete.",
                        "tasks_to_spawn": [],
                        "progress_update": "Gravity definition obtained.",
                        "world_model_patch": {
                            "gaps_closed": ["What is gravity?"],
                            "gaps_added": [],
                            "domain_updates": {"physics": "Gravity is a fundamental interaction."},
                            "belief_updates": {}
                        },
                        "updated_objectives": [
                            {
                                "id": "obj-1",
                                "description": "Obtain physics context",
                                "status": "complete",
                                "spawned_cycle": 1
                            }
                        ],
                        "updated_open_questions": [],
                        "is_goal_complete": True,
                        "final_answer": "Gravity pulls objects together."
                    })
                }
            }
        ]
    }

    # Set up mock completion sequence
    # Call 1: Manager LLM decision 1
    # Call 2: Critic LLM evaluating the subtask answer
    # Call 3: Manager LLM decision 2
    llm_service.get_chat_completion = AsyncMock()
    llm_service.get_chat_completion.side_effect = [decision_1, critic_response, decision_2]

    # Mock AgentExecutor.run_generator to yield a successful run directly
    async def mock_run_generator(enriched_task_input):
        yield {"event": "iteration_start", "iteration": 1}
        yield {"event": "thought", "thought": "I will answer this.", "step_index": 0}
        yield {"event": "final_answer", "answer": "Gravity is a force.", "step_index": 0}

    # Initialize manager
    manager = GoalManager(llm_service=llm_service, model="test-model", max_cycles=2)

    # Use patch to mock AgentExecutor.run_generator and EpisodicStore/Writer
    with patch("agents.goal_manager.AgentExecutor.run_generator", side_effect=mock_run_generator), \
         patch("agents.goal_manager.EpisodicStore.load_relevant", return_value=[]) as mock_load, \
         patch("agents.goal_manager.EpisodicWriter.write_summary") as mock_write:
        
        events = []
        async for event in manager.run_stream("Define gravity"):
            events.append(event)

        # Assertions
        assert len(events) > 0
        assert mock_load.called
        assert mock_write.called
        
        # Check that we received a 'goal_complete' event
        goal_complete_event = next((e for e in events if e["event"] == "goal_complete"), None)
        assert goal_complete_event is not None
        assert goal_complete_event["final_answer"] == "Gravity pulls objects together."
        
        # Verify call counts: 2 for GoalManager, 1 for Critic
        assert llm_service.get_chat_completion.call_count == 3
