import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.model_router import ModelRouter, ModelResponse, ModelTier
from agents.goal_manager import GoalManager
from agents.schemas import GoalState, ManagerDecision, SubTask, WorldModel, WorldModelPatch, ExecutorType, Objective
from agents.memory.episodic_store import EpisodicStore, EpisodicEntry
from agents.memory.episodic_writer import EpisodicWriter
from agents.tools import build_specialized_registry


def _mock_router(responses: list[str]) -> ModelRouter:
    """
    Build a mock ModelRouter whose .complete() sequentially returns ModelResponse
    objects constructed from the provided content strings.

    The first two responses cover STRONG tier (manager) calls;
    the middle response covers NANO tier (critic) calls.
    All use the same side_effect list so call order matches the test sequence.
    """
    router = MagicMock(spec=ModelRouter)
    router.complete = AsyncMock(side_effect=[
        ModelResponse(
            content=c,
            model_used="test-model",
            tier=ModelTier.STRONG,
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            latency_ms=1.0,
        )
        for c in responses
    ])
    router.resolve_model = MagicMock(return_value="test-model")
    router.config = MagicMock()
    router.config.tiers = {
        ModelTier.STRONG: MagicMock(context_budget=32000),
        ModelTier.NANO:   MagicMock(context_budget=3000),
    }
    return router


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
    # Manager decision for cycle 1 — spawns one subtask
    decision_1_content = json.dumps({
        "thought": "I need to query information about gravity.",
        "tasks_to_spawn": [
            {
                "goal": "Find the definition of gravity",
                "context": "None",
                "executor_type": "research",
            }
        ],
        "progress_update": "Spawning gravity query.",
        "world_model_patch": {
            "gaps_closed": [],
            "gaps_added": ["What is gravity?"],
            "domain_updates": {"physics": "Exploring basic laws of motion."},
            "belief_updates": {"Gravity exists": 0.99},
        },
        "updated_objectives": [
            {
                "id": "obj-1",
                "description": "Obtain physics context",
                "status": "active",
                "spawned_cycle": 1,
            }
        ],
        "updated_open_questions": [],
        "is_goal_complete": False,
        "final_answer": None,
    })

    # Critic response (NANO tier in production, same mock here)
    critic_content = json.dumps({
        "score": 0.95,
        "reason": "Accurate definition of gravity.",
    })

    # Manager decision for cycle 2 — declares completion
    decision_2_content = json.dumps({
        "thought": "I have the gravity definition. Goal complete.",
        "tasks_to_spawn": [],
        "progress_update": "Gravity definition obtained.",
        "world_model_patch": {
            "gaps_closed": ["What is gravity?"],
            "gaps_added": [],
            "domain_updates": {"physics": "Gravity is a fundamental interaction."},
            "belief_updates": {},
        },
        "updated_objectives": [
            {
                "id": "obj-1",
                "description": "Obtain physics context",
                "status": "complete",
                "spawned_cycle": 1,
            }
        ],
        "updated_open_questions": [],
        "is_goal_complete": True,
        "final_answer": "Gravity pulls objects together.",
    })

    # Call order: manager decision 1, critic evaluation, manager decision 2
    router = _mock_router([decision_1_content, critic_content, decision_2_content])

    # Mock AgentExecutor.run_generator to yield a successful subtask directly
    async def mock_run_generator(enriched_task_input):
        yield {"event": "iteration_start", "iteration": 1}
        yield {"event": "thought", "thought": "I will answer this.", "step_index": 0}
        yield {"event": "final_answer", "answer": "Gravity is a force.", "step_index": 0}

    manager = GoalManager(router=router, max_cycles=2)

    with patch("agents.goal_manager.AgentExecutor.run_generator", side_effect=mock_run_generator), \
         patch("agents.goal_manager.EpisodicStore.load_relevant", return_value=[]) as mock_load, \
         patch("agents.goal_manager.EpisodicWriter.write_summary") as mock_write:

        events = []
        async for event in manager.run_stream("Define gravity"):
            events.append(event)

        assert len(events) > 0
        assert mock_load.called
        assert mock_write.called

        # Check that we received a 'goal_complete' event
        goal_complete_event = next((e for e in events if e["event"] == "goal_complete"), None)
        assert goal_complete_event is not None
        assert goal_complete_event["final_answer"] == "Gravity pulls objects together."

        # Verify total router.complete call count: 2 manager + 1 critic
        assert router.complete.call_count == 3
