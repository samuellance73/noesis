import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.events import UnifiedIngestEvent, SenderClass, PriorityLevel
from perception.schemas import ScoredSignal, PerceptionWorldModel
from triggers.triage import BatchTriageDecision, SlowPathEscalation
from perception.dispatcher import BatchActionDispatcher
from utils.event_bus import event_bus
from core.model_router import ModelRouter


@pytest.mark.asyncio
async def test_batch_action_dispatcher_slow_path_metadata_injection():
    # Setup test event
    event = UnifiedIngestEvent(
        source_channel="discord",
        sender_identifier="test_user",
        sender_class=SenderClass.EXTERNAL,
        raw_content="hello",
        target_conversation_identifier="123456",
        priority_level=PriorityLevel.NORMAL,
        metadata={"message_id": 999},
    )
    sig = ScoredSignal(
        representative=event,
        frequency=1,
        perception_type=None,
        authority_score=0.5,
    )

    decision = BatchTriageDecision(
        rationale="testing metadata injection",
        fast_path_actions=[],
        slow_path_escalations=[
            SlowPathEscalation(signal_index=0, refined_goal="refined test goal")
        ],
    )

    # Subscribe to event bus to capture published events
    q = event_bus.subscribe()

    # Mock GoalManager
    mock_goal_manager = MagicMock()
    
    async def mock_run_stream(goal, run_id):
        yield {"event": "goal_set", "goal": goal}
        yield {"event": "goal_complete", "final_answer": "done!"}

    mock_goal_manager.run_stream = mock_run_stream

    # Mock ModelRouter and WorldModel
    mock_router = MagicMock(spec=ModelRouter)
    mock_world_model = MagicMock(spec=PerceptionWorldModel)

    with patch("perception.dispatcher.GoalManager", return_value=mock_goal_manager):
        # Run dispatcher
        await BatchActionDispatcher.dispatch([sig], decision, mock_world_model, mock_router)
        
        # Give some time for background tasks to process
        await asyncio.sleep(0.1)

    # Retrieve published events
    events = []
    while not q.empty():
        events.append(q.get_nowait())

    event_bus.unsubscribe(q)

    # Verify the event metadata and source injection
    assert len(events) == 2
    for ev in events:
        assert ev["trigger_source"] == "perception"
        assert ev["trigger_metadata"]["message_id"] == 999
        # Fallback channel_id was resolved from target_conversation_identifier
        assert ev["trigger_metadata"]["channel_id"] == 123456
