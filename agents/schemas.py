from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any


# ─── Tool-level schemas ───────────────────────────────────────────────────────

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to execute.")
    tool_input: Any = Field(..., description="The parameters to pass to the tool.")


class AgentStep(BaseModel):
    thought: str = Field(..., description="The agent's internal reasoning about the next step.")
    # Supports a list of tool calls to run concurrently.
    # Single tool_call (legacy) is normalised into this list by the validator below.
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="Tools to call concurrently. Leave empty if none.",
    )
    final_answer: Optional[str] = Field(None, description="The final response to the user, if complete.")

    @model_validator(mode="before")
    @classmethod
    def normalize_tool_calls(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # ------------------------------------------------------------------
        # Normalise legacy singular `tool_call` key into `tool_calls` list
        # ------------------------------------------------------------------
        singular = data.get("tool_call")
        plural   = data.get("tool_calls")

        if singular is not None and not plural:
            if isinstance(singular, dict):
                data["tool_calls"] = [singular]
            elif isinstance(singular, list):
                data["tool_calls"] = singular

        if data.get("tool_calls") is None:
            data["tool_calls"] = []

        # ------------------------------------------------------------------
        # Normalise alternative key names inside each tool call entry
        # ------------------------------------------------------------------
        normalised_calls = []
        for tc in data.get("tool_calls", []):
            if isinstance(tc, dict):
                if "tool" in tc and "tool_name" not in tc:
                    tc["tool_name"] = tc["tool"]
                if "query" in tc and "tool_input" not in tc:
                    tc["tool_input"] = tc["query"]
                if "input" in tc and "tool_input" not in tc:
                    tc["tool_input"] = tc["input"]
            normalised_calls.append(tc)
        data["tool_calls"] = normalised_calls

        return data


class AgentState(BaseModel):
    user_input: str = ""
    history: List[dict] = []
    steps: List[dict] = []
    max_iterations: int = 5


# ─── Goal-level schemas (autonomous loop) ────────────────────────────────────

class SubTask(BaseModel):
    """A focused unit of work to be handed to an AgentExecutor."""
    goal: str = Field(..., description="The specific sub-goal for this execution.")
    context: str = Field(
        default="",
        description="Relevant background the executor needs to know (prior findings, constraints, etc.).",
    )


class GoalState(BaseModel):
    """
    Persistent state owned by the GoalManager across autonomous cycles.

    The `reflection` field is intentionally kept for future metacognition:
    the agent can store a self-assessment of its own reasoning quality here.
    """
    ultimate_goal: str
    progress_summary: str = ""
    completed_tasks: List[str] = Field(default_factory=list)
    # Parallel to completed_tasks — stores the executor's concrete answer for each task.
    # Both lists grow together so completed_tasks[i] ↔ completed_answers[i].
    completed_answers: List[str] = Field(default_factory=list)
    # Tasks that executors attempted but failed to produce an answer for.
    # Surfaced to the manager so it can retry, reframe, or de-prioritise them.
    failed_tasks: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    cycle: int = 0
    is_complete: bool = False
    # Monotonic task counter — increments across cycles so IDs never reset.
    task_counter: int = 0
    # Hook for metacognition — populated by a future reflection step
    reflection: Optional[str] = None


class ManagerDecision(BaseModel):
    """
    What the GoalManager decides to do on each autonomous cycle.

    - If `is_goal_complete` is True the loop stops.
    - If `tasks_to_spawn` is non-empty, executors are launched in parallel.
    - `progress_update` is streamed to the user / UI after each cycle.
    - `goal_state_update` optionally replaces fields in the current GoalState.
    """
    thought: str = Field(..., description="Manager's internal reasoning about the current state of the goal.")
    tasks_to_spawn: List[SubTask] = Field(
        default_factory=list,
        description="Sub-tasks to execute concurrently this cycle. Empty means synthesise/respond.",
    )
    progress_update: str = Field(
        ...,
        description="A human-readable status message to stream to the user after this cycle.",
    )
    updated_progress_summary: Optional[str] = Field(
        None,
        description="New value for GoalState.progress_summary (omit to keep unchanged).",
    )
    updated_open_questions: Optional[List[str]] = Field(
        None,
        description="New list of open questions (omit to keep unchanged).",
    )
    is_goal_complete: bool = Field(
        False,
        description="Set to true when the ultimate goal has been fully achieved.",
    )
    final_answer: Optional[str] = Field(
        None,
        description="The complete final response to deliver to the user when is_goal_complete is true.",
    )
