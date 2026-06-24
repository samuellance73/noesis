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


class CompletedTask(BaseModel):
    """A sub-task that an executor has successfully answered."""
    goal: str = Field(..., description="The sub-task goal that was executed.")
    answer: str = Field(..., description="The executor's concrete answer for this task.")


class FailedTask(BaseModel):
    """A sub-task that an executor attempted but could not answer."""
    goal: str = Field(..., description="The sub-task goal that failed.")
    attempts: int = Field(1, description="How many times this task has been attempted.")
    give_up_after: int = Field(2, description="Stop retrying once attempts reaches this value.")
    last_error: str = Field("", description="Last error or failure reason (if available).")

    @property
    def is_abandoned(self) -> bool:
        """True when the retry budget is exhausted — manager should re-plan around this."""
        return self.attempts >= self.give_up_after


class WorldModel(BaseModel):
    domain_map: dict[str, str] = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)
    findings: list[dict] = Field(default_factory=list)
    beliefs: dict[str, float] = Field(default_factory=dict)
    # Per-cycle manager notes — used by EpisodicWriter for long-term storage
    cycle_summaries: list[str] = Field(default_factory=list)

class WorldModelPatch(BaseModel):
    gaps_closed: list[str] = Field(default_factory=list)
    gaps_added: list[str] = Field(default_factory=list)
    domain_updates: dict[str, str] = Field(default_factory=dict)
    belief_updates: dict[str, float] = Field(default_factory=dict)

class GoalState(BaseModel):
    """
    Persistent state owned by the GoalManager across autonomous cycles.

    The `reflection` field is intentionally kept for future metacognition:
    the agent can store a self-assessment of its own reasoning quality here.
    """
    ultimate_goal: str
    cycle: int = 0
    task_counter: int = 0
    world_model: WorldModel = Field(default_factory=WorldModel)
    # Completed sub-tasks with their verified answers — kept together so they
    # can never fall out of sync.
    completed: List[CompletedTask] = Field(default_factory=list)
    # Tasks that executors attempted but failed to produce an answer for.
    # Surfaced to the manager so it can retry, reframe, or permanently abandon them.
    failed_tasks: List[FailedTask] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    is_complete: bool = False
    # Hook for metacognition — populated by a future reflection step
    reflection: Optional[str] = None

    def record_success(self, goal: str, answer: str) -> None:
        """Mark a sub-task as successfully completed and remove from failed list."""
        self.completed.append(CompletedTask(goal=goal, answer=answer))
        self.failed_tasks = [f for f in self.failed_tasks if f.goal != goal]

    def record_failure(self, goal: str) -> None:
        """Increment the failure counter for a sub-task, or add it if new."""
        existing = next((f for f in self.failed_tasks if f.goal == goal), None)
        if existing:
            existing.attempts += 1
        else:
            self.failed_tasks.append(FailedTask(goal=goal))


class ManagerDecision(BaseModel):
    """
    What the GoalManager decides to do on each autonomous cycle.

    - If `is_goal_complete` is True the loop stops.
    - If `tasks_to_spawn` is non-empty, executors are launched in parallel.
    - `progress_update` is streamed to the user / UI after each cycle.
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
    world_model_patch: Optional[WorldModelPatch] = Field(
        None,
        description="New patches to apply to the GoalState's WorldModel (omit or leave empty to keep unchanged).",
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

