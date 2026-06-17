from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to execute.")
    tool_input: Any = Field(..., description="The parameters to pass to the tool.")

class AgentStep(BaseModel):
    thought: str = Field(..., description="The agent's internal reasoning about the next step.")
    tool_call: Optional[ToolCall] = Field(None, description="The tool the agent wants to run.")
    final_answer: Optional[str] = Field(None, description="The final response to the user, if complete.")

    @model_validator(mode="before")
    @classmethod
    def normalize_tool_call(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Support plural 'tool_calls' from some models
        if "tool_calls" in data and data["tool_calls"] is not None and data.get("tool_call") is None:
            tc = data["tool_calls"]
            if isinstance(tc, list) and len(tc) > 0:
                data["tool_call"] = tc[0]
            elif isinstance(tc, dict):
                data["tool_call"] = tc

        # Support when tool_call itself is mistakenly returned as a list
        if "tool_call" in data and isinstance(data["tool_call"], list):
            tc_list = data["tool_call"]
            if len(tc_list) > 0:
                data["tool_call"] = tc_list[0]
            else:
                data["tool_call"] = None

        # Support mapping older/alternative keys inside tool_call
        if "tool_call" in data and isinstance(data["tool_call"], dict):
            tc = data["tool_call"]
            if "tool" in tc and "tool_name" not in tc:
                tc["tool_name"] = tc["tool"]
            if "query" in tc and "tool_input" not in tc:
                tc["tool_input"] = tc["query"]
            if "input" in tc and "tool_input" not in tc:
                tc["tool_input"] = tc["input"]

        return data

class AgentState(BaseModel):
    user_input: str = ""
    history: List[dict] = []
    steps: List[dict] = []  # Logs of previous thoughts and tool execution results
    max_iterations: int = 6

