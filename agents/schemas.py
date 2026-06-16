from pydantic import BaseModel, Field
from typing import List, Optional, Any

class ToolCall(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to execute.")
    tool_input: Any = Field(..., description="The parameters to pass to the tool.")

class AgentStep(BaseModel):
    thought: str = Field(..., description="The agent's internal reasoning about the next step.")
    tool_call: Optional[ToolCall] = Field(None, description="The tool the agent wants to run.")
    final_answer: Optional[str] = Field(None, description="The final response to the user, if complete.")

class AgentState(BaseModel):
    user_input: str = ""
    history: List[dict] = []
    steps: List[dict] = []  # Logs of previous thoughts and tool execution results
    max_iterations: int = 5
