import json
import logging
from typing import Optional, AsyncGenerator
from integrations.llm.service import UpstreamService
from integrations.llm.schemas import ChatMessage, ChatPayload
from .schemas import AgentState, AgentStep
from .tools import tools_registry

logger = logging.getLogger(__name__)

class AgentExecutor:
    def __init__(self, llm_service: UpstreamService, model: str):
        self.llm_service = llm_service
        self.model = model
        self.state = AgentState()

    @staticmethod
    def _parse_agent_step(raw: str) -> AgentStep:
        clean = raw.strip()
        
        # 1. Remove <think>...</think> tags and content if present
        if "<think>" in clean and "</think>" in clean:
            parts = clean.split("</think>", 1)
            clean = parts[1].strip()
        elif "</think>" in clean: # Handle edge case where <think> was truncated
            clean = clean.split("</think>", 1)[1].strip()
            
        # 2. Strip out markdown code blocks if the model wrapped the JSON
        if clean.startswith("```json"):
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif clean.startswith("```"):
            clean = clean.split("```")[1].split("```")[0].strip()
            
        # 3. Robust extraction: locate the outermost JSON object braces
        # This strips away any stray conversational text the model appended before or after the JSON
        try:
            start_idx = clean.index("{")
            end_idx = clean.rindex("}")
            clean = clean[start_idx:end_idx + 1]
        except ValueError:
            # If braces are missing altogether, let the validation pass 
            # to let Pydantic raise a standard schema validation error
            pass

        # 4. Parse with strict=False to allow raw control characters
        parsed_dict = json.loads(clean, strict=False)
        return AgentStep.model_validate(parsed_dict)

    def _build_system_prompt(self) -> str:
        tool_docs = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in tools_registry.tools.items()
        )
        return (
            "You are an agent with access to tools. You must execute steps sequentially: "
            "Think about what to do, select a tool, analyze the observation, and decide the next step.\n"
            f"You have access to the following tools:\n{tool_docs}\n\n"
            "You MUST respond ONLY with a JSON object in the following format:\n"
            "{\n"
            '  "thought": "your reasoning here",\n'
            '  "tool_call": {"tool_name": "web_search", "tool_input": "query"} or null,\n'
            '  "final_answer": "your response to the user" or null\n'
            "}\n"
            "Do not include any text outside the JSON block."
        )

    async def run(self, user_input: str) -> str:
        async for event in self.run_generator(user_input):
            if event["event"] == "final_answer":
                return event["answer"]
            if event["event"] == "error":
                return event["message"]
        return "Agent reached execution limit."

    async def run_generator(self, user_input: str) -> AsyncGenerator[dict, None]:
        self.state.user_input = user_input
        state = self.state
        
        # Build prompt messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": state.user_input}
        ]

        for i in range(state.max_iterations):
            # Send status update of current iteration
            yield {"event": "iteration_start", "iteration": i + 1}

            payload = ChatPayload(
                model=self.model,
                messages=[ChatMessage(**m) for m in messages],
                temperature=0.1,
                stream=False
            )
            
            try:
                raw_response = await self.llm_service.get_chat_completion(
                    payload.model_dump(exclude_none=True)
                )
            except Exception as e:
                yield {"event": "error", "message": f"Upstream call failed: {str(e)}"}
                return

            assistant_content = raw_response["choices"][0]["message"]["content"]
            
            try:
                parsed_step = self._parse_agent_step(assistant_content)
            except Exception as e:
                yield {"event": "error", "message": f"Failed to parse model instructions: {str(e)}"}
                return

            messages.append({"role": "assistant", "content": assistant_content})

            # Stream the reasoning thought to the UI
            yield {
                "event": "thought", 
                "thought": parsed_step.thought, 
                "step_index": i
            }

            if parsed_step.final_answer:
                state.steps.append({
                    "step": parsed_step.model_dump(),
                    "observation": None
                })
                yield {"event": "final_answer", "answer": parsed_step.final_answer}
                return

            if parsed_step.tool_call:
                tool_name = parsed_step.tool_call.tool_name
                tool_input = parsed_step.tool_call.tool_input

                # Notify the UI that a tool execution is beginning
                yield {
                    "event": "tool_start", 
                    "tool_name": tool_name, 
                    "tool_input": tool_input,
                    "step_index": i
                }
                
                observation = await tools_registry.execute(tool_name, tool_input)
                
                # Notify the UI with the tool's result
                yield {
                    "event": "tool_observation", 
                    "tool_name": tool_name, 
                    "observation": observation,
                    "step_index": i
                }

                observation_message = {
                    "role": "user", 
                    "content": f"Observation from '{tool_name}': {observation}"
                }
                messages.append(observation_message)
                
                state.steps.append({
                    "step": parsed_step.model_dump(),
                    "observation": observation
                })
            else:
                yield {"event": "error", "message": "No tool call or final answer was provided."}
                break

        yield {"event": "error", "message": "Execution limit reached without finding a final answer."}
