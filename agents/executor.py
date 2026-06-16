import json
import logging
from typing import Optional
from integrations.llm.service import UpstreamService
from integrations.llm.schemas import ChatMessage, ChatPayload
from .schemas import AgentState, AgentStep
from .tools import tools_registry

logger = logging.getLogger(__name__)

class AgentExecutor:
    def __init__(self, llm_service: UpstreamService, model: str):
        self.llm_service = llm_service
        self.model = model
        self.state = None

    def _build_system_prompt(self) -> str:
        return (
            "You are an agent with access to tools. You must execute steps sequentially: "
            "Think about what to do, select a tool, analyze the observation, and decide the next step.\n"
            "You have access to the following tools:\n"
            "- web_search: Use this to search information. Input is a search query string.\n\n"
            "You MUST respond ONLY with a JSON object in the following format:\n"
            "{\n"
            '  "thought": "your reasoning here",\n'
            '  "tool_call": {"tool_name": "web_search", "tool_input": "query"} or null,\n'
            '  "final_answer": "your response to the user" or null\n'
            "}\n"
            "Do not include any text outside the JSON block."
        )

    async def run(self, user_input: str) -> str:
        self.state = AgentState(user_input=user_input)
        state = self.state
        
        # Build prompt messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": state.user_input}
        ]

        for i in range(state.max_iterations):
            logger.info(f"Agent Loop Iteration {i + 1}")

            # 1. Ask the LLM what to do next
            payload = ChatPayload(
                model=self.model,
                messages=[ChatMessage(**m) for m in messages],
                temperature=0.1,  # Lower temperature helps with structured logic output
                stream=False
            )
            
            raw_response = await self.llm_service.get_chat_completion(
                payload.model_dump(exclude_none=True)
            )
            
            # Extract content (assuming OpenAI format)
            assistant_content = raw_response["choices"][0]["message"]["content"]
            
            # 2. Parse the LLM output
            try:
                # Basic JSON sanitization if the LLM includes markdown code blocks
                clean_json = assistant_content.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                elif clean_json.startswith("```"):
                    clean_json = clean_json.split("```")[1].split("```")[0].strip()

                parsed_step = AgentStep.model_validate_json(clean_json)
            except Exception as e:
                logger.error(f"Failed to parse agent step: {e}. Content: {assistant_content}")
                return "The agent failed to parse its internal instructions."

            # Append the LLM's thought processes to history to preserve agent state
            messages.append({"role": "assistant", "content": assistant_content})

            # Check for termination
            if parsed_step.final_answer:
                state.steps.append({
                    "step": parsed_step.model_dump(),
                    "observation": None
                })
                return parsed_step.final_answer

            if parsed_step.tool_call:
                tool_name = parsed_step.tool_call.tool_name
                tool_input = parsed_step.tool_call.tool_input

                logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                
                # 3. Act: Execute the tool
                observation = await tools_registry.execute(tool_name, tool_input)
                
                # 4. Perceive: Put the action observation back into context
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
                # If there's no tool call and no final answer, stop the loop to prevent deadlocks
                logger.warning("Agent returned no tool call or final answer.")
                break

        return "The agent reached its execution limit without finding a final answer."
