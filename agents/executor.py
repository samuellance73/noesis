import json
import logging
from typing import AsyncGenerator
from integrations.llm.service import UpstreamService
from integrations.llm.schemas import ChatMessage, ChatPayload
from .schemas import AgentState, AgentStep
from .tools import tools_registry
from utils.tracer import current_aspan

logger = logging.getLogger(__name__)

# Observations longer than this are truncated before being fed back into
# the conversation — large observations are the primary driver of model looping.
_MAX_OBSERVATION_CHARS = 2_000

# Upstream APIs sometimes return error payloads inside a 200 OK response.
_UPSTREAM_ERROR_PREFIXES = (
    "error:",
    "model output error",
    "your output is flagged",
    "content policy",
    "rate limit",
    "context length",
)


def _is_upstream_error(content: str) -> bool:
    lowered = content.strip().lower()
    return any(lowered.startswith(p) for p in _UPSTREAM_ERROR_PREFIXES)


class AgentExecutor:
    def __init__(self, llm_service: UpstreamService, model: str):
        self.llm_service = llm_service
        self.model       = model
        self.state       = AgentState()

    @staticmethod
    def _parse_agent_step(raw: str) -> AgentStep:
        clean = raw.strip()
        if "<think>" in clean and "</think>" in clean:
            clean = clean.split("</think>", 1)[1].strip()
        elif "</think>" in clean:
            clean = clean.split("</think>", 1)[1].strip()
        if clean.startswith("```json"):
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif clean.startswith("```"):
            clean = clean.split("```")[1].split("```")[0].strip()
        try:
            clean = clean[clean.index("{"):clean.rindex("}") + 1]
        except ValueError:
            pass
        return AgentStep.model_validate(json.loads(clean, strict=False))

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

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user",   "content": state.user_input},
        ]

        for i in range(state.max_iterations):
            iteration_num = i + 1
            yield {"event": "iteration_start", "iteration": iteration_num}

            async with current_aspan(f"iteration[{iteration_num}]", model=self.model) as span:
                payload = ChatPayload(
                    model=self.model,
                    messages=[ChatMessage(**m) for m in messages],
                    temperature=0.1,
                    stream=False,
                )

                try:
                    raw_response = await self.llm_service.get_chat_completion(
                        payload.model_dump(exclude_none=True)
                    )
                except Exception as e:
                    span.log_error(str(e))
                    yield {"event": "error", "message": f"Upstream call failed: {str(e)}"}
                    return

                assistant_content = raw_response["choices"][0]["message"]["content"]

                # Detect upstream error payloads returned inside a 200 OK body
                if _is_upstream_error(assistant_content):
                    msg = assistant_content.strip()
                    logger.error("Upstream error payload in iteration %d: %r", iteration_num, msg)
                    span.log_error(f"Upstream error: {msg}")
                    yield {"event": "error", "message": f"Upstream model error: {msg}"}
                    return

                try:
                    parsed_step = self._parse_agent_step(assistant_content)
                except Exception as e:
                    span.log_error(f"Parse failure: {e}")
                    yield {"event": "error", "message": f"Failed to parse model instructions: {str(e)}"}
                    return

                messages.append({"role": "assistant", "content": assistant_content})

                logger.info("Thought [iter %d]: %s", iteration_num, parsed_step.thought)
                yield {"event": "thought", "thought": parsed_step.thought, "step_index": i}

                if parsed_step.final_answer:
                    state.steps.append({"step": parsed_step.model_dump(), "observation": None})
                    logger.info("Final answer reached after %d iteration(s).", iteration_num)
                    span.log_close(status="final_answer")
                    yield {"event": "final_answer", "answer": parsed_step.final_answer}
                    return

                if parsed_step.tool_call:
                    tool_name  = parsed_step.tool_call.tool_name
                    tool_input = parsed_step.tool_call.tool_input

                    logger.info("Tool call [iter %d]: %r  input=%r", iteration_num, tool_name, tool_input)
                    yield {"event": "tool_start", "tool_name": tool_name, "tool_input": tool_input, "step_index": i}

                    observation = await tools_registry.execute(tool_name, tool_input)

                    # Truncate to prevent context bloat → model looping
                    if len(observation) > _MAX_OBSERVATION_CHARS:
                        extra = len(observation) - _MAX_OBSERVATION_CHARS
                        logger.warning(
                            "Observation from %r truncated: %d → %d chars (+%d omitted)",
                            tool_name, len(observation), _MAX_OBSERVATION_CHARS, extra,
                        )
                        observation = observation[:_MAX_OBSERVATION_CHARS] + f"\n... [truncated: {extra} chars omitted]"

                    yield {"event": "tool_observation", "tool_name": tool_name, "observation": observation, "step_index": i}
                    messages.append({"role": "user", "content": f"Observation from '{tool_name}': {observation}"})
                    state.steps.append({"step": parsed_step.model_dump(), "observation": observation})
                    span.log_close(status="tool_called", tool=tool_name)

                else:
                    span.log_error("No tool call or final answer was provided.")
                    yield {"event": "error", "message": "No tool call or final answer was provided."}
                    break

        yield {"event": "error", "message": "Execution limit reached without finding a final answer."}
