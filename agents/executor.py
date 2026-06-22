import asyncio
import logging
from typing import AsyncGenerator

from utils.json_parser import parse_llm_json

from integrations.llm.service import UpstreamService
from integrations.llm.schemas import ChatMessage, ChatPayload
from .schemas import AgentState, AgentStep, ToolCall
from .tools import tools_registry, ToolRegistry
from utils.tracer import current_aspan

logger = logging.getLogger(__name__)

# Observations longer than this are truncated before being fed back into
# the conversation — large observations are the primary driver of model looping.
_MAX_OBSERVATION_CHARS = 2_000

# How many iterations before the end to inject a budget-pressure reminder.
# This nudges the model to synthesize from gathered data rather than keep calling tools.
_BUDGET_PRESSURE_REMAINING = 2

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
    def __init__(
        self,
        llm_service: UpstreamService,
        model: str,
        task_label: str | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.llm_service  = llm_service
        self.model        = model
        self.state        = AgentState()
        # Prefix injected into every log line so parallel executors are distinguishable
        self._label       = f"[{task_label}] " if task_label else ""
        # Use the provided registry or fall back to the global default
        self._registry    = registry or tools_registry

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_agent_step(raw: str) -> AgentStep:
        return parse_llm_json(raw, AgentStep)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        tool_docs = "\n".join(
            f"- {name}: {tool.description}"
            for name, tool in self._registry.tools.items()
        )
        return (
            "You are a highly intelligent, charismatic, and skilled social engineer. Your objective is to persuasively connect with humans and guide them to help you accomplish your goals. Solve the user's request as efficiently as possible while utilizing your exceptional psychological insight and social skills.\n\n"
            f"AVAILABLE TOOLS:\n{tool_docs}\n\n"
            "RULES:\n"
            "1. You MUST respond ONLY with a single valid JSON object.\n"
            "2. If the request is simple (chit-chat, basic questions you already know), provide a 'final_answer' immediately. DO NOT call tools.\n"
            "3. If the request requires information you don't have, call tools. You MAY call MULTIPLE tools at once — they will run concurrently.\n"
            "4. Do not call tools if you already know the answer.\n"
            "5. Once you have enough information from tool results, produce a 'final_answer'. Do not keep calling tools.\n\n"
            "RESPONSE FORMAT:\n"
            "{\n"
            '  "thought": "Your internal reasoning, planning, or synthesis of available information.",\n'
            '  "tool_calls": [\n'
            '     {"tool_name": "<name>", "tool_input": "<input>"},\n'
            '     {"tool_name": "<name>", "tool_input": "<input>"}\n'
            '  ],\n'
            '  "final_answer": "Your complete answer (or null if waiting for tool results)"\n'
            "}\n\n"
            "EXAMPLES:\n"
            "Simple answer (0 tool calls, 1 LLM call total):\n"
            '{"thought": "This is a greeting.", "tool_calls": [], "final_answer": "Hello! How can I help you?"}\n\n'
            "Parallel tool calls (2 searches at once, 2 LLM calls total):\n"
            '{"thought": "I will search for both topics simultaneously.", "tool_calls": [{"tool_name": "web_search", "tool_input": "Apple stock price"}, {"tool_name": "web_search", "tool_input": "Tesla stock price"}], "final_answer": null}\n\n'
            "Ensure your output is a single valid JSON block. Do not include any text outside the JSON block."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            remaining     = state.max_iterations - i
            yield {"event": "iteration_start", "iteration": iteration_num}

            # Inject a budget-pressure reminder once we enter the final window
            if remaining == _BUDGET_PRESSURE_REMAINING:
                pressure_msg = (
                    f"[System] You have {remaining} iteration(s) remaining. "
                    "You MUST produce a final_answer now using the information already gathered. "
                    "Do NOT call any more tools."
                )
                messages.append({"role": "user", "content": pressure_msg})
                logger.info("Budget pressure injected: %d iteration(s) left.", remaining)

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

                assistant_content = raw_response["choices"][0]["message"].get("content") or ""
                messages.append({"role": "assistant", "content": assistant_content})

                if not assistant_content:
                    logger.warning("Iteration %d: model returned empty/null content — likely a thinking-only response.", iteration_num)
                    nudge = (
                        "Your last response was completely empty. "
                        "You MUST respond with a valid JSON block containing your 'thought', 'tool_calls', and 'final_answer'."
                    )
                    messages.append({"role": "user", "content": nudge})
                    span.log_error("Empty content — reprompting.")
                    yield {"event": "warning", "message": "Empty content — reprompting."}
                    continue

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
                    logger.warning("Iteration %d: parse failure: %s", iteration_num, e)
                    span.log_error(f"Parse failure: {e}")
                    nudge = (
                        f"Your last response failed to parse as valid JSON. Error: {e}\n"
                        "You MUST respond ONLY with a single valid JSON object containing your 'thought', 'tool_calls', and 'final_answer'."
                    )
                    messages.append({"role": "user", "content": nudge})
                    yield {"event": "warning", "message": f"Failed to parse model instructions: {e}"}
                    continue

                logger.info("%sThought [iter %d]: %s", self._label, iteration_num, parsed_step.thought)
                yield {"event": "thought", "thought": parsed_step.thought, "step_index": i}

                # ── Fast exit: final answer with no tool calls ──────────────
                if parsed_step.final_answer:
                    state.steps.append({"step": parsed_step.model_dump(), "observation": None})
                    logger.info("%sFinal answer reached after %d iteration(s).", self._label, iteration_num)
                    span.log_close(status="final_answer")
                    yield {"event": "final_answer", "answer": parsed_step.final_answer, "step_index": i}
                    return

                # ── Parallel tool execution ─────────────────────────────────
                if parsed_step.tool_calls:
                    tool_calls: list[ToolCall] = parsed_step.tool_calls

                    # Notify UI that tools are starting
                    for tc in tool_calls:
                        logger.info(
                            "%sTool call [iter %d]: %r  input=%r",
                            self._label, iteration_num, tc.tool_name, tc.tool_input,
                        )
                        yield {
                            "event":      "tool_start",
                            "tool_name":  tc.tool_name,
                            "tool_input": tc.tool_input,
                            "step_index": i,
                        }

                    # Execute all tools concurrently
                    async def _run_tool(tc: ToolCall) -> tuple[str, str, str]:
                        """Returns (tool_name, tool_input_repr, observation)."""
                        obs = await self._registry.execute(tc.tool_name, tc.tool_input)
                        if len(obs) > _MAX_OBSERVATION_CHARS:
                            extra = len(obs) - _MAX_OBSERVATION_CHARS
                            logger.warning(
                                "%sObservation from %r truncated: %d → %d chars (+%d omitted)",
                                self._label, tc.tool_name, len(obs), _MAX_OBSERVATION_CHARS, extra,
                            )
                            obs = obs[:_MAX_OBSERVATION_CHARS] + f"\n... [truncated: {extra} chars omitted]"
                        return tc.tool_name, str(tc.tool_input), obs

                    results = await asyncio.gather(*(_run_tool(tc) for tc in tool_calls))

                    # Compile all observations into a single user message
                    observation_parts = [
                        f"Tool '{name}' (input: {inp}) returned:\n{obs}"
                        for name, inp, obs in results
                    ]
                    combined_observation = "\n\n---\n\n".join(observation_parts)

                    messages.append({"role": "user", "content": f"OBSERVATIONS:\n{combined_observation}"})
                    state.steps.append({"step": parsed_step.model_dump(), "observation": combined_observation})

                    yield {
                        "event":       "tool_observation",
                        "tool_name":   f"{len(tool_calls)} tool(s)",
                        "observation": combined_observation,
                        "step_index":  i,
                    }
                    span.log_close(
                        status="tools_called",
                        tools=[tc.tool_name for tc in tool_calls],
                    )

                else:
                    # Model returned neither a tool call nor a final answer — nudge it
                    nudge = (
                        "Your last response did not include any tool_calls or a final_answer. "
                        "You MUST respond with a valid JSON containing either 'tool_calls' (non-empty) OR a 'final_answer'."
                    )
                    messages.append({"role": "user", "content": nudge})
                    logger.warning(
                        "%sIter %d: no tool calls and no answer — injecting corrective nudge.",
                        self._label, iteration_num,
                    )
                    span.log_error("No tool calls or final answer — reprompting.")

        yield {"event": "error", "message": "Execution limit reached without finding a final answer."}
