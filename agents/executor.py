import asyncio
from typing import AsyncGenerator

from utils.json_parser import parse_llm_json
from utils.prompts import load_prompt

from core.model_router import ModelRouter, ModelRequest, ModelTier
from .schemas import AgentState, AgentStep, ToolCall
from .tools import tools_registry, ToolRegistry
from utils.log_writer import emit

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
        router: ModelRouter,
        task_label: str | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.router       = router
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
        base_prompt = load_prompt("executor_system.txt")
        return base_prompt.replace("{tool_docs}", tool_docs)

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
                emit("tactical.budget_pressure", "tactical", {"remaining": remaining}, level="warn")

            model_name = self.router.resolve_model(ModelTier.STANDARD)
            emit("llm.request", "llm", {"model": model_name, "tier": "STANDARD", "component": f"AgentExecutor.iter={iteration_num}"}, level="debug")
            
            request = ModelRequest(
                tier=ModelTier.STANDARD,
                messages=messages,
                component=f"AgentExecutor.iter={iteration_num}",
            )

            import time
            start = time.perf_counter()
            try:
                raw_response = await self.router.complete(request)
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "error": str(e)}, level="error")
                yield {"event": "error", "message": f"Upstream call failed: {str(e)}"}
                return

            elapsed_ms = (time.perf_counter() - start) * 1000
            assistant_content = raw_response.content
            messages.append({"role": "assistant", "content": assistant_content})

            if not assistant_content:
                emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "error": "empty content"}, level="error")
                emit("tactical.warning", "tactical", {"msg": f"Iteration {iteration_num}: model returned empty/null content — likely a thinking-only response."}, level="warn")
                nudge = (
                    "Your last response was completely empty. "
                    "You MUST respond with a valid JSON block containing your 'thought', 'tool_calls', and 'final_answer'."
                )
                messages.append({"role": "user", "content": nudge})
                yield {"event": "warning", "message": "Empty content — reprompting."}
                continue

            # Detect upstream error payloads returned inside a 200 OK body
            if _is_upstream_error(assistant_content):
                msg = assistant_content.strip()
                emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "error": f"upstream error: {msg}"}, level="error")
                emit("tactical.error", "tactical", {"msg": f"Upstream error payload in iteration {iteration_num}: {msg}"}, level="error")
                yield {"event": "error", "message": f"Upstream model error: {msg}"}
                return

            try:
                parsed_step = self._parse_agent_step(assistant_content)
            except Exception as e:
                emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "error": f"parse failure: {e}"}, level="error")
                emit("tactical.warning", "tactical", {"msg": f"Iteration {iteration_num}: parse failure: {e}"}, level="warn")
                nudge = (
                    f"Your last response failed to parse as valid JSON. Error: {e}\n"
                    "You MUST respond ONLY with a single valid JSON object containing your 'thought', 'tool_calls', and 'final_answer'."
                )
                messages.append({"role": "user", "content": nudge})
                yield {"event": "warning", "message": f"Failed to parse model instructions: {e}"}
                continue

            emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "usage": getattr(raw_response, 'usage', None)}, level="debug")
            emit("tactical.thought", "tactical", {"thought": parsed_step.thought, "iteration": iteration_num})
            yield {"event": "thought", "thought": parsed_step.thought, "step_index": i}

            # ── Fast exit: final answer with no tool calls ──────────────
            if parsed_step.final_answer and not parsed_step.tool_calls:
                state.steps.append({"step": parsed_step.model_dump(), "observation": None})
                emit("tactical.final_answer", "tactical", {"iteration": iteration_num, "answer": parsed_step.final_answer})
                yield {"event": "final_answer", "answer": parsed_step.final_answer, "step_index": i}
                return

            # ── Parallel tool execution ─────────────────────────────────
            if parsed_step.tool_calls:
                tool_calls: list[ToolCall] = parsed_step.tool_calls

                # Notify UI that tools are starting
                for tc in tool_calls:
                    emit(
                        event="tactical.tool_call",
                        layer="tactical",
                        data={
                            "tool": tc.tool_name,
                            "input": tc.tool_input,
                            "iteration": iteration_num,
                        }
                    )
                    yield {
                        "event":      "tool_start",
                        "tool_name":  tc.tool_name,
                        "tool_input": tc.tool_input,
                        "step_index": i,
                    }

                # Execute all tools concurrently
                results = await asyncio.gather(*(self._run_tool(tc) for tc in tool_calls))

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

                # If the model also provided a final answer in the same step, emit it now
                if parsed_step.final_answer:
                    emit("tactical.final_answer", "tactical", {"iteration": iteration_num, "answer": parsed_step.final_answer})
                    yield {"event": "final_answer", "answer": parsed_step.final_answer, "step_index": i}
                    return

            else:
                # Model returned neither a tool call nor a final answer — nudge it
                nudge = (
                    "Your last response did not include any tool_calls or a final_answer. "
                    "You MUST respond with a valid JSON containing either 'tool_calls' (non-empty) OR a 'final_answer'."
                )
                messages.append({"role": "user", "content": nudge})
                emit("tactical.warning", "tactical", {"msg": f"Iter {iteration_num}: no tool calls and no answer — injecting corrective nudge."}, level="warn")

        yield {"event": "error", "message": "Execution limit reached without finding a final answer."}

    async def _run_tool(self, tc: ToolCall) -> tuple[str, str, str]:
        """Returns (tool_name, tool_input_repr, observation)."""
        import time
        start = time.perf_counter()
        obs = await self._registry.execute(tc.tool_name, tc.tool_input)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        if len(obs) > _MAX_OBSERVATION_CHARS:
            extra = len(obs) - _MAX_OBSERVATION_CHARS
            emit("tactical.warning", "tactical", {"msg": f"Observation from {tc.tool_name!r} truncated: {len(obs)} → {_MAX_OBSERVATION_CHARS} chars (+{extra} omitted)"}, level="warn")
            obs = obs[:_MAX_OBSERVATION_CHARS] + f"\n... [truncated: {extra} chars omitted]"
            
        emit(
            event="tactical.tool_result",
            layer="tactical",
            data={
                "tool": tc.tool_name,
                "ok": not _is_upstream_error(obs),
                "elapsed_ms": elapsed_ms,
                "result": obs[:80],
            }
        )
        return tc.tool_name, str(tc.tool_input), obs

