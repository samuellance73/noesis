import logging
from typing import AsyncGenerator
from .executor import AgentExecutor
from .planner import plan

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Owns the end-to-end agentic pipeline:
      1. Plan  – decompose the user request into milestones
      2. Execute – run each milestone through an AgentExecutor
      3. Collect – accumulate results and emit a final summary event

    All yielded dicts are plain, JSON-serialisable event payloads.
    The transport layer (router) is responsible only for framing them
    as SSE lines or returning them as a JSON response body.
    """

    def __init__(self, llm_service, model: str):
        self.llm_service = llm_service
        self.model = model

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    async def run_stream(self, user_input: str) -> AsyncGenerator[dict, None]:
        """Yield structured event dicts for the entire plan → execute cycle."""
        try:
            yield {"event": "planning_start"}

            milestones = await plan(user_input, self.llm_service)
            yield {"event": "plan_ready", "milestones": milestones}

            results = []
            for idx, milestone in enumerate(milestones):
                yield {
                    "event": "step_start",
                    "step_index": idx,
                    "milestone_index": idx,
                    "milestone_goal": milestone["goal"],
                    "step_goal": milestone["goal"],
                }

                executor = AgentExecutor(llm_service=self.llm_service, model=self.model)
                final_result = None

                async for step_update in executor.run_generator(milestone["goal"]):
                    # Preserve the executor's own step_index (iteration within milestone).
                    # Add milestone_index separately so the frontend can group correctly.
                    step_update["milestone_index"] = idx
                    if step_update["event"] == "final_answer":
                        final_result = step_update["answer"]
                    yield step_update

                results.append({"milestone": milestone["goal"], "result": final_result})

            yield {"event": "done", "milestones": milestones, "results": results}

        except Exception as exc:
            logger.error("Orchestrator error: %s", exc, exc_info=True)
            yield {"event": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    async def run(self, user_input: str) -> dict:
        """Execute the full pipeline and return a single result dict."""
        milestones = await plan(user_input, self.llm_service)
        results = []
        for milestone in milestones:
            executor = AgentExecutor(llm_service=self.llm_service, model=self.model)
            result = await executor.run(milestone["goal"])
            results.append({"milestone": milestone["goal"], "result": result})
        return {"milestones": milestones, "results": results}
