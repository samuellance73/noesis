import logging
from typing import AsyncGenerator

from .executor import AgentExecutor
from .planner import plan
from utils.tracer import Trace, set_current_trace

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
        self.model       = model

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    async def run_stream(self, user_input: str) -> AsyncGenerator[dict, None]:
        """Yield structured event dicts for the entire plan → execute cycle."""
        trace = Trace(query=user_input)
        set_current_trace(trace)

        try:
            yield {"event": "planning_start"}
            milestones = await plan(user_input, self.llm_service)
            yield {"event": "plan_ready", "milestones": milestones}

            results = []
            for idx, milestone in enumerate(milestones):
                yield {
                    "event":           "step_start",
                    "step_index":      idx,
                    "milestone_index": idx,
                    "milestone_goal":  milestone["goal"],
                    "step_goal":       milestone["goal"],
                }

                executor      = AgentExecutor(llm_service=self.llm_service, model=self.model)
                final_result  = None
                enriched_goal = self._build_enriched_goal(milestone["goal"], results)

                milestone_failed = False
                async for step_update in executor.run_generator(enriched_goal):
                    step_update["milestone_index"] = idx
                    if step_update["event"] == "final_answer":
                        final_result = step_update["answer"]
                    elif step_update["event"] == "error":
                        milestone_failed = True
                    yield step_update

                # FAIL FAST: Abort orchestration if the current milestone failed
                if milestone_failed or final_result is None:
                    logger.error(f"Aborting orchestration. Milestone {idx + 1} failed.")
                    yield {"event": "error", "message": f"Execution aborted: Milestone {idx + 1} failed."}
                    return

                results.append({"milestone": milestone["goal"], "result": final_result})

            trace.done(milestones=len(milestones))
            yield {"event": "done", "milestones": milestones, "results": results}

        except Exception as exc:
            logger.error("Orchestrator error: %s", exc, exc_info=True)
            trace.error(str(exc))
            yield {"event": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Non-streaming path
    # ------------------------------------------------------------------

    async def run(self, user_input: str) -> dict:
        """Execute the full pipeline and return a single result dict."""
        trace = Trace(query=user_input)
        set_current_trace(trace)

        milestones = await plan(user_input, self.llm_service)
        results    = []
        for idx, milestone in enumerate(milestones):
            executor      = AgentExecutor(llm_service=self.llm_service, model=self.model)
            enriched_goal = self._build_enriched_goal(milestone["goal"], results)
            result        = await executor.run(enriched_goal)
            
            # FAIL FAST: Check if the result indicates execution failure
            if "failed" in result.lower() or "error" in result.lower() or "limit reached" in result.lower():
                logger.error(f"Aborting execution. Milestone {idx + 1} failed: {result}")
                results.append({"milestone": milestone["goal"], "result": f"Aborted: Dependency failed. Error: {result}"})
                break
                
            results.append({"milestone": milestone["goal"], "result": result})

        trace.done(milestones=len(milestones))
        return {"milestones": milestones, "results": results}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_enriched_goal(goal: str, prior_results: list[dict]) -> str:
        """Prepend a summary of completed milestones so the executor has context."""
        if not prior_results:
            return goal
        context_lines = ["Context from previously completed milestones:"]
        for i, entry in enumerate(prior_results, start=1):
            result_text = entry["result"] or "(no result)"
            context_lines.append(f"  {i}. Goal: {entry['milestone']}")
            context_lines.append(f"     Finding: {result_text}")
        context_lines.append("")
        context_lines.append(f"Current milestone goal: {goal}")
        return "\n".join(context_lines)
