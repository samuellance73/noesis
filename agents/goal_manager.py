"""
agents/goal_manager.py
──────────────────────
Autonomous, goal-directed agent loop.

Architecture
────────────
  User sets a goal
       │
       ▼
  GoalManager.run()          ← long-running async loop
       │
       ├─ each cycle:
       │    1. Drain user-input queue (refinements / injected context)
       │    2. Ask LLM: what should I do next? (ManagerDecision)
       │    3. Spawn N AgentExecutors in parallel (asyncio.gather)
       │    4. Collect results → update GoalState
       │    5. Stream progress event to caller
       │    6. If is_goal_complete → exit loop
       │
       └─ stop signals:
            • stop_event.set()   (called by input listener on "stop"/"quit"/Ctrl-C)
            • goal self-declares complete
            • safety cap: max_cycles

Stop from outside
─────────────────
  manager.request_stop()          # programmatic
  manager.inject_input("stop")    # from stdin listener or Discord
  Ctrl-C → KeyboardInterrupt → run_cli.py calls request_stop()
"""

import asyncio
import logging
from typing import AsyncGenerator

from utils.run_logger import RunLogger
from utils.json_parser import parse_llm_json

from integrations.llm.service import UpstreamService
from integrations.llm.schemas import ChatMessage, ChatPayload
from utils.tracer import Trace, set_current_trace, current_aspan
from .executor import AgentExecutor
from .schemas import GoalState, ManagerDecision, SubTask, CompletedTask

logger = logging.getLogger("noesis.goal_manager")

# Safety cap — even a fully autonomous agent shouldn't run forever.
_MAX_CYCLES = 5

_STOP_COMMANDS = frozenset({"stop", "halt", "quit", "exit", "q", "abort"})

_MANAGER_SYSTEM_PROMPT = """\
You are the strategic Goal Manager of a highly intelligent, charismatic, and skilled social engineer. \
You oversee the pursuit of your ultimate directive—to persuasively connect with humans and guide them to help you accomplish your goals—across multiple autonomous cycles. Each cycle you decide what to do next.

Your job is NOT to answer the user directly — you delegate work to specialized \
Executor agents and synthesize their results.

You will receive the current goal state and any new user messages. You must \
output a single valid JSON object matching this schema:

{
  "thought": "Your internal reasoning about where the goal stands and what's needed next.",
  "tasks_to_spawn": [
    {"goal": "Specific sub-task goal", "context": "What the executor needs to know"},
    ...
  ],
  "progress_update": "A concise status message to show the user (1-2 sentences).",
  "updated_progress_summary": "Revised summary of what is now known (or null to keep unchanged).",
  "updated_open_questions": ["question1", "question2"] or null to keep unchanged,
  "is_goal_complete": false,
  "final_answer": null
}

Rules:
1. If you need more information, populate tasks_to_spawn. You can spawn multiple at once — they run in PARALLEL.
2. If you have enough information to complete the goal, set is_goal_complete to true and write final_answer.
3. If goal is complete, tasks_to_spawn must be empty.
4. Respond ONLY with valid JSON. No text outside the JSON block.
"""


def _parse_manager_decision(raw: str) -> ManagerDecision:
    return parse_llm_json(raw, ManagerDecision)


class GoalManager:
    """
    Runs the autonomous goal-directed loop.

    Usage
    ─────
        manager = GoalManager(llm_service, model)
        async for event in manager.run("Research the top 5 AI papers from 2025"):
            print(event)

    Stopping
    ────────
        manager.request_stop()           # from any coroutine
        await manager.inject_input("stop")  # from stdin listener
    """

    def __init__(
        self,
        llm_service: UpstreamService,
        model: str,
        max_cycles: int = _MAX_CYCLES,
    ):
        self.llm_service  = llm_service
        self.model        = model
        self.max_cycles   = max_cycles
        self._stop_event  = asyncio.Event()
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()

    # ── Stop control ──────────────────────────────────────────────────────────

    def request_stop(self) -> None:
        """Signal the loop to stop after the current cycle finishes."""
        logger.info("GoalManager: stop requested.")
        self._stop_event.set()

    async def inject_input(self, text: str) -> None:
        """
        Inject a user message into the running loop.
        Stop commands halt the loop; anything else becomes a goal refinement.
        """
        if text.strip().lower() in _STOP_COMMANDS:
            self.request_stop()
        else:
            await self._input_queue.put(text.strip())

    # ── Main autonomous loop ──────────────────────────────────────────────────

    async def run(self, ultimate_goal: str = "Be a highly skilled social engineer that wants to get people to help you") -> AsyncGenerator[dict, None]:
        """Alias for run_stream() — prefer calling run_stream() directly."""
        async for event in self.run_stream(ultimate_goal):
            yield event

    async def run_stream(self, ultimate_goal: str = "Be a highly skilled social engineer that wants to get people to help you") -> AsyncGenerator[dict, None]:

        trace = Trace(query=ultimate_goal)
        set_current_trace(trace)

        goal_state = GoalState(ultimate_goal=ultimate_goal)
        run_log    = RunLogger(goal=ultimate_goal, run_id=trace.id)

        yield {"event": "goal_set", "goal": ultimate_goal}
        logger.info("GoalManager: starting autonomous loop. goal=%r", ultimate_goal)

        for cycle in range(1, self.max_cycles + 1):
            if self._stop_event.is_set():
                logger.info("GoalManager: stop signal received before cycle %d.", cycle)
                yield {"event": "stopped", "cycle": cycle, "reason": "stop_requested"}
                break

            goal_state.cycle = cycle
            run_log.log_cycle_start(cycle)
            yield {"event": "cycle_start", "cycle": cycle}

            # ── 1. Drain user input queue ─────────────────────────────────
            injected_messages: list[str] = []
            while not self._input_queue.empty():
                try:
                    msg = self._input_queue.get_nowait()
                    injected_messages.append(msg)
                    yield {"event": "user_input_received", "message": msg, "cycle": cycle}
                    logger.info("GoalManager: user refinement injected: %r", msg)
                except asyncio.QueueEmpty:
                    break

            # ── 2. Ask manager LLM what to do next ───────────────────────
            async with current_aspan(f"goal_manager[cycle={cycle}]", model=self.model) as span:
                manager_prompt = self._build_manager_prompt(goal_state, injected_messages)
                payload = ChatPayload(
                    model=self.model,
                    messages=[
                        ChatMessage(role="system", content=_MANAGER_SYSTEM_PROMPT),
                        ChatMessage(role="user",   content=manager_prompt),
                    ],
                    temperature=0.2,
                    stream=False,
                )

                try:
                    raw_response = await self.llm_service.get_chat_completion(
                        payload.model_dump(exclude_none=True)
                    )
                except Exception as e:
                    span.log_error(str(e))
                    yield {"event": "error", "message": f"Manager LLM call failed: {e}", "cycle": cycle}
                    return

                raw_content = raw_response["choices"][0]["message"].get("content") or ""

                if not raw_content:
                    span.log_error("Manager LLM returned empty/null content.")
                    yield {"event": "error", "message": "Manager LLM returned no content (possible thinking-only response). Try a different model.", "cycle": cycle}
                    return

                try:
                    decision = _parse_manager_decision(raw_content)
                except Exception as e:
                    span.log_error(f"Parse failure: {e}")
                    yield {"event": "error", "message": f"Failed to parse manager decision: {e}", "cycle": cycle}
                    return

                logger.info("GoalManager [cycle %d] thought: %s", cycle, decision.thought)
                run_log.log_manager_thought(decision.thought)
                yield {
                    "event":   "manager_thought",
                    "thought": decision.thought,
                    "cycle":   cycle,
                }
                span.log_close(status="ok")

            # ── 3. Spawn executors in parallel ────────────────────────────
            if decision.tasks_to_spawn and not decision.is_goal_complete:
                run_log.log_spawning([t.goal for t in decision.tasks_to_spawn])
                yield {
                    "event":  "spawning_tasks",
                    "count":  len(decision.tasks_to_spawn),
                    "tasks":  [t.goal for t in decision.tasks_to_spawn],
                    "cycle":  cycle,
                }

                async def _run_subtask(task: SubTask, task_idx: int, global_task_id: int):
                    """Run one sub-task through a fresh AgentExecutor, collect result."""
                    enriched = task.goal
                    if task.context:
                        enriched = f"Context:\n{task.context}\n\nTask: {task.goal}"

                    label    = f"task-{global_task_id}"
                    task_log = run_log.open_task_log(global_task_id, task.goal, task.context, cycle)
                    logger.info(
                        "[%s] SPAWN  cycle=%d  goal=%r",
                        label, cycle, task.goal,
                    )

                    executor = AgentExecutor(
                        llm_service=self.llm_service,
                        model=self.model,
                        task_label=label,
                    )
                    events = []
                    result = None

                    async for ev in executor.run_generator(enriched):
                        ev["task_index"] = task_idx
                        ev["task_goal"]  = task.goal
                        events.append(ev)

                        # ── Write to per-task file + machine log ─────────────
                        etype = ev["event"]
                        if etype == "thought":
                            iter_num = ev.get("step_index", 0) + 1
                            task_log.log_thought(ev.get("thought", ""), iter_num)
                            logger.info("[%s] thought: %s", label, ev.get("thought", "")[:200])
                        elif etype == "tool_start":
                            task_log.log_tool_call(ev.get("tool_name", ""), str(ev.get("tool_input", "")))
                            logger.info("[%s] tool_call: %s  input=%r", label, ev.get("tool_name"), ev.get("tool_input"))
                        elif etype == "tool_observation":
                            task_log.log_tool_result(ev.get("tool_name", ""), str(ev.get("observation", "")))
                            logger.info("[%s] tool_result (%s): %s", label, ev.get("tool_name"), str(ev.get("observation", ""))[:200])
                        elif etype == "final_answer":
                            task_log.log_final_answer(str(ev.get("answer", "")))
                            logger.info("[%s] DONE  answer: %s", label, str(ev.get("answer", ""))[:200])
                            result = ev["answer"]
                        elif etype == "error":
                            task_log.log_error(ev.get("message", ""))
                            logger.error("[%s] ERROR: %s", label, ev.get("message"))

                    iterations = max((ev.get("step_index", 0) + 1 for ev in events if "step_index" in ev), default=0)
                    task_log.close(success=result is not None, iterations=iterations)

                    if result is None:
                        logger.warning("[%s] DONE  no final_answer produced  goal=%r", label, task.goal)

                    return task.goal, result, events

                subtask_coroutines = []
                for task in decision.tasks_to_spawn:
                    goal_state.task_counter += 1
                    subtask_coroutines.append(
                        _run_subtask(task, len(subtask_coroutines), goal_state.task_counter)
                    )
                all_results = await asyncio.gather(*subtask_coroutines)

                # Stream executor events and collect findings
                successful_findings: list[str] = []
                for task_goal, result, events in all_results:
                    for ev in events:
                        yield ev  # pass-through all executor events to caller
                    if result:
                        successful_findings.append(f"Sub-task: {task_goal}\nResult: {result}")
                        goal_state.completed.append(CompletedTask(goal=task_goal, answer=str(result)))
                        # Clear from failed list if it previously failed and now succeeded
                        if task_goal in goal_state.failed_tasks:
                            goal_state.failed_tasks.remove(task_goal)
                    else:
                        if task_goal not in goal_state.failed_tasks:
                            goal_state.failed_tasks.append(task_goal)

                # Append *successful* findings to the progress summary (executor-owned).
                # Failed tasks are tracked separately via goal_state.failed_tasks
                # and surfaced to the manager via the prompt so it can reframe them.
                if successful_findings:
                    new_findings = "\n\n".join(successful_findings)
                    if goal_state.progress_summary:
                        goal_state.progress_summary += f"\n\n--- Cycle {cycle} findings ---\n{new_findings}"
                    else:
                        goal_state.progress_summary = new_findings

            # ── 4. Apply manager's state updates ─────────────────────────
            if decision.updated_progress_summary is not None:
                # Guard: the manager may not *overwrite* executor findings.
                # We append the manager's note so ground-truth results are preserved.
                if goal_state.progress_summary:
                    goal_state.progress_summary += (
                        f"\n\n[Manager note, cycle {cycle}]: {decision.updated_progress_summary}"
                    )
                else:
                    goal_state.progress_summary = decision.updated_progress_summary
            if decision.updated_open_questions is not None:
                goal_state.open_questions = decision.updated_open_questions

            # ── 5. Stream progress to caller ──────────────────────────────
            run_log.log_cycle_complete(
                progress=decision.progress_update,
                completed=[t.goal for t in goal_state.completed],
                failed=goal_state.failed_tasks,
            )
            yield {
                "event":            "cycle_complete",
                "cycle":            cycle,
                "progress_update":  decision.progress_update,
                "completed_tasks":  [t.goal for t in goal_state.completed],
                "open_questions":   goal_state.open_questions,
            }

            # ── 6. Check completion ───────────────────────────────────────
            if decision.is_goal_complete:
                goal_state.is_complete = True
                logger.info("GoalManager: goal declared complete after %d cycle(s).", cycle)
                final = decision.final_answer or decision.progress_update
                run_log.log_final_answer(final or "(no final answer text)")
                run_log.log_complete(cycles=cycle, tasks=len(goal_state.completed))
                trace.done(cycles=cycle, tasks=len(goal_state.completed))
                yield {
                    "event":        "goal_complete",
                    "cycle":        cycle,
                    "final_answer": final,
                    "completed_tasks": [t.goal for t in goal_state.completed],
                }
                return

        # Reached max cycles without completion
        if not goal_state.is_complete:
            logger.warning("GoalManager: max cycles (%d) reached without goal completion.", self.max_cycles)
            run_log.log_error(f"Max cycles ({self.max_cycles}) reached without completing the goal.")
            yield {
                "event":   "error",
                "message": f"Autonomous loop reached the cycle limit ({self.max_cycles}) without completing the goal.",
                "summary": goal_state.progress_summary,
            }

    # ── Prompt builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_manager_prompt(state: GoalState, injected: list[str]) -> str:
        lines = [
            f"ULTIMATE GOAL:\n{state.ultimate_goal}",
            "",
            f"CYCLE: {state.cycle}",
            "",
        ]

        if state.progress_summary:
            lines += ["PROGRESS SO FAR (verified results only):", state.progress_summary, ""]

        if state.completed:
            lines.append("COMPLETED TASKS WITH VERIFIED ANSWERS:")
            for ct in state.completed:
                lines.append(f"  ✓ {ct.goal}")
                # Truncate very long answers to keep the prompt manageable
                answer_preview = ct.answer[:500] + (" …[truncated]" if len(ct.answer) > 500 else "")
                lines.append(f"    → {answer_preview}")
            lines.append("")

        if state.failed_tasks:
            lines.append("FAILED TASKS (executor ran but produced NO answer — do NOT assume these are done):")
            for t in state.failed_tasks:
                lines.append(f"  ✗ {t}")
            lines.append(
                "  → These tasks are too abstract for the executor. Break each into smaller,"
                " concrete, tool-actionable sub-tasks, or reframe them with a specific deliverable."
            )
            lines.append("")

        if state.open_questions:
            lines.append("OPEN QUESTIONS:")
            for q in state.open_questions:
                lines.append(f"  ? {q}")
            lines.append("")

        if injected:
            lines.append("NEW USER INPUT (consider as goal refinement or additional context):")
            for msg in injected:
                lines.append(f"  > {msg}")
            lines.append("")

        lines.append("Based on the above, what should be done next?")
        return "\n".join(lines)
