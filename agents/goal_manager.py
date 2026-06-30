"""
agents/goal_manager.py
──────────────────────
Autonomous, goal-directed agent loop.

Architecture
────────────
  User sets a goal
       │
       ▼
  GoalManager.run_stream()          ← long-running async generator
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
from typing import Any, AsyncGenerator

from utils.json_parser import parse_llm_json
from utils.log_writer import emit
from utils.prompts import load_prompt

from core.model_router import ModelRouter, ModelRequest, ModelTier
from .executor import AgentExecutor
from .schemas import GoalState, ManagerDecision, SubTask, Mission, Objective
from .tools import build_specialized_registry
from agents.memory.base import MemoryStore, EpisodicMemoryAdapter
from agents.critic import score_result

# Safety cap — even a fully autonomous agent shouldn't run forever.
_MAX_CYCLES = 5

# Hard timeout for the parallel subtask batch within a single cycle.
# A stuck tool call (network hang, infinite loop) will be cancelled after this.
_CYCLE_TIMEOUT_SECONDS = 120

_STOP_COMMANDS = frozenset({"stop", "halt", "quit", "exit", "q", "abort"})

# ── System prompt ─────────────────────────────────────────────────────────────
# Loaded from prompts/manager_system.txt for easier editing and version control
_MANAGER_SYSTEM_PROMPT = load_prompt("manager_system.txt")


def _parse_manager_decision(raw: str) -> ManagerDecision:
    return parse_llm_json(raw, ManagerDecision)


class GoalManager:
    """
    Runs the autonomous goal-directed loop.

    Usage
    ─────
        manager = GoalManager(router)
        async for event in manager.run_stream("Research the top 5 AI papers from 2025"):
            print(event)

    Stopping
    ────────
        manager.request_stop()              # from any coroutine
        await manager.inject_input("stop")  # from stdin listener
    """

    def __init__(
        self,
        router: ModelRouter,
        memory_store: MemoryStore | None = None,
        max_cycles: int = _MAX_CYCLES,
        cycle_interval_seconds: float | None = None,
    ):
        self.router                     = router
        self.memory_store               = memory_store
        self.max_cycles                 = max_cycles
        self.cycle_interval_seconds     = cycle_interval_seconds  # min wall-time per cycle
        self._stop_event                = asyncio.Event()
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()

    # ── Stop control ──────────────────────────────────────────────────────────

    def request_stop(self) -> None:
        """Signal the loop to stop after the current cycle finishes."""
        emit("strategic.stop_requested", "strategic", {})
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

    async def run_stream(self, ultimate_goal: str, run_id: str | None = None) -> AsyncGenerator[dict, None]:
        from datetime import datetime
        import uuid

        if not run_id:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"

        goal_state = GoalState(ultimate_goal=ultimate_goal)
        goal_state.mission = Mission(statement=ultimate_goal, domain="general")

        from pathlib import Path
        runs_root = Path("logs") / "runs"
        run_dir   = runs_root / run_id

        # Lazy directory creation — EpisodicWriter.write_summary() calls
        # run_dir.mkdir(parents=True, exist_ok=True) on first write.
        # We do NOT create the directory here so that fast-path runs leave no
        # empty placeholder folders on disk (spec Component 6).
        # Use injected memory_store or create default EpisodicMemoryAdapter
        if self.memory_store is None:
            self.memory_store = EpisodicMemoryAdapter(runs_root=str(runs_root))
        
        # If using EpisodicMemoryAdapter, set the run_dir for writing
        if isinstance(self.memory_store, EpisodicMemoryAdapter):
            self.memory_store.set_run_dir(run_dir)

        # Working memory priming from memory
        prior_runs = self.memory_store.load_relevant(ultimate_goal, limit=3)
        if prior_runs:
            for run in prior_runs:
                topic = f"Prior Knowledge (Run ID {run.run_id[:8]})"
                details = f"Goal: {run.goal}\nFinal Answer: {run.final_answer or 'None'}"
                goal_state.world_model.domain_map[topic] = details
                # merge domain map from prior run
                if run.domain_map:
                    for k, v in run.domain_map.items():
                        if not k.startswith("Prior Knowledge"):
                            goal_state.world_model.domain_map[f"{k} (from run {run.run_id[:8]})"] = v
                # merge beliefs from prior run
                if run.beliefs:
                    for k, v in run.beliefs.items():
                        goal_state.world_model.beliefs[k] = v

        yield {"event": "goal_set", "goal": ultimate_goal}
        emit("strategic.loop_started", "strategic", {"goal": ultimate_goal}, run_id=run_id)

        for cycle in range(1, self.max_cycles + 1):
            if self._stop_event.is_set():
                emit("strategic.stop_signal_received", "strategic", {"cycle": cycle}, run_id=run_id)
                yield {"event": "stopped", "cycle": cycle, "reason": "stop_requested"}
                break

            cycle_start_time = asyncio.get_event_loop().time()
            goal_state.cycle = cycle
            emit("strategic.cycle_start", "strategic", {"cycle": cycle}, run_id=run_id)
            yield {"event": "cycle_start", "cycle": cycle}

            # ── 1. Drain user input queue ─────────────────────────────────
            injected_messages = await self._drain_input_queue(cycle)
            for msg in injected_messages:
                # Persist user instructions so they aren't forgotten in future cycles
                goal_state.ultimate_goal += f"\n[User Refinement]: {msg}"
                if goal_state.mission:
                    goal_state.mission.statement += f"\n[User Refinement]: {msg}"
                yield {"event": "user_input_received", "message": msg, "cycle": cycle}

            # ── 2. Ask manager LLM what to do next ───────────────────────
            decision = await self._get_manager_decision(goal_state, injected_messages, cycle, run_id)
            if decision is None:
                return  # error already yielded inside helper

            # Yield the decision events separately to keep run_stream a clean generator
            async for event in self._yield_decision_events(decision, cycle, run_id):
                yield event

            # ── 3. Spawn executors in parallel ────────────────────────────
            if decision.tasks_to_spawn and not decision.is_goal_complete:
                async for event in self._run_subtask_batch(
                    decision.tasks_to_spawn, goal_state, run_id, cycle
                ):
                    yield event

            # ── 4. Apply manager's state updates ─────────────────────────
            self._apply_state_updates(decision, goal_state, cycle)

            # Write run summary to memory (intermediate cycle update)
            self.memory_store.write_summary(
                run_id=run_id,
                data={
                    "goal": ultimate_goal,
                    "cycle_summaries": goal_state.world_model.cycle_summaries,
                    "completed_tasks": [{"goal": ct.goal, "answer": ct.answer} for ct in goal_state.completed],
                    "final_answer": decision.final_answer if decision.is_goal_complete else None,
                    "is_complete": goal_state.is_complete,
                    "domain_map": goal_state.world_model.domain_map,
                    "beliefs": goal_state.world_model.beliefs
                }
            )

            # ── 5. Stream progress to caller ──────────────────────────────
            emit(
                event="strategic.cycle_complete", 
                layer="strategic", 
                data={
                    "progress": decision.progress_update,
                    "completed": [t.goal for t in goal_state.completed],
                    "failed": [f.goal for f in goal_state.failed_tasks],
                    "cycle": cycle,
                    # Serialize the full active mind state to the JSONL log line:
                    "objectives": [obj.model_dump() for obj in goal_state.objectives],
                    "world_model": goal_state.world_model.model_dump(),
                }, 
                run_id=run_id
            )
            
            yield {
                "event":            "cycle_complete",
                "cycle":            cycle,
                "progress_update":  decision.progress_update,
                "completed_tasks":  [t.goal for t in goal_state.completed],
                "open_questions":   goal_state.open_questions,
                "objectives":       [obj.model_dump() for obj in goal_state.objectives],
                "world_model":      goal_state.world_model.model_dump(),
            }

            # ── 6. Check completion ───────────────────────────────────────
            if decision.is_goal_complete:
                goal_state.is_complete = True
                emit("strategic.goal_complete", "strategic", {"cycle": cycle}, run_id=run_id)
                final = decision.final_answer or decision.progress_update
                emit("strategic.final_answer", "strategic", {"answer": final, "cycles": cycle, "tasks": len(goal_state.completed)}, run_id=run_id)
                
                # Write final run summary to memory
                self.memory_store.write_summary(
                    run_id=run_id,
                    data={
                        "goal": ultimate_goal,
                        "cycle_summaries": goal_state.world_model.cycle_summaries,
                        "completed_tasks": [{"goal": ct.goal, "answer": ct.answer} for ct in goal_state.completed],
                        "final_answer": final,
                        "is_complete": True,
                        "domain_map": goal_state.world_model.domain_map,
                        "beliefs": goal_state.world_model.beliefs
                    }
                )
                
                yield {
                    "event":           "goal_complete",
                    "cycle":           cycle,
                    "final_answer":    final,
                    "completed_tasks": [t.goal for t in goal_state.completed],
                }
                return

            # ── 7. Enforce minimum cycle interval (pacing) ────────────────
            await self._pace_cycle(cycle, cycle_start_time, run_id)
            if self.cycle_interval_seconds and cycle < self.max_cycles:
                elapsed  = asyncio.get_event_loop().time() - cycle_start_time
                throttle = self.cycle_interval_seconds - elapsed
                if throttle > 0:
                    yield {
                        "event":     "cycle_throttle",
                        "cycle":     cycle,
                        "sleep_for": round(throttle, 1),
                    }

        # Reached max cycles without completion
        if not goal_state.is_complete:
            emit("strategic.max_cycles_reached", "strategic", {"max_cycles": self.max_cycles}, level="warn", run_id=run_id)
            emit("strategic.error", "strategic", {"msg": f"Max cycles ({self.max_cycles}) reached without completing the goal."}, level="error", run_id=run_id)
            yield {
                "event":   "error",
                "message": f"Autonomous loop reached the cycle limit ({self.max_cycles}) without completing the goal.",
                "summary": f"Findings: {len(goal_state.world_model.findings)} completed, gaps: {len(goal_state.world_model.gaps)} remaining",
            }

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _drain_input_queue(self, cycle: int) -> list[str]:
        """Drain all pending user messages and return them as a list."""
        messages: list[str] = []
        while not self._input_queue.empty():
            try:
                msg = self._input_queue.get_nowait()
                messages.append(msg)
                emit("strategic.user_input_injected", "strategic", {"msg": msg, "cycle": cycle})
            except asyncio.QueueEmpty:
                break
        return messages

    async def _get_manager_decision(
        self,
        goal_state: GoalState,
        injected_messages: list[str],
        cycle: int,
        run_id: str,
    ) -> ManagerDecision | None:
        """Call the manager LLM (STRONG tier) and parse its decision. Returns None on failure."""
        model_name = self.router.resolve_model(ModelTier.STRONG)
        emit("llm.request", "llm", {"model": model_name, "tier": "STRONG", "component": f"GoalManager.cycle={cycle}"}, level="debug", run_id=run_id)
        
        prompt = self._build_manager_prompt(goal_state, injected_messages)
        request = ModelRequest(
            tier=ModelTier.STRONG,
            messages=[
                {"role": "system", "content": _MANAGER_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            component=f"GoalManager.cycle={cycle}",
        )

        import time
        start = time.perf_counter()
        try:
            raw_response = await self.router.complete(request)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "error": str(e)}, level="error", run_id=run_id)
            emit("strategic.error", "strategic", {"msg": f"LLM call failed: {e}", "cycle": cycle}, level="error", run_id=run_id)
            return None

        elapsed_ms = (time.perf_counter() - start) * 1000
        raw_content = raw_response.content

        if not raw_content:
            emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "error": "empty content"}, level="error", run_id=run_id)
            emit("strategic.error", "strategic", {"msg": "empty LLM content.", "cycle": cycle}, level="error", run_id=run_id)
            return None

        try:
            decision = _parse_manager_decision(raw_content)
        except Exception as e:
            emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "error": f"parse failure: {e}"}, level="error", run_id=run_id)
            emit("strategic.error", "strategic", {"msg": f"parse failure: {e}", "cycle": cycle}, level="error", run_id=run_id)
            return None

        emit("llm.response", "llm", {"model": model_name, "elapsed_ms": elapsed_ms, "usage": getattr(raw_response, 'usage', None)}, level="debug", run_id=run_id)
        emit("strategic.thought", "strategic", {"thought": decision.thought, "cycle": cycle}, run_id=run_id)
        return decision

    async def _yield_decision_events(
        self,
        decision: ManagerDecision,
        cycle: int,
        run_id: str,
    ) -> AsyncGenerator[dict, None]:
        """Yield events derived from the manager's decision."""
        emit("strategic.manager_thought", "strategic", {"thought": decision.thought, "cycle": cycle}, run_id=run_id)
        yield {
            "event":   "manager_thought",
            "thought": decision.thought,
            "cycle":   cycle,
        }

        if decision.tasks_to_spawn and not decision.is_goal_complete:
            emit("strategic.plan_received", "strategic", {"tasks": [t.goal for t in decision.tasks_to_spawn], "cycle": cycle}, run_id=run_id)
            yield {
                "event":  "spawning_tasks",
                "count":  len(decision.tasks_to_spawn),
                "tasks":  [t.goal for t in decision.tasks_to_spawn],
                "cycle":  cycle,
            }

    async def _run_subtask_batch(
        self,
        tasks: list[SubTask],
        goal_state: GoalState,
        run_id: str,
        cycle: int,
    ) -> AsyncGenerator[dict, None]:
        """Run all sub-tasks in parallel and stream their events."""
        coroutines = []
        for task in tasks:
            goal_state.task_counter += 1
            coroutines.append(
                self._run_subtask(task, len(coroutines), goal_state.task_counter, cycle, run_id)
            )

        try:
            all_results = await asyncio.wait_for(
                asyncio.gather(*coroutines, return_exceptions=True),
                timeout=_CYCLE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            emit("strategic.error", "strategic", {"msg": f"subtask batch timed out after {_CYCLE_TIMEOUT_SECONDS}s.", "cycle": cycle}, level="error", run_id=run_id)
            yield {
                "event":   "error",
                "message": f"Cycle {cycle} subtasks timed out after {_CYCLE_TIMEOUT_SECONDS}s. "
                           "The daemon will retry on the next trigger.",
                "cycle":   cycle,
            }
            return

        critic_tasks = []
        successful_subtasks = []

        for item in all_results:
            if isinstance(item, BaseException):
                emit("strategic.error", "strategic", {"msg": f"subtask raised: {item}", "cycle": cycle}, level="error", run_id=run_id)
                continue
            task_goal, result, events = item
            for ev in events:
                yield ev
            if result:
                successful_subtasks.append((task_goal, result, events))
                critic_tasks.append(score_result(self.router, task_goal, str(result)))
            else:
                goal_state.record_failure(task_goal)
                failed = next((f for f in goal_state.failed_tasks if f.goal == task_goal), None)
                if failed:
                    emit("strategic.warning", "strategic", {"msg": f"task {task_goal!r} failed (attempt {failed.attempts}/{failed.give_up_after}).", "cycle": cycle}, level="warn", run_id=run_id)

        scores = []
        if critic_tasks:
            scores = await asyncio.gather(*critic_tasks)

        for (task_goal, result, events), score in zip(successful_subtasks, scores):
            goal_state.record_success(task_goal, str(result))
            goal_state.world_model.findings.append({
                "cycle": cycle,
                "task_goal": task_goal,
                "answer": str(result),
                "quality_score": score
            })

    async def _run_subtask(
        self,
        task: SubTask,
        task_idx: int,
        global_task_id: int,
        cycle: int,
        run_id: str,
    ) -> tuple[str, Any, list[dict]]:
        """Run one sub-task through a fresh AgentExecutor and collect results."""
        enriched = (
            f"Context:\n{task.context}\n\nTask: {task.goal}"
            if task.context
            else task.goal
        )
        label    = f"task-{global_task_id}"
        task_id  = f"task-{global_task_id}"
        emit("strategic.task_spawned", "strategic", {"label": label, "cycle": cycle, "goal": task.goal}, run_id=run_id, task_id=task_id)

        registry = build_specialized_registry(task.executor_type)
        executor = AgentExecutor(
            router=self.router,
            task_label=label,
            registry=registry,
        )

        events: list[dict] = []
        result = None

        async for ev in executor.run_generator(enriched):
            ev["task_index"] = task_idx
            ev["task_goal"]  = task.goal
            events.append(ev)
            self._log_executor_event(ev, label, task_id, run_id)
            if ev["event"] == "final_answer":
                result = ev["answer"]

        iterations = max(
            (ev.get("step_index", 0) + 1 for ev in events if "step_index" in ev),
            default=0,
        )
        emit("strategic.task_complete", "strategic", {"label": label, "success": result is not None, "iterations": iterations}, run_id=run_id, task_id=task_id)

        if result is None:
            emit("strategic.warning", "strategic", {"msg": f"[{label}] DONE  no final_answer produced  goal={task.goal!r}"}, level="warn", run_id=run_id, task_id=task_id)

        return task.goal, result, events

    @staticmethod
    def _log_executor_event(ev: dict, label: str, task_id: str, run_id: str) -> None:
        """Route a single executor event to the appropriate emit call."""
        etype = ev["event"]
        if etype == "thought":
            iter_num = ev.get("step_index", 0) + 1
            emit("tactical.thought", "tactical", {"thought": ev.get("thought", ""), "iteration": iter_num}, run_id=run_id, task_id=task_id)
        elif etype == "tool_start":
            emit("tactical.tool_call", "tactical", {"tool": ev.get("tool_name", ""), "input": str(ev.get("tool_input", ""))}, run_id=run_id, task_id=task_id)
        elif etype == "tool_observation":
            emit("tactical.tool_result", "tactical", {"tool": ev.get("tool_name", ""), "result": str(ev.get("observation", ""))[:80]}, run_id=run_id, task_id=task_id)
        elif etype == "final_answer":
            emit("tactical.final_answer", "tactical", {"answer": str(ev.get("answer", ""))}, run_id=run_id, task_id=task_id)
        elif etype == "error":
            emit("tactical.error", "tactical", {"msg": ev.get("message", "")}, level="error", run_id=run_id, task_id=task_id)

    def _apply_state_updates(self, decision: ManagerDecision, goal_state: GoalState, cycle: int) -> None:
        """Apply the manager's optional state field updates."""
        if decision.world_model_patch is not None:
            patch = decision.world_model_patch
            wm = goal_state.world_model
            # Remove closed gaps
            wm.gaps = [g for g in wm.gaps if g not in patch.gaps_closed]
            # Add new gaps
            wm.gaps.extend(patch.gaps_added)
            # Update domain map
            wm.domain_map.update(patch.domain_updates)
            # Update beliefs
            wm.beliefs.update(patch.belief_updates)

        if decision.updated_objectives is not None:
            goal_state.objectives = decision.updated_objectives

        # Record cycle summary
        goal_state.world_model.cycle_summaries.append(
            f"[Cycle {cycle}] {decision.progress_update}"
        )

        if decision.updated_open_questions is not None:
            goal_state.open_questions = decision.updated_open_questions


    async def _pace_cycle(self, cycle: int, cycle_start_time: float, run_id: str) -> None:
        """Sleep to enforce the minimum cycle wall-time, if configured."""
        if not self.cycle_interval_seconds or cycle >= self.max_cycles:
            return
        elapsed  = asyncio.get_event_loop().time() - cycle_start_time
        throttle = self.cycle_interval_seconds - elapsed
        if throttle > 0:
            emit(
                event="strategic.cycle_throttle",
                layer="strategic",
                data={
                    "cycle": cycle,
                    "elapsed": elapsed,
                    "throttle": throttle,
                    "interval": self.cycle_interval_seconds,
                },
                run_id=run_id
            )
            await asyncio.sleep(throttle)

    # ── Prompt builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_manager_prompt(state: GoalState, injected: list[str]) -> str:
        lines = [
            f"ULTIMATE GOAL:\n{state.ultimate_goal}",
            "",
            f"CYCLE: {state.cycle}",
            "",
        ]

        if state.mission:
            lines.append(f"MISSION ({state.mission.domain.upper()}):")
            lines.append(f"  {state.mission.statement}")
            lines.append("")

        if state.objectives:
            lines.append("OBJECTIVES:")
            for obj in state.objectives:
                lines.append(f"  * [{obj.id}] {obj.description} (status: {obj.status}, spawned cycle: {obj.spawned_cycle})")
            lines.append("")

        wm = state.world_model

        if wm.domain_map:
            lines.append("WORLD MODEL - DOMAIN MAP:")
            for topic, details in wm.domain_map.items():
                lines.append(f"  * {topic}: {details}")
            lines.append("")

        if wm.gaps:
            lines.append("WORLD MODEL - IDENTIFIED GAPS/UNKNOWNS:")
            for gap in wm.gaps:
                lines.append(f"  * {gap}")
            lines.append("")

        if wm.beliefs:
            lines.append("WORLD MODEL - BELIEFS (confidence 0.0-1.0):")
            for belief, confidence in wm.beliefs.items():
                lines.append(f"  * {belief} [confidence: {confidence:.2f}]")
            lines.append("")

        if wm.findings:
            lines.append("WORLD MODEL - RECENT FINDINGS (with quality scores):")
            for finding in wm.findings:
                score = finding.get("quality_score", 1.0)
                lines.append(f"  * Task: {finding.get('task_goal', '')}")
                lines.append(f"    Answer: {finding.get('answer', '')[:300]}")
                lines.append(f"    Quality Score: {score:.2f}")
            lines.append("")

        if state.completed:
            lines.append("COMPLETED TASKS WITH VERIFIED ANSWERS:")
            for ct in state.completed:
                lines.append(f"  ✓ {ct.goal}")
                answer_preview = ct.answer[:500] + (" …[truncated]" if len(ct.answer) > 500 else "")
                lines.append(f"    → {answer_preview}")
            lines.append("")

        if state.failed_tasks:
            retryable = [f for f in state.failed_tasks if not f.is_abandoned]
            abandoned = [f for f in state.failed_tasks if f.is_abandoned]

            if retryable:
                lines.append("FAILED TASKS (retryable — break into smaller, concrete sub-tasks):")
                for f in retryable:
                    lines.append(f"  ✗ {f.goal}  [attempt {f.attempts}/{f.give_up_after}]")
                lines.append(
                    "  → These tasks are too abstract. Break each into smaller,"
                    " concrete, tool-actionable sub-tasks with a specific deliverable."
                )
                lines.append("")

            if abandoned:
                lines.append("PERMANENTLY BLOCKED (do NOT retry — re-plan around these):")
                for f in abandoned:
                    lines.append(f"  ✗ {f.goal}  [gave up after {f.attempts} attempt(s)]")
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
