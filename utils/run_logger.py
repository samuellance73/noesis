"""
utils/run_logger.py
───────────────────
Human-readable, plain-text log files for each goal run and each executor
sub-task.  Writes directly to files — no Python logging plumbing involved.

Output layout
─────────────
    logs/runs/<timestamp>_<run_id>/
        _manager.log              ← manager thoughts, cycle decisions, final answer
        task-0_<slug>.log         ← one file per executor sub-task
        task-1_<slug>.log
        ...

Sample task file
────────────────
    ════════════════════════════════════════════════════════════════════════
    TASK    : Summarize philosophical perspectives on the meaning of life
    CYCLE   : 1
    STARTED : 00:18:47
    ════════════════════════════════════════════════════════════════════════

    [iter 1] THOUGHT
      I will synthesize Nihilism, Existentialism, Absurdism, and Stoicism.

    [iter 1] FINAL ANSWER
      The meaning of life across philosophical schools:
      • Nihilism: no inherent meaning exists …
      …

    ════════════════════════════════════════════════════════════════════════
    RESULT  : ✓ SUCCESS  |  1 iteration(s)  |  1.70s
    ════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import datetime
import pathlib
import re
import textwrap
import time
from typing import TextIO

_HEAVY = "═" * 72
_LIGHT = "─" * 72
_W     = 70   # text wrap width


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _slugify(text: str, max_len: int = 35) -> str:
    """Turn arbitrary text into a filesystem-safe lowercase slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:max_len].rstrip("-")


def _wrap(text: str, indent: int = 2) -> str:
    """Word-wrap text and indent every line."""
    prefix = " " * indent
    return textwrap.fill(
        text,
        width=_W,
        initial_indent=prefix,
        subsequent_indent=prefix,
    )


# ─── RunLogger ───────────────────────────────────────────────────────────────

class RunLogger:
    """
    Creates the per-run directory and exposes helpers for writing the
    manager log.  Call ``open_task_log()`` to get a ``TaskLog`` for each
    executor sub-task.

    Usage (inside GoalManager.run_stream):
        run_log = RunLogger(goal=ultimate_goal, run_id=trace.id)
        ...
        run_log.log_cycle_start(cycle)
        run_log.log_manager_thought(decision.thought)
        run_log.log_spawning([t.goal for t in decision.tasks_to_spawn])
        ...
        task_log = run_log.open_task_log(idx, task.goal, task.context, cycle)
        ...
        run_log.log_cycle_complete(decision.progress_update, ...)
        run_log.log_complete(cycle, tasks)   # or run_log.log_error(msg)
    """

    def __init__(self, goal: str, run_id: str) -> None:
        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = pathlib.Path("logs") / "runs" / f"{ts}_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._f: TextIO = (self.run_dir / "_manager.log").open("w", encoding="utf-8")
        self._start     = time.perf_counter()

        self._w(f"{_HEAVY}\n")
        self._w(f"GOAL    : {goal}\n")
        self._w(f"RUN ID  : {run_id}\n")
        self._w(f"STARTED : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._w(f"{_HEAVY}\n")

    # ── internal ──────────────────────────────────────────────────────────

    def _w(self, text: str) -> None:
        self._f.write(text)
        self._f.flush()

    def _close(self) -> None:
        if not self._f.closed:
            self._f.close()

    # ── manager events ────────────────────────────────────────────────────

    def log_cycle_start(self, cycle: int) -> None:
        dashes = _LIGHT[:66 - len(str(cycle))]
        self._w(f"\n── CYCLE {cycle} {dashes}\n\n")

    def log_manager_thought(self, thought: str) -> None:
        self._w(f"THOUGHT:\n{_wrap(thought)}\n\n")

    def log_spawning(self, tasks: list[str]) -> None:
        self._w(f"SPAWNING {len(tasks)} TASK(S):\n")
        for i, goal in enumerate(tasks):
            self._w(f"  [task-{i}] {goal}\n")
        self._w("\n")

    def log_cycle_complete(
        self,
        progress: str,
        completed: list[str],
        failed: list[str],
    ) -> None:
        self._w(f"PROGRESS : {progress}\n")
        if completed:
            self._w(f"DONE     : {len(completed)} task(s) completed so far\n")
        if failed:
            self._w(f"FAILED   : {len(failed)} task(s) produced no answer\n")
            for t in failed:
                self._w(f"  ✗ {t}\n")
        self._w("\n")

    def log_final_answer(self, answer: str) -> None:
        self._w(f"FINAL ANSWER:\n")
        for line in answer.splitlines():
            self._w(f"  {line}\n")
        self._w("\n")

    def log_complete(self, cycles: int, tasks: int) -> None:
        elapsed = time.perf_counter() - self._start
        self._w(f"{_HEAVY}\n")
        self._w(
            f"COMPLETED  cycles={cycles}  tasks_done={tasks}"
            f"  total={elapsed:.2f}s\n"
        )
        self._w(f"{_HEAVY}\n")
        self._close()

    def log_error(self, message: str) -> None:
        elapsed = time.perf_counter() - self._start
        self._w(f"\n{_HEAVY}\n")
        self._w(f"ERROR : {message}\n")
        self._w(f"AFTER : {elapsed:.2f}s\n")
        self._w(f"{_HEAVY}\n")
        self._close()

    # ── per-task ──────────────────────────────────────────────────────────

    def open_task_log(
        self,
        task_idx: int,
        goal: str,
        context: str | None,
        cycle: int,
    ) -> "TaskLog":
        """Return a TaskLog writing to task-<N>_<slug>.log in this run dir."""
        slug     = _slugify(goal)
        log_path = self.run_dir / f"task-{task_idx}_{slug}.log"
        return TaskLog(
            path=log_path,
            goal=goal,
            context=context,
            cycle=cycle,
        )


# ─── TaskLog ─────────────────────────────────────────────────────────────────

class TaskLog:
    """
    Writes a human-readable log for a single executor sub-task.
    Opened by RunLogger.open_task_log() and closed at end of _run_subtask.
    """

    def __init__(
        self,
        path: pathlib.Path,
        goal: str,
        context: str | None,
        cycle: int,
    ) -> None:
        self._f: TextIO  = path.open("w", encoding="utf-8")
        self._start      = time.perf_counter()
        self._iter: int  = 0

        self._w(f"{_HEAVY}\n")
        self._w(f"TASK    : {goal}\n")
        if context:
            # Show first 300 chars of context — it can be long
            ctx_preview = context[:300] + ("…" if len(context) > 300 else "")
            self._w(f"CONTEXT : {ctx_preview}\n")
        self._w(f"CYCLE   : {cycle}\n")
        self._w(f"STARTED : {_now()}\n")
        self._w(f"{_HEAVY}\n\n")

    # ── internal ──────────────────────────────────────────────────────────

    def _w(self, text: str) -> None:
        self._f.write(text)
        self._f.flush()

    # ── events ────────────────────────────────────────────────────────────

    def log_thought(self, thought: str, iteration: int) -> None:
        self._iter = iteration
        self._w(f"[iter {iteration}] THOUGHT\n{_wrap(thought)}\n\n")

    def log_tool_call(self, tool_name: str, tool_input: str) -> None:
        self._w(f"[iter {self._iter}] TOOL CALL → {tool_name}\n")
        self._w(f"  input : {str(tool_input)[:300]}\n\n")

    def log_tool_result(self, tool_name: str, observation: str) -> None:
        obs = str(observation)
        preview = obs[:600]
        if len(obs) > 600:
            preview += f"\n  … [{len(obs) - 600} chars omitted]"
        self._w(f"[iter {self._iter}] TOOL RESULT ← {tool_name}\n")
        for line in preview.splitlines():
            self._w(f"  {line}\n")
        self._w("\n")

    def log_final_answer(self, answer: str) -> None:
        self._w(f"[iter {self._iter}] FINAL ANSWER\n")
        for line in answer.splitlines():
            self._w(f"  {line}\n")
        self._w("\n")

    def log_error(self, message: str) -> None:
        self._w(f"[iter {self._iter}] ERROR\n  {message}\n\n")

    def close(self, success: bool, iterations: int) -> None:
        elapsed = time.perf_counter() - self._start
        status  = "✓ SUCCESS" if success else "✗ NO ANSWER"
        self._w(f"{_HEAVY}\n")
        self._w(
            f"RESULT  : {status}  |  {iterations} iteration(s)"
            f"  |  {elapsed:.2f}s\n"
        )
        self._w(f"{_HEAVY}\n")
        self._f.close()
