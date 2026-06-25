"""
perception/reactive_pool.py
────────────────────────────
ReactivePool — processes ResponseJob objects with a fixed worker pool.

Sits adjacent to the perception layer. Receives ResponseJob objects from the
Router and processes them with up to `max_workers` concurrent coroutines.

Worker contract
───────────────
  • Single LLM call via ReactiveExecutor.run().
  • No tool use except `send_message`-style output.
  • No memory writeback — responses are ephemeral by design.
  • Max response time: reactive_executor_timeout_seconds from config.
    Timeout → mark job failed, log, continue.

The pool runs forever as a background asyncio task started from PerceptionLayer.
It accepts new jobs via the public `enqueue()` coroutine even while workers are
busy (asyncio.PriorityQueue buffers unbounded by default).

Priority queue ordering
───────────────────────
PriorityQueue pops the *smallest* item first.  We negate the urgency score so
that higher urgency = lower number = popped first.
"""

from __future__ import annotations

import asyncio
import logging

from perception.schemas import ResponseJob

logger = logging.getLogger("noesis.perception")


class ReactivePool:
    """
    Fixed-size async worker pool for handling ResponseJob objects.

    Parameters
    ──────────
    max_workers           : int   — concurrent worker coroutines.
    executor_timeout      : float — seconds before a job is abandoned.
    executor_factory      : optional callable(job) → awaitable
                            Defaults to a no-op stub; replace in production.
    """

    def __init__(
        self,
        max_workers: int = 5,
        executor_timeout: float = 15.0,
        executor_factory=None,
    ) -> None:
        self.max_workers = max_workers
        self.executor_timeout = executor_timeout
        self._executor_factory = executor_factory or _noop_executor
        # PriorityQueue items: (neg_urgency, job)
        self._queue: asyncio.PriorityQueue[tuple[float, ResponseJob]] = asyncio.PriorityQueue()
        self._worker_tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Launch all worker coroutines as background tasks."""
        if self._running:
            return
        self._running = True
        logger.info("ReactivePool: starting %d worker(s).", self.max_workers)
        for i in range(self.max_workers):
            task = asyncio.create_task(
                self._worker(i), name=f"reactive-worker-{i}"
            )
            self._worker_tasks.append(task)

    async def stop(self) -> None:
        """Cancel all workers and wait for them to finish."""
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("ReactivePool: stopped.")

    async def enqueue(self, job: ResponseJob) -> None:
        """Add a job to the priority queue. Higher urgency = processed first."""
        priority = -job.priority  # negate so highest urgency pops first
        await self._queue.put((priority, job))
        logger.debug(
            "ReactivePool: enqueued job_id=%s  urgency=%.2f  queue_size=%d",
            job.id, job.priority, self._queue.qsize(),
        )

    # ── Private ───────────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """Long-running worker coroutine — pulls jobs and executes them."""
        label = f"worker-{worker_id}"
        logger.debug("ReactivePool: %s started.", label)
        while True:
            try:
                _, job = await self._queue.get()
                job.assigned_worker = label
                job.status = "running"
                logger.info(
                    "ReactivePool: %s handling job_id=%s  event_type=%s  urgency=%.2f",
                    label, job.id, job.event.type.value, job.priority,
                )
                try:
                    await asyncio.wait_for(
                        self._executor_factory(job),
                        timeout=self.executor_timeout,
                    )
                    job.status = "done"
                    logger.info("ReactivePool: %s job_id=%s done.", label, job.id)
                except asyncio.TimeoutError:
                    job.status = "failed"
                    logger.error(
                        "ReactivePool: %s job_id=%s timed out after %.1fs.",
                        label, job.id, self.executor_timeout,
                    )
                except Exception as exc:
                    job.status = "failed"
                    logger.error(
                        "ReactivePool: %s job_id=%s failed: %s",
                        label, job.id, exc, exc_info=True,
                    )
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                logger.debug("ReactivePool: %s cancelled — shutting down.", label)
                raise


async def _noop_executor(job: ResponseJob) -> None:
    """
    Default stub executor — logs the job but takes no action.

    Replace this with a real ReactiveExecutor in production by passing
    `executor_factory=my_factory` to ReactivePool().

    Example factory:
        async def my_factory(job: ResponseJob) -> None:
            executor = ReactiveExecutor(job, llm_service, model)
            await executor.run()
    """
    logger.info(
        "ReactivePool[noop]: job_id=%s  event_type=%s  summary=%r",
        job.id, job.event.type.value, job.event.summary[:100],
    )
