"""
app/lifespan.py
───────────────
FastAPI lifespan context manager — manages startup and shutdown of all
long-running background components:

  • HTTP connection pool (httpx)
  • Autonomous trigger daemon
  • Perception layer (signal intake → classify → synthesize → route)
  • Discord selfbot (optional, gated on DISCORD_BOT_TOKEN)

Extracted from main.py so the application entry-point stays thin.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from integrations.llm.client import get_client
from integrations.llm.service import UpstreamService
from core.model_router import ModelRouter, load_config
from triggers.daemon import start_daemon
from perception import PerceptionLayer, PerceptionConfig
from perception.schemas import PerceptionWorldModel, ResponseJob
from utils.log_writer import emit


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with get_client(timeout=30.0) as client:
        app.state.upstream_client  = client
        app.state.upstream_service = UpstreamService(client)

        router_config = load_config("config/model_router.yaml")
        app.state.model_router     = ModelRouter(router_config, app.state.upstream_service)

        # ── Perception layer ──────────────────────────────────────────────
        perception_config     = PerceptionConfig()
        perception_world      = PerceptionWorldModel()
        app.state.perception_world = perception_world

        executor_factory      = _make_executor_factory(app.state.model_router)
        app.state.perception  = PerceptionLayer(
            config=perception_config,
            router=app.state.model_router,
            world_model=perception_world,
            executor_factory=executor_factory,
        )
        await app.state.perception.start()

        # ── Daemon ────────────────────────────────────────────────────────
        daemon_task = asyncio.create_task(
            start_daemon(router=app.state.model_router, interval_seconds=30)
        )
        app.state.daemon_task = daemon_task

        # ── Discord bot (optional) ────────────────────────────────────────
        discord_task = await _start_discord(app)

        yield

        # ── Shutdown ─────────────────────────────────────────────────────
        await app.state.perception.stop()
        await _cancel(daemon_task)
        await _stop_discord(discord_task)


def _make_executor_factory(router: ModelRouter):
    """
    Returns an async callable(job: ResponseJob) that runs an AgentExecutor for
    each reactive perception job and publishes its events to the EventBus.

    This is the bridge between the perception layer's ReactivePool and the
    existing agent infrastructure — no TriggerStore, no daemon overhead.
    """
    from agents.executor import AgentExecutor
    from utils.event_bus import event_bus
    from utils.log_writer import emit

    async def executor_factory(job: ResponseJob) -> None:
        import uuid
        event_id = str(job.id)[:8]
        run_id = f"perception-{event_id}"
        task_description = job.event.response_context or job.event.summary

        # Create run directory for perception jobs
        from pathlib import Path
        runs_root = Path("logs") / "runs"
        run_dir = runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        executor = AgentExecutor(
            router=router,
            task_label=f"perception-{event_id}",
        )

        iteration = 0
        try:
            emit("perception.job_started", "perception", {"job_id": event_id, "description": task_description[:80]}, run_id=run_id)
            async for event in executor.run_generator(task_description):
                event["perception_job_id"] = event_id
                event["perception_type"]   = job.event.type.value
                await event_bus.publish(event)

                # Log events to agent.jsonl
                ev = event.get("event")
                if ev == "thought":
                    emit("tactical.thought", "tactical", {"thought": event.get("thought", "")}, run_id=run_id)
                elif ev == "tool_start":
                    emit("tactical.tool_call", "tactical", {"tool": event.get("tool_name", "unknown"), "input": str(event.get("tool_input", ""))}, run_id=run_id)
                elif ev == "tool_observation":
                    emit("tactical.tool_result", "tactical", {"tool": event.get("tool_name", "unknown"), "result": str(event.get("observation", ""))[:80]}, run_id=run_id)
                elif ev == "final_answer":
                    emit("tactical.final_answer", "tactical", {"answer": event.get("answer", "")}, run_id=run_id)
                elif ev == "error":
                    emit("tactical.error", "tactical", {"msg": event.get("message", "Unknown error")}, level="error", run_id=run_id)
                    raise

                iteration += 1

            emit("perception.job_complete", "perception", {"job_id": event_id, "iterations": iteration}, run_id=run_id)
        except Exception as e:
            emit("perception.job_failed", "perception", {"job_id": event_id, "error": str(e)}, level="error", run_id=run_id)
            raise

    return executor_factory


async def _start_discord(app: FastAPI) -> "asyncio.Task | None":
    """Start the Discord bot task if a token is configured."""
    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        emit("system.warning", "system", {"msg": "DISCORD_BOT_TOKEN not set — Discord interface disabled."}, level="warn")
        return None

    if "PYTEST_CURRENT_TEST" in os.environ:
        emit("system.warning", "system", {"msg": "Discord bot disabled (PYTEST running)."}, level="warn")
        return None

    from interfaces.discord.bot import bot as discord_bot
    task = asyncio.create_task(discord_bot.start(token))
    app.state.discord_task = task
    return task


async def _stop_discord(task: "asyncio.Task | None") -> None:
    """Gracefully close the Discord bot and cancel its task."""
    if task is None:
        return
    from interfaces.discord.bot import bot as discord_bot
    await discord_bot.close()
    await _cancel(task)


async def _cancel(task: asyncio.Task) -> None:
    """Cancel a task and swallow the expected CancelledError."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
