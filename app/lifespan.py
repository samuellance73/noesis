"""
app/lifespan.py
───────────────
FastAPI lifespan context manager — manages startup and shutdown of all
long-running background components:

  • HTTP connection pool (httpx)
  • Unified daemon (feeder + processing + self-initiative tasks)
  • Discord selfbot (optional, gated on DISCORD_BOT_TOKEN)
  • Reddit poller  (optional, gated on REDDIT_CLIENT_ID)

Design note — ownership of IntakeBuffer / PerceptionLayer
──────────────────────────────────────────────────────────
core.daemon.start_daemon() owns and constructs both the IntakeBuffer and
the PerceptionLayer internally.  The PerceptionLayer is also exposed on
app.state.perception so that the Discord bot adapter can call
`perception.ingest(event)` to drop events into the ingest_queue.

The PerceptionLayer.ingest() method pushes to the global `ingest_queue`
which the daemon's Feeder Task drains — there is no direct coupling.

Lazy run-directory creation (spec Component 6)
──────────────────────────────────────────────
The executor_factory does NOT create the logs/runs/{run_id} directory
up-front.  It passes run_id to the GoalManager / AgentExecutor and lets the
EpisodicMemoryAdapter create the folder on-demand when it actually writes
a summary.json.  Fast-path ReactivePool jobs never write to disk.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from integrations.llm.client import get_client
from integrations.llm.service import UpstreamService
from core.model_router import ModelRouter, load_config
from core.daemon import start_daemon
from perception import PerceptionLayer, PerceptionConfig
from perception.schemas import PerceptionWorldModel, ResponseJob
from utils.log_writer import emit


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with get_client(timeout=30.0) as client:
        app.state.upstream_client  = client
        app.state.upstream_service = UpstreamService(client)

        router_config = load_config("config/model_router.yaml")
        app.state.model_router = ModelRouter(router_config, app.state.upstream_service)

        # ── Perception config & world model ───────────────────────────────
        perception_config = PerceptionConfig()
        perception_world  = PerceptionWorldModel()
        app.state.perception_world = perception_world

        # Build the executor factory — used by the ReactivePool inside the daemon
        executor_factory = _make_executor_factory(app.state.model_router)

        # Expose a PerceptionLayer facade on app.state so adapters can call
        # perception.ingest(event).  The daemon creates its own internal instance
        # that shares the same ingest_queue, so events always flow correctly.
        app.state.perception = PerceptionLayer(
            config=perception_config,
            router=app.state.model_router,
            intake_buffer=_dummy_intake_buffer(),
            world_model=perception_world,
            executor_factory=executor_factory,
        )

        # ── Unified Daemon ─────────────────────────────────────────────────
        daemon_task = asyncio.create_task(
            start_daemon(
                config=perception_config,
                router=app.state.model_router,
                world_model=perception_world,
                executor_factory=executor_factory,
            ),
            name="noesis-daemon",
        )
        app.state.daemon_task = daemon_task

        # ── Discord bot (optional) ────────────────────────────────────────
        discord_task = await _start_discord(app)

        # ── Reddit poller (optional) ─────────────────────────────────────
        reddit_task = await _start_reddit(app)

        yield

        # ── Shutdown ──────────────────────────────────────────────────────
        await _cancel(daemon_task)
        await _stop_discord(discord_task)
        if reddit_task:
            await _cancel(reddit_task)


def _dummy_intake_buffer():
    """
    Return a minimal IntakeBuffer for the facade PerceptionLayer on app.state.
    This instance is never used for actual buffering — the daemon owns the real one.
    All events submitted via perception.ingest() go directly into the ingest_queue
    which the daemon's Feeder Task picks up.
    """
    from perception.stages.intake import IntakeBuffer
    return IntakeBuffer(window_seconds=60.0, max_buffer_size=3)


def _make_executor_factory(router: ModelRouter):
    """
    Returns an async callable(job: ResponseJob) -> None used by the ReactivePool
    to process high-urgency perception jobs inline.

    Lazy run-directory creation: we pass run_id to AgentExecutor but do NOT
    create the directory ourselves.  EpisodicMemoryAdapter handles creation
    on first write, so fast-path jobs that never write leave no disk footprint.
    """
    from agents.executor import AgentExecutor
    from utils.event_bus import event_bus

    async def executor_factory(job: ResponseJob) -> None:
        from datetime import datetime
        event_id = str(job.id)[:8]
        run_id   = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_reactive_{event_id}"
        task_description = job.event.response_context or job.event.summary

        executor = AgentExecutor(
            router=router,
            task_label=f"reactive-{event_id}",
        )

        emit("perception.job_started", "perception",
             {"job_id": event_id, "description": task_description[:80]}, run_id=run_id)

        try:
            emit("strategic.loop_started", "strategic",
                 {"goal": task_description}, run_id=run_id)

            async for event in executor.run_generator(task_description):
                event["perception_job_id"] = event_id
                event["perception_type"]   = job.event.type.value
                await event_bus.publish(event)

                ev = event.get("event")
                if ev == "thought":
                    emit("tactical.thought", "tactical",
                         {"thought": event.get("thought", "")}, run_id=run_id)
                elif ev == "tool_start":
                    emit("tactical.tool_call", "tactical",
                         {"tool": event.get("tool_name", "unknown"),
                          "input": str(event.get("tool_input", ""))}, run_id=run_id)
                elif ev == "tool_observation":
                    emit("tactical.tool_result", "tactical",
                         {"tool": event.get("tool_name", "unknown"),
                          "result": str(event.get("observation", ""))[:80]}, run_id=run_id)
                elif ev == "final_answer":
                    emit("tactical.final_answer", "tactical",
                         {"answer": event.get("answer", "")}, run_id=run_id)
                elif ev == "error":
                    emit("tactical.error", "tactical",
                         {"msg": event.get("message", "Unknown error")},
                         level="error", run_id=run_id)

            emit("perception.job_complete", "perception",
                 {"job_id": event_id}, run_id=run_id)
        except Exception as e:
            emit("perception.job_failed", "perception",
                 {"job_id": event_id, "error": str(e)}, level="error", run_id=run_id)
            raise

    return executor_factory


async def _start_discord(app: FastAPI) -> "asyncio.Task | None":
    """Start the Discord selfbot task if a token is configured."""
    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        emit("system.warning", "system",
             {"msg": "DISCORD_BOT_TOKEN not set — Discord interface disabled."}, level="warn")
        return None

    if "PYTEST_CURRENT_TEST" in os.environ:
        emit("system.warning", "system",
             {"msg": "Discord bot disabled (PYTEST running)."}, level="warn")
        return None

    from interfaces.discord.bot import bot as discord_bot

    # Inject shared state so the bot's event handlers can reach the agent core
    discord_bot.perception_layer = app.state.perception
    discord_bot.model_router     = app.state.model_router

    task = asyncio.create_task(discord_bot.start(token), name="noesis-discord")
    app.state.discord_task = task
    return task


async def _stop_discord(task: "asyncio.Task | None") -> None:
    """Gracefully close the Discord bot and cancel its task."""
    if task is None:
        return
    from interfaces.discord.bot import bot as discord_bot
    await discord_bot.close()
    await _cancel(task)


async def _start_reddit(app: FastAPI) -> "asyncio.Task | None":
    """Start the Reddit poller task if credentials are configured."""
    from integrations.reddit.config import RedditConfig
    from integrations.reddit.poller import run_reddit_poller

    config = RedditConfig()
    if not config.enabled:
        emit(
            "system.warning",
            "system",
            {"msg": "Reddit credentials not set — Reddit interface disabled."},
            level="warn",
        )
        return None

    if "PYTEST_CURRENT_TEST" in os.environ:
        emit("system.warning", "system",
             {"msg": "Reddit poller disabled (PYTEST running)."}, level="warn")
        return None

    task = asyncio.create_task(
        run_reddit_poller(config, app.state.perception),
        name="noesis-reddit",
    )
    app.state.reddit_task = task
    return task


async def _cancel(task: asyncio.Task) -> None:
    """Cancel a task and swallow the expected CancelledError."""
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
