"""
app/lifespan.py
───────────────
FastAPI lifespan context manager — manages startup and shutdown of all
long-running background components:

  • HTTP connection pool (httpx)
  • Autonomous trigger daemon
  • Discord selfbot (optional, gated on DISCORD_BOT_TOKEN)

Extracted from main.py so the application entry-point stays thin.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from integrations.llm.client import get_client
from integrations.llm.service import UpstreamService
from core.model_router import ModelRouter, load_config
from triggers.daemon import start_daemon

logger = logging.getLogger("noesis")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with get_client(timeout=30.0) as client:
        app.state.upstream_client  = client
        app.state.upstream_service = UpstreamService(client)

        router_config = load_config("main/config/model_router.yaml")
        app.state.model_router     = ModelRouter(router_config, app.state.upstream_service)

        # ── Daemon ────────────────────────────────────────────────────────
        daemon_task = asyncio.create_task(
            start_daemon(router=app.state.model_router, interval_seconds=60)
        )
        app.state.daemon_task = daemon_task

        # ── Discord bot (optional) ────────────────────────────────────────
        discord_task = await _start_discord(app)

        yield

        # ── Shutdown ─────────────────────────────────────────────────────
        await _cancel(daemon_task)
        await _stop_discord(discord_task)


async def _start_discord(app: FastAPI) -> "asyncio.Task | None":
    """Start the Discord bot task if a token is configured."""
    token = os.getenv("DISCORD_BOT_TOKEN")

    if not token:
        logger.warning("DISCORD_BOT_TOKEN not set — Discord interface disabled.")
        return None

    if "PYTEST_CURRENT_TEST" in os.environ:
        logger.warning("Discord bot disabled (PYTEST running).")
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
