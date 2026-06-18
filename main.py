import logging
import os
import ssl
from dotenv import load_dotenv

# Monkey-patch ssl.create_default_context globally to bypass strict SSL verification
# This handles cases where NixOS / restricted environments lack proper CA bundles.
_orig_create_default_context = ssl.create_default_context
def _unverified_context(*args, **kwargs):
    ctx = _orig_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _unverified_context

# Also patch the default https context for urllib
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Load environment variables from .env
load_dotenv(override=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from utils.logging_setup import setup_global_logging
from interfaces.web.router import router
from integrations.llm.client import get_client
from integrations.llm.service import UpstreamService
from triggers.daemon import start_daemon

# Show the full trace tree in the terminal at INFO level.
# Reduce to logging.WARNING to silence the trace and keep only errors.
# WARNING = quiet normal operation. Set env LOG_LEVEL=INFO for full verbose output.
_console_level = logging.INFO if os.getenv("LOG_LEVEL", "").upper() == "INFO" else logging.WARNING
setup_global_logging(console_level=_console_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open connection pool on startup; build a single shared UpstreamService
    async with get_client(timeout=30.0) as client:
        app.state.upstream_client  = client
        app.state.upstream_service = UpstreamService(client)

        # Spawn the autonomous daemon — picks up triggers every 60s (human
        # triggers fire immediately via fast-lane).
        daemon_task = asyncio.create_task(
            start_daemon(service=app.state.upstream_service, interval_seconds=60)
        )
        app.state.daemon_task = daemon_task

        # Spawn the Discord bot alongside the web server in the same event loop.
        discord_token = os.getenv("DISCORD_BOT_TOKEN")
        discord_task = None
        if discord_token and "PYTEST_CURRENT_TEST" not in os.environ:
            from interfaces.discord.bot import bot as discord_bot
            discord_task = asyncio.create_task(discord_bot.start(discord_token))
            app.state.discord_task = discord_task
        else:
            msg = "Discord bot disabled (PYTEST running)." if "PYTEST_CURRENT_TEST" in os.environ else "DISCORD_BOT_TOKEN not set — Discord interface disabled."
            logging.getLogger("noesis").warning(msg)

        yield

        # Clean shutdown: cancel daemon and Discord bot
        daemon_task.cancel()
        try:
            await daemon_task
        except asyncio.CancelledError:
            pass

        if discord_task:
            from interfaces.discord.bot import bot as discord_bot
            await discord_bot.close()
            discord_task.cancel()
            try:
                await discord_task
            except asyncio.CancelledError:
                pass

    # Closes connection pool cleanly on shutdown


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response: Response = await call_next(request)
    # Tell browsers never to serve stale static assets from cache.
    if request.url.path.startswith("/") and not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.include_router(router)
app.mount("/", StaticFiles(directory="interfaces/web/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
