import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from utils.logging_setup import setup_global_logging
from interfaces.web.router import router
from client import get_client

# Show the full trace tree in the terminal at INFO level.
# Reduce to logging.WARNING to silence the trace and keep only errors.
setup_global_logging(console_level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open connection pool on startup using the shared client module
    async with get_client(timeout=30.0) as client:
        app.state.upstream_client = client
        yield
    # Closes connection pool cleanly on shutdown

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/", StaticFiles(directory="interfaces/web/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
