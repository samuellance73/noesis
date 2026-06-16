from contextlib import asynccontextmanager
import httpx2
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from integrations.llm.config import settings
from integrations.llm.router import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure the persistent client headers globally
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json"
    }
    
    # Ensure base_url ends with a trailing slash so that relative URL resolution works properly
    base_url = settings.upstream_api_url
    if not base_url.endswith("/"):
        base_url += "/"

    # Open connection pool on startup
    async with httpx2.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=30.0
    ) as client:
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
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
