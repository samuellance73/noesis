import asyncio
import os
import sys
import httpx2

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integrations.llm.config import settings
from integrations.llm.service import UpstreamService
from agents.planner import plan

async def main():
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json"
    }
    base_url = settings.upstream_api_url
    if not base_url.endswith("/"):
        base_url += "/"
        
    async with httpx2.AsyncClient(base_url=base_url, headers=headers) as client:
        service = UpstreamService(client)
        steps = await plan("Research Tesla Q2 earnings and compare to analyst expectations", service)
        for step in steps:
            print(step)

if __name__ == "__main__":
    asyncio.run(main())
