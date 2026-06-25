"""
tests/test_live_models.py
─────────────────────────
Integration tests that make live API calls to verify the models and tiers
configured in model_router.yaml are active, reachable, and performant.

Run with:
    uv run pytest tests/test_live_models.py -s
or:
    python tests/test_live_models.py
"""

import asyncio
import os
import sys
import time
import pytest

# Add the project root to sys.path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from integrations.llm.client import get_client
from integrations.llm.service import UpstreamService
from core.model_router import ModelRouter, load_config, ModelTier, ModelRequest

from rich.console import Console
from rich.table import Table

console = Console()


async def run_live_tests():
    console.print("\n[bold purple]🔍 Testing Live API Models from model_router.yaml[/bold purple]\n")
    
    # 1. Load config
    config = load_config("config/model_router.yaml")
    
    # 2. Extract unique models from the config
    unique_models = set()
    for tier_config in config.tiers.values():
        unique_models.add(tier_config.primary)
        for fb in tier_config.fallbacks:
            unique_models.add(fb)
            
    console.print(f"Loaded config from [yellow]config/model_router.yaml[/yellow]")
    console.print(f"Found [bold]{len(unique_models)}[/bold] unique model strings configured across tiers:")
    for model in sorted(unique_models):
        console.print(f"  • [blue]{model}[/blue]")
    console.print("")

    async with get_client(timeout=30.0) as client:
        service = UpstreamService(client)
        router = ModelRouter(config, service)
        
        # We will test two things:
        # A. Direct connection to each unique model
        # B. End-to-end routing via ModelRouter tiersz

        # --- A. Direct Model Tests ---
        console.print("[bold cyan]=== Part A: Testing Direct Upstream Model Connections ===[/bold cyan]")
        direct_table = Table(title="Direct Model Verification")
        direct_table.add_column("Model String", style="blue")
        direct_table.add_column("Status", justify="center")
        direct_table.add_column("Latency (ms)", justify="right")
        direct_table.add_column("Response Snippet", style="italic")
        direct_table.add_column("Tokens (P/C/T)", justify="center")

        for model in sorted(unique_models):
            start_time = time.perf_counter()
            try:
                # Call model directly
                response = await service.chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": "Respond with only: 'pong'"}],
                    temperature=0.1,
                    max_tokens=10,
                )
                latency = int((time.perf_counter() - start_time) * 1000)
                usage = f"{response.usage.prompt_tokens}/{response.usage.completion_tokens}/{response.usage.total_tokens}"
                snippet = response.content.replace("\n", " ").strip()
                direct_table.add_row(model, "[green]SUCCESS[/green]", f"{latency}ms", snippet[:35], usage)
            except Exception as e:
                latency = int((time.perf_counter() - start_time) * 1000)
                direct_table.add_row(model, f"[red]FAIL ({type(e).__name__})[/red]", f"{latency}ms", str(e)[:35], "N/A")

        console.print(direct_table)
        console.print("")

        # --- B. Tier-based Routing Tests ---
        console.print("[bold cyan]=== Part B: Testing ModelRouter Tier Interface ===[/bold cyan]")
        router_table = Table(title="ModelRouter Tier Routing")
        router_table.add_column("Tier", style="magenta")
        router_table.add_column("Primary Model", style="blue")
        router_table.add_column("Actual Model Used", style="cyan")
        router_table.add_column("Status", justify="center")
        router_table.add_column("Latency (ms)", justify="right")
        router_table.add_column("Tokens (P/C/T)", justify="center")
        router_table.add_column("Fallback Used?", justify="center")

        for tier in ModelTier:
            start_time = time.perf_counter()
            primary_model = config.tiers[tier].primary
            try:
                request = ModelRequest(
                    tier=tier,
                    messages=[{"role": "user", "content": "Hello. Answer in under 5 words."}],
                    component="test.live_script"
                )
                response = await router.complete(request)
                latency = int(response.latency_ms)
                usage = f"{response.prompt_tokens}/{response.completion_tokens}/{response.total_tokens}"
                fallback_status = "[yellow]Yes[/yellow]" if response.fallback_used else "No"
                router_table.add_row(
                    tier.value,
                    primary_model,
                    response.model_used,
                    "[green]SUCCESS[/green]",
                    f"{latency}ms",
                    usage,
                    fallback_status
                )
            except Exception as e:
                latency = int((time.perf_counter() - start_time) * 1000)
                router_table.add_row(
                    tier.value,
                    primary_model,
                    "N/A",
                    f"[red]FAIL ({type(e).__name__})[/red]",
                    f"{latency}ms",
                    "N/A",
                    "N/A"
                )

        console.print(router_table)
        console.print("")


@pytest.mark.asyncio
async def test_live_model_router_apis():
    """Pytest entrypoint for running live model tests."""
    await run_live_tests()


if __name__ == "__main__":
    asyncio.run(run_live_tests())
