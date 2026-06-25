"""
interfaces/cli/main.py
──────────────────────
Terminal interface for the autonomous GoalManager loop.

Usage
─────
  python run_cli.py

Lifecycle
─────────
  1. User enters the ultimate goal.
  2. GoalManager starts its autonomous loop in a background task.
  3. A foreground input listener reads stdin and feeds messages into the manager.
  4. Typing "stop", "quit", "exit", or pressing Ctrl-C halts the loop gracefully.
  5. Any other text is injected as a goal refinement mid-run.
"""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from integrations.llm.service import UpstreamService
from integrations.llm.client import get_client
from core.model_router import ModelRouter, load_config
from agents.goal_manager import GoalManager
from utils.dashboard import LiveDashboard, SimpleDashboard
from utils.log_writer import emit

console = Console()

_STOP_HINT = "[dim]Type [bold]stop[/bold] to stop · any other text refines the goal[/dim]"


def _render_event(event: dict) -> None:
    """Pretty-print a single GoalManager event to the terminal."""
    ev = event.get("event")

    if ev == "goal_set":
        console.print(Rule(f"[bold purple]🎯 Goal Set[/bold purple]"))
        console.print(f"[bold]{event['goal']}[/bold]\n")

    elif ev == "cycle_start":
        console.print(Rule(f"[cyan]Cycle {event['cycle']}[/cyan]", style="cyan"))

    elif ev == "manager_thought":
        console.print(f"[yellow]🧠 Manager:[/yellow] [italic]{event['thought']}[/italic]")

    elif ev == "spawning_tasks":
        count = event["count"]
        console.print(f"[magenta]⚡ Spawning {count} executor(s) in parallel:[/magenta]")
        for t in event.get("tasks", []):
            console.print(f"   [dim]→ {t}[/dim]")

    elif ev == "iteration_start":
        task_goal = event.get("task_goal", "")
        label = f" [{task_goal[:40]}]" if task_goal else ""
        console.print(f"[dim]  ↻ Executor iter {event['iteration']}{label}[/dim]")

    elif ev == "thought":
        console.print(f"[yellow]  Thought:[/yellow] [italic]{event['thought']}[/italic]")

    elif ev == "tool_start":
        console.print(
            f"[cyan]  ⚙  {event['tool_name']}[/cyan] ← [dim]{str(event['tool_input'])[:80]}[/dim]"
        )

    elif ev == "tool_observation":
        obs = event.get("observation", "")
        cropped = obs if len(obs) < 300 else f"{obs[:300]}… [cropped]"
        console.print(f"[grey50]  Obs:[/grey50] {cropped}\n")

    elif ev == "final_answer":
        task_goal = event.get("task_goal", "sub-task")
        console.print(
            Panel(
                event["answer"],
                title=f"[green]✓ {task_goal[:60]}[/green]",
                border_style="green",
            )
        )

    elif ev == "cycle_complete":
        console.print(f"\n[bold blue]📊 Cycle {event['cycle']} complete:[/bold blue] {event['progress_update']}")
        if event.get("open_questions"):
            console.print("[dim]  Open questions:[/dim]")
            for q in event["open_questions"]:
                console.print(f"  [dim]? {q}[/dim]")
        console.print()

    elif ev == "user_input_received":
        console.print(f"[bold green]↩ Injected:[/bold green] {event['message']}")

    elif ev == "goal_complete":
        console.print(Rule("[bold green]✅ Goal Complete[/bold green]", style="green"))
        if event.get("final_answer"):
            console.print(
                Panel(event["final_answer"], title="Final Answer", border_style="bright_green")
            )
        console.print(f"[dim]Completed in {event['cycle']} cycle(s).[/dim]")

    elif ev == "stopped":
        console.print(f"\n[bold red]⏹  Stopped[/bold red] (cycle {event['cycle']}): {event.get('reason', '')}")

    elif ev == "error":
        console.print(f"[bold red]❌ Error:[/bold red] {event.get('message', event)}")
        if event.get("summary"):
            console.print(Panel(event["summary"], title="Progress so far", border_style="yellow"))


def _update_dashboard_full(dash: LiveDashboard, event: dict) -> None:
    """Update the full dashboard based on event type."""
    ev = event.get("event")
    
    if ev == "goal_set":
        dash.update_goal(event.get("goal", ""))
    elif ev == "spawning_tasks":
        tasks = [{"label": t[:40], "status": "pending"} for t in event.get("tasks", [])]
        dash.update_plan(tasks)
    elif ev == "cycle_start":
        dash.add_log_event("strategic", "cycle_start", {"cycle": event.get("cycle")})
    elif ev == "manager_thought":
        dash.add_log_event("strategic", "manager_thought", {"thought": event.get("thought", "")[:50]})
    elif ev == "thought":
        dash.add_log_event("tactical", "thought", {"thought": event.get("thought", "")[:50]})
    elif ev == "tool_start":
        dash.add_log_event("tactical", "tool_call", {"tool": event.get("tool_name"), "input": str(event.get("tool_input", ""))[:30]})
    elif ev == "tool_observation":
        dash.add_log_event("tactical", "tool_result", {"result": event.get("observation", "")[:50]})
    elif ev == "final_answer":
        dash.add_log_event("tactical", "final_answer", {"answer": event.get("answer", "")[:50]})


def _update_dashboard_simple(dash: SimpleDashboard, event: dict) -> None:
    """Update the simple dashboard based on event type."""
    ev = event.get("event")
    
    if ev == "thought":
        dash.add_log_event("tactical", "thought", {"thought": event.get("thought", "")[:50]})
    elif ev == "tool_start":
        dash.add_log_event("tactical", "tool_call", {"tool": event.get("tool_name"), "input": str(event.get("tool_input", ""))[:30]})
    elif ev == "tool_observation":
        dash.add_log_event("tactical", "tool_result", {"result": event.get("observation", "")[:50]})
    elif ev == "final_answer":
        dash.add_log_event("tactical", "final_answer", {"answer": event.get("answer", "")[:50]})


async def _input_listener(manager: GoalManager) -> None:
    """
    Reads lines from stdin without blocking the event loop.
    Each line is forwarded to manager.inject_input().
    Exits when the manager's stop event fires or EOF is reached.
    """
    loop = asyncio.get_running_loop()
    while not manager._stop_event.is_set():
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except Exception:
            break
        if not line:        # EOF (e.g. pipe closed)
            break
        text = line.strip()
        if text:
            await manager.inject_input(text)
            if manager._stop_event.is_set():
                break


async def run_terminal_interface() -> None:
    router_config = load_config("config/model_router.yaml")

    async with get_client(timeout=60.0) as client:
        transport = UpstreamService(client)
        router    = ModelRouter(router_config, transport)

        console.print(
            Panel(
                "[bold purple]Noesis — Autonomous Agent[/bold purple]\n"
                "[dim]Set an ultimate goal and let the agent work autonomously.\n"
                "Inject refinements at any time. Type [bold]stop[/bold] to halt.[/dim]",
                expand=False,
            )
        )

        while True:
            # ── Get ultimate goal ──────────────────────────────────────
            try:
                goal = console.input("\n[bold blue]Goal > [/bold blue]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[bold red]Goodbye![/bold red]")
                break

            if not goal:
                continue
            if goal.lower() in ("exit", "quit", "stop"):
                console.print("[bold red]Goodbye![/bold red]")
                break

            manager = GoalManager(router=router)

            console.print(f"\n{_STOP_HINT}\n")

            # ── Run manager loop + input listener concurrently ─────────
            async def stream_events():
                # Check if this is a quick command (! prefix)
                is_quick = goal.startswith("!")
                
                if is_quick:
                    # Use simple dashboard for quick commands
                    with SimpleDashboard() as dash:
                        async for event in manager.run_stream(goal):
                            _render_event(event)
                            _update_dashboard_simple(dash, event)
                else:
                    # Use full dashboard for autonomous runs
                    import uuid
                    run_id = str(uuid.uuid4())[:8]
                    with LiveDashboard(goal=goal, run_id=run_id) as dash:
                        async for event in manager.run_stream(goal):
                            _render_event(event)
                            _update_dashboard_full(dash, event)

            try:
                await asyncio.gather(
                    stream_events(),
                    _input_listener(manager),
                )
            except KeyboardInterrupt:
                manager.request_stop()
                console.print("\n[bold red]Interrupted — stopping agent.[/bold red]")
            except Exception as e:
                console.print(f"[bold red]System error: {e}[/bold red]")

            console.print(Rule(style="dim"))
            console.print("[dim]Agent loop ended. Enter a new goal or type exit.[/dim]")
