"""
utils/dashboard.py
──────────────────
Rich live dashboard for real-time observability during agent runs.

Displays:
- GOAL: static mission string
- PLAN: cycle progress with task status
- ACTIVE TASK: live tactical ReAct loop
- WORLD MODEL: key beliefs, gaps, blocked tasks
- LOG: rolling tail of recent events
"""

import asyncio
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class LiveDashboard:
    """
    Live-rendered terminal dashboard for agent observability.
    
    Usage:
        with LiveDashboard(goal="Research X", run_id="abc123") as dash:
            dash.update_goal(goal)
            dash.update_plan(tasks)
            dash.update_active_task(task_id, thought, tool, result)
            dash.update_world_model(beliefs, gaps, blocked)
            dash.add_log_event(layer, event, data)
    """
    
    def __init__(
        self,
        goal: str,
        run_id: str,
        refresh_per_second: int = 4,
    ):
        self.goal = goal
        self.run_id = run_id
        self.console = Console()
        self.refresh_per_second = refresh_per_second
        
        # State
        self.start_time = time.time()
        self.plan_tasks: list[dict] = []  # [{"label": "task-1", "status": "pending|running|done|failed", "elapsed": 1.2}]
        self.active_task: Optional[dict] = None  # {"task_id": "task-1", "iterations": [...]}
        self.world_model: dict = {"beliefs": {}, "gaps": [], "blocked": []}
        self.log_events: deque = deque(maxlen=10)
        self._live: Optional[Live] = None
        self._running = False
        
    @contextmanager
    def __enter__(self):
        self._running = True
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args):
        self._running = False
        if self._live:
            self._live.__exit__(*args)
    
    def _render(self):
        """Render the full dashboard layout."""
        elapsed = time.time() - self.start_time
        
        # Header
        header = Text()
        header.append(" NOESIS ", style="bold white on blue")
        header.append(f" run: {self.run_id} ", style="dim")
        header.append(f" {self._format_elapsed(elapsed)} ", style="dim")
        
        # Build panels
        goal_panel = Panel(
            Text(self.goal, style="bold cyan"),
            title="GOAL",
            border_style="blue",
        )
        
        plan_panel = self._render_plan()
        active_panel = self._render_active_task()
        world_panel = self._render_world_model()
        log_panel = self._render_log()
        
        # Layout using vertical stack
        layout = Table.grid()
        layout.add_row(header)
        layout.add_row("")
        layout.add_row(goal_panel)
        layout.add_row("")
        layout.add_row(plan_panel)
        layout.add_row("")
        layout.add_row(active_panel)
        layout.add_row("")
        layout.add_row(world_panel)
        layout.add_row("")
        layout.add_row(log_panel)
        
        return layout
    
    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time as MM:SS."""
        mins, secs = divmod(int(seconds), 60)
        return f"{mins:02d}:{secs:02d}"
    
    def _render_plan(self):
        """Render the PLAN panel with task status."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Cycle", width=8)
        table.add_column("Task", width=40)
        table.add_column("Status", width=12)
        table.add_column("Time", width=8)
        
        for task in self.plan_tasks:
            status_icon = {
                "pending": "●",
                "running": "▶",
                "done": "✓",
                "failed": "✗",
            }.get(task.get("status", "pending"), "?")
            
            status_style = {
                "pending": "dim",
                "running": "bold yellow",
                "done": "bold green",
                "failed": "bold red",
            }.get(task.get("status", "pending"), "dim")
            
            elapsed = task.get("elapsed", 0)
            elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
            
            table.add_row(
                str(task.get("cycle", "?")),
                task.get("label", "")[:38],
                Text(f"{status_icon} {task.get('status', 'pending')}", style=status_style),
                elapsed_str,
            )
        
        return Panel(table, title="PLAN", border_style="magenta")
    
    def _render_active_task(self):
        """Render the ACTIVE TASK panel with live iterations."""
        if not self.active_task:
            return Panel(Text("No active task", style="dim italic"), title="ACTIVE TASK", border_style="yellow")
        
        task_id = self.active_task.get("task_id", "unknown")
        iterations = self.active_task.get("iterations", [])
        
        table = Table(show_header=False)
        table.add_column("", width=6)
        table.add_column("Content", width=70)
        
        for i, iter_data in enumerate(iterations[-5:]):  # Show last 5 iterations
            thought = iter_data.get("thought", "")[:70]
            tool = iter_data.get("tool", "")
            result = iter_data.get("result", "")[:70]
            
            if thought:
                table.add_row(f"iter {i+1}", Text(f"thought: {thought}", style="cyan"))
            if tool:
                table.add_row("", Text(f"→ tool: {tool}", style="yellow"))
            if result:
                table.add_row("", Text(f"← {result}", style="green"))
        
        return Panel(table, title=f"ACTIVE TASK · {task_id}", border_style="yellow")
    
    def _render_world_model(self):
        """Render the WORLD MODEL panel."""
        beliefs = self.world_model.get("beliefs", {})
        gaps = self.world_model.get("gaps", [])
        blocked = self.world_model.get("blocked", [])
        
        lines = []
        
        if beliefs:
            lines.append(Text("Beliefs:", style="bold cyan"))
            for claim, conf in list(beliefs.items())[:3]:
                lines.append(Text(f"  {claim[:60]}: {conf:.2f}", style="dim"))
        
        if gaps:
            lines.append(Text("Gaps:", style="bold yellow"))
            for gap in gaps[:3]:
                lines.append(Text(f"  • {gap[:60]}", style="dim"))
        
        if blocked:
            lines.append(Text("Blocked:", style="bold red"))
            for b in blocked[:2]:
                lines.append(Text(f"  ✗ {b[:60]}", style="dim"))
        
        if not lines:
            lines.append(Text("No world model data", style="dim italic"))
        
        content = Text("\n").join(lines)
        return Panel(content, title="WORLD MODEL", border_style="cyan")
    
    def _render_log(self):
        """Render the LOG panel with recent events."""
        table = Table(show_header=False)
        table.add_column("Time", width=8)
        table.add_column("Layer", width=10)
        table.add_column("Event", width=15)
        table.add_column("Data", width=40)
        
        for event in self.log_events:
            ts = event.get("ts", "")[:8]  # HH:MM:SS
            layer = event.get("layer", "")
            event_name = event.get("event", "")
            data_str = str(event.get("data", {}))[:38]
            
            layer_style = {
                "system": "bold blue",
                "strategic": "bold magenta",
                "tactical": "bold yellow",
                "llm": "dim",
                "perception": "bold cyan",
            }.get(layer, "dim")
            
            table.add_row(
                ts,
                Text(layer, style=layer_style),
                Text(event_name, style="green"),
                Text(data_str, style="dim"),
            )
        
        return Panel(table, title="LOG", border_style="green")
    
    def refresh(self):
        """Force a dashboard refresh."""
        if self._live and self._running:
            self._live.update(self._render())
    
    def update_goal(self, goal: str):
        """Update the goal string."""
        self.goal = goal
        self.refresh()
    
    def update_plan(self, tasks: list[dict]):
        """Update the plan with current task list."""
        self.plan_tasks = tasks
        self.refresh()
    
    def update_active_task(self, task_id: str, iterations: list[dict]):
        """Update the active task with iteration data."""
        self.active_task = {"task_id": task_id, "iterations": iterations}
        self.refresh()
    
    def update_world_model(self, beliefs: dict, gaps: list, blocked: list):
        """Update the world model state."""
        self.world_model = {"beliefs": beliefs, "gaps": gaps, "blocked": blocked}
        self.refresh()
    
    def add_log_event(self, layer: str, event: str, data: dict):
        """Add a log event to the rolling log."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log_events.append({"ts": ts, "layer": layer, "event": event, "data": data})
        self.refresh()


class SimpleDashboard:
    """
    Simplified two-panel dashboard for quick commands (! prefix).
    Only shows ACTIVE TASK and LOG panels.
    """
    
    def __init__(self, refresh_per_second: int = 4):
        self.console = Console()
        self.refresh_per_second = refresh_per_second
        self.active_task: Optional[dict] = None
        self.log_events: deque = deque(maxlen=10)
        self._live: Optional[Live] = None
        self._running = False
    
    @contextmanager
    def __enter__(self):
        self._running = True
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args):
        self._running = False
        if self._live:
            self._live.__exit__(*args)
    
    def _render(self):
        """Render simplified two-panel layout."""
        active_panel = self._render_active_task()
        log_panel = self._render_log()
        
        layout = Table.grid()
        layout.add_row(active_panel)
        layout.add_row("")
        layout.add_row(log_panel)
        
        return layout
    
    def _render_active_task(self):
        """Render ACTIVE TASK panel."""
        if not self.active_task:
            return Panel(Text("No active task", style="dim italic"), title="ACTIVE TASK", border_style="yellow")
        
        iterations = self.active_task.get("iterations", [])
        table = Table(show_header=False)
        table.add_column("", width=6)
        table.add_column("Content", width=70)
        
        for i, iter_data in enumerate(iterations[-5:]):
            thought = iter_data.get("thought", "")[:70]
            tool = iter_data.get("tool", "")
            result = iter_data.get("result", "")[:70]
            
            if thought:
                table.add_row(f"iter {i+1}", Text(f"thought: {thought}", style="cyan"))
            if tool:
                table.add_row("", Text(f"→ tool: {tool}", style="yellow"))
            if result:
                table.add_row("", Text(f"← {result}", style="green"))
        
        return Panel(table, title="ACTIVE TASK", border_style="yellow")
    
    def _render_log(self):
        """Render LOG panel."""
        table = Table(show_header=False)
        table.add_column("Time", width=8)
        table.add_column("Layer", width=10)
        table.add_column("Event", width=15)
        table.add_column("Data", width=40)
        
        for event in self.log_events:
            ts = event.get("ts", "")[:8]
            layer = event.get("layer", "")
            event_name = event.get("event", "")
            data_str = str(event.get("data", {}))[:38]
            
            layer_style = {
                "tactical": "bold yellow",
                "llm": "dim",
            }.get(layer, "dim")
            
            table.add_row(
                ts,
                Text(layer, style=layer_style),
                Text(event_name, style="green"),
                Text(data_str, style="dim"),
            )
        
        return Panel(table, title="LOG", border_style="green")
    
    def refresh(self):
        """Force a dashboard refresh."""
        if self._live and self._running:
            self._live.update(self._render())
    
    def update_active_task(self, task_id: str, iterations: list[dict]):
        """Update the active task."""
        self.active_task = {"task_id": task_id, "iterations": iterations}
        self.refresh()
    
    def add_log_event(self, layer: str, event: str, data: dict):
        """Add a log event."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log_events.append({"ts": ts, "layer": layer, "event": event, "data": data})
        self.refresh()
