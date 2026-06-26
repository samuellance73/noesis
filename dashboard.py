import os
import json
import asyncio
from pathlib import Path
from utils.dashboard import LiveDashboard

LOG_FILE = Path("logs/agent.jsonl")

async def tail_log_file(filepath: Path):
    """Wait for the log file to exist, read existing lines, and stream new lines as they arrive."""
    while not filepath.exists():
        await asyncio.sleep(0.5)
        
    with open(filepath, "r", encoding="utf-8") as f:
        # 1. Read historical lines to reconstruct run state
        for line in f:
            yield line
            
        # 2. Infinite loop to tail fresh appends
        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
            yield line

async def run_dashboard():
    dash = LiveDashboard(goal="Waiting for backend active run...", run_id="N/A")
    
    active_task_id = "N/A"
    active_iterations = []
    plan_tasks = []
    beliefs = {}
    gaps = []
    deferred_objectives = []

    with dash:
        async for line in tail_log_file(LOG_FILE):
            try:
                entry = json.loads(line.strip())
            except Exception:
                continue
                
            event = entry.get("event")
            layer = entry.get("layer")
            data = entry.get("data", {})
            run_id = entry.get("run_id") or "N/A"
            task_id = entry.get("task_id") or "N/A"
            
            # Feed raw logs to bottom panel
            dash.add_log_event(layer, event, data)
            
            # --- Strategic Run Init ---
            if event == "strategic.loop_started":
                dash.run_id = run_id
                dash.update_goal(data.get("goal", ""))
                plan_tasks = []
                active_iterations = []
                beliefs = {}
                gaps = []
                deferred_objectives = []
                dash.update_plan([])
                dash.update_active_task("N/A", [])
                dash.update_world_model({}, [], [])
                
            # --- Plan Received ---
            elif event == "strategic.plan_received":
                plan_tasks = [
                    {"label": t, "status": "pending", "cycle": data.get("cycle", 1), "elapsed": 0.0} 
                    for t in data.get("tasks", [])
                ]
                dash.update_plan(plan_tasks)
                
            # --- Task Spawned ---
            elif event == "strategic.task_spawned":
                active_task_id = data.get("label", "unknown")
                active_iterations = []
                for task in plan_tasks:
                    if task["label"] == data.get("goal", ""):
                        task["status"] = "running"
                dash.update_plan(plan_tasks)
                dash.update_active_task(active_task_id, active_iterations)
                
            # --- Tactical Thoughts & Tool Outputs ---
            elif event == "tactical.thought":
                active_iterations.append({"thought": data.get("thought", ""), "tool": "", "result": ""})
                dash.update_active_task(active_task_id, active_iterations)
                
            elif event == "tactical.tool_call":
                if active_iterations:
                    active_iterations[-1]["tool"] = f"{data.get('tool', '')} ← {str(data.get('input', ''))}"
                else:
                    active_iterations.append({"thought": "Running...", "tool": f"{data.get('tool', '')} ← {str(data.get('input', ''))}", "result": ""})
                dash.update_active_task(active_task_id, active_iterations)
                
            elif event == "tactical.tool_result":
                if active_iterations:
                    active_iterations[-1]["result"] = data.get("result", "")
                else:
                    active_iterations.append({"thought": "", "tool": "", "result": data.get("result", "")})
                dash.update_active_task(active_task_id, active_iterations)
                
            # --- Task Completed ---
            elif event == "strategic.task_complete":
                status = "done" if data.get("success", True) else "failed"
                for task in plan_tasks:
                    if data.get("label", "") in task.get("label", "") or task.get("label", "") in data.get("label", ""):
                        task["status"] = status
                dash.update_plan(plan_tasks)
                
            # --- Cycle Completed ---
            elif event == "strategic.cycle_complete":
                objectives = data.get("objectives", [])
                deferred_objectives = [obj["description"] for obj in objectives if obj["status"] == "deferred"]
                
                wm = data.get("world_model", {})
                beliefs = wm.get("beliefs", {})
                gaps = wm.get("gaps", [])
                
                dash.update_world_model(beliefs, gaps, deferred_objectives)

if __name__ == "__main__":
    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        print("\nExiting dashboard...")
