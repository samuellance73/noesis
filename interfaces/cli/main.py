import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from integrations.llm.service import UpstreamService
from agents.planner import plan
from agents.executor import AgentExecutor
from client import get_client

console = Console()

async def run_terminal_interface():
    # 1. Initialize our connection "Outlet" (HTTP client) using the shared client module
    async with get_client(timeout=45.0) as client:
        service = UpstreamService(client)
        
        console.print(Panel("[bold purple]Noesis CLI Agent Client Ready[/bold purple]", expand=False))
        
        # Get default model from environment or fallback to groq/openai/gpt-oss-120b
        default_model = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")
        
        while True:
            try:
                # Prompt the user for input
                user_input = console.input("\n[bold blue]User > [/bold blue]").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[bold red]Goodbye![/bold red]")
                    break
                
                # --- PHASE 1: Planning ---
                console.print("\n[bold magenta]🧠 Generating Task Plan...[/bold magenta]")
                steps = await plan(user_input, service)
                
                # Print the generated plan as a clean table
                table = Table(title="Execution Roadmap")
                table.add_column("Step ID", justify="center", style="cyan")
                table.add_column("Sub-Goal", style="green")
                table.add_column("Depends On", justify="center", style="yellow")
                
                for step in steps:
                    deps = ", ".join(map(str, step.get("depends_on", []))) or "None"
                    table.add_row(str(step["id"]), step["goal"], deps)
                console.print(table)
                
                # --- PHASE 2: Step-by-Step Execution ---
                results = []
                for idx, step in enumerate(steps):
                    console.print(f"\n[bold purple]🚀 [Step {idx+1}/{len(steps)}] Executing: {step['goal']}[/bold purple]")
                    
                    # Context Injection
                    step_input = step["goal"]
                    if results:
                        context_str = "Context of completed steps:\n"
                        for prev in results:
                            context_str += f"- Task: {prev['step']}\n  Result: {prev['result']}\n\n"
                        step_input = f"{context_str}Current Task: {step_input}"
                    
                    # Create a fresh, isolated Executor for this sub-task
                    executor = AgentExecutor(llm_service=service, model=default_model)
                    final_result = None
                    
                    # Stream and print step updates in real-time
                    async for event in executor.run_generator(step_input):
                        if event["event"] == "iteration_start":
                            console.print(f"[dim]  --- Iteration {event['iteration']} ---[/dim]")
                            
                        elif event["event"] == "thought":
                            console.print(f"[yellow]  Thought:[/yellow] [italic]{event['thought']}[/italic]")
                            
                        elif event["event"] == "tool_start":
                            console.print(f"[cyan]  ⚙️  Calling tool [bold]{event['tool_name']}[/bold] with input:[/cyan] {event['tool_input']}")
                            
                        elif event["event"] == "tool_observation":
                            # Crop long observations to avoid terminal flooding
                            obs_text = event["observation"]
                            cropped = obs_text if len(obs_text) < 400 else f"{obs_text[:400]}... [cropped]"
                            console.print(f"[grey50]  Observation Result:[/grey50]\n{cropped}\n")
                            
                        elif event["event"] == "final_answer":
                            final_result = event["answer"]
                            console.print(Panel(final_result, title=f"Step {idx+1} Complete", border_style="green"))
                            
                        elif event["event"] == "error":
                            console.print(f"[bold red]❌ Step Error: {event['message']}[/bold red]")
                    
                    results.append({"step": step["goal"], "result": final_result})
                    
                # Final complete summary
                console.print("\n[bold green]✅ ALL TASKS COMPLETE![/bold green]")
                
            except Exception as e:
                console.print(f"[bold red]System Error: {str(e)}[/bold red]")
