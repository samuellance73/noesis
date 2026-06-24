import json
import pathlib
from typing import List, Dict, Any, Optional

class EpisodicWriter:
    """Writes a structured JSON summary of a run to its run directory."""

    def __init__(self, run_dir: pathlib.Path):
        self.run_dir = run_dir
        self.summary_file = self.run_dir / "summary.json"

    def write_summary(
        self,
        run_id: str,
        goal: str,
        cycle_summaries: List[str],
        completed_tasks: List[Dict[str, Any]],
        final_answer: Optional[str],
        is_complete: bool,
        domain_map: Dict[str, str],
        beliefs: Dict[str, float]
    ) -> None:
        """Saves a structured JSON file describing the current/final state of the run."""
        data = {
            "run_id": run_id,
            "goal": goal,
            "cycle_summaries": cycle_summaries,
            "completed_tasks": completed_tasks,
            "final_answer": final_answer,
            "is_complete": is_complete,
            "domain_map": domain_map,
            "beliefs": beliefs
        }
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            # Let exceptions in logging pass silently so as not to crash the agent
            pass

