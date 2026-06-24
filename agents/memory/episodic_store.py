import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EpisodicEntry:
    run_id: str
    goal: str
    cycle_summaries: List[str] = field(default_factory=list)
    completed_tasks: List[Dict[str, Any]] = field(default_factory=list)  # list of {"goal": ..., "answer": ...}
    final_answer: Optional[str] = None
    is_complete: bool = False
    domain_map: Dict[str, str] = field(default_factory=dict)
    beliefs: Dict[str, float] = field(default_factory=dict)

class EpisodicStore:
    """Reads past run records from the structured summaries in run directories."""
    
    def __init__(self, runs_root: str = "logs/runs"):
        self.runs_root = pathlib.Path(runs_root)

    def load_relevant(self, current_goal: str, limit: int = 5) -> List[EpisodicEntry]:
        """Return the most recent runs whose goals overlap with current_goal."""
        if not self.runs_root.exists():
            return []

        entries: List[EpisodicEntry] = []
        # Sort directories descending by name (timestamp format YYYYMMDD_HHMMSS matches lexicographical sort)
        run_dirs = sorted(
            [d for d in self.runs_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True
        )

        for run_dir in run_dirs:
            summary_file = run_dir / "summary.json"
            if not summary_file.exists():
                continue

            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                entry = EpisodicEntry(
                    run_id=data.get("run_id", ""),
                    goal=data.get("goal", ""),
                    cycle_summaries=data.get("cycle_summaries", []),
                    completed_tasks=data.get("completed_tasks", []),
                    final_answer=data.get("final_answer"),
                    is_complete=data.get("is_complete", False),
                    domain_map=data.get("domain_map", {}),
                    beliefs=data.get("beliefs", {})
                )

                if self._is_relevant(entry.goal, current_goal):
                    entries.append(entry)
                    if len(entries) >= limit:
                        break
            except Exception:
                # Silently ignore corrupted or unreadable summary files
                continue

        return entries

    def _is_relevant(self, past_goal: str, current_goal: str) -> bool:
        """Determines if the past goal is relevant to the current goal."""
        past_tokens = set(past_goal.lower().split())
        curr_tokens = set(current_goal.lower().split())
        # If there's an overlap of at least 2 tokens, consider it relevant
        return len(past_tokens & curr_tokens) >= 2

