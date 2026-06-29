"""
agents/memory/base.py
────────────────────
Abstract memory interface for dependency injection.
"""

from typing import Protocol, List, Dict, Any, Optional
from pathlib import Path


class MemoryStore(Protocol):
    """Abstract interface for memory storage and retrieval."""
    
    def load_relevant(self, goal: str, limit: int = 5) -> List[Any]:
        """Load relevant past entries based on goal similarity."""
        ...

    def write_summary(self, run_id: str, data: Dict[str, Any]) -> None:
        """Write a summary entry for a run."""
        ...


class EpisodicMemoryAdapter:
    """
    Adapter that combines EpisodicStore and EpisodicWriter to implement MemoryStore protocol.
    This allows the existing JSON-based episodic memory to work with the new interface.
    """
    
    def __init__(self, runs_root: str = "logs/runs", run_dir: Path | None = None):
        from .episodic_store import EpisodicStore
        from .episodic_writer import EpisodicWriter
        
        self._store = EpisodicStore(runs_root=runs_root)
        self._writer = EpisodicWriter(run_dir) if run_dir else None
    
    def set_run_dir(self, run_dir: Path) -> None:
        """Set the run directory for the writer (called after run_dir is created)."""
        from .episodic_writer import EpisodicWriter
        self._writer = EpisodicWriter(run_dir)
    
    def load_relevant(self, goal: str, limit: int = 5) -> List[Any]:
        """Load relevant past entries using EpisodicStore."""
        return self._store.load_relevant(goal, limit=limit)
    
    def write_summary(self, run_id: str, data: Dict[str, Any]) -> None:
        """Write summary using EpisodicWriter."""
        if self._writer is None:
            raise RuntimeError("EpisodicMemoryAdapter: run_dir not set. Call set_run_dir() first.")
        self._writer.write_summary(
            run_id=run_id,
            goal=data.get("goal", ""),
            cycle_summaries=data.get("cycle_summaries", []),
            completed_tasks=data.get("completed_tasks", []),
            final_answer=data.get("final_answer"),
            is_complete=data.get("is_complete", False),
            domain_map=data.get("domain_map", {}),
            beliefs=data.get("beliefs", {})
        )
