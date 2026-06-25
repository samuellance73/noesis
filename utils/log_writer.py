import json
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()
_log_file = Path("logs/agent.jsonl")

def clear_log() -> None:
    """Clear the agent.jsonl log file at the start of a new run."""
    try:
        with _lock:
            _log_file.parent.mkdir(parents=True, exist_ok=True)
            _log_file.write_text("", encoding="utf-8")
    except Exception:
        pass

def emit(
    event: str,
    layer: str,
    data: dict,
    level: str = "info",
    run_id: str | None = None,
    task_id: str | None = None,
) -> None:
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "task_id": task_id,
            "layer": layer,
            "event": event,
            "level": level,
            "data": data,
        }
        line = json.dumps(entry) + "\n"
        
        with _lock:
            _log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass
