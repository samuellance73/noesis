import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

_lock = threading.Lock()
# Anchor to this file's location so the path is correct regardless of CWD.
_log_file = Path(__file__).resolve().parent.parent / "logs" / "agent.jsonl"

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)

def clear_log() -> None:
    """
    Mark a new run boundary in agent.jsonl.

    Instead of fully truncating the file (which would silently discard
    daemon/perception infrastructure events logged before run_stream()
    fires), we truncate to zero so the new run starts clean.  The caller
    (GoalManager.run_stream) immediately writes a strategic.loop_started
    entry right after, making the boundary obvious in the log.
    """
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
        line = json.dumps(entry, cls=CustomJSONEncoder) + "\n"

        with _lock:
            _log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass
