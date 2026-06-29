"""
utils/prompts.py
────────────────
Prompt loading utility for decoupling prompt strings from application logic.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt from the prompts/ directory."""
    try:
        return (PROMPTS_DIR / filename).read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to load prompt {filename}: {e}")
