"""
utils/json_parser.py
────────────────────
Shared LLM JSON output parser.

LLM responses frequently contain extra noise around the JSON payload:
- <think>…</think> reasoning blocks
- Markdown code fences (```json … ```)
- Trailing commas (invalid in strict JSON)

This module normalises all of that into a clean JSON string and then
validates it against a Pydantic schema.

Usage
─────
    from utils.json_parser import parse_llm_json
    from agents.schemas import AgentStep

    step = parse_llm_json(raw_response, AgentStep)
"""

import json
import re
from typing import TypeVar, Type

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _clean_llm_json(raw: str) -> str:
    """
    Strip LLM noise and return a best-effort clean JSON string.

    Steps (in order):
    1. Strip <think>…</think> reasoning blocks.
    2. Extract content from markdown code fences wrapping the JSON (if any).
    3. Isolate the outermost JSON object { … } or list [ … ].
    4. Remove trailing commas before } or ] (invalid in strict JSON).
    """
    clean = raw.strip()

    # 1. Strip thinking tags
    if "<think>" in clean and "</think>" in clean:
        clean = clean.split("</think>", 1)[1].strip()
    elif "</think>" in clean:
        clean = clean.split("</think>", 1)[1].strip()

    # 2. Extract from markdown code fences wrapping the JSON
    first_idx = -1
    for char in ["{", "["]:
        idx = clean.find(char)
        if idx != -1 and (first_idx == -1 or idx < first_idx):
            first_idx = idx

    last_idx = -1
    for char in ["}", "]"]:
        idx = clean.rfind(char)
        if idx != -1 and idx > last_idx:
            last_idx = idx

    if first_idx != -1 and last_idx != -1:
        first_fence = clean.find("```")
        last_fence = clean.rfind("```")
        if first_fence != -1 and first_fence < first_idx and last_fence != -1 and last_fence > last_idx:
            fence_content = clean[first_fence:last_fence].strip()
            match = re.match(r'^```(?:json)?\s*(.*)$', fence_content, re.DOTALL)
            if match:
                clean = match.group(1).strip()

    # 3. Isolate outermost JSON object/list
    first_idx = -1
    for char in ["{", "["]:
        idx = clean.find(char)
        if idx != -1 and (first_idx == -1 or idx < first_idx):
            first_idx = idx

    last_idx = -1
    for char in ["}", "]"]:
        idx = clean.rfind(char)
        if idx != -1 and idx > last_idx:
            last_idx = idx

    if first_idx != -1 and last_idx != -1:
        clean = clean[first_idx:last_idx + 1]

    # 4. Remove trailing commas
    clean = re.sub(r',(\s*[}\]])', r'\1', clean)

    return clean


def parse_llm_json(raw: str, schema: Type[T]) -> T:
    """
    Parse a raw LLM response string into a validated Pydantic model.

    Raises
    ------
    json.JSONDecodeError  — if the JSON cannot be parsed even after cleaning.
    pydantic.ValidationError — if the parsed JSON doesn't match the schema.
    """
    clean = _clean_llm_json(raw)

    if not clean:
        raise ValueError("No JSON content found in your response (only reasoning/thinking blocks). You MUST output the JSON object after your thinking block.")

    try:
        return schema.model_validate(json.loads(clean, strict=False))
    except Exception as json_err:
        # Fallback: attempt to repair single-quote usage
        try:
            repaired = re.sub(r"'\s*:\s*", '": ', clean)
            repaired = re.sub(r"([{,]\s*)'", r'\1"', repaired)
            repaired = re.sub(r":\s*'(.*?)'\s*([,}])", r': "\1" \2', repaired)
            return schema.model_validate(json.loads(repaired, strict=False))
        except Exception:
            raise json_err
