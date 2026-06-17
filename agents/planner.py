import json
from integrations.llm.service import UpstreamService

PLANNER_PROMPT = """
You are a planning agent. Given a user goal, break it into concrete steps.

Rules:
- Each step must be a specific, self-contained sub-goal
- Steps with no dependencies can run in parallel
- Steps that need prior results use depends_on
- Maximum 5 steps
- Respond ONLY with valid JSON, nothing else

Format:
[
  {"id": 1, "goal": "...", "depends_on": []},
  {"id": 2, "goal": "...", "depends_on": []},
  {"id": 3, "goal": "...", "depends_on": [1, 2]}
]
"""

async def plan(goal: str, service: UpstreamService) -> list[dict]:
    payload = {
        "model": "groq/openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": goal}
        ],
        "temperature": 0.1,
        "stream": False
    }
    response = await service.get_chat_completion(payload)

    raw = response["choices"][0]["message"]["content"].strip()

    # 1. Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    # 2. Try to extract a JSON array first, then fall back to an object
    if "[" in raw:
        try:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            steps = json.loads(raw[start:end])
            if isinstance(steps, list):
                return steps
        except (ValueError, json.JSONDecodeError):
            pass

    # 3. Try parsing the whole thing as JSON (model may return {"steps": [...]})
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("steps", "plan", "goals"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Planner returned unparseable response: {raw[:200]}")
