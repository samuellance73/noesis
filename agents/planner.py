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

    raw = response["choices"][0]["message"]["content"]
    start = raw.index("[")
    end = raw.rindex("]") + 1
    steps = json.loads(raw[start:end])
    return steps
