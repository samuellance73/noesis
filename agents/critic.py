from pydantic import BaseModel, Field

from core.model_router import ModelRouter, ModelRequest, ModelTier
from utils.json_parser import parse_llm_json
from utils.log_writer import emit


class CriticScore(BaseModel):
    score: float = Field(..., description="The quality score of the answer from 0.0 to 1.0.")
    reason: str = Field(..., description="Brief explanation for the score.")


async def score_result(
    router: ModelRouter,
    task_goal: str,
    result: str,
) -> float:
    """Evaluate subtask results using the LLM critic (NANO tier) and return a score 0.0–1.0."""
    prompt = f"""Evaluate the following subtask result and provide a quality score between 0.0 (completely wrong, failed, or empty) and 1.0 (correct, complete, and high-quality).

Task Goal:
{task_goal}

Result to Evaluate:
{result}

Respond with a single valid JSON object containing 'score' (float) and 'reason' (string) fields.
Example:
{{
  "score": 0.85,
  "reason": "The answer provides all requested details but lacks formatting."
}}
"""

    request = ModelRequest(
        tier=ModelTier.NANO,
        messages=[
            {"role": "system", "content": "You are a rigorous quality assurance critic evaluating the outputs of AI agent executions."},
            {"role": "user",   "content": prompt},
        ],
        component="Critic.score_result",
    )

    try:
        raw_response = await router.complete(request)
        content = raw_response.content

        if not content:
            emit("critic.error", "critic", {"msg": "Critic LLM returned empty content."}, level="error")
            return 0.5

        parsed = parse_llm_json(content, CriticScore)
        score = max(0.0, min(1.0, parsed.score))
        emit(
            event="critic.score",
            layer="critic",
            data={
                "task_goal": task_goal,
                "score": score,
                "reason": parsed.reason,
            }
        )
        return score
    except Exception as e:
        emit("critic.error", "critic", {"msg": f"Critic evaluation failed for task {task_goal!r}: {e}"}, level="error")
        return 0.5  # default neutral score on failure
