import logging
from typing import Dict, Any
from integrations.llm.service import UpstreamService
from integrations.llm.schemas import ChatMessage, ChatPayload
from utils.json_parser import parse_llm_json
from pydantic import BaseModel, Field

logger = logging.getLogger("noesis.critic")

class CriticScore(BaseModel):
    score: float = Field(..., description="The quality score of the answer from 0.0 to 1.0.")
    reason: str = Field(..., description="Brief explanation for the score.")

async def score_result(
    llm_service: UpstreamService,
    model: str,
    task_goal: str,
    result: str,
) -> float:
    """Evaluate subtask results using the LLM critic and return a score between 0.0 and 1.0."""
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

    payload = ChatPayload(
        model=model,
        messages=[
            ChatMessage(role="system", content="You are a rigorous quality assurance critic evaluating the outputs of AI agent executions."),
            ChatMessage(role="user", content=prompt),
        ],
        temperature=0.1,
        stream=False,
    )

    try:
        raw_response = await llm_service.get_chat_completion(
            payload.model_dump(exclude_none=True)
        )
        content = raw_response["choices"][0]["message"].get("content") or ""
        
        if not content:
            logger.error("Critic LLM returned empty content.")
            return 0.5

        parsed = parse_llm_json(content, CriticScore)
        score = max(0.0, min(1.0, parsed.score))
        logger.info("Critic score for task %r: %.2f (Reason: %s)", task_goal, score, parsed.reason)
        return score
    except Exception as e:
        logger.error("Critic evaluation failed for task %r: %s", task_goal, e)
        return 0.5  # default neutral score on failure
