"""
core/model_router.py
────────────────────
ModelRouter is the single authority over which model is used for any LLM call
in the system. No agent, executor, or utility may reference a model name directly.
All LLM calls go through the router, which resolves a logical tier to a concrete
model, enforces token budgets, and handles fallback chains transparently.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Literal

import yaml
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from utils.log_writer import emit


# ── Tier enum ──────────────────────────────────────────────────────────────

class ModelTier(str, Enum):
    NANO = "nano"
    STANDARD = "standard"
    STRONG = "strong"


# ── Config models ──────────────────────────────────────────────────────────

class TierConfig(BaseModel):
    primary: str
    fallbacks: list[str]
    context_budget: int
    max_response_tokens: int
    temperature: float
    timeout_seconds: float

class RetryConfig(BaseModel):
    max_attempts: int = 3
    wait_min_seconds: float = 1.0
    wait_max_seconds: float = 30.0
    retryable_status_codes: list[int] = [429, 500, 502, 503, 504]

class EnforcementConfig(BaseModel):
    hard_limit: bool = True
    truncation_strategy: Literal["tail", "head"] = "tail"

class ModelRouterConfig(BaseModel):
    tiers: dict[ModelTier, TierConfig]
    retry: RetryConfig
    enforcement: EnforcementConfig


# ── Request / Response ─────────────────────────────────────────────────────

class ModelRequest(BaseModel):
    tier: ModelTier
    messages: list[dict]  # standard chat format [{role, content}]
    system: str | None = None
    stream: bool = False
    # callers may override temperature per-call if needed
    temperature_override: float | None = None
    # optional metadata for observability
    component: str | None = None

class ModelResponse(BaseModel):
    content: str
    model_used: str  # which model actually answered
    tier: ModelTier
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    fallback_used: bool = False  # was a fallback model used?
    fallback_reason: str | None = None
    latency_ms: float


# ── Exceptions ─────────────────────────────────────────────────────────────

class BudgetExceededError(Exception):
    def __init__(self, tier: ModelTier, token_count: int, budget: int):
        self.tier = tier
        self.token_count = token_count
        self.budget = budget
        super().__init__(
            f"Prompt token count {token_count} exceeds {tier} budget {budget}"
        )


# ── ModelRouter Implementation ───────────────────────────────────────────────

class ModelRouter:
    def __init__(self, config: ModelRouterConfig, transport: "UpstreamService"):
        self.config = config
        self.transport = transport

    # ── Public API ────────────────────────────────────────────────────────

    async def complete(self, request: ModelRequest) -> ModelResponse:
        tier_config = self.config.tiers[request.tier]
        messages = self._prepare_messages(request, tier_config)
        
        emit(
            event="llm.request",
            layer="llm",
            level="debug",
            data={
                "tier": request.tier.value,
                "model": tier_config.primary,
                "messages": messages,
                "system": request.system,
                "component": request.component,
            }
        )
        
        return await self._complete_with_fallback(request, messages, tier_config)

    def resolve_model(self, tier: ModelTier) -> str:
        """Returns the primary model name for a tier. Use for logging only."""
        return self.config.tiers[tier].primary

    def budget_for(self, tier: ModelTier) -> int:
        return self.config.tiers[tier].context_budget

    # ── Budget enforcement ────────────────────────────────────────────────

    def _prepare_messages(
        self,
        request: ModelRequest,
        tier_config: TierConfig
    ) -> list[dict]:
        token_count = self._count_tokens(request.messages, request.system)
        budget = tier_config.context_budget

        if token_count <= budget:
            return request.messages

        if self.config.enforcement.hard_limit:
            raise BudgetExceededError(request.tier, token_count, budget)

        # soft limit: truncate to fit
        return self._truncate(
            request.messages,
            request.system,
            budget,
            self.config.enforcement.truncation_strategy
        )

    def _count_tokens(self, messages: list[dict], system: str | None) -> int:
        # approximate: 4 chars per token
        # replace with tiktoken or model-specific counter if available
        total = sum(len(m.get("content", "")) for m in messages)
        if system:
            total += len(system)
        return total // 4

    def _truncate(
        self,
        messages: list[dict],
        system: str | None,
        budget: int,
        strategy: str
    ) -> list[dict]:
        # always preserve system message and the most recent user message
        # "tail" drops oldest non-system messages first
        # "head" drops newest messages first (rare, but useful for some flows)
        if strategy == "tail":
            result = []
            token_remaining = budget
            if system:
                token_remaining -= len(system) // 4
            # walk from newest to oldest, keep until budget exhausted
            for msg in reversed(messages):
                cost = len(msg.get("content", "")) // 4
                if token_remaining - cost >= 0:
                    result.insert(0, msg)
                    token_remaining -= cost
            return result
        else:
            raise NotImplementedError(f"Truncation strategy '{strategy}' not implemented")

    # ── Fallback chain ────────────────────────────────────────────────────

    async def _complete_with_fallback(
        self,
        request: ModelRequest,
        messages: list[dict],
        tier_config: TierConfig
    ) -> ModelResponse:
        models_to_try = [tier_config.primary] + tier_config.fallbacks
        last_exception = None

        for i, model in enumerate(models_to_try):
            fallback_used = i > 0
            fallback_reason = str(last_exception) if fallback_used else None

            if fallback_used:
                emit(
                    event="transport.fallback",
                    layer="transport",
                    level="warn",
                    data={
                        "model": model,
                        "tier": request.tier.value,
                        "reason": fallback_reason,
                    }
                )

            try:
                return await self._call_model(
                    model=model,
                    request=request,
                    messages=messages,
                    tier_config=tier_config,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                )
            except Exception as e:
                last_exception = e
                if not self._is_fallback_eligible(e):
                    # non-retryable error (auth, malformed request) — don't try fallbacks
                    raise
                continue

        raise last_exception

    def _is_fallback_eligible(self, exc: Exception) -> bool:
        # fall back on rate limits, capacity errors, timeouts
        # do NOT fall back on auth errors or malformed request errors
        if hasattr(exc, "response") and exc.response is not None:
            return exc.response.status_code in {429, 413, 500, 502, 503, 504}
        return isinstance(exc, (TimeoutError, ConnectionError))

    # ── Model call with retry ─────────────────────────────────────────────

    async def _call_model(
        self,
        model: str,
        request: ModelRequest,
        messages: list[dict],
        tier_config: TierConfig,
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> ModelResponse:
        retry_config = self.config.retry
        start = time.perf_counter()

        @retry(
            retry=retry_if_exception(self._is_retryable),
            stop=stop_after_attempt(retry_config.max_attempts),
            wait=wait_exponential(
                multiplier=1,
                min=retry_config.wait_min_seconds,
                max=retry_config.wait_max_seconds
            ),
            reraise=True,
        )
        async def _attempt():
            return await self.transport.chat_completion(
                model=model,
                messages=messages,
                system=request.system,
                max_tokens=tier_config.max_response_tokens,
                temperature=(
                    request.temperature_override
                    if request.temperature_override is not None
                    else tier_config.temperature
                ),
                timeout=tier_config.timeout_seconds,
            )

        raw = await _attempt()
        latency_ms = (time.perf_counter() - start) * 1000

        response = ModelResponse(
            content=raw.content,
            model_used=model,
            tier=request.tier,
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            total_tokens=raw.usage.total_tokens,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            latency_ms=latency_ms,
        )

        # Emit structured log entry
        self._log_call(response, tier_config, request.component)

        return response

    def _is_retryable(self, exc: Exception) -> bool:
        if hasattr(exc, "response") and exc.response is not None:
            return exc.response.status_code in set(
                self.config.retry.retryable_status_codes
            )
        return isinstance(exc, (TimeoutError, ConnectionError))

    # ── Observability ───────────────────────────────────────────────────────

    def _log_call(
        self,
        response: ModelResponse,
        tier_config: TierConfig,
        component: str | None
    ) -> None:
        """Emit structured log entry for observability."""
        emit(
            event="llm.response",
            layer="llm",
            level="debug",
            data={
                "tier": response.tier.value,
                "model_requested": self.config.tiers[response.tier].primary,
                "model_used": response.model_used,
                "fallback_used": response.fallback_used,
                "fallback_reason": response.fallback_reason,
                "usage": {
                    "prompt": response.prompt_tokens,
                    "completion": response.completion_tokens,
                    "total": response.total_tokens,
                },
                "budget": tier_config.context_budget,
                "budget_utilization": response.total_tokens / tier_config.context_budget,
                "elapsed_ms": response.latency_ms,
                "component": component or "unknown",
                "status": "success",
            }
        )


# ── Prompt Budget Guard ─────────────────────────────────────────────────────

def assert_fits_budget(
    content: str,
    tier: ModelTier,
    router: ModelRouter,
    label: str = "prompt"
) -> None:
    """Utility function that components call before building a prompt.
    Prevents wasted work building a large prompt that the router will reject.
    """
    token_estimate = len(content) // 4
    budget = router.budget_for(tier)
    if token_estimate > budget:
        raise BudgetExceededError(tier, token_estimate, budget)
    if token_estimate > budget * 0.85:
        emit(
            event="transport.budget_pressure",
            layer="transport",
            level="warn",
            data={
                "label": label,
                "token_estimate": token_estimate,
                "budget": budget,
                "tier": tier.value,
            }
        )


# ── Config Loading ─────────────────────────────────────────────────────────

def load_config(config_path: str = "config/model_router.yaml") -> ModelRouterConfig:
    """Load ModelRouter configuration from YAML file."""
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    # Convert tier keys to ModelTier enums
    tiers_dict = {}
    for tier_name, tier_data in data["tiers"].items():
        tier_enum = ModelTier(tier_name)
        tiers_dict[tier_enum] = TierConfig(**tier_data)
    
    return ModelRouterConfig(
        tiers=tiers_dict,
        retry=RetryConfig(**data["retry"]),
        enforcement=EnforcementConfig(**data["enforcement"]),
    )
