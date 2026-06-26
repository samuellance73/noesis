"""
tests/test_model_router.py
──────────────────────────
Unit tests for ModelRouter configuration and functionality.

Covers:
  Config loading   — YAML deserialization, all tier/retry/enforcement fields
  Budget           — token counting, hard-limit rejection, soft-limit truncation
  Per-tier routing — correct upstream model string sent for NANO / STANDARD / STRONG
  Fallback chain   — primary fail → correct fallback model used, fallback_used flag
  Retry            — retryable vs non-retryable error classification
  Fallback eligib. — eligible vs non-eligible error classification
  assert_fits_budget utility

Run with:
    uv run pytest tests/test_model_router.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call

import httpx2

from core.model_router import (
    ModelTier,
    ModelRequest,
    ModelResponse,
    ModelRouterConfig,
    TierConfig,
    RetryConfig,
    EnforcementConfig,
    BudgetExceededError,
    ModelRouter,
    load_config,
    assert_fits_budget,
)
from integrations.llm.schemas import ChatCompletionResponse, Usage


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_router(transport=None) -> tuple[ModelRouter, MagicMock]:
    """Return (router, transport_mock) using the real YAML config."""
    config = load_config("config/model_router.yaml")
    if transport is None:
        transport = MagicMock()
    return ModelRouter(config, transport), transport


def _ok_response(content: str = "ok") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ── Config loading ────────────────────────────────────────────────────────────

class TestConfigLoading:
    """Test YAML configuration loading and validation."""

    def test_load_config_from_yaml(self):
        config = load_config("config/model_router.yaml")
        assert isinstance(config, ModelRouterConfig)
        assert ModelTier.NANO     in config.tiers
        assert ModelTier.STANDARD in config.tiers
        assert ModelTier.STRONG   in config.tiers

    def test_nano_tier_config(self):
        config = load_config("config/model_router.yaml")
        nano = config.tiers[ModelTier.NANO]
        assert isinstance(nano.primary, str)
        assert isinstance(nano.fallbacks, list)
        assert nano.context_budget     == 3000
        assert nano.max_response_tokens == 500
        assert nano.temperature        == 0.1
        assert nano.timeout_seconds    == 10.0

    def test_standard_tier_config(self):
        config = load_config("config/model_router.yaml")
        std = config.tiers[ModelTier.STANDARD]
        assert std.context_budget      == 8000
        assert std.max_response_tokens == 2000
        assert std.temperature         == 0.3
        assert std.timeout_seconds     == 30.0

    def test_strong_tier_config(self):
        config = load_config("config/model_router.yaml")
        strong = config.tiers[ModelTier.STRONG]
        assert strong.context_budget      == 64000
        assert strong.max_response_tokens == 4000
        assert strong.temperature         == 0.4
        assert strong.timeout_seconds     == 60.0

    def test_retry_config(self):
        config = load_config("config/model_router.yaml")
        assert config.retry.max_attempts         == 3
        assert config.retry.wait_min_seconds     == 1
        assert config.retry.wait_max_seconds     == 30
        assert config.retry.retryable_status_codes == [429, 500, 502, 503, 504]

    def test_enforcement_config(self):
        config = load_config("config/model_router.yaml")
        assert config.enforcement.hard_limit         is True
        assert config.enforcement.truncation_strategy == "tail"


# ── Budget enforcement ────────────────────────────────────────────────────────

class TestBudgetEnforcement:

    def test_count_tokens_simple(self):
        router, _ = _make_router()
        messages = [{"content": "hello world"}]   # 11 chars → 2 tokens
        assert router._count_tokens(messages, None) == 2

    def test_count_tokens_with_system(self):
        router, _ = _make_router()
        messages = [{"content": "hello"}]
        system = "system prompt"
        # 5 + 13 = 18 chars → 4 tokens
        assert router._count_tokens(messages, system) == 4

    def test_budget_not_exceeded(self):
        router, _ = _make_router()
        request = ModelRequest(
            tier=ModelTier.NANO,
            messages=[{"content": "short message"}],
        )
        tier_config = router.config.tiers[ModelTier.NANO]
        result = router._prepare_messages(request, tier_config)
        assert result == request.messages

    def test_budget_exceeded_hard_limit(self):
        router, _ = _make_router()
        long_content = "x" * 15000   # ~3750 tokens > NANO budget of 3000
        request = ModelRequest(
            tier=ModelTier.NANO,
            messages=[{"content": long_content}],
        )
        tier_config = router.config.tiers[ModelTier.NANO]
        with pytest.raises(BudgetExceededError) as exc_info:
            router._prepare_messages(request, tier_config)
        assert exc_info.value.tier   == ModelTier.NANO
        assert exc_info.value.budget == 3000

    def test_budget_exceeded_soft_limit_truncation(self):
        config = load_config("config/model_router.yaml")
        config.enforcement.hard_limit = False
        router = ModelRouter(config, MagicMock())

        messages = [
            {"content": "first message"},
            {"content": "x" * 15000},   # should be dropped
            {"content": "last message"},
        ]
        request = ModelRequest(tier=ModelTier.NANO, messages=messages)
        tier_config = config.tiers[ModelTier.NANO]

        result = router._prepare_messages(request, tier_config)
        assert len(result) < len(messages)
        assert router._count_tokens(result, None) <= tier_config.context_budget


# ── Per-tier upstream model routing ──────────────────────────────────────────

class TestPerTierUpstreamModel:
    """
    These are the key tests you asked about — verifying that the correct upstream
    model string from the YAML is actually sent to transport.chat_completion for
    each tier, not just that the router holds some model name.
    """

    @pytest.mark.asyncio
    async def test_nano_tier_sends_correct_model_to_upstream(self):
        """transport.chat_completion receives the NANO primary model string."""
        config = load_config("config/model_router.yaml")
        expected_model = config.tiers[ModelTier.NANO].primary

        transport = AsyncMock()
        transport.chat_completion = AsyncMock(return_value=_ok_response())
        router = ModelRouter(config, transport)

        request = ModelRequest(
            tier=ModelTier.NANO,
            messages=[{"role": "user", "content": "hi"}],
            component="test.nano",
        )
        response = await router.complete(request)

        transport.chat_completion.assert_called_once()
        call_kwargs = transport.chat_completion.call_args
        assert call_kwargs.kwargs["model"] == expected_model
        assert response.model_used == expected_model
        assert response.tier == ModelTier.NANO

    @pytest.mark.asyncio
    async def test_standard_tier_sends_correct_model_to_upstream(self):
        """transport.chat_completion receives the STANDARD primary model string."""
        config = load_config("config/model_router.yaml")
        expected_model = config.tiers[ModelTier.STANDARD].primary

        transport = AsyncMock()
        transport.chat_completion = AsyncMock(return_value=_ok_response())
        router = ModelRouter(config, transport)

        request = ModelRequest(
            tier=ModelTier.STANDARD,
            messages=[{"role": "user", "content": "hi"}],
            component="test.standard",
        )
        response = await router.complete(request)

        call_kwargs = transport.chat_completion.call_args
        assert call_kwargs.kwargs["model"] == expected_model
        assert response.model_used == expected_model
        assert response.tier == ModelTier.STANDARD

    @pytest.mark.asyncio
    async def test_strong_tier_sends_correct_model_to_upstream(self):
        """transport.chat_completion receives the STRONG primary model string."""
        config = load_config("config/model_router.yaml")
        expected_model = config.tiers[ModelTier.STRONG].primary

        transport = AsyncMock()
        transport.chat_completion = AsyncMock(return_value=_ok_response())
        router = ModelRouter(config, transport)

        request = ModelRequest(
            tier=ModelTier.STRONG,
            messages=[{"role": "user", "content": "hi"}],
            component="test.strong",
        )
        response = await router.complete(request)

        call_kwargs = transport.chat_completion.call_args
        assert call_kwargs.kwargs["model"] == expected_model
        assert response.model_used == expected_model
        assert response.tier == ModelTier.STRONG

    @pytest.mark.asyncio
    async def test_nano_fallback_model_used_when_primary_fails(self):
        """When NANO primary returns a rate-limit error, the first fallback model is used."""
        config = load_config("config/model_router.yaml")
        primary  = config.tiers[ModelTier.NANO].primary
        fallback = config.tiers[ModelTier.NANO].fallbacks[0]

        # Disable retry backoff delays so the test is fast
        config.retry.max_attempts    = 1
        config.retry.wait_min_seconds = 0
        config.retry.wait_max_seconds = 0

        call_order: list[str] = []

        async def mock_chat_completion(model: str, **kwargs):
            call_order.append(model)
            if model == primary:
                raise httpx2.HTTPStatusError(
                    "429",
                    request=MagicMock(),
                    response=MagicMock(status_code=429),
                )
            return _ok_response("fallback worked")

        transport = AsyncMock()
        transport.chat_completion = mock_chat_completion
        router = ModelRouter(config, transport)

        request = ModelRequest(
            tier=ModelTier.NANO,
            messages=[{"role": "user", "content": "hi"}],
        )
        response = await router.complete(request)

        assert response.content == "fallback worked"
        assert response.model_used == fallback
        assert response.fallback_used is True
        assert response.fallback_reason is not None
        assert primary  in call_order
        assert fallback in call_order

    @pytest.mark.asyncio
    async def test_standard_fallback_model_used_when_primary_fails(self):
        """When STANDARD primary times out, the first fallback model is used."""
        config = load_config("config/model_router.yaml")
        primary  = config.tiers[ModelTier.STANDARD].primary
        fallback = config.tiers[ModelTier.STANDARD].fallbacks[0]

        config.retry.max_attempts    = 1
        config.retry.wait_min_seconds = 0
        config.retry.wait_max_seconds = 0

        async def mock_chat_completion(model: str, **kwargs):
            if model == primary:
                raise TimeoutError("upstream timeout")
            return _ok_response("standard fallback")

        transport = AsyncMock()
        transport.chat_completion = mock_chat_completion
        router = ModelRouter(config, transport)

        request = ModelRequest(
            tier=ModelTier.STANDARD,
            messages=[{"role": "user", "content": "query"}],
        )
        response = await router.complete(request)

        assert response.content == "standard fallback"
        assert response.model_used == fallback
        assert response.fallback_used is True

    @pytest.mark.asyncio
    async def test_strong_fallback_model_used_when_primary_fails(self):
        """When STRONG primary returns 503, the first fallback model is used."""
        config = load_config("config/model_router.yaml")
        primary  = config.tiers[ModelTier.STRONG].primary
        fallback = config.tiers[ModelTier.STRONG].fallbacks[0]

        config.retry.max_attempts    = 1
        config.retry.wait_min_seconds = 0
        config.retry.wait_max_seconds = 0

        async def mock_chat_completion(model: str, **kwargs):
            if model == primary:
                raise httpx2.HTTPStatusError(
                    "503",
                    request=MagicMock(),
                    response=MagicMock(status_code=503),
                )
            return _ok_response("strong fallback")

        transport = AsyncMock()
        transport.chat_completion = mock_chat_completion
        router = ModelRouter(config, transport)

        request = ModelRequest(
            tier=ModelTier.STRONG,
            messages=[{"role": "user", "content": "analyse this"}],
        )
        response = await router.complete(request)

        assert response.content == "strong fallback"
        assert response.model_used == fallback
        assert response.fallback_used is True

    def test_resolve_model_returns_primary_per_tier(self):
        """resolve_model returns the exact primary string from config for each tier."""
        config = load_config("config/model_router.yaml")
        router = ModelRouter(config, MagicMock())

        for tier in ModelTier:
            assert router.resolve_model(tier) == config.tiers[tier].primary

    def test_budget_for_matches_config(self):
        """budget_for returns the correct context_budget for each tier."""
        config = load_config("config/model_router.yaml")
        router = ModelRouter(config, MagicMock())

        assert router.budget_for(ModelTier.NANO)     == 3000
        assert router.budget_for(ModelTier.STANDARD) == 8000
        assert router.budget_for(ModelTier.STRONG)   == 64000


# ── Full round-trip (primary success) ────────────────────────────────────────

class TestModelRouterCompletion:

    @pytest.mark.asyncio
    async def test_complete_success_response_fields(self):
        """Successful complete() populates all ModelResponse fields correctly."""
        config = load_config("config/model_router.yaml")
        transport = AsyncMock()
        transport.chat_completion = AsyncMock(
            return_value=ChatCompletionResponse(
                content="Test response",
                usage=Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            )
        )
        router = ModelRouter(config, transport)

        request = ModelRequest(
            tier=ModelTier.NANO,
            messages=[{"role": "user", "content": "test"}],
            component="TestComponent.test",
        )
        response = await router.complete(request)

        assert response.content          == "Test response"
        assert response.model_used       == config.tiers[ModelTier.NANO].primary
        assert response.tier             == ModelTier.NANO
        assert response.prompt_tokens    == 100
        assert response.completion_tokens == 50
        assert response.total_tokens     == 150
        assert response.fallback_used    is False
        assert response.fallback_reason  is None


# ── Fallback eligibility ──────────────────────────────────────────────────────

class TestFallbackEligibility:

    def test_fallback_eligible_rate_limit(self):
        router, _ = _make_router()
        exc = httpx2.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock(status_code=429)
        )
        assert router._is_fallback_eligible(exc) is True

    def test_fallback_eligible_timeout(self):
        router, _ = _make_router()
        assert router._is_fallback_eligible(TimeoutError()) is True

    def test_fallback_not_eligible_auth_error(self):
        router, _ = _make_router()
        exc = httpx2.HTTPStatusError(
            "401", request=MagicMock(), response=MagicMock(status_code=401)
        )
        assert router._is_fallback_eligible(exc) is False

    def test_fallback_not_eligible_bad_request(self):
        router, _ = _make_router()
        exc = httpx2.HTTPStatusError(
            "400", request=MagicMock(), response=MagicMock(status_code=400)
        )
        assert router._is_fallback_eligible(exc) is False


# ── Retry eligibility ─────────────────────────────────────────────────────────

class TestRetryEligibility:

    def test_retry_eligible_rate_limit(self):
        router, _ = _make_router()
        exc = httpx2.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock(status_code=429)
        )
        assert router._is_retryable(exc) is True

    def test_retry_eligible_server_error(self):
        router, _ = _make_router()
        exc = httpx2.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(status_code=500)
        )
        assert router._is_retryable(exc) is True

    def test_retry_not_eligible_client_error(self):
        router, _ = _make_router()
        exc = httpx2.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404)
        )
        assert router._is_retryable(exc) is False


# ── assert_fits_budget utility ────────────────────────────────────────────────

class TestAssertFitsBudget:

    def test_assert_fits_budget_pass(self):
        router, _ = _make_router()
        # Should not raise
        assert_fits_budget("short content", ModelTier.NANO, router)

    def test_assert_fits_budget_exceeds(self):
        router, _ = _make_router()
        content = "x" * 15000   # ~3750 tokens, exceeds NANO budget of 3000
        with pytest.raises(BudgetExceededError):
            assert_fits_budget(content, ModelTier.NANO, router)
