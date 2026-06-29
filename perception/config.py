"""
perception/config.py
────────────────────
Pydantic-settings configuration for the entire perception layer.

All settings are read from environment variables with the PERCEPTION_ prefix,
or from the project's .env file (via pydantic-settings auto-load).

Example .env entries:
    PERCEPTION_INTAKE_WINDOW_SECONDS=3.0
    PERCEPTION_OPERATOR_IDS=user123,user456
    PERCEPTION_SYNTHESIZER_MODEL=claude-haiku
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PerceptionConfig(BaseSettings):
    # ── Intake buffer ──────────────────────────────────────────────────────────
    intake_window_seconds: float = 60.0
    intake_max_buffer_size: int = 100

    # ── Authority scoring ──────────────────────────────────────────────────────
    # Loaded from env — never elevatable by message content.
    operator_ids: list[str] = []
    trusted_ids: list[str] = []

    # ── Synthesizer ───────────────────────────────────────────────────────────
    synthesizer_model: str = "claude-haiku"
    synthesizer_timeout_seconds: float = 8.0
    synthesizer_max_tokens: int = 800

    # ── Reactive pool ──────────────────────────────────────────────────────────
    reactive_pool_size: int = 5
    reactive_executor_timeout_seconds: float = 15.0

    model_config = SettingsConfigDict(
        env_prefix="PERCEPTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("operator_ids", "trusted_ids", mode="before")
    @classmethod
    def _parse_csv(cls, v: object) -> list[str]:
        """
        Accept either a Python list or a comma-separated string from env.
        PERCEPTION_OPERATOR_IDS=user123,user456
        """
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v  # type: ignore[return-value]
