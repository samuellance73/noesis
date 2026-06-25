"""
perception/stages/authority.py
───────────────────────────────
Stage 4 — Authority Scorer

Assigns a 0.0–1.0 authority score to each signal based on its source provenance.
Does NOT evaluate content — only origin metadata.

Scoring components
──────────────────
  base_score      : fixed per SourceType (operator=1.0, trusted=0.75, ...)
  frequency_bonus : min(0.20, frequency × 0.01)  — more people = slightly higher
  recency_bonus   : up to 0.05 for signals arriving in the last 30 seconds

Operator IDs are supplied at construction time from PerceptionConfig (loaded
from env).  A message claiming to be from an operator is NOT an operator message
— SourceType determines the base score, not text content.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from perception.schemas import DeduplicatedSignal, ScoredSignal, SourceType

logger = logging.getLogger("noesis.perception")

# Base scores per source type — keep in sync with spec §4
_BASE_SCORES: dict[SourceType, float] = {
    SourceType.OPERATOR:  1.00,
    SourceType.TRUSTED:   0.75,
    SourceType.USER:      0.50,
    SourceType.ANONYMOUS: 0.20,
    SourceType.AGENT:     0.60,
}

_DEFAULT_BASE_SCORE = 0.30

# Frequency bonus cap and per-signal increment
_FREQ_BONUS_CAP = 0.20
_FREQ_BONUS_PER_SIGNAL = 0.01

# Recency bonus: signals arriving within this many seconds get a small boost
_RECENCY_WINDOW_SECONDS = 30.0
_RECENCY_BONUS_MAX = 0.05


class AuthorityScorer:
    """
    Scores deduplicated signals and returns ScoredSignal objects ready for the
    Synthesizer stage.

    Parameters
    ──────────
    operator_ids : list[str]
        Source identifiers that map to SourceType.OPERATOR. Loaded from env.
        Not used to override SourceType — just for informational logging.
    """

    def __init__(self, operator_ids: list[str] | None = None) -> None:
        self._operator_ids: frozenset[str] = frozenset(operator_ids or [])

    def score(self, signal: DeduplicatedSignal) -> ScoredSignal:
        """Compute authority score and return a ScoredSignal."""
        if signal.perception_type is None:
            raise ValueError(
                "AuthorityScorer received a signal without a perception_type. "
                "Ensure the Classifier runs before the AuthorityScorer."
            )

        base = self._base_score(signal.representative.source.type)
        freq_bonus = min(_FREQ_BONUS_CAP, signal.frequency * _FREQ_BONUS_PER_SIGNAL)
        rec_bonus = self._recency_bonus(signal.representative.timestamp)
        final = min(1.0, base + freq_bonus + rec_bonus)

        logger.debug(
            "AuthorityScorer: signal id=%s  source_type=%s  "
            "base=%.2f  freq_bonus=%.2f  rec_bonus=%.2f  → score=%.2f",
            signal.representative.id,
            signal.representative.source.type.value,
            base, freq_bonus, rec_bonus, final,
        )

        return ScoredSignal(
            representative=signal.representative,
            frequency=signal.frequency,
            sources=signal.sources,
            perception_type=signal.perception_type,
            authority_score=final,
        )

    def score_batch(
        self, signals: list[DeduplicatedSignal]
    ) -> list[ScoredSignal]:
        return [self.score(s) for s in signals]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _base_score(self, source_type: SourceType) -> float:
        return _BASE_SCORES.get(source_type, _DEFAULT_BASE_SCORE)

    def _recency_bonus(self, timestamp: datetime) -> float:
        """
        Give a small bonus to very recent signals.  Uses UTC-aware comparison
        when the timestamp is timezone-aware, falls back to naive UTC otherwise.
        """
        now = datetime.now(timezone.utc)
        # Make naive timestamps comparable by assuming UTC
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        age_seconds = (now - ts).total_seconds()
        if age_seconds < 0:
            age_seconds = 0.0

        if age_seconds <= _RECENCY_WINDOW_SECONDS:
            # Linear decay: newest → full bonus, at window boundary → 0
            fraction = 1.0 - (age_seconds / _RECENCY_WINDOW_SECONDS)
            return _RECENCY_BONUS_MAX * fraction
        return 0.0
