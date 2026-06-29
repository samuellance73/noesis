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

from datetime import datetime, timedelta, timezone

from perception.schemas import RawSignal, ScoredSignal, SourceType
from utils.log_writer import emit

# Base scores per source type — keep in sync with spec §4
_BASE_SCORES: dict[SourceType, float] = {
    SourceType.OPERATOR:  1.00,
    SourceType.TRUSTED:   0.75,
    SourceType.USER:      0.50,
    SourceType.ANONYMOUS: 0.20,
    SourceType.AGENT:     0.60,
}

_DEFAULT_BASE_SCORE = 0.30

# Frequency bonus cap and per-signal increment (removed - no dedup)
# _FREQ_BONUS_CAP = 0.20
# _FREQ_BONUS_PER_SIGNAL = 0.01

# Recency bonus: signals arriving within this many seconds get a small boost
_RECENCY_WINDOW_SECONDS = 30.0
_RECENCY_BONUS_MAX = 0.05


class AuthorityScorer:
    """
    Scores raw signals and returns ScoredSignal objects ready for the
    Synthesizer stage.

    Parameters
    ──────────
    operator_ids : list[str]
        Source identifiers that map to SourceType.OPERATOR. Loaded from env.
        Not used to override SourceType — just for informational logging.
    """

    def __init__(self, operator_ids: list[str] | None = None) -> None:
        self._operator_ids: frozenset[str] = frozenset(operator_ids or [])

    def score(self, signal: RawSignal) -> ScoredSignal:
        """Compute authority score and return a ScoredSignal."""
        base = self._base_score(signal.source.type)
        rec_bonus = self._recency_bonus(signal.timestamp)
        final = min(1.0, base + rec_bonus)

        emit(
            event="perception.scored",
            layer="perception",
            level="debug",
            data={
                "signal_id": signal.id,
                "source_type": signal.source.type.value,
                "base": base,
                "rec_bonus": rec_bonus,
                "score": final,
            }
        )

        return ScoredSignal(
            representative=signal,
            frequency=1,
            sources=[signal.source],
            perception_type=None,
            authority_score=final,
        )

    def score_batch(
        self, signals: list[RawSignal]
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
