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

from core.events import UnifiedIngestEvent, SenderClass, PriorityLevel
from perception.schemas import ScoredSignal
from utils.log_writer import emit

# Base scores per source type — keep in sync with spec §4
_BASE_SCORES: dict[SenderClass, float] = {
    SenderClass.OPERATOR:  1.00,
    SenderClass.TRUSTED:   0.75,
    SenderClass.EXTERNAL:  0.50, # Corresponds to old SourceType.USER
    # No direct mapping for ANONYMOUS or AGENT, use EXTERNAL for now or define new SenderClasses
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

    def score(self, event: UnifiedIngestEvent) -> ScoredSignal:
        """Compute authority score and return a ScoredSignal."""
        base = self._base_score(event.sender_class)
        rec_bonus = self._recency_bonus(event.monotonic_timestamp)
        final = min(1.0, base + rec_bonus)

        emit(
            event="perception.scored",
            layer="perception",
            level="debug",
            data={
                "event_identifier": event.event_identifier,
                "sender_class": event.sender_class.value,
                "base": base,
                "rec_bonus": rec_bonus,
                "score": final,
            }
        )

        return ScoredSignal(
            representative=event,
            frequency=1,
            perception_type=None,
            authority_score=final,
        )

    def score_batch(
        self, events: list[UnifiedIngestEvent]
    ) -> list[ScoredSignal]:
        return [self.score(e) for e in events]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _base_score(self, sender_class: SenderClass) -> float:
        return _BASE_SCORES.get(sender_class, _DEFAULT_BASE_SCORE)

    def _recency_bonus(self, event_monotonic_ts: float) -> float:
        """
        Give a small bonus to very recent signals.

        Both event.monotonic_timestamp and time.monotonic() use the same
        monotonic clock (no timezone, no epoch), so subtraction gives the
        true elapsed seconds since the event was created.
        """
        import time
        age_seconds = time.monotonic() - event_monotonic_ts
        if age_seconds < 0:
            age_seconds = 0.0

        if age_seconds <= _RECENCY_WINDOW_SECONDS:
            fraction = 1.0 - (age_seconds / _RECENCY_WINDOW_SECONDS)
            return _RECENCY_BONUS_MAX * fraction
        return 0.0
