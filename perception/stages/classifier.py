"""
perception/stages/classifier.py
─────────────────────────────────
Stage 3 — Classifier

Labels each deduplicated signal with a PerceptionType. Determines downstream
routing without requiring an LLM call — uses keyword patterns and source metadata.

Noise patterns include:
  • bot commands not addressed to this agent (e.g. "!command", "/slash")
  • messages shorter than 3 tokens
  • pure emoji / reaction strings
  • messages from known-bot SourceType.AGENT sources that are purely status ticks

Correction detection:
  Checks if the message references a prior agent output ("you said", "that's
  wrong", "actually", "incorrect") AND contains a negation/contradiction.

Directive vs Query vs Information heuristics:
  • Directive   — imperative verb at start OR explicit request verbs ("please do",
                  "can you", "I need you to", "make sure", "update", "create",
                  "delete", "run", "send", "fix").
  • Query       — interrogative structure: starts with wh-word or contains "?".
  • Information — default for declarative sentences that don't fit above.
"""

from __future__ import annotations

import logging
import re

from perception.schemas import DeduplicatedSignal, PerceptionType, SourceType

logger = logging.getLogger("noesis.perception")

# ── Pattern banks ─────────────────────────────────────────────────────────────

_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^[!/][\w]+"),              # bot commands (!/slash)
    re.compile(r"^[\U00010000-\U0010ffff\U00002600-\U000027BF\s]+$"),  # pure emoji
    re.compile(r"^[\W\d\s]*$"),             # only punctuation / numbers
]

_CORRECTION_TRIGGERS: list[str] = [
    "you said", "you told", "you mentioned", "you stated",
    "that's wrong", "thats wrong", "that is wrong",
    "you're wrong", "youre wrong", "you are wrong",
    "actually,", "actually.", "actually ",
    "incorrect", "that's incorrect", "not quite",
    "you got it wrong", "wrong answer", "that's not right",
    "no, ", "nope", "not true", "false",
]

_DIRECTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:please\s+)?(?:can you|could you|would you|i(?:'d| would) like you to)\b", re.I),
    re.compile(r"^(?:make sure|ensure|don'?t forget|remember to)\b", re.I),
    re.compile(
        r"^(?:create|update|delete|remove|add|send|run|execute|deploy|"
        r"fix|check|fetch|get|post|publish|push|pull|generate|write|build|"
        r"start|stop|restart|enable|disable|install|uninstall)\b",
        re.I,
    ),
    re.compile(r"\b(?:please\s+(?:do|make|create|update|send|run|fix|check))\b", re.I),
]

_QUERY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:what|who|where|when|why|how|which|is|are|do|does|did|can|could|would|should|will)\b", re.I),
    re.compile(r"\?$"),
    re.compile(r"\?\s*$"),
]

_FEEDBACK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:good job|great job|well done|nice work|perfect|excellent|"
               r"love it|loved it|awesome|amazing|thank(?:s| you)|👍|⭐)\b", re.I),
    re.compile(r"\b(?:bad job|terrible|awful|horrible|useless|disappointing|"
               r"failed|waste|garbage|trash|👎)\b", re.I),
    re.compile(r"\b(?:rating|score|feedback|review|rate)\b", re.I),
]

_MIN_TOKEN_COUNT = 3


class Classifier:
    """
    Pure rule-based signal classifier.  No model calls — fast enough to run
    synchronously within the pipeline loop.
    """

    def classify(self, signal: DeduplicatedSignal) -> PerceptionType:
        text = signal.representative.text
        lower = text.lower().strip()

        # ── 0. Noise: structural checks first ─────────────────────────────────
        if self._is_noise(lower, signal):
            return PerceptionType.NOISE

        # ── 1. Correction: must be identified before directive/query ──────────
        if self._is_correction(lower, signal):
            return PerceptionType.CORRECTION

        # ── 2. Feedback: evaluative statements ────────────────────────────────
        if self._is_feedback(lower):
            return PerceptionType.FEEDBACK

        # ── 3. Directive: action request ──────────────────────────────────────
        if self._is_directive(lower):
            return PerceptionType.DIRECTIVE

        # ── 4. Query: question ────────────────────────────────────────────────
        if self._is_query(lower):
            return PerceptionType.QUERY

        # ── 5. Default: treat as informational context ────────────────────────
        return PerceptionType.INFORMATION

    # ── Pattern matchers ──────────────────────────────────────────────────────

    def _is_noise(self, lower: str, signal: DeduplicatedSignal) -> bool:
        # Too short
        token_count = len(re.findall(r"\b\w+\b", lower))
        if token_count < _MIN_TOKEN_COUNT:
            return True

        # Matches explicit noise patterns
        for pattern in _NOISE_PATTERNS:
            if pattern.match(lower):
                return True

        return False

    def _is_correction(self, lower: str, signal: DeduplicatedSignal) -> bool:
        for trigger in _CORRECTION_TRIGGERS:
            if trigger in lower:
                return True
        return False

    def _is_feedback(self, lower: str) -> bool:
        for pattern in _FEEDBACK_PATTERNS:
            if pattern.search(lower):
                return True
        return False

    def _is_directive(self, lower: str) -> bool:
        for pattern in _DIRECTIVE_PATTERNS:
            if pattern.search(lower):
                return True
        return False

    def _is_query(self, lower: str) -> bool:
        for pattern in _QUERY_PATTERNS:
            if pattern.search(lower):
                return True
        return False
