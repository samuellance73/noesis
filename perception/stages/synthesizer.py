"""
perception/stages/synthesizer.py
──────────────────────────────────
Stage 5 — Synthesizer

The ONLY stage that calls an LLM.  Takes the classified, scored batch of signals
and produces a list of PerceptionEvent objects — one per semantically distinct
concern.

Single-signal optimisation
───────────────────────────
If the batch contains exactly one signal after dedup+classification, the LLM is
skipped entirely and a trivial PerceptionEvent is constructed directly.

Fallback strategy (§7)
───────────────────────
  1. LLM timeout (> synthesizer_timeout_seconds)
     → Each ScoredSignal becomes its own PerceptionEvent without merging.
  2. LLM returns invalid JSON → json_parser repair → if still invalid,
     fall back to trivial events.
  3. Any other exception → fall back to trivial events + log error.

Prompt contract
───────────────
  Temperature : 0.1  (determinism preferred)
  Max tokens  : 800  (events are compact)
  Model       : synthesizer_model from config (lightest available)
  Response    : JSON array of PerceptionEvent-shaped dicts. No preamble.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from perception.schemas import PerceptionEvent, PerceptionType, ScoredSignal
from core.model_router import ModelRouter, ModelRequest, ModelTier
from utils.json_parser import _clean_llm_json  # reuse existing noise stripper

logger = logging.getLogger("noesis.perception")

# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the perception synthesizer for an autonomous agent system.

Given the following batch of incoming signals (already deduplicated and
classified), produce a list of PerceptionEvents. Each event represents one
semantically distinct concern. Merge signals that address the same topic.
Do not split a single concern into multiple events.

Return ONLY a JSON array of PerceptionEvent objects. No preamble. No markdown.
No explanations. The array must be parseable by json.loads() directly.

PerceptionEvent schema:
{
  "summary": "<one sentence describing the concern>",
  "type": "<directive|query|information|correction|feedback|noise>",
  "urgency": <float 0.0-1.0>,
  "authority_score": <float 0.0-1.0, weighted average of contributing signals>,
  "requires_immediate_response": <true|false>,
  "affects_objectives": <true|false>,
  "frequency": <int, total raw signals that contributed>,
  "source_ids": ["<uuid string>", ...],
  "response_context": "<string or null, only if requires_immediate_response>"
}
"""

_USER_PROMPT_TEMPLATE = """\
Current WorldModel summary:
{world_model_summary}

Signals (JSON):
{signals_json}

Produce the PerceptionEvent array now.
"""


class Synthesizer:
    """
    Converts a batch of ScoredSignals into PerceptionEvents via a single LLM
    call (or trivial construction for single-signal batches).

    Parameters
    ──────────
    router    : ModelRouter — the shared router; Synthesizer uses NANO tier.
    timeout   : float — seconds before fallback is triggered.
    max_tokens: int — output token cap (passed as temperature override).
    """

    def __init__(
        self,
        router: ModelRouter,
        timeout: float = 8.0,
        max_tokens: int = 800,
    ) -> None:
        self.router     = router
        self.timeout    = timeout
        self.max_tokens = max_tokens

    async def synthesize(
        self,
        signals: list[ScoredSignal],
        world_model_summary: str = "",
    ) -> tuple[list[PerceptionEvent], float]:
        """
        Returns (events, latency_seconds).

        Latency is logged by the PerceptionLayer for monitoring.
        """
        if not signals:
            return [], 0.0

        t0 = time.perf_counter()

        # ── Single-signal fast path ────────────────────────────────────────────
        if len(signals) == 1:
            event = self._trivial_event(signals[0])
            return [event], time.perf_counter() - t0

        # ── Multi-signal LLM path ─────────────────────────────────────────────
        try:
            events = await asyncio.wait_for(
                self._llm_synthesize(signals, world_model_summary),
                timeout=self.timeout,
            )
            latency = time.perf_counter() - t0
            logger.debug(
                "Synthesizer: LLM produced %d events in %.2fs for %d signals.",
                len(events), latency, len(signals),
            )
            return events, latency

        except asyncio.TimeoutError:
            latency = time.perf_counter() - t0
            logger.warning(
                "Synthesizer: LLM timed out after %.1fs — falling back to trivial events.",
                latency,
            )
            return [self._trivial_event(s) for s in signals], latency

        except Exception as exc:
            latency = time.perf_counter() - t0
            logger.error(
                "Synthesizer: unexpected error after %.2fs — falling back to trivial events. error=%s",
                latency, exc, exc_info=True,
            )
            return [self._trivial_event(s) for s in signals], latency

    # ── Private ───────────────────────────────────────────────────────────────

    async def _llm_synthesize(
        self,
        signals: list[ScoredSignal],
        world_model_summary: str,
    ) -> list[PerceptionEvent]:
        """Call the LLM (NANO tier) and parse the result into PerceptionEvent objects."""
        signals_json = json.dumps(
            [self._signal_to_dict(s) for s in signals],
            default=str,
            indent=2,
        )

        request = ModelRequest(
            tier=ModelTier.NANO,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_PROMPT_TEMPLATE.format(
                        world_model_summary=world_model_summary or "(none)",
                        signals_json=signals_json,
                    ),
                },
            ],
            component="Synthesizer._llm_synthesize",
        )

        response = await self.router.complete(request)
        raw_content = response.content

        if not raw_content:
            logger.warning("Synthesizer: LLM returned empty content — falling back to trivial events.")
            return [self._trivial_event(s) for s in signals]

        return self._parse_events(raw_content, signals)

    def _parse_events(
        self,
        raw: str,
        fallback_signals: list[ScoredSignal],
    ) -> list[PerceptionEvent]:
        """
        Attempt to parse LLM output into PerceptionEvent objects.

        Repair strategy (§7):
          1. Pass through _clean_llm_json (strips thinking tags, fences, trailing commas).
          2. json.loads the cleaned string.
          3. Build PerceptionEvent from each dict.
          4. If any step fails → fall back to trivial events.
        """
        try:
            cleaned = _clean_llm_json(raw)
            data = json.loads(cleaned)

            if not isinstance(data, list):
                data = [data]

            events: list[PerceptionEvent] = []
            for item in data:
                try:
                    event = PerceptionEvent(
                        summary=item.get("summary", ""),
                        type=PerceptionType(item.get("type", "information")),
                        urgency=float(item.get("urgency", 0.5)),
                        authority_score=float(item.get("authority_score", 0.5)),
                        requires_immediate_response=bool(item.get("requires_immediate_response", False)),
                        affects_objectives=bool(item.get("affects_objectives", False)),
                        frequency=int(item.get("frequency", 1)),
                        source_ids=[],  # UUIDs from LLM are strings — omit for safety
                        response_context=item.get("response_context"),
                    )
                    events.append(event)
                except Exception as item_err:
                    logger.warning("Synthesizer: skipping malformed event item: %s", item_err)

            if events:
                return events

            logger.warning("Synthesizer: parsed 0 valid events — falling back.")
            return [self._trivial_event(s) for s in fallback_signals]

        except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
            logger.warning(
                "Synthesizer: JSON parse failed (%s) — falling back to trivial events.",
                parse_err,
            )
            return [self._trivial_event(s) for s in fallback_signals]

    @staticmethod
    def _trivial_event(signal: ScoredSignal) -> PerceptionEvent:
        """
        Construct a PerceptionEvent directly from a ScoredSignal without LLM.
        Used for single-signal batches and all fallback paths.
        """
        perception_type = signal.perception_type
        requires_immediate = perception_type in (
            PerceptionType.DIRECTIVE,
            PerceptionType.QUERY,
            PerceptionType.CORRECTION,
        )
        affects_objectives = perception_type in (
            PerceptionType.DIRECTIVE,
            PerceptionType.INFORMATION,
            PerceptionType.CORRECTION,
            PerceptionType.FEEDBACK,
        )

        return PerceptionEvent(
            summary=signal.representative.text,
            type=perception_type,
            urgency=signal.authority_score,  # authority as a proxy for urgency
            authority_score=signal.authority_score,
            requires_immediate_response=requires_immediate,
            affects_objectives=affects_objectives,
            frequency=signal.frequency,
            source_ids=[signal.representative.id],
            response_context=signal.representative.text if requires_immediate else None,
        )

    @staticmethod
    def _signal_to_dict(signal: ScoredSignal) -> dict:
        """Serialise a ScoredSignal into a compact dict for the LLM prompt."""
        return {
            "id": str(signal.id),
            "text": signal.representative.text,
            "type": signal.perception_type.value,
            "authority_score": signal.authority_score,
            "frequency": signal.frequency,
            "source_type": signal.representative.source.type.value,
            "channel_id": signal.representative.channel_id,
            "timestamp": signal.representative.timestamp.isoformat(),
        }
