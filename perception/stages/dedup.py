"""
perception/stages/dedup.py
───────────────────────────
Stage 2 — Deduplicator

Collapses signals that are semantically equivalent before any LLM work is done.
Operates purely on raw text — fast, no model calls.

Similarity method
─────────────────
1. Exact match on normalised text (lowercase, stripped, whitespace-collapsed).
2. Fuzzy token-overlap ratio (Jaccard on word sets) above `similarity_threshold`.
3. Secondary grouping: same channel + arrival within a short time proximity
   window is treated as a secondary similarity signal (not a hard rule).

Output
──────
Each DeduplicatedSignal carries:
  - representative : the most-recent signal in the group
  - frequency      : how many raw signals were merged (feeds importance scoring)
  - sources        : all contributing RawSignalSources
  - raw_signals    : all raw signals in the cluster (for audit / debugging)
"""

from __future__ import annotations

import re
from datetime import timedelta

from perception.schemas import DeduplicatedSignal, RawSignal
from utils.log_writer import emit

# Secondary-key proximity window: two signals in the same channel that arrive
# within this many seconds are considered "contextually related" for grouping.
_CHANNEL_PROXIMITY_SECONDS = 10.0


def _normalise(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_set(text: str) -> set[str]:
    """Split normalised text into a word set for Jaccard similarity."""
    return set(re.findall(r"\b\w+\b", text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class Deduplicator:
    """
    Groups semantically equivalent RawSignals and returns one
    DeduplicatedSignal per group.

    Parameters
    ──────────
    similarity_threshold : float (default 0.85)
        Jaccard threshold above which two signals are considered duplicates.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold

    def deduplicate(self, signals: list[RawSignal]) -> list[DeduplicatedSignal]:
        """
        Return a de-duplicated list.  The representative of each group is the
        *most-recent* signal (highest timestamp), so the downstream stages see
        the freshest version of repeated content.
        """
        if not signals:
            return []

        groups = self._cluster(signals)

        result: list[DeduplicatedSignal] = []
        for group in groups:
            # Pick most-recent as representative
            representative = max(group, key=lambda s: s.timestamp)
            result.append(
                DeduplicatedSignal(
                    representative=representative,
                    frequency=len(group),
                    sources=[s.source for s in group],
                    raw_signals=group,
                )
            )

        emit(
            event="perception.deduped",
            layer="perception",
            level="debug",
            data={
                "raw_count": len(signals),
                "deduped_count": len(result),
                "threshold": self.similarity_threshold,
            }
        )
        return result

    # ── Private ───────────────────────────────────────────────────────────────

    def _cluster(self, signals: list[RawSignal]) -> list[list[RawSignal]]:
        """
        Greedy single-pass clustering.

        For each signal we try to merge it into the first existing cluster
        whose representative it matches.  O(n²) in the worst case, but
        perception batches are small (< 100 signals) so this is fine.
        """
        clusters: list[list[RawSignal]] = []
        cluster_norms: list[str] = []          # normalised text of cluster head
        cluster_tokens: list[set[str]] = []    # token set of cluster head

        for signal in signals:
            norm = _normalise(signal.text)
            toks = _token_set(norm)
            placed = False

            for idx, (c_norm, c_toks) in enumerate(zip(cluster_norms, cluster_tokens)):
                # 1. Exact match
                if norm == c_norm:
                    clusters[idx].append(signal)
                    placed = True
                    break

                # 2. Fuzzy token-overlap
                if _jaccard(toks, c_toks) >= self.similarity_threshold:
                    clusters[idx].append(signal)
                    placed = True
                    break

                # 3. Channel + time-proximity secondary key (informational — may
                #    push borderline signals together but never separates exact matches)
                head = clusters[idx][0]
                if (
                    signal.channel_id
                    and signal.channel_id == head.channel_id
                    and abs(
                        (signal.timestamp - head.timestamp).total_seconds()
                    ) <= _CHANNEL_PROXIMITY_SECONDS
                    and _jaccard(toks, c_toks) >= 0.5   # lower bar when same channel
                ):
                    clusters[idx].append(signal)
                    placed = True
                    break

            if not placed:
                clusters.append([signal])
                cluster_norms.append(norm)
                cluster_tokens.append(toks)

        return clusters
