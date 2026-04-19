"""
Pair sampling engine for human labeling.

Given an org corpus and (optionally) model correlation scores, selects
incident pairs for SRE labeling. Samples across the confidence spectrum
to calibrate both true positives and false positives.

Sampling strategy:
  - High confidence (top 25%): pairs the model thinks are related — verify TPs
  - Medium confidence (middle 25%): the interesting boundary
  - Low confidence (bottom 25%): pairs the model thinks are unrelated — verify TNs
  - Random (25%): calibration anchors, unbiased by model
"""

from __future__ import annotations

import random
import uuid
from itertools import combinations

from .models import CorrelationScore, IncidentPair, LabelingSession, OrgCorpus


def _all_pair_ids(corpus: OrgCorpus) -> list[tuple[str, str]]:
    """Generate all unique incident pairs."""
    ids = [inc.id for inc in corpus.incidents]
    return list(combinations(ids, 2))


def sample_pairs(
    corpus: OrgCorpus,
    model_scores: list[CorrelationScore] | None = None,
    total_pairs: int = 200,
    seed: int = 42,
) -> LabelingSession:
    """Sample incident pairs for human labeling.

    Args:
        corpus: The org's incident corpus.
        model_scores: Optional model correlation scores. If provided,
            pairs are stratified across the confidence spectrum.
            If not provided, all pairs are sampled randomly.
        total_pairs: Target number of pairs to label.
        seed: Random seed.

    Returns:
        A LabelingSession ready for labeling.
    """
    rng = random.Random(seed)
    all_pairs = _all_pair_ids(corpus)

    if len(all_pairs) < total_pairs:
        total_pairs = len(all_pairs)

    pairs: list[IncidentPair] = []

    if model_scores:
        # Build score lookup
        score_map: dict[tuple[str, str], float] = {}
        for s in model_scores:
            key = tuple(sorted([s.incident_a_id, s.incident_b_id]))
            score_map[key] = s.confidence

        # Score all pairs (unscored pairs get 0.0)
        scored_pairs = []
        for a_id, b_id in all_pairs:
            key = tuple(sorted([a_id, b_id]))
            conf = score_map.get(key, 0.0)
            scored_pairs.append((a_id, b_id, conf))

        # Sort by confidence
        scored_pairs.sort(key=lambda x: x[2], reverse=True)

        # Allocate across buckets
        per_bucket = total_pairs // 4
        remainder = total_pairs - per_bucket * 4

        # High confidence (top)
        high = scored_pairs[:max(per_bucket * 2, len(scored_pairs) // 4)]
        rng.shuffle(high)
        for a_id, b_id, conf in high[:per_bucket]:
            pairs.append(IncidentPair(
                pair_id=uuid.uuid4().hex[:12],
                incident_a_id=a_id,
                incident_b_id=b_id,
                model_confidence=round(conf, 3),
                sampling_bucket="high_confidence",
            ))

        # Medium confidence (middle)
        mid_start = len(scored_pairs) // 3
        mid_end = 2 * len(scored_pairs) // 3
        medium = scored_pairs[mid_start:mid_end]
        rng.shuffle(medium)
        for a_id, b_id, conf in medium[:per_bucket]:
            pairs.append(IncidentPair(
                pair_id=uuid.uuid4().hex[:12],
                incident_a_id=a_id,
                incident_b_id=b_id,
                model_confidence=round(conf, 3),
                sampling_bucket="medium",
            ))

        # Low confidence (bottom)
        low = scored_pairs[-max(per_bucket * 2, len(scored_pairs) // 4):]
        rng.shuffle(low)
        for a_id, b_id, conf in low[:per_bucket]:
            pairs.append(IncidentPair(
                pair_id=uuid.uuid4().hex[:12],
                incident_a_id=a_id,
                incident_b_id=b_id,
                model_confidence=round(conf, 3),
                sampling_bucket="low_confidence",
            ))

        # Random (from any bucket, ignore model scores)
        used_pair_keys = {(p.incident_a_id, p.incident_b_id) for p in pairs}
        remaining = [
            (a, b, c) for a, b, c in scored_pairs
            if (a, b) not in used_pair_keys
        ]
        rng.shuffle(remaining)
        random_count = per_bucket + remainder
        for a_id, b_id, conf in remaining[:random_count]:
            pairs.append(IncidentPair(
                pair_id=uuid.uuid4().hex[:12],
                incident_a_id=a_id,
                incident_b_id=b_id,
                model_confidence=round(conf, 3),
                sampling_bucket="random",
            ))
    else:
        # No model scores — all pairs are random
        rng.shuffle(all_pairs)
        for a_id, b_id in all_pairs[:total_pairs]:
            pairs.append(IncidentPair(
                pair_id=uuid.uuid4().hex[:12],
                incident_a_id=a_id,
                incident_b_id=b_id,
                sampling_bucket="random",
            ))

    # Shuffle final order so labelers don't see buckets in sequence
    rng.shuffle(pairs)

    return LabelingSession(
        session_id=uuid.uuid4().hex[:16],
        corpus_path="",
        pairs=pairs,
    )
