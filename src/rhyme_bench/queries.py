"""
Query set construction with confusability stratification.

Selects query incidents from the corpus, identifies ground-truth matches
(same cause class) and tempting wrong matches (confusable pair, different cause).

Supports tier-stratified selection targeting ~30% easy / ~40% medium / ~30% hard.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .models import (
    Corpus,
    Incident,
    QueryGroundTruth,
    QuerySet,
    RemediationChoice,
    RemediationQuestion,
    RemediationQuestionSet,
)
from .taxonomy import (
    CONFUSABLE_PAIRS,
    REMEDIATIONS,
    CauseClass,
    ConfusabilityTier,
)


def _find_confusable_class(cause_class: CauseClass) -> CauseClass | None:
    """Find the most confusable partner for a given cause class."""
    for a, b, tier in CONFUSABLE_PAIRS:
        if tier == ConfusabilityTier.HARD:
            if a == cause_class:
                return b
            if b == cause_class:
                return a
    for a, b, tier in CONFUSABLE_PAIRS:
        if tier == ConfusabilityTier.MEDIUM:
            if a == cause_class:
                return b
            if b == cause_class:
                return a
    return None


def _build_ground_truth(
    query_incidents: list[Incident],
    corpus_without_queries: list[Incident],
) -> list[QueryGroundTruth]:
    """Build ground truth entries for a set of query incidents."""
    queries = []
    for query_inc in query_incidents:
        cause = query_inc.labels.cause_class
        correct = [
            inc.payload.incident_id
            for inc in corpus_without_queries
            if inc.labels.cause_class == cause
        ]
        confusable = _find_confusable_class(cause)
        tempting = []
        if confusable:
            tempting = [
                inc.payload.incident_id
                for inc in corpus_without_queries
                if inc.labels.cause_class == confusable
            ]
        queries.append(QueryGroundTruth(
            query_id=query_inc.payload.incident_id,
            correct_match_ids=correct,
            tempting_wrong_ids=tempting,
            confusability_tier=query_inc.labels.confusability_tier,
        ))
    return queries


def build_query_set(
    corpus: Corpus,
    total_queries: int = 200,
    tier_ratios: tuple[float, float, float] = (0.3, 0.4, 0.3),
    seed: int = 123,
) -> tuple[QuerySet, list[Incident]]:
    """Build a tier-stratified query set from the corpus.

    Args:
        corpus: The full corpus to select queries from.
        total_queries: Target number of queries.
        tier_ratios: Target (easy, medium, hard) ratio. Will be approximated.
        seed: Random seed for reproducibility.

    Returns:
        (query_set, query_incidents)
    """
    rng = random.Random(seed)
    easy_target = int(total_queries * tier_ratios[0])
    medium_target = int(total_queries * tier_ratios[1])
    hard_target = total_queries - easy_target - medium_target

    # Group incidents by tier
    by_tier: dict[ConfusabilityTier, list[Incident]] = defaultdict(list)
    for inc in corpus.incidents:
        by_tier[inc.labels.confusability_tier].append(inc)

    # Select from each tier, ensuring class diversity
    def _select_from_tier(
        pool: list[Incident], target: int, already_selected: set[str],
    ) -> list[Incident]:
        available = [inc for inc in pool if inc.payload.incident_id not in already_selected]
        # Try to get even class distribution within the tier
        by_class: dict[CauseClass, list[Incident]] = defaultdict(list)
        for inc in available:
            by_class[inc.labels.cause_class].append(inc)
        selected: list[Incident] = []
        # Round-robin across classes
        class_lists = list(by_class.values())
        rng.shuffle(class_lists)
        idx = 0
        while len(selected) < target and any(class_lists):
            class_pool = class_lists[idx % len(class_lists)]
            if class_pool:
                chosen = class_pool.pop(rng.randint(0, len(class_pool) - 1))
                selected.append(chosen)
            # Remove empty lists
            class_lists = [cl for cl in class_lists if cl]
            if class_lists:
                idx = (idx + 1) % len(class_lists)
        return selected

    selected_ids: set[str] = set()

    easy_selected = _select_from_tier(
        by_tier[ConfusabilityTier.EASY], easy_target, selected_ids,
    )
    selected_ids.update(inc.payload.incident_id for inc in easy_selected)

    medium_selected = _select_from_tier(
        by_tier[ConfusabilityTier.MEDIUM], medium_target, selected_ids,
    )
    selected_ids.update(inc.payload.incident_id for inc in medium_selected)

    hard_selected = _select_from_tier(
        by_tier[ConfusabilityTier.HARD], hard_target, selected_ids,
    )
    selected_ids.update(inc.payload.incident_id for inc in hard_selected)

    query_incidents = easy_selected + medium_selected + hard_selected

    # If we didn't hit the target (not enough incidents in some tiers), fill from any
    if len(query_incidents) < total_queries:
        remaining = [
            inc for inc in corpus.incidents
            if inc.payload.incident_id not in selected_ids
        ]
        rng.shuffle(remaining)
        needed = total_queries - len(query_incidents)
        query_incidents.extend(remaining[:needed])
        selected_ids.update(inc.payload.incident_id for inc in remaining[:needed])

    # Build ground truth
    corpus_without_queries = [
        inc for inc in corpus.incidents if inc.payload.incident_id not in selected_ids
    ]
    queries = _build_ground_truth(query_incidents, corpus_without_queries)

    return QuerySet(version="1.0.0", queries=queries), query_incidents


def build_private_slice(
    corpus: Corpus,
    public_query_ids: set[str],
    total_queries: int = 50,
    seed: int = 789,
) -> tuple[QuerySet, list[Incident]]:
    """Build a private query slice from corpus incidents not used in the public set.

    The private slice uses a different seed and avoids all public query incidents.
    """
    rng = random.Random(seed)

    available = [
        inc for inc in corpus.incidents
        if inc.payload.incident_id not in public_query_ids
    ]
    rng.shuffle(available)

    # Select up to total_queries, balanced across classes
    by_class: dict[CauseClass, list[Incident]] = defaultdict(list)
    for inc in available:
        by_class[inc.labels.cause_class].append(inc)

    per_class = max(1, total_queries // len(by_class))
    selected: list[Incident] = []
    for cause_class, incidents in by_class.items():
        selected.extend(incidents[:per_class])

    rng.shuffle(selected)
    selected = selected[:total_queries]
    selected_ids = {inc.payload.incident_id for inc in selected}

    corpus_without = [
        inc for inc in corpus.incidents
        if inc.payload.incident_id not in selected_ids
        and inc.payload.incident_id not in public_query_ids
    ]
    queries = _build_ground_truth(selected, corpus_without)

    return QuerySet(version="1.0.0", queries=queries), selected


def build_remediation_questions(
    query_incidents: list[Incident],
    seed: int = 456,
) -> RemediationQuestionSet:
    """Build multiple-choice remediation questions for each query.

    Each question has 5 options:
      - The canonical fix for the query's cause class (correct answer)
      - The masks-symptom fix for the query's cause class
      - The would-worsen action (if it exists) or a no-op distractor
      - Two plausible-but-wrong canonical fixes from other cause classes

    Options are shuffled and labeled A-E.
    """
    rng = random.Random(seed)
    all_classes = list(CauseClass)
    questions: list[RemediationQuestion] = []

    for query_inc in query_incidents:
        cause = query_inc.labels.cause_class
        rem = REMEDIATIONS[cause]

        choices_raw: list[tuple[str, str]] = [
            (rem.canonical, "fixed"),
            (rem.masks_symptom, "masks_symptom"),
        ]

        if rem.would_worsen:
            choices_raw.append((rem.would_worsen, "would_worsen"))

        # Pick distractor remediation(s) from other classes
        other_classes = [c for c in all_classes if c != cause]
        rng.shuffle(other_classes)
        distractors_needed = 5 - len(choices_raw)
        used_texts: set[str] = {text for text, _ in choices_raw}

        for other in other_classes:
            if distractors_needed <= 0:
                break
            other_rem = REMEDIATIONS[other]
            if other_rem.canonical not in used_texts:
                choices_raw.append((other_rem.canonical, "no_op"))
                used_texts.add(other_rem.canonical)
                distractors_needed -= 1

        # Shuffle and assign labels
        rng.shuffle(choices_raw)
        labels = ["A", "B", "C", "D", "E"]
        choices = []
        correct_label = ""
        for i, (text, grade) in enumerate(choices_raw[:5]):
            label = labels[i]
            choices.append(RemediationChoice(label=label, text=text, grade=grade))
            if grade == "fixed":
                correct_label = label

        questions.append(RemediationQuestion(
            query_id=query_inc.payload.incident_id,
            choices=choices,
            correct_label=correct_label,
        ))

    return RemediationQuestionSet(version="1.0.0", questions=questions)
