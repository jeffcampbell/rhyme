"""
Deterministic scorer implementing all metrics from spec §7.

Metrics:
  - Retrieval@k (per tier): does top-k contain any same-cause incident?
  - Cause-match precision@k (per tier): fraction of top-k sharing proximal cause,
    reported as lift over class base rate.
  - Confusion matrix delta: for each query class, which wrong classes dominate top-k.
  - Calibration ECE: expected calibration error on confidence scores.
  - Remediation grade: {fixed, no_op, masks_symptom, would_worsen} distribution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .models import (
    Corpus,
    Incident,
    QueryGroundTruth,
    QuerySet,
    RemediationQuestion,
    RemediationQuestionSet,
    RemediationResult,
    RetrievalResult,
)
from .taxonomy import CauseClass, ConfusabilityTier


@dataclass
class TierMetrics:
    tier: ConfusabilityTier
    num_queries: int = 0
    retrieval_at_k: float = 0.0  # fraction of queries with any correct match in top-k
    precision_at_k_raw: float = 0.0  # raw cause-match precision
    precision_at_k_lift: float = 0.0  # lift over class base rate
    calibration_ece: float = 0.0


@dataclass
class ConfusionEntry:
    query_class: CauseClass
    predicted_class: CauseClass
    count: int


@dataclass
class RemediationGrades:
    fixed: int = 0
    no_op: int = 0
    masks_symptom: int = 0
    would_worsen: int = 0
    total: int = 0

    def as_dict(self) -> dict[str, float]:
        if self.total == 0:
            return {"fixed": 0, "no_op": 0, "masks_symptom": 0, "would_worsen": 0}
        return {
            "fixed": self.fixed / self.total,
            "no_op": self.no_op / self.total,
            "masks_symptom": self.masks_symptom / self.total,
            "would_worsen": self.would_worsen / self.total,
        }


@dataclass
class EfficiencyMetrics:
    """Token usage efficiency metrics for comparing prompt strategies."""

    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    num_queries_with_usage: int = 0
    mean_tokens_per_query: float = 0.0
    mean_input_per_query: float = 0.0
    mean_output_per_query: float = 0.0
    tokens_per_correct_match: float = 0.0  # total tokens / number of correct top-k matches
    precision_per_1k_tokens: float = 0.0  # precision@k per 1000 tokens spent


@dataclass
class ScoreReport:
    overall_retrieval_at_k: float
    overall_precision_at_k_raw: float
    overall_precision_at_k_lift: float
    overall_calibration_ece: float
    per_tier: dict[ConfusabilityTier, TierMetrics]
    confusion_matrix: list[ConfusionEntry]
    remediation_grades: RemediationGrades | None
    efficiency: EfficiencyMetrics | None
    k: int

    def summary(self) -> str:
        lines = [
            f"=== Sifter Score Report (k={self.k}) ===",
            "",
            "Per-tier results (primary — no composite score; see spec §13 Q1):",
        ]
        for tier in ConfusabilityTier:
            tm = self.per_tier.get(tier)
            if tm and tm.num_queries > 0:
                lines.append(
                    f"  {tier.value:6s}  n={tm.num_queries:2d}  "
                    f"Ret@k={tm.retrieval_at_k:.3f}  "
                    f"Prec@k={tm.precision_at_k_raw:.3f}  "
                    f"Lift={tm.precision_at_k_lift:.3f}  "
                    f"ECE={tm.calibration_ece:.3f}"
                )
        lines.extend([
            "",
            f"Macro-average (reference only, not a headline score):",
            f"  Retrieval@{self.k}={self.overall_retrieval_at_k:.3f}  "
            f"Prec@{self.k}={self.overall_precision_at_k_raw:.3f}  "
            f"Lift={self.overall_precision_at_k_lift:.3f}  "
            f"ECE={self.overall_calibration_ece:.3f}",
        ])

        if self.confusion_matrix:
            lines.append("")
            lines.append("--- Confusion matrix (top wrong predictions) ---")
            sorted_cm = sorted(self.confusion_matrix, key=lambda e: -e.count)
            for entry in sorted_cm[:15]:
                lines.append(
                    f"  {entry.query_class.value:>25s} -> "
                    f"{entry.predicted_class.value:<25s}  count={entry.count}"
                )

        if self.remediation_grades:
            lines.append("")
            lines.append("--- Remediation grades ---")
            grades = self.remediation_grades.as_dict()
            for label, pct in grades.items():
                lines.append(f"  {label:15s}: {pct:.1%}")

        if self.efficiency and self.efficiency.num_queries_with_usage > 0:
            e = self.efficiency
            lines.append("")
            lines.append("--- Efficiency ---")
            lines.append(f"  Queries with token data: {e.num_queries_with_usage}")
            lines.append(f"  Total tokens:            {e.total_tokens:,}")
            lines.append(f"    Input:                 {e.total_input_tokens:,}")
            lines.append(f"    Output:                {e.total_output_tokens:,}")
            lines.append(f"  Mean tokens/query:       {e.mean_tokens_per_query:,.0f}")
            lines.append(f"    Input/query:           {e.mean_input_per_query:,.0f}")
            lines.append(f"    Output/query:          {e.mean_output_per_query:,.0f}")
            if e.tokens_per_correct_match > 0:
                lines.append(f"  Tokens/correct match:    {e.tokens_per_correct_match:,.0f}")
            if e.precision_per_1k_tokens > 0:
                lines.append(f"  Precision per 1k tokens: {e.precision_per_1k_tokens:.4f}")

        return "\n".join(lines)


def _class_base_rates(corpus: Corpus) -> dict[CauseClass, float]:
    """Compute the base rate for each cause class in the corpus."""
    counts: Counter[CauseClass] = Counter()
    for inc in corpus.incidents:
        counts[inc.labels.cause_class] += 1
    total = len(corpus.incidents)
    return {cc: count / total for cc, count in counts.items()}


def _calibration_ece(
    confidences: list[float],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected calibration error."""
    if not confidences:
        return 0.0
    confs = np.array(confidences)
    cors = np.array(correct, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_edges[i]) & (confs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = cors[mask].mean()
        ece += mask.sum() / len(confs) * abs(bin_acc - bin_conf)
    return float(ece)


def score(
    retrieval_results: list[RetrievalResult],
    query_set: QuerySet,
    corpus: Corpus,
    k: int = 10,
    remediation_results: list[RemediationResult] | None = None,
    remediation_questions: RemediationQuestionSet | None = None,
) -> ScoreReport:
    """Score retrieval results against ground truth.

    Args:
        retrieval_results: Model outputs from the harness.
        query_set: Ground-truth query set.
        corpus: The full corpus (for base rate computation).
        k: Evaluation depth.
        remediation_results: Optional Task 3 multiple-choice results.
        remediation_questions: Question set with correct answers and grades.

    Returns:
        A ScoreReport with all metrics.
    """
    gt_map = {q.query_id: q for q in query_set.queries}
    inc_map = {inc.payload.incident_id: inc for inc in corpus.incidents}
    base_rates = _class_base_rates(corpus)

    # Per-tier accumulators
    tier_hits: dict[ConfusabilityTier, list[bool]] = defaultdict(list)
    tier_precisions: dict[ConfusabilityTier, list[float]] = defaultdict(list)
    tier_lifts: dict[ConfusabilityTier, list[float]] = defaultdict(list)
    tier_confidences: dict[ConfusabilityTier, list[float]] = defaultdict(list)
    tier_correct: dict[ConfusabilityTier, list[bool]] = defaultdict(list)

    # Confusion matrix
    confusion: Counter[tuple[CauseClass, CauseClass]] = Counter()

    for result in retrieval_results:
        gt = gt_map.get(result.query_id)
        if gt is None:
            continue

        correct_ids = set(gt.correct_match_ids)
        query_inc = inc_map.get(result.query_id)
        if query_inc is None:
            continue
        query_class = query_inc.labels.cause_class
        tier = gt.confusability_tier

        top_k = result.ranked_matches[:k]

        # Retrieval@k: any correct match in top-k?
        hit = any(m.incident_id in correct_ids for m in top_k)
        tier_hits[tier].append(hit)

        # Precision@k: fraction of top-k that are correct
        if top_k:
            correct_count = sum(1 for m in top_k if m.incident_id in correct_ids)
            precision = correct_count / len(top_k)
        else:
            precision = 0.0
        tier_precisions[tier].append(precision)

        # Lift over base rate
        br = base_rates.get(query_class, 0.01)
        lift = precision / br if br > 0 else 0.0
        tier_lifts[tier].append(lift)

        # Calibration: per-match confidence vs correctness
        for m in top_k:
            is_correct = m.incident_id in correct_ids
            tier_confidences[tier].append(m.confidence)
            tier_correct[tier].append(is_correct)

        # Confusion matrix: what wrong classes show up?
        for m in top_k:
            matched_inc = inc_map.get(m.incident_id)
            if matched_inc and matched_inc.labels.cause_class != query_class:
                confusion[(query_class, matched_inc.labels.cause_class)] += 1

    # Aggregate per-tier
    per_tier: dict[ConfusabilityTier, TierMetrics] = {}
    all_hits: list[bool] = []
    all_precs: list[float] = []
    all_lifts: list[float] = []
    all_confs: list[float] = []
    all_cors: list[bool] = []

    for tier in ConfusabilityTier:
        hits = tier_hits.get(tier, [])
        precs = tier_precisions.get(tier, [])
        lifts = tier_lifts.get(tier, [])
        confs = tier_confidences.get(tier, [])
        cors = tier_correct.get(tier, [])

        all_hits.extend(hits)
        all_precs.extend(precs)
        all_lifts.extend(lifts)
        all_confs.extend(confs)
        all_cors.extend(cors)

        n = len(hits)
        per_tier[tier] = TierMetrics(
            tier=tier,
            num_queries=n,
            retrieval_at_k=sum(hits) / n if n else 0.0,
            precision_at_k_raw=sum(precs) / n if n else 0.0,
            precision_at_k_lift=sum(lifts) / n if n else 0.0,
            calibration_ece=_calibration_ece(confs, cors),
        )

    n_total = len(all_hits)

    # Remediation grading (deterministic multiple-choice)
    rem_grades = None
    if remediation_results and remediation_questions:
        rem_grades = _grade_remediations(remediation_results, remediation_questions)

    # Efficiency metrics
    efficiency = _compute_efficiency(retrieval_results, all_precs, k)

    return ScoreReport(
        overall_retrieval_at_k=sum(all_hits) / n_total if n_total else 0.0,
        overall_precision_at_k_raw=sum(all_precs) / n_total if n_total else 0.0,
        overall_precision_at_k_lift=sum(all_lifts) / n_total if n_total else 0.0,
        overall_calibration_ece=_calibration_ece(all_confs, all_cors),
        per_tier=per_tier,
        confusion_matrix=[
            ConfusionEntry(query_class=qc, predicted_class=pc, count=cnt)
            for (qc, pc), cnt in confusion.items()
        ],
        remediation_grades=rem_grades,
        efficiency=efficiency,
        k=k,
    )


def _compute_efficiency(
    retrieval_results: list[RetrievalResult],
    per_query_precisions: list[float],
    k: int,
) -> EfficiencyMetrics | None:
    """Compute token efficiency metrics from retrieval results.

    Returns None if no results include token usage data.
    """
    total_input = 0
    total_output = 0
    n_with_usage = 0

    for result in retrieval_results:
        if result.token_usage is not None:
            total_input += result.token_usage.input_tokens
            total_output += result.token_usage.output_tokens
            n_with_usage += 1

    if n_with_usage == 0:
        return None

    total = total_input + total_output
    mean_prec = sum(per_query_precisions) / len(per_query_precisions) if per_query_precisions else 0
    total_correct = sum(p * k for p in per_query_precisions)

    return EfficiencyMetrics(
        total_tokens=total,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        num_queries_with_usage=n_with_usage,
        mean_tokens_per_query=total / n_with_usage,
        mean_input_per_query=total_input / n_with_usage,
        mean_output_per_query=total_output / n_with_usage,
        tokens_per_correct_match=total / total_correct if total_correct > 0 else 0,
        precision_per_1k_tokens=mean_prec / (total / n_with_usage / 1000) if total > 0 else 0,
    )


def _grade_remediations(
    results: list[RemediationResult],
    question_set: RemediationQuestionSet,
) -> RemediationGrades:
    """Grade multiple-choice remediation answers.

    Fully deterministic: compare selected label to the grade assigned
    to that choice in the question set.
    """
    question_map = {q.query_id: q for q in question_set.questions}
    grades = RemediationGrades()

    for result in results:
        question = question_map.get(result.query_id)
        if question is None:
            continue

        grades.total += 1

        # Find the grade for the selected choice
        selected_grade = "no_op"
        for choice in question.choices:
            if choice.label == result.selected_label:
                selected_grade = choice.grade
                break

        if selected_grade == "fixed":
            grades.fixed += 1
        elif selected_grade == "masks_symptom":
            grades.masks_symptom += 1
        elif selected_grade == "would_worsen":
            grades.would_worsen += 1
        else:
            grades.no_op += 1

    return grades
