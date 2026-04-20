"""
Score model correlation predictions against human labels.

Computes precision, recall, F1 at various confidence thresholds,
calibration curves, inter-annotator agreement, and red-herring detection.
"""

from __future__ import annotations

from collections import defaultdict

from .models import HumanVsModelReport, LabelingSession


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_against_humans(session: LabelingSession) -> HumanVsModelReport:
    """Score model predictions against human labels from a labeling session.

    Uses model_confidence from the session pairs (import-time scores).
    """
    pair_conf = {
        p.pair_id: p.model_confidence if p.model_confidence is not None else 0.5
        for p in session.pairs
    }
    return _score_with_confidences(session, pair_conf)


def score_model_against_humans(
    session: LabelingSession,
    model_confidences: dict[str, float],
) -> HumanVsModelReport:
    """Score a named model's confidences against human labels.

    Args:
        session: The labeling session with human labels.
        model_confidences: Mapping of pair_id -> model confidence score.
    """
    # Fill in 0.5 for any pairs not scored by this model
    pair_conf = {p.pair_id: model_confidences.get(p.pair_id, 0.5) for p in session.pairs}
    return _score_with_confidences(session, pair_conf)


def _score_with_confidences(
    session: LabelingSession,
    pair_conf: dict[str, float],
) -> HumanVsModelReport:
    """Core scoring logic given a pair_id -> confidence mapping.

    Human judgment mapping:
      'yes'   = incidents are correlated (positive)
      'no'    = not correlated (negative)
      'maybe' = uncertain (excluded from precision/recall, counted separately)
    """
    if not session.labels:
        return _empty_report()

    # Aggregate labels per pair (majority vote if multiple labelers)
    pair_labels: dict[str, list[str]] = defaultdict(list)
    labelers: set[str] = set()
    for label in session.labels:
        pair_labels[label.pair_id].append(label.judgment)
        labelers.add(label.labeler_id)

    # Resolve to majority judgment per pair
    resolved: dict[str, str] = {}
    for pair_id, judgments in pair_labels.items():
        yes_count = judgments.count("yes")
        no_count = judgments.count("no")
        if yes_count > no_count:
            resolved[pair_id] = "yes"
        elif no_count > yes_count:
            resolved[pair_id] = "no"
        else:
            resolved[pair_id] = "maybe"

    # Compute metrics at different thresholds
    def _pr_at_threshold(threshold: float) -> tuple[float, float]:
        tp = fp = fn = 0
        for pair_id, human in resolved.items():
            if human == "maybe":
                continue
            conf = pair_conf.get(pair_id, 0.5)
            model_positive = conf >= threshold
            human_positive = human == "yes"
            if model_positive and human_positive:
                tp += 1
            elif model_positive and not human_positive:
                fp += 1
            elif not model_positive and human_positive:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    p50, r50 = _pr_at_threshold(0.5)
    p70, r70 = _pr_at_threshold(0.7)
    p90, r90 = _pr_at_threshold(0.9)

    # Calibration buckets
    n_bins = 5
    cal_buckets = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bucket_confs = []
        bucket_agrees = []
        for pair_id, human in resolved.items():
            if human == "maybe":
                continue
            conf = pair_conf.get(pair_id, 0.5)
            if lo <= conf < hi or (i == n_bins - 1 and conf == hi):
                bucket_confs.append(conf)
                bucket_agrees.append(1.0 if human == "yes" else 0.0)
        if bucket_confs:
            cal_buckets.append({
                "conf_range": f"{lo:.1f}-{hi:.1f}",
                "mean_conf": round(sum(bucket_confs) / len(bucket_confs), 3),
                "human_agree_rate": round(sum(bucket_agrees) / len(bucket_agrees), 3),
                "count": len(bucket_confs),
            })

    # Human label distribution
    yes_count = sum(1 for v in resolved.values() if v == "yes")
    maybe_count = sum(1 for v in resolved.values() if v == "maybe")
    total_resolved = len(resolved)

    # Red herring detection
    high_conf_disagree = 0
    low_conf_agree = 0
    for pair_id, human in resolved.items():
        if human == "maybe":
            continue
        conf = pair_conf.get(pair_id, 0.5)
        if conf >= 0.7 and human == "no":
            high_conf_disagree += 1
        if conf < 0.3 and human == "yes":
            low_conf_agree += 1

    # Inter-annotator agreement (Cohen's kappa for pairs labeled by 2+ labelers)
    iaa = _compute_inter_annotator(session)

    return HumanVsModelReport(
        total_pairs=len(session.pairs),
        labeled_pairs=total_resolved,
        labelers=sorted(labelers),
        precision_at_50=round(p50, 3),
        recall_at_50=round(r50, 3),
        f1_at_50=round(_f1(p50, r50), 3),
        precision_at_70=round(p70, 3),
        recall_at_70=round(r70, 3),
        f1_at_70=round(_f1(p70, r70), 3),
        precision_at_90=round(p90, 3),
        recall_at_90=round(r90, 3),
        f1_at_90=round(_f1(p90, r90), 3),
        calibration_buckets=cal_buckets,
        human_yes_rate=round(yes_count / total_resolved, 3) if total_resolved else 0,
        human_maybe_rate=round(maybe_count / total_resolved, 3) if total_resolved else 0,
        inter_annotator_agreement=iaa,
        high_conf_disagreements=high_conf_disagree,
        low_conf_agreements=low_conf_agree,
    )


def _compute_inter_annotator(session: LabelingSession) -> float | None:
    """Compute Cohen's kappa for pairs labeled by exactly 2 labelers."""
    pair_by_labeler: dict[str, dict[str, str]] = defaultdict(dict)
    for label in session.labels:
        pair_by_labeler[label.pair_id][label.labeler_id] = label.judgment

    # Find pairs with exactly 2 labelers
    dual_labeled = [
        (labelers_map, pair_id)
        for pair_id, labelers_map in pair_by_labeler.items()
        if len(labelers_map) >= 2
    ]

    if len(dual_labeled) < 5:
        return None

    # Compute kappa between first two labelers
    agreements = 0
    total = 0
    label_counts_a: dict[str, int] = defaultdict(int)
    label_counts_b: dict[str, int] = defaultdict(int)

    for labelers_map, pair_id in dual_labeled:
        labeler_ids = sorted(labelers_map.keys())[:2]
        la = labelers_map[labeler_ids[0]]
        lb = labelers_map[labeler_ids[1]]
        if la == lb:
            agreements += 1
        label_counts_a[la] += 1
        label_counts_b[lb] += 1
        total += 1

    if total == 0:
        return None

    po = agreements / total  # observed agreement
    # Expected agreement by chance
    pe = sum(
        (label_counts_a.get(cat, 0) / total) * (label_counts_b.get(cat, 0) / total)
        for cat in {"yes", "no", "maybe"}
    )

    if pe >= 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 3)


def _empty_report() -> HumanVsModelReport:
    return HumanVsModelReport(
        total_pairs=0,
        labeled_pairs=0,
        labelers=[],
        precision_at_50=0, recall_at_50=0, f1_at_50=0,
        precision_at_70=0, recall_at_70=0, f1_at_70=0,
        precision_at_90=0, recall_at_90=0, f1_at_90=0,
        calibration_buckets=[],
        human_yes_rate=0,
        human_maybe_rate=0,
        high_conf_disagreements=0,
        low_conf_agreements=0,
    )
