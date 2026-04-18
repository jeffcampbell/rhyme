"""Tests for human-label scoring."""

from sifter_web.models import (
    IncidentPair,
    LabelingSession,
    PairLabel,
)
from sifter_web.scorer_human import score_against_humans


def _make_session(pairs_and_labels):
    """Helper: pairs_and_labels is [(pair_id, model_conf, judgment, labeler_id), ...]"""
    pairs = []
    labels = []
    for pair_id, conf, judgment, labeler in pairs_and_labels:
        pairs.append(IncidentPair(
            pair_id=pair_id,
            incident_a_id=f"a-{pair_id}",
            incident_b_id=f"b-{pair_id}",
            model_confidence=conf,
            sampling_bucket="random",
        ))
        labels.append(PairLabel(
            pair_id=pair_id,
            labeler_id=labeler,
            judgment=judgment,
        ))
    return LabelingSession(
        session_id="test",
        corpus_path="/tmp/test.json",
        pairs=pairs,
        labels=labels,
    )


def test_empty_session():
    session = LabelingSession(session_id="t", corpus_path="", pairs=[], labels=[])
    report = score_against_humans(session)
    assert report.labeled_pairs == 0


def test_perfect_model():
    """Model says high confidence for all, humans agree."""
    session = _make_session([
        ("p1", 0.9, "yes", "jeff"),
        ("p2", 0.9, "yes", "jeff"),
        ("p3", 0.1, "no", "jeff"),
        ("p4", 0.1, "no", "jeff"),
    ])
    report = score_against_humans(session)
    assert report.precision_at_50 == 1.0
    assert report.recall_at_50 == 1.0


def test_terrible_model():
    """Model says high confidence for all, humans all say no."""
    session = _make_session([
        ("p1", 0.9, "no", "jeff"),
        ("p2", 0.9, "no", "jeff"),
        ("p3", 0.9, "no", "jeff"),
    ])
    report = score_against_humans(session)
    assert report.precision_at_50 == 0.0
    assert report.high_conf_disagreements == 3


def test_maybe_excluded_from_precision():
    session = _make_session([
        ("p1", 0.9, "maybe", "jeff"),
        ("p2", 0.9, "yes", "jeff"),
        ("p3", 0.1, "no", "jeff"),
    ])
    report = score_against_humans(session)
    # Only p2 and p3 count for precision/recall
    assert report.precision_at_50 == 1.0  # 1 TP, 0 FP
    assert report.human_maybe_rate > 0


def test_red_herring_detection():
    session = _make_session([
        ("p1", 0.95, "no", "jeff"),   # high conf, human says no
        ("p2", 0.05, "yes", "jeff"),  # low conf, human says yes
        ("p3", 0.5, "yes", "jeff"),
    ])
    report = score_against_humans(session)
    assert report.high_conf_disagreements == 1
    assert report.low_conf_agreements == 1


def test_calibration_buckets():
    session = _make_session([
        ("p1", 0.1, "no", "jeff"),
        ("p2", 0.3, "no", "jeff"),
        ("p3", 0.5, "yes", "jeff"),
        ("p4", 0.7, "yes", "jeff"),
        ("p5", 0.9, "yes", "jeff"),
    ])
    report = score_against_humans(session)
    assert len(report.calibration_buckets) > 0
    for b in report.calibration_buckets:
        assert "mean_conf" in b
        assert "human_agree_rate" in b
        assert "count" in b


def test_inter_annotator_agreement():
    """Two labelers, some agreement, some not."""
    pairs = [
        IncidentPair(pair_id="p1", incident_a_id="a1", incident_b_id="b1", sampling_bucket="random"),
        IncidentPair(pair_id="p2", incident_a_id="a2", incident_b_id="b2", sampling_bucket="random"),
        IncidentPair(pair_id="p3", incident_a_id="a3", incident_b_id="b3", sampling_bucket="random"),
        IncidentPair(pair_id="p4", incident_a_id="a4", incident_b_id="b4", sampling_bucket="random"),
        IncidentPair(pair_id="p5", incident_a_id="a5", incident_b_id="b5", sampling_bucket="random"),
    ]
    labels = [
        PairLabel(pair_id="p1", labeler_id="jeff", judgment="yes"),
        PairLabel(pair_id="p1", labeler_id="alice", judgment="yes"),   # agree
        PairLabel(pair_id="p2", labeler_id="jeff", judgment="no"),
        PairLabel(pair_id="p2", labeler_id="alice", judgment="no"),    # agree
        PairLabel(pair_id="p3", labeler_id="jeff", judgment="yes"),
        PairLabel(pair_id="p3", labeler_id="alice", judgment="no"),    # disagree
        PairLabel(pair_id="p4", labeler_id="jeff", judgment="no"),
        PairLabel(pair_id="p4", labeler_id="alice", judgment="yes"),   # disagree
        PairLabel(pair_id="p5", labeler_id="jeff", judgment="yes"),
        PairLabel(pair_id="p5", labeler_id="alice", judgment="yes"),   # agree
    ]
    session = LabelingSession(session_id="t", corpus_path="", pairs=pairs, labels=labels)
    report = score_against_humans(session)
    assert report.inter_annotator_agreement is not None
    assert -1.0 <= report.inter_annotator_agreement <= 1.0
    assert len(report.labelers) == 2


def test_human_yes_rate():
    session = _make_session([
        ("p1", 0.5, "yes", "jeff"),
        ("p2", 0.5, "no", "jeff"),
        ("p3", 0.5, "yes", "jeff"),
        ("p4", 0.5, "no", "jeff"),
    ])
    report = score_against_humans(session)
    assert report.human_yes_rate == 0.5
