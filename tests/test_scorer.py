"""Tests for the scorer."""

from rhyme_bench.baselines import BM25Baseline, RandomBaseline
from rhyme_bench.harness import run_retrieval
from rhyme_bench.models import (
    RankedMatch,
    RemediationResult,
    RetrievalResult,
)
from rhyme_bench.scorer import score, EfficiencyMetrics
from rhyme_bench.taxonomy import ConfusabilityTier


def test_score_produces_all_tiers(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    adapter = BM25Baseline()
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)
    report = score(results, qs, small_corpus, k=10)

    assert report.k == 10
    # At least some tiers should have queries
    tiers_with_queries = [t for t, m in report.per_tier.items() if m.num_queries > 0]
    assert len(tiers_with_queries) >= 1


def test_bm25_beats_random(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    payloads = [q.payload for q in qi]

    random_results = run_retrieval(RandomBaseline(seed=1), payloads, small_corpus, k=10)
    bm25_results = run_retrieval(BM25Baseline(), payloads, small_corpus, k=10)

    random_report = score(random_results, qs, small_corpus, k=10)
    bm25_report = score(bm25_results, qs, small_corpus, k=10)

    assert bm25_report.overall_precision_at_k_raw > random_report.overall_precision_at_k_raw


def test_precision_lift_normalized(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    adapter = BM25Baseline()
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)
    report = score(results, qs, small_corpus, k=10)

    # Lift should be >= raw precision (since base rate < 1)
    assert report.overall_precision_at_k_lift >= report.overall_precision_at_k_raw


def test_confusion_matrix_populated(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    adapter = RandomBaseline(seed=1)
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)
    report = score(results, qs, small_corpus, k=10)

    # Random baseline should produce some wrong predictions
    assert len(report.confusion_matrix) > 0
    for entry in report.confusion_matrix:
        assert entry.query_class != entry.predicted_class
        assert entry.count > 0


def test_remediation_grading(small_corpus, query_set_and_incidents, remediation_questions):
    qs, qi = query_set_and_incidents
    adapter = RandomBaseline(seed=1)
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)

    # Simulate random remediation answers
    from rhyme_bench.harness import run_remediation
    rem_results = run_remediation(adapter, payloads, results, small_corpus, remediation_questions)

    report = score(
        results, qs, small_corpus, k=10,
        remediation_results=rem_results,
        remediation_questions=remediation_questions,
    )

    assert report.remediation_grades is not None
    assert report.remediation_grades.total > 0
    grades = report.remediation_grades.as_dict()
    assert abs(sum(grades.values()) - 1.0) < 0.01


def test_no_remediation_when_not_provided(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    adapter = BM25Baseline()
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)
    report = score(results, qs, small_corpus, k=10)
    assert report.remediation_grades is None


def test_efficiency_none_without_token_usage(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    adapter = BM25Baseline()
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)
    report = score(results, qs, small_corpus, k=10)
    assert report.efficiency is None


def test_score_report_summary_string(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    adapter = BM25Baseline()
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)
    report = score(results, qs, small_corpus, k=10)
    summary = report.summary()
    assert "Rhyme Score Report" in summary
    assert "Per-tier" in summary
    assert "Macro-average" in summary


def test_calibration_ece_bounded(small_corpus, query_set_and_incidents):
    qs, qi = query_set_and_incidents
    adapter = BM25Baseline()
    payloads = [q.payload for q in qi]
    results = run_retrieval(adapter, payloads, small_corpus, k=10)
    report = score(results, qs, small_corpus, k=10)
    assert 0.0 <= report.overall_calibration_ece <= 1.0
