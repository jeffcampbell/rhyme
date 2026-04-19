"""Integration test: full generate → run → score pipeline."""

from rhyme_bench.baselines import BM25Baseline, RandomBaseline
from rhyme_bench.generator import generate_corpus
from rhyme_bench.harness import run_reasoning_only, run_remediation, run_retrieval
from rhyme_bench.queries import build_query_set, build_remediation_questions
from rhyme_bench.scorer import score
from rhyme_bench.taxonomy import CauseClass, ConfusabilityTier


def test_full_pipeline():
    """Generate a small corpus, run baselines, score, and assert results make sense."""
    # Generate
    corpus = generate_corpus(incidents_per_class=5, seed=77)
    assert len(corpus.incidents) == 5 * len(CauseClass)

    # All classes present
    classes = {inc.labels.cause_class for inc in corpus.incidents}
    assert classes == set(CauseClass)

    # Build query set
    qs, qi = build_query_set(corpus, total_queries=20, seed=77)
    assert len(qs.queries) == 20
    payloads = [q.payload for q in qi]

    # Build remediation questions
    rem_qs = build_remediation_questions(qi)
    assert len(rem_qs.questions) == 20

    # Run BM25 (Task 1)
    bm25 = BM25Baseline()
    bm25_results = run_retrieval(bm25, payloads, corpus, k=10)
    assert len(bm25_results) == 20

    # Run Random (Task 1)
    random_adapter = RandomBaseline(seed=77)
    random_results = run_retrieval(random_adapter, payloads, corpus, k=10)

    # Score both
    bm25_report = score(bm25_results, qs, corpus, k=10)
    random_report = score(random_results, qs, corpus, k=10)

    # BM25 should beat random
    assert bm25_report.overall_precision_at_k_raw > random_report.overall_precision_at_k_raw
    assert bm25_report.overall_retrieval_at_k >= random_report.overall_retrieval_at_k

    # All metrics should be bounded
    assert 0.0 <= bm25_report.overall_calibration_ece <= 1.0
    assert bm25_report.overall_precision_at_k_lift >= 1.0  # BM25 should beat base rate

    # Per-tier should exist
    tiers_with_data = [t for t, m in bm25_report.per_tier.items() if m.num_queries > 0]
    assert len(tiers_with_data) >= 1

    # Task 2: reasoning-only
    bm25_t2 = run_reasoning_only(bm25, payloads, corpus, k=10)
    assert len(bm25_t2) == 20
    bm25_t2_report = score(bm25_t2, qs, corpus, k=10)
    assert bm25_t2_report.overall_retrieval_at_k == 1.0  # BM25 re-ranking BM25 pre-retrieval

    # Task 3: remediation
    rem_results = run_remediation(random_adapter, payloads, random_results, corpus, rem_qs)
    report_with_rem = score(
        random_results, qs, corpus, k=10,
        remediation_results=rem_results,
        remediation_questions=rem_qs,
    )
    assert report_with_rem.remediation_grades is not None
    assert report_with_rem.remediation_grades.total > 0
    grades = report_with_rem.remediation_grades.as_dict()
    assert abs(sum(grades.values()) - 1.0) < 0.01

    # Summary string should be well-formed
    summary = bm25_report.summary()
    assert "Rhyme" in summary
    assert "Per-tier" in summary


def test_pipeline_deterministic():
    """Same seed should produce identical scores."""
    def run_once(seed):
        corpus = generate_corpus(incidents_per_class=3, seed=seed)
        qs, qi = build_query_set(corpus, total_queries=10, seed=seed)
        payloads = [q.payload for q in qi]
        results = run_retrieval(BM25Baseline(), payloads, corpus, k=5)
        report = score(results, qs, corpus, k=5)
        return report.overall_precision_at_k_raw

    score_a = run_once(42)
    score_b = run_once(42)
    assert score_a == score_b
