"""Tests for the evaluation harness."""

from sifter_bench.baselines import BM25Baseline, RandomBaseline, TfidfBaseline
from sifter_bench.harness import (
    Adapter,
    RetrieveOutput,
    run_reasoning_only,
    run_remediation,
    run_retrieval,
)
from sifter_bench.models import (
    IncidentPayload,
    RankedMatch,
    RemediationChoice,
    TokenUsage,
)


def test_run_retrieval_excludes_self(small_corpus):
    adapter = BM25Baseline()
    payloads = small_corpus.payloads()[:3]
    results = run_retrieval(adapter, payloads, small_corpus, k=5)
    assert len(results) == 3
    for query_payload, result in zip(payloads, results):
        match_ids = {m.incident_id for m in result.ranked_matches}
        assert query_payload.incident_id not in match_ids


def test_run_retrieval_respects_k(small_corpus):
    adapter = BM25Baseline()
    payloads = small_corpus.payloads()[:2]
    results = run_retrieval(adapter, payloads, small_corpus, k=3)
    for r in results:
        assert len(r.ranked_matches) <= 3


def test_run_reasoning_only_narrows_candidates(small_corpus):
    adapter = RandomBaseline(seed=1)
    payloads = small_corpus.payloads()[:2]
    results = run_reasoning_only(adapter, payloads, small_corpus, pre_retrieve_k=20, k=5)
    assert len(results) == 2
    for r in results:
        assert len(r.ranked_matches) <= 5


def test_run_remediation(small_corpus, query_set_and_incidents, remediation_questions):
    qs, qi = query_set_and_incidents
    adapter = RandomBaseline(seed=1)
    payloads = [q.payload for q in qi]
    retrieval = run_retrieval(adapter, payloads, small_corpus, k=5)
    rem_results = run_remediation(adapter, payloads, retrieval, small_corpus, remediation_questions)
    assert len(rem_results) > 0
    for r in rem_results:
        assert r.selected_label in {"A", "B", "C", "D", "E"}


class TokenTrackingAdapter(Adapter):
    """Test adapter that returns token usage."""

    def retrieve(self, query, corpus, k=10):
        matches = [
            RankedMatch(incident_id=corpus[0].incident_id, confidence=0.5)
        ]
        return RetrieveOutput(
            matches=matches,
            token_usage=TokenUsage(input_tokens=1000, output_tokens=200),
        )


def test_token_usage_propagated(small_corpus):
    adapter = TokenTrackingAdapter()
    payloads = small_corpus.payloads()[:1]
    results = run_retrieval(adapter, payloads, small_corpus, k=5)
    assert results[0].token_usage is not None
    assert results[0].token_usage.input_tokens == 1000
    assert results[0].token_usage.output_tokens == 200


def test_plain_list_return_still_works(small_corpus):
    """Adapters returning plain list[RankedMatch] should still work."""
    adapter = BM25Baseline()
    payloads = small_corpus.payloads()[:1]
    results = run_retrieval(adapter, payloads, small_corpus, k=5)
    assert results[0].token_usage is None
    assert len(results[0].ranked_matches) > 0
